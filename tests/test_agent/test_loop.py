"""Tests for the low-level bampy.agent loop."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import pytest

from bampy.agent import (
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentToolResult,
    BeforeToolCallResult,
    ToolExecutionMode,
    agent_loop,
    agent_loop_continue,
    clear_message_converters,
    default_convert_to_llm,
    register_message_converter,
)
from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    Model,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
    },
    "required": ["value"],
}


def create_model() -> Model:
    return Model(
        id="mock-model",
        name="Mock Model",
        api="mock-api",
        provider="mock",
    )


def create_user_message(text: str) -> UserMessage:
    return UserMessage(content=text)


def create_assistant_message(
    content: list,
    stop_reason: StopReason = StopReason.STOP,
) -> AssistantMessage:
    return AssistantMessage(
        api="mock-api",
        provider="mock",
        model="mock-model",
        content=content,
        stop_reason=stop_reason,
    )


def done_stream(message: AssistantMessage) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    asyncio.get_running_loop().call_soon(
        stream.push,
        DoneEvent(reason=message.stop_reason, message=message),
    )
    return stream


@pytest.fixture(autouse=True)
def _clear_message_registry():
    clear_message_converters()
    yield
    clear_message_converters()


class TestAgentLoop:
    async def test_emits_expected_events_for_basic_prompt(self):
        context = AgentContext(system_prompt="You are helpful.")
        config = AgentLoopConfig(
            model=create_model(),
            convert_to_llm=default_convert_to_llm,
        )

        events = []
        stream = agent_loop(
            [create_user_message("Hello")],
            context,
            config,
            stream_fn=lambda model, ctx, options: done_stream(
                create_assistant_message([TextContent(text="Hi there!")])
            ),
        )

        async for event in stream:
            events.append(event)

        messages = await stream.result()
        assert [message.role for message in messages] == ["user", "assistant"]
        assert [event.type for event in events] == [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]

    async def test_applies_transform_context_before_convert_to_llm(self):
        transformed_messages = []
        converted_messages = []

        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[
                create_user_message("old-1"),
                create_assistant_message([TextContent(text="old-response-1")]),
                create_user_message("old-2"),
                create_assistant_message([TextContent(text="old-response-2")]),
            ],
        )
        config = AgentLoopConfig(
            model=create_model(),
            convert_to_llm=lambda messages: converted_messages.extend(messages) or default_convert_to_llm(messages),
            transform_context=lambda messages, cancellation: transformed_messages.extend(messages[-2:]) or messages[-2:],
        )

        stream = agent_loop(
            [create_user_message("new")],
            context,
            config,
            stream_fn=lambda model, ctx, options: done_stream(
                create_assistant_message([TextContent(text="done")])
            ),
        )

        async for _ in stream:
            pass

        assert len(transformed_messages) == 2
        assert len(converted_messages) == 2

    async def test_executes_tools_in_parallel_and_preserves_source_order(self):
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        observed_parallel = {"value": False}
        executed: list[str] = []

        @dataclass
        class EchoTool:
            name: str = "echo"
            label: str = "Echo"
            description: str = "Echo tool"
            parameters: dict = field(default_factory=lambda: TOOL_SCHEMA)

            async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
                executed.append(params["value"])
                if params["value"] == "first":
                    first_started.set()
                    await release_first.wait()
                else:
                    await first_started.wait()
                    if not release_first.is_set():
                        observed_parallel["value"] = True
                return AgentToolResult(
                    content=[TextContent(text=f"echo:{params['value']}")],
                    details={"value": params["value"]},
                )

        context = AgentContext(tools=[EchoTool()])
        config = AgentLoopConfig(
            model=create_model(),
            convert_to_llm=default_convert_to_llm,
            tool_execution=ToolExecutionMode.PARALLEL,
        )

        call_index = 0

        def stream_fn(model: Model, llm_context: Context, options):
            nonlocal call_index
            if call_index == 0:
                asyncio.get_running_loop().call_later(0.02, release_first.set)
                message = create_assistant_message(
                    [
                        ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                        ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                    ],
                    stop_reason=StopReason.TOOL_USE,
                )
            else:
                message = create_assistant_message([TextContent(text="done")])
            call_index += 1
            return done_stream(message)

        events = []
        stream = agent_loop(
            [create_user_message("echo both")],
            context,
            config,
            stream_fn=stream_fn,
        )

        async for event in stream:
            events.append(event)

        tool_result_ids = [
            event.message.tool_call_id
            for event in events
            if event.type == "message_end" and isinstance(event.message, ToolResultMessage)
        ]

        assert executed == ["first", "second"]
        assert observed_parallel["value"] is True
        assert tool_result_ids == ["tool-1", "tool-2"]

    async def test_before_and_after_tool_hooks(self):
        @dataclass
        class EchoTool:
            name: str = "echo"
            label: str = "Echo"
            description: str = "Echo tool"
            parameters: dict = field(default_factory=lambda: TOOL_SCHEMA)

            async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
                return AgentToolResult(
                    content=[TextContent(text=f"raw:{params['value']}")],
                    details={"value": params["value"]},
                )

        context = AgentContext(tools=[EchoTool()])
        config = AgentLoopConfig(
            model=create_model(),
            convert_to_llm=default_convert_to_llm,
            before_tool_call=lambda payload, cancellation: BeforeToolCallResult(
                block=payload.args["value"] == "blocked",
                reason="blocked by policy",
            ),
            after_tool_call=lambda payload, cancellation: AfterToolCallResult(
                content=[TextContent(text="audited")],
                details={"audited": True},
            ),
        )

        call_index = 0

        def stream_fn(model: Model, llm_context: Context, options):
            nonlocal call_index
            if call_index == 0:
                message = create_assistant_message(
                    [
                        ToolCall(id="tool-1", name="echo", arguments={"value": "blocked"}),
                        ToolCall(id="tool-2", name="echo", arguments={"value": "ok"}),
                    ],
                    stop_reason=StopReason.TOOL_USE,
                )
            else:
                message = create_assistant_message([TextContent(text="done")])
            call_index += 1
            return done_stream(message)

        stream = agent_loop(
            [create_user_message("run hooks")],
            context,
            config,
            stream_fn=stream_fn,
        )

        async for _ in stream:
            pass

        messages = await stream.result()
        tool_results = [message for message in messages if isinstance(message, ToolResultMessage)]

        assert tool_results[0].is_error is True
        assert tool_results[0].content[0].text == "blocked by policy"
        assert tool_results[1].is_error is False
        assert tool_results[1].content[0].text == "audited"
        assert tool_results[1].details == {"audited": True}

    async def test_stops_when_max_turns_is_exceeded(self):
        @dataclass
        class EchoTool:
            name: str = "echo"
            label: str = "Echo"
            description: str = "Echo tool"
            parameters: dict = field(default_factory=lambda: TOOL_SCHEMA)

            async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
                return AgentToolResult(
                    content=[TextContent(text="ok")],
                    details={"value": params["value"]},
                )

        context = AgentContext(tools=[EchoTool()])
        config = AgentLoopConfig(
            model=create_model(),
            convert_to_llm=default_convert_to_llm,
            max_turns=1,
        )

        call_index = 0

        def stream_fn(model: Model, llm_context: Context, options):
            nonlocal call_index
            if call_index == 0:
                message = create_assistant_message(
                    [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})],
                    stop_reason=StopReason.TOOL_USE,
                )
            else:
                message = create_assistant_message([TextContent(text="should not happen")])
            call_index += 1
            return done_stream(message)

        stream = agent_loop(
            [create_user_message("loop forever")],
            context,
            config,
            stream_fn=stream_fn,
        )

        async for _ in stream:
            pass

        with pytest.raises(RuntimeError, match="max_turns=1"):
            await stream.result()


class TestAgentLoopContinue:
    def test_rejects_empty_context(self):
        config = AgentLoopConfig(
            model=create_model(),
            convert_to_llm=default_convert_to_llm,
        )
        with pytest.raises(ValueError, match="Cannot continue: no messages in context"):
            agent_loop_continue(AgentContext(), config)

    async def test_supports_custom_message_converter_registry(self):
        @dataclass
        class CustomMessage:
            role: str
            text: str
            timestamp: float

        register_message_converter(
            "custom",
            lambda message: [UserMessage(content=message.text, timestamp=message.timestamp)],
        )

        context = AgentContext(
            messages=[
                CustomMessage(
                    role="custom",
                    text="hello from custom",
                    timestamp=time.time() * 1000,
                )
            ]
        )
        config = AgentLoopConfig(
            model=create_model(),
            convert_to_llm=default_convert_to_llm,
        )

        stream = agent_loop_continue(
            context,
            config,
            stream_fn=lambda model, llm_context, options: done_stream(
                create_assistant_message([TextContent(text="response")])
            ),
        )

        async for _ in stream:
            pass

        messages = await stream.result()
        assert len(messages) == 1
        assert isinstance(messages[0], AssistantMessage)
        assert messages[0].content[0].text == "response"
