"""Tests for the high-level bampy.agent.Agent."""

from __future__ import annotations

import asyncio

from bampy.agent import Agent, AgentEvent
from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import AssistantMessage, DoneEvent, ErrorEvent, Model, StopReason, TextContent, UserMessage


def create_model() -> Model:
    return Model(
        id="mock-model",
        name="Mock Model",
        api="mock-api",
        provider="mock",
    )


def create_assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        api="mock-api",
        provider="mock",
        model="mock-model",
        content=[TextContent(text=text)],
    )


def done_stream(message: AssistantMessage) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason=message.stop_reason, message=message))
    return stream


class TestAgent:
    def test_default_state_and_mutators(self):
        agent = Agent(initial_state={"model": create_model()})

        assert agent.state.system_prompt == ""
        assert agent.state.thinking_level == "off"
        assert agent.state.tools == []
        assert agent.state.messages == []
        assert agent.state.is_streaming is False
        assert agent.state.stream_message is None
        assert agent.state.pending_tool_calls == set()
        assert agent.state.error is None

        agent.set_system_prompt("You are helpful.")
        agent.set_model(create_model())
        agent.set_thinking_level("high")
        agent.set_tools([])
        agent.replace_messages([UserMessage(content="hello")])

        assert agent.state.system_prompt == "You are helpful."
        assert agent.state.thinking_level == "high"
        assert len(agent.state.messages) == 1

    async def test_prompt_updates_state_and_subscribers(self):
        events: list[AgentEvent] = []
        agent = Agent(
            initial_state={"model": create_model()},
            stream_fn=lambda model, context, options: done_stream(
                create_assistant_message("Processed")
            ),
        )
        agent.subscribe(events.append)

        await agent.prompt("Hello")

        assert [message.role for message in agent.state.messages] == ["user", "assistant"]
        assert agent.state.messages[-1].content[0].text == "Processed"
        assert agent.state.is_streaming is False
        assert agent.state.error is None
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

    async def test_continue_processes_follow_up_from_assistant_tail(self):
        agent = Agent(
            initial_state={
                "model": create_model(),
                "messages": [
                    UserMessage(content="Initial"),
                    create_assistant_message("Initial response"),
                ],
            },
            stream_fn=lambda model, context, options: done_stream(
                create_assistant_message("Processed")
            ),
        )

        agent.follow_up(UserMessage(content="Queued follow-up"))
        await agent.continue_()

        recent_roles = [message.role for message in agent.state.messages[-4:]]
        assert recent_roles == ["user", "assistant", "user", "assistant"]
        assert agent.state.messages[-2].content == "Queued follow-up"
        assert agent.state.messages[-1].content[0].text == "Processed"

    async def test_continue_keeps_one_at_a_time_steering_semantics(self):
        response_count = {"value": 0}

        def stream_fn(model, context, options):
            response_count["value"] += 1
            return done_stream(create_assistant_message(f"Processed {response_count['value']}"))

        agent = Agent(
            initial_state={
                "model": create_model(),
                "messages": [
                    UserMessage(content="Initial"),
                    create_assistant_message("Initial response"),
                ],
            },
            stream_fn=stream_fn,
        )

        agent.steer(UserMessage(content="Steering 1"))
        agent.steer(UserMessage(content="Steering 2"))

        await agent.continue_()

        recent_roles = [message.role for message in agent.state.messages[-4:]]
        assert recent_roles == ["user", "assistant", "user", "assistant"]
        assert response_count["value"] == 2

    async def test_prompt_handles_stream_function_failure(self):
        agent = Agent(
            initial_state={"model": create_model()},
            stream_fn=lambda model, context, options: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        await agent.prompt("Hello")

        assert agent.state.error == "boom"
        assert agent.state.messages[-1].role == "assistant"
        assert agent.state.messages[-1].error_message == "boom"
        assert agent.state.messages[-1].stop_reason == StopReason.ERROR

    async def test_abort_propagates_cancellation_and_wait_for_idle(self):
        def stream_fn(model, context, options):
            stream = AssistantMessageEventStream()
            output = AssistantMessage(
                api=model.api,
                provider=model.provider,
                model=model.id,
                content=[],
            )

            async def runner():
                while not options.cancellation.cancelled:
                    await asyncio.sleep(0.01)
                output.stop_reason = StopReason.ABORTED
                output.error_message = options.cancellation.reason
                stream.push(ErrorEvent(reason=StopReason.ABORTED, error=output))

            asyncio.create_task(runner())
            return stream

        agent = Agent(
            initial_state={"model": create_model()},
            stream_fn=stream_fn,
        )

        prompt_task = asyncio.create_task(agent.prompt("Hello"))
        await asyncio.sleep(0.02)
        wait_task = asyncio.create_task(agent.wait_for_idle())
        agent.abort("user aborted")

        await prompt_task
        await wait_task

        assert agent.state.is_streaming is False
        assert agent.state.error == "user aborted"
        assert agent.state.messages[-1].stop_reason == StopReason.ABORTED

    async def test_runtime_failure_from_max_turns_surfaces_as_error_message(self):
        call_index = {"value": 0}

        def stream_fn(model, context, options):
            if call_index["value"] == 0:
                message = AssistantMessage(
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    content=[{
                        "type": "tool_call",
                        "id": "tool-1",
                        "name": "missing",
                        "arguments": {},
                    }],
                    stop_reason=StopReason.TOOL_USE,
                )
            else:
                message = create_assistant_message("should not happen")
            call_index["value"] += 1
            return done_stream(message)

        agent = Agent(
            initial_state={"model": create_model()},
            stream_fn=stream_fn,
            max_turns=1,
        )

        await agent.prompt("Hello")

        assert agent.state.error == "Agent loop exceeded max_turns=1"
        assert agent.state.messages[-1].stop_reason == StopReason.ERROR
