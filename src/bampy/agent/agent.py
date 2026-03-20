"""High-level stateful agent built on the layer2 runtime."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from bampy.ai import ImageContent, SimpleStreamOptions, TextContent, get_model
from bampy.ai.types import AssistantMessage, Model, StopReason, UserMessage
from bampy.agent.cancellation import CancellationError, CancellationToken
from bampy.agent.loop import build_terminal_assistant_message, run_agent_loop, run_agent_loop_continue
from bampy.agent.messages import default_convert_to_llm, is_assistant_message
from bampy.agent.types import (
    AfterToolCallHook,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentThinkingLevel,
    AgentTool,
    ApiKeyResolverFn,
    BeforeToolCallHook,
    ConvertToLlmFn,
    StreamFn,
    TransformContextFn,
    ToolExecutionMode,
)


def _default_model():
    return get_model("google", "gemini-2.5-flash-lite")


class Agent:
    """Stateful async-first agent runtime."""

    def __init__(
        self,
        *,
        initial_state: AgentState | Mapping[str, Any] | None = None,
        convert_to_llm: ConvertToLlmFn | None = None,
        transform_context: TransformContextFn | None = None,
        steering_mode: str = "one-at-a-time",
        follow_up_mode: str = "one-at-a-time",
        stream_fn: StreamFn | None = None,
        stream_options: SimpleStreamOptions | None = None,
        get_api_key: ApiKeyResolverFn | None = None,
        tool_execution: ToolExecutionMode | str = ToolExecutionMode.PARALLEL,
        before_tool_call: BeforeToolCallHook | None = None,
        after_tool_call: AfterToolCallHook | None = None,
        max_turns: int = 50,
    ) -> None:
        self._state = self._build_initial_state(initial_state)
        self._listeners: set[Callable[[AgentEvent], None]] = set()
        self._convert_to_llm = convert_to_llm or default_convert_to_llm
        self._transform_context = transform_context
        self._steering_queue: deque[AgentMessage] = deque()
        self._follow_up_queue: deque[AgentMessage] = deque()
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode
        self.stream_fn = stream_fn
        self._stream_options = stream_options.model_copy(deep=True) if stream_options else SimpleStreamOptions()
        self._get_api_key = get_api_key
        self._tool_execution = ToolExecutionMode(tool_execution)
        self._before_tool_call = before_tool_call
        self._after_tool_call = after_tool_call
        self._max_turns = 50
        self.set_max_turns(max_turns)
        self._cancellation: CancellationToken | None = None
        self._idle_waiter: asyncio.Future[None] | None = None

    @staticmethod
    def _build_initial_state(
        initial_state: AgentState | Mapping[str, Any] | None,
    ) -> AgentState:
        if initial_state is None:
            return AgentState(system_prompt="", model=_default_model())
        if isinstance(initial_state, AgentState):
            return deepcopy(initial_state)

        data = dict(initial_state)
        return AgentState(
            system_prompt=str(data.get("system_prompt", "")),
            model=data.get("model", _default_model()),
            thinking_level=AgentThinkingLevel(data.get("thinking_level", AgentThinkingLevel.OFF)),
            tools=list(data.get("tools", [])),
            messages=list(data.get("messages", [])),
            is_streaming=bool(data.get("is_streaming", False)),
            stream_message=data.get("stream_message"),
            pending_tool_calls=set(data.get("pending_tool_calls", set())),
            error=data.get("error"),
        )

    @property
    def state(self) -> AgentState:
        return self._state

    def subscribe(self, listener: Callable[[AgentEvent], None]):
        self._listeners.add(listener)

        def unsubscribe() -> None:
            self._listeners.discard(listener)

        return unsubscribe

    def set_system_prompt(self, value: str) -> None:
        self._state.system_prompt = value

    def set_model(self, model: Model) -> None:
        self._state.model = model

    def set_thinking_level(self, level: AgentThinkingLevel | str) -> None:
        self._state.thinking_level = AgentThinkingLevel(level)

    def set_tools(self, tools: list[AgentTool]) -> None:
        self._state.tools = list(tools)

    def set_stream_options(self, options: SimpleStreamOptions) -> None:
        self._stream_options = options.model_copy(deep=True)

    def set_tool_execution(self, value: ToolExecutionMode | str) -> None:
        self._tool_execution = ToolExecutionMode(value)

    def set_before_tool_call(self, hook: BeforeToolCallHook | None) -> None:
        self._before_tool_call = hook

    def set_after_tool_call(self, hook: AfterToolCallHook | None) -> None:
        self._after_tool_call = hook

    def set_max_turns(self, value: int) -> None:
        if value < 1:
            raise ValueError("max_turns must be at least 1")
        self._max_turns = value

    def set_steering_mode(self, mode: str) -> None:
        self._steering_mode = mode

    def get_steering_mode(self) -> str:
        return self._steering_mode

    def set_follow_up_mode(self, mode: str) -> None:
        self._follow_up_mode = mode

    def get_follow_up_mode(self) -> str:
        return self._follow_up_mode

    def replace_messages(self, messages: list[AgentMessage]) -> None:
        self._state.messages = list(messages)

    def append_message(self, message: AgentMessage) -> None:
        self._state.messages = [*self._state.messages, message]

    def clear_messages(self) -> None:
        self._state.messages = []

    def steer(self, message: AgentMessage) -> None:
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage) -> None:
        self._follow_up_queue.append(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        return bool(self._steering_queue or self._follow_up_queue)

    def abort(self, reason: str | None = None) -> None:
        if self._cancellation is not None:
            self._cancellation.cancel(reason or "Operation cancelled")

    async def wait_for_idle(self) -> None:
        if self._idle_waiter is not None:
            await self._idle_waiter

    def reset(self) -> None:
        self._state.messages = []
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None
        self.clear_all_queues()

    async def prompt(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, or wait for completion."
            )

        messages = self._normalize_prompt_input(input, images)
        await self._run_loop(messages)

    async def continue_(self) -> None:
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing. Wait for completion before continuing."
            )

        if not self._state.messages:
            raise RuntimeError("No messages to continue from")

        if is_assistant_message(self._state.messages[-1]):
            queued_steering = self._dequeue_steering_messages()
            if queued_steering:
                await self._run_loop(queued_steering, skip_initial_steering_poll=True)
                return

            queued_follow_up = self._dequeue_follow_up_messages()
            if queued_follow_up:
                await self._run_loop(queued_follow_up)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_loop(None)

    resume = continue_

    def _normalize_prompt_input(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None,
    ) -> list[AgentMessage]:
        if isinstance(input, list):
            return list(input)
        if isinstance(input, str):
            if images:
                return [
                    UserMessage(
                        content=[TextContent(text=input), *images],
                    )
                ]
            return [UserMessage(content=input)]
        return [input]

    def _dequeue_steering_messages(self) -> list[AgentMessage]:
        if self._steering_mode == "one-at-a-time":
            if not self._steering_queue:
                return []
            return [self._steering_queue.popleft()]
        messages = list(self._steering_queue)
        self._steering_queue.clear()
        return messages

    def _dequeue_follow_up_messages(self) -> list[AgentMessage]:
        if self._follow_up_mode == "one-at-a-time":
            if not self._follow_up_queue:
                return []
            return [self._follow_up_queue.popleft()]
        messages = list(self._follow_up_queue)
        self._follow_up_queue.clear()
        return messages

    async def _run_loop(
        self,
        messages: list[AgentMessage] | None,
        *,
        skip_initial_steering_poll: bool = False,
    ) -> None:
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None
        self._cancellation = CancellationToken()
        loop = asyncio.get_running_loop()
        self._idle_waiter = loop.create_future()

        skip_poll = skip_initial_steering_poll

        async def _get_steering_messages() -> list[AgentMessage]:
            nonlocal skip_poll
            if skip_poll:
                skip_poll = False
                return []
            return self._dequeue_steering_messages()

        config = AgentLoopConfig(
            model=self._state.model,
            convert_to_llm=self._convert_to_llm,
            stream_options=self._stream_options.model_copy(
                update={
                    "reasoning": self._state.thinking_level.to_ai_reasoning(),
                },
                deep=True,
            ),
            transform_context=self._transform_context,
            get_api_key=self._get_api_key,
            get_steering_messages=_get_steering_messages,
            get_follow_up_messages=self._get_follow_up_messages,
            tool_execution=self._tool_execution,
            before_tool_call=self._before_tool_call,
            after_tool_call=self._after_tool_call,
            max_turns=self._max_turns,
        )

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

        try:
            if messages is not None:
                await run_agent_loop(
                    prompts=messages,
                    context=context,
                    config=config,
                    emit=self._process_loop_event,
                    cancellation=self._cancellation,
                    stream_fn=self.stream_fn,
                )
            else:
                await run_agent_loop_continue(
                    context=context,
                    config=config,
                    emit=self._process_loop_event,
                    cancellation=self._cancellation,
                    stream_fn=self.stream_fn,
                )
        except CancellationError as exc:
            self._handle_runtime_failure(
                stop_reason=StopReason.ABORTED,
                error_message=str(exc),
            )
        except Exception as exc:
            self._handle_runtime_failure(
                stop_reason=StopReason.ERROR,
                error_message=str(exc),
            )
        finally:
            self._state.is_streaming = False
            self._state.stream_message = None
            self._state.pending_tool_calls = set()
            self._cancellation = None
            if self._idle_waiter is not None and not self._idle_waiter.done():
                self._idle_waiter.set_result(None)

    def _handle_runtime_failure(
        self,
        *,
        stop_reason: StopReason,
        error_message: str,
    ) -> None:
        error_message_obj = build_terminal_assistant_message(
            model=self._state.model,
            stop_reason=stop_reason,
            error_message=error_message,
        )
        self.append_message(error_message_obj)
        self._state.error = error_message
        self._emit(AgentEndEvent(messages=[error_message_obj]))

    async def _get_steering_messages(self) -> list[AgentMessage]:
        return self._dequeue_steering_messages()

    async def _get_follow_up_messages(self) -> list[AgentMessage]:
        return self._dequeue_follow_up_messages()

    def _process_loop_event(self, event: AgentEvent) -> None:
        if event.type == "message_start":
            self._state.stream_message = event.message
        elif event.type == "message_update":
            self._state.stream_message = event.message
        elif event.type == "message_end":
            self._state.stream_message = None
            self.append_message(event.message)
        elif event.type == "tool_execution_start":
            pending = set(self._state.pending_tool_calls)
            pending.add(event.tool_call_id)
            self._state.pending_tool_calls = pending
        elif event.type == "tool_execution_end":
            pending = set(self._state.pending_tool_calls)
            pending.discard(event.tool_call_id)
            self._state.pending_tool_calls = pending
        elif event.type == "turn_end":
            if isinstance(event.message, AssistantMessage) and event.message.error_message:
                self._state.error = event.message.error_message
        elif event.type == "agent_end":
            self._state.is_streaming = False
            self._state.stream_message = None

        self._emit(event)

    def _emit(self, event: AgentEvent) -> None:
        for listener in tuple(self._listeners):
            listener(event)
