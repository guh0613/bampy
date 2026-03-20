"""Low-level agent loop implementation."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from bampy.ai import AssistantMessageEventStream
from bampy.ai.stream import EventStream, stream_simple
from bampy.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from bampy.ai.validation import ToolValidationError, validate_tool_arguments
from bampy.agent.cancellation import CancellationError, CancellationToken
from bampy.agent.messages import clone_message
from bampy.agent.types import (
    AfterToolCallContext,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionMode,
    TurnEndEvent,
    TurnStartEvent,
    _UNSET,
    to_ai_tool,
    tool_schema,
)

AgentEventSink = Callable[[AgentEvent], Awaitable[None] | None]


class AgentEventStream(EventStream[AgentEvent, list[AgentMessage]]):
    """Event stream specialized for agent runtime events."""

    def __init__(self) -> None:
        super().__init__(
            is_complete=lambda event: isinstance(event, AgentEndEvent),
            extract_result=lambda event: event.messages if isinstance(event, AgentEndEvent) else [],
        )


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    cancellation: CancellationToken | None = None,
    stream_fn: StreamFn | None = None,
) -> AgentEventStream:
    """Start an agent loop with one or more prompt messages."""
    stream = AgentEventStream()

    async def _run() -> None:
        try:
            messages = await run_agent_loop(
                prompts,
                context,
                config,
                emit=stream.push,
                cancellation=cancellation,
                stream_fn=stream_fn,
            )
        except BaseException as exc:  # pragma: no cover - defensive stream bridge
            stream.error(exc)
            return
        stream.end(messages)

    stream._task = asyncio.create_task(_run())  # type: ignore[attr-defined]
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancellation: CancellationToken | None = None,
    stream_fn: StreamFn | None = None,
) -> AgentEventStream:
    """Continue an existing agent loop without adding a new prompt."""
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if _message_role(context.messages[-1]) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = AgentEventStream()

    async def _run() -> None:
        try:
            messages = await run_agent_loop_continue(
                context,
                config,
                emit=stream.push,
                cancellation=cancellation,
                stream_fn=stream_fn,
            )
        except BaseException as exc:  # pragma: no cover - defensive stream bridge
            stream.error(exc)
            return
        stream.end(messages)

    stream._task = asyncio.create_task(_run())  # type: ignore[attr-defined]
    return stream


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancellation: CancellationToken | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Run the full loop for new prompt messages."""
    new_messages = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=list(context.tools),
    )

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())
    for prompt in prompts:
        await _emit(emit, MessageStartEvent(message=clone_message(prompt)))
        await _emit(emit, MessageEndEvent(message=clone_message(prompt)))

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        config=config,
        cancellation=cancellation,
        emit=emit,
        stream_fn=stream_fn,
    )
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancellation: CancellationToken | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Continue the loop from existing context."""
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if _message_role(context.messages[-1]) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=list(context.tools),
    )

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        config=config,
        cancellation=cancellation,
        emit=emit,
        stream_fn=stream_fn,
    )
    return new_messages


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
    stream_fn: StreamFn | None = None,
) -> None:
    first_turn = True
    turn_count = 0
    pending_messages = await _get_optional_messages(config.get_steering_messages)

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            _raise_if_cancelled(cancellation)

            if first_turn:
                first_turn = False
            else:
                await _emit(emit, TurnStartEvent())

            if pending_messages:
                for message in pending_messages:
                    await _emit(emit, MessageStartEvent(message=clone_message(message)))
                    await _emit(emit, MessageEndEvent(message=clone_message(message)))
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            turn_count += 1
            if turn_count > config.max_turns:
                raise RuntimeError(
                    f"Agent loop exceeded max_turns={config.max_turns}"
                )

            assistant_message = await _stream_assistant_response(
                context=current_context,
                config=config,
                cancellation=cancellation,
                emit=emit,
                stream_fn=stream_fn,
            )
            new_messages.append(assistant_message)

            if assistant_message.stop_reason in (StopReason.ERROR, StopReason.ABORTED):
                await _emit(
                    emit,
                    TurnEndEvent(
                        message=clone_message(assistant_message),
                        tool_results=[],
                    ),
                )
                await _emit(emit, AgentEndEvent(messages=[clone_message(m) for m in new_messages]))
                return

            tool_calls = [
                content
                for content in assistant_message.content
                if isinstance(content, ToolCall)
            ]
            has_more_tool_calls = bool(tool_calls)

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                tool_results = await _execute_tool_calls(
                    current_context=current_context,
                    assistant_message=assistant_message,
                    tool_calls=tool_calls,
                    config=config,
                    cancellation=cancellation,
                    emit=emit,
                )
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _emit(
                emit,
                TurnEndEvent(
                    message=clone_message(assistant_message),
                    tool_results=[result.model_copy(deep=True) for result in tool_results],
                ),
            )

            pending_messages = await _get_optional_messages(config.get_steering_messages)

        follow_up_messages = await _get_optional_messages(config.get_follow_up_messages)
        if follow_up_messages:
            pending_messages = follow_up_messages
            continue
        break

    await _emit(emit, AgentEndEvent(messages=[clone_message(m) for m in new_messages]))


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
    stream_fn: StreamFn | None = None,
) -> AssistantMessage:
    _raise_if_cancelled(cancellation)

    messages = context.messages
    if config.transform_context is not None:
        transformed = await _maybe_await(config.transform_context(messages, cancellation))
        messages = transformed if transformed is not None else messages

    llm_messages = await _maybe_await(config.convert_to_llm(messages))

    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=list(llm_messages),
        tools=[to_ai_tool(tool) for tool in context.tools] or None,
    )

    resolved_api_key = config.stream_options.api_key
    if config.get_api_key is not None:
        dynamic_api_key = await _maybe_await(config.get_api_key(config.model.provider))
        if dynamic_api_key:
            resolved_api_key = dynamic_api_key

    stream_options = config.stream_options.model_copy(deep=True)
    if resolved_api_key is not None:
        stream_options.api_key = resolved_api_key
    stream_options.cancellation = cancellation

    stream_callable = stream_fn or stream_simple
    response = await _resolve_stream(
        stream_callable(config.model, llm_context, stream_options)
    )

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        _raise_if_cancelled(cancellation)

        match event:
            case _ if event.type == "start":
                partial_message = event.partial
                context.messages.append(partial_message)
                added_partial = True
                await _emit(
                    emit,
                    MessageStartEvent(message=partial_message.model_copy(deep=True)),
                )
            case _ if event.type in {
                "text_start",
                "text_delta",
                "text_end",
                "thinking_start",
                "thinking_delta",
                "thinking_end",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_end",
            }:
                if partial_message is None:
                    continue
                partial_message = event.partial
                context.messages[-1] = partial_message
                await _emit(
                    emit,
                    MessageUpdateEvent(
                        message=partial_message.model_copy(deep=True),
                        assistant_message_event=event.model_copy(deep=True),
                    ),
                )
            case DoneEvent() | ErrorEvent():
                final_message = await response.result()
                if added_partial:
                    context.messages[-1] = final_message
                else:
                    context.messages.append(final_message)
                    await _emit(
                        emit,
                        MessageStartEvent(message=final_message.model_copy(deep=True)),
                    )
                await _emit(
                    emit,
                    MessageEndEvent(message=final_message.model_copy(deep=True)),
                )
                return final_message

    final_message = await response.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _emit(
            emit,
            MessageStartEvent(message=final_message.model_copy(deep=True)),
        )
    await _emit(
        emit,
        MessageEndEvent(message=final_message.model_copy(deep=True)),
    )
    return final_message


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    if config.tool_execution == ToolExecutionMode.SEQUENTIAL:
        return await _execute_tool_calls_sequential(
            current_context=current_context,
            assistant_message=assistant_message,
            tool_calls=tool_calls,
            config=config,
            cancellation=cancellation,
            emit=emit,
        )

    return await _execute_tool_calls_parallel(
        current_context=current_context,
        assistant_message=assistant_message,
        tool_calls=tool_calls,
        config=config,
        cancellation=cancellation,
        emit=emit,
    )


@dataclass(slots=True)
class _PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool
    args: Any


@dataclass(slots=True)
class _ImmediateToolCallOutcome:
    result: AgentToolResult
    is_error: bool


@dataclass(slots=True)
class _ExecutedToolCallOutcome:
    result: AgentToolResult
    is_error: bool


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    results: list[ToolResultMessage] = []

    for tool_call in tool_calls:
        _raise_if_cancelled(cancellation)
        await _emit(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            ),
        )
        preparation = await _prepare_tool_call(
            current_context=current_context,
            assistant_message=assistant_message,
            tool_call=tool_call,
            config=config,
            cancellation=cancellation,
        )
        if isinstance(preparation, _ImmediateToolCallOutcome):
            results.append(
                await _emit_tool_call_outcome(
                    tool_call=tool_call,
                    result=preparation.result,
                    is_error=preparation.is_error,
                    emit=emit,
                )
            )
            continue

        executed = await _execute_prepared_tool_call(
            prepared=preparation,
            cancellation=cancellation,
            emit=emit,
        )
        results.append(
            await _finalize_executed_tool_call(
                current_context=current_context,
                assistant_message=assistant_message,
                prepared=preparation,
                executed=executed,
                config=config,
                cancellation=cancellation,
                emit=emit,
            )
        )

    return results


async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    results: list[ToolResultMessage] = []
    runnable_calls: list[_PreparedToolCall] = []

    for tool_call in tool_calls:
        _raise_if_cancelled(cancellation)
        await _emit(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            ),
        )
        preparation = await _prepare_tool_call(
            current_context=current_context,
            assistant_message=assistant_message,
            tool_call=tool_call,
            config=config,
            cancellation=cancellation,
        )
        if isinstance(preparation, _ImmediateToolCallOutcome):
            results.append(
                await _emit_tool_call_outcome(
                    tool_call=tool_call,
                    result=preparation.result,
                    is_error=preparation.is_error,
                    emit=emit,
                )
            )
            continue
        runnable_calls.append(preparation)

    running = [
        (
            prepared,
            asyncio.create_task(
                _execute_prepared_tool_call(
                    prepared=prepared,
                    cancellation=cancellation,
                    emit=emit,
                )
            ),
        )
        for prepared in runnable_calls
    ]

    try:
        for prepared, task in running:
            executed = await task
            results.append(
                await _finalize_executed_tool_call(
                    current_context=current_context,
                    assistant_message=assistant_message,
                    prepared=prepared,
                    executed=executed,
                    config=config,
                    cancellation=cancellation,
                    emit=emit,
                )
            )
    except BaseException:
        for _, task in running:
            if not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for _, task in running),
            return_exceptions=True,
        )
        raise

    return results


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
) -> _PreparedToolCall | _ImmediateToolCallOutcome:
    tool = next((item for item in current_context.tools if item.name == tool_call.name), None)
    if tool is None:
        return _ImmediateToolCallOutcome(
            result=_create_error_tool_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        validated_args = validate_tool_arguments(
            tool_call.arguments,
            tool_schema(tool.parameters),
        )
    except (ToolValidationError, Exception) as exc:
        return _ImmediateToolCallOutcome(
            result=_create_error_tool_result(str(exc)),
            is_error=True,
        )

    if config.before_tool_call is not None:
        before_result = await _maybe_await(
            config.before_tool_call(
                BeforeToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=tool_call,
                    args=validated_args,
                    context=current_context,
                ),
                cancellation,
            )
        )
        if before_result is not None and before_result.block:
            return _ImmediateToolCallOutcome(
                result=_create_error_tool_result(
                    before_result.reason or "Tool execution was blocked"
                ),
                is_error=True,
            )

    return _PreparedToolCall(
        tool_call=tool_call,
        tool=tool,
        args=validated_args,
    )


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallOutcome:
    _raise_if_cancelled(cancellation)
    pending_updates: list[asyncio.Task[None]] = []

    def on_update(partial_result: AgentToolResult) -> asyncio.Task[None]:
        task = asyncio.create_task(
            _emit(
                emit,
                ToolExecutionUpdateEvent(
                    tool_call_id=prepared.tool_call.id,
                    tool_name=prepared.tool_call.name,
                    args=prepared.tool_call.arguments,
                    partial_result=partial_result,
                ),
            )
        )
        pending_updates.append(task)
        return task

    try:
        result = await prepared.tool.execute(
            prepared.tool_call.id,
            prepared.args,
            cancellation,
            on_update,
        )
        await asyncio.gather(*pending_updates)
        return _ExecutedToolCallOutcome(
            result=_coerce_tool_result(result),
            is_error=False,
        )
    except CancellationError:
        await asyncio.gather(*pending_updates, return_exceptions=True)
        raise
    except Exception as exc:
        await asyncio.gather(*pending_updates, return_exceptions=True)
        return _ExecutedToolCallOutcome(
            result=_create_error_tool_result(str(exc)),
            is_error=True,
        )


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCallOutcome,
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
) -> ToolResultMessage:
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call is not None:
        after_result = await _maybe_await(
            config.after_tool_call(
                AfterToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=prepared.tool_call,
                    args=prepared.args,
                    result=result,
                    is_error=is_error,
                    context=current_context,
                ),
                cancellation,
            )
        )
        if after_result is not None:
            if after_result.content is not _UNSET:
                result = AgentToolResult(
                    content=list(after_result.content),  # type: ignore[arg-type]
                    details=result.details,
                )
            if after_result.details is not _UNSET:
                result = AgentToolResult(
                    content=list(result.content),
                    details=after_result.details,
                )
            if after_result.is_error is not _UNSET:
                is_error = bool(after_result.is_error)

    return await _emit_tool_call_outcome(
        tool_call=prepared.tool_call,
        result=result,
        is_error=is_error,
        emit=emit,
    )


def _coerce_tool_result(result: Any) -> AgentToolResult:
    if isinstance(result, AgentToolResult):
        return result
    if isinstance(result, str):
        return AgentToolResult(
            content=[TextContent(text=result)],
            details=None,
        )
    if isinstance(result, dict):
        return AgentToolResult(
            content=list(result.get("content", [])),
            details=result.get("details"),
        )
    raise TypeError(f"Unsupported tool result type: {type(result)!r}")


def _create_error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(text=message)],
        details={},
    )


async def _emit_tool_call_outcome(
    tool_call: ToolCall,
    result: AgentToolResult,
    is_error: bool,
    emit: AgentEventSink,
) -> ToolResultMessage:
    await _emit(
        emit,
        ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
            is_error=is_error,
        ),
    )

    tool_result_message = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=list(result.content),
        details=result.details,
        is_error=is_error,
    )
    await _emit(
        emit,
        MessageStartEvent(message=tool_result_message.model_copy(deep=True)),
    )
    await _emit(
        emit,
        MessageEndEvent(message=tool_result_message.model_copy(deep=True)),
    )
    return tool_result_message


async def _resolve_stream(
    maybe_stream: AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
) -> AssistantMessageEventStream:
    if inspect.isawaitable(maybe_stream):
        return await maybe_stream
    return maybe_stream


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _emit(emit: AgentEventSink, event: AgentEvent) -> None:
    result = emit(event)
    if inspect.isawaitable(result):
        await result


async def _get_optional_messages(supplier: Any) -> list[AgentMessage]:
    if supplier is None:
        return []
    messages = await _maybe_await(supplier())
    return list(messages or [])


def _message_role(message: Any) -> str | None:
    if hasattr(message, "role"):
        role = getattr(message, "role")
        return str(role) if role is not None else None
    if isinstance(message, dict):
        role = message.get("role")
        return str(role) if role is not None else None
    return None


def _raise_if_cancelled(cancellation: CancellationToken | None) -> None:
    if cancellation is not None:
        cancellation.raise_if_cancelled()


def build_terminal_assistant_message(
    *,
    model: Any,
    stop_reason: StopReason,
    error_message: str | None = None,
) -> AssistantMessage:
    """Create a synthetic terminal assistant message for runtime failures."""
    return AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        content=[],
        usage=Usage(),
        stop_reason=stop_reason,
        error_message=error_message,
    )
