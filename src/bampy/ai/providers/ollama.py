"""Ollama-compatible Responses API provider adapter.

This adapter is intentionally scoped to OpenAI-compatible gateways that may
emit overlapping text deltas. It keeps those compatibility workarounds out of
the standard OpenAI provider.
"""

from __future__ import annotations

import json
from typing import Any

from bampy.ai.api_registry import ApiProviderEntry
from bampy.ai.models import calculate_cost
from bampy.ai.providers._cancellation import spawn_provider_task
from bampy.ai.providers._transform import sanitize_tool_call_id
from bampy.ai.providers.openai import (
    _convert_messages,
    _convert_tools,
    _emit_error,
    _resolve_reasoning_effort,
    _serialize_sdk_item,
)
from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    OpenAIOptions,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/136.0.0.0 Safari/537.36"
)

_MISSING_TERMINAL_EVENT_ERROR = (
    "Response stream ended before a terminal response.completed event was received."
)


def _normalize_stream_delta(existing: str, delta: str) -> str:
    """Collapse overlapping chunks into the suffix that still needs appending."""
    if not delta:
        return ""
    if not existing:
        return delta
    if delta.startswith(existing):
        return delta[len(existing) :]
    if existing.endswith(delta) or existing.startswith(delta):
        return ""

    max_overlap = min(len(existing), len(delta))
    for overlap in range(max_overlap, 0, -1):
        if existing.endswith(delta[:overlap]):
            return delta[overlap:]
    return delta


def stream_ollama(
    model: Model,
    context: Context,
    options: OpenAIOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream an OpenAI-compatible Ollama response with overlap normalization."""
    event_stream = AssistantMessageEventStream()
    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        content=[],
    )

    async def _run() -> None:
        try:
            import openai as openai_sdk
        except ImportError as e:
            _emit_error(event_stream, model, f"openai SDK not installed: {e}")
            return

        try:
            api_key = options.api_key if options else None
            base_url = model.base_url or None
            if base_url and not base_url.rstrip("/").endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            client_kwargs: dict[str, Any] = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if base_url:
                client_kwargs["base_url"] = base_url
            extra_headers = {}
            if model.headers:
                extra_headers.update(model.headers)
            if options and options.headers:
                extra_headers.update(options.headers)
            extra_headers.setdefault("User-Agent", _DEFAULT_USER_AGENT)
            if extra_headers:
                client_kwargs["default_headers"] = extra_headers

            client = openai_sdk.AsyncOpenAI(**client_kwargs)

            input_items = _convert_messages(context)
            max_tokens = (
                (options.max_tokens if options and options.max_tokens else None)
                or model.max_tokens
            )

            params: dict[str, Any] = {
                "model": model.id,
                "input": input_items,
                "stream": True,
                "max_output_tokens": max_tokens,
            }

            if options and options.temperature is not None and not model.reasoning:
                params["temperature"] = options.temperature

            tools = _convert_tools(context.tools)
            if tools:
                params["tools"] = tools

            if (
                model.reasoning
                and options
                and isinstance(options, OpenAIOptions)
                and options.reasoning_effort
            ):
                params["reasoning"] = {"effort": options.reasoning_effort}
                params["include"] = ["reasoning.encrypted_content"]

            event_stream.push(StartEvent(partial=output))

            output_to_content: dict[int, int] = {}
            tool_json_bufs: dict[int, str] = {}
            saw_completion_event = False

            response = await client.responses.create(**params)
            async for event in response:
                saw_completion_event = _handle_stream_event(
                    event,
                    output,
                    event_stream,
                    output_to_content,
                    tool_json_bufs,
                ) or saw_completion_event

            if not saw_completion_event:
                output.stop_reason = StopReason.ERROR
                output.error_message = _MISSING_TERMINAL_EVENT_ERROR
                event_stream.push(ErrorEvent(reason=StopReason.ERROR, error=output))
                event_stream.end(output)
                return

            output.usage.cost = calculate_cost(model, output.usage)

            event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            event_stream.end(output)

        except Exception as e:
            output.stop_reason = StopReason.ERROR
            output.error_message = str(e)
            event_stream.push(ErrorEvent(reason=StopReason.ERROR, error=output))
            event_stream.end(output)

    spawn_provider_task(
        event_stream=event_stream,
        output=output,
        options=options,
        runner=_run,
    )
    return event_stream


def stream_simple_ollama(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Simplified streaming -- maps SimpleStreamOptions to OpenAIOptions."""
    ollama_opts: OpenAIOptions | None = None

    if options is not None:
        reasoning_effort = (
            _resolve_reasoning_effort(model, options.reasoning)
            if model.reasoning
            else None
        )

        ollama_opts = OpenAIOptions(
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            api_key=options.api_key,
            max_retry_delay_ms=options.max_retry_delay_ms,
            headers=options.headers,
            cancellation=options.cancellation,
            reasoning_effort=reasoning_effort,
        )

    return stream_ollama(model, context, ollama_opts)


def _handle_stream_event(
    event: Any,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    output_to_content: dict[int, int],
    tool_json_bufs: dict[int, str],
) -> bool:
    """Map a single Responses API SSE event to bampy events."""
    etype = getattr(event, "type", "")

    if etype == "response.output_item.added":
        item = getattr(event, "item", None)
        out_idx = getattr(event, "output_index", 0)
        if item is None:
            return False

        item_type = getattr(item, "type", "")

        if item_type == "function_call":
            call_id = sanitize_tool_call_id(
                getattr(item, "call_id", "") or getattr(item, "id", f"call_{out_idx}")
            )
            item_id = getattr(item, "id", "") or ""
            tc_name = getattr(item, "name", "")
            tool_call_id = (
                f"{call_id}|{sanitize_tool_call_id(item_id)}"
                if item_id
                else call_id
            )
            tool_call = ToolCall(id=tool_call_id, name=tc_name, arguments={})
            output.content.append(tool_call)
            content_idx = len(output.content) - 1
            output_to_content[out_idx] = content_idx
            tool_json_bufs[out_idx] = ""
            stream.push(ToolCallStartEvent(
                content_index=content_idx,
                content=tool_call,
                partial=output,
            ))

        elif item_type == "reasoning":
            thinking = ThinkingContent(thinking="")
            output.content.append(thinking)
            content_idx = len(output.content) - 1
            output_to_content[out_idx] = content_idx
            stream.push(ThinkingStartEvent(
                content_index=content_idx,
                content=thinking,
                partial=output,
            ))

    elif etype == "response.content_part.added":
        out_idx = getattr(event, "output_index", 0)
        part = getattr(event, "part", None)
        if part and getattr(part, "type", "") == "output_text":
            text_content = TextContent(text="")
            output.content.append(text_content)
            content_idx = len(output.content) - 1
            output_to_content[out_idx] = content_idx
            stream.push(TextStartEvent(
                content_index=content_idx,
                content=text_content,
                partial=output,
            ))

    elif etype == "response.output_text.delta":
        out_idx = getattr(event, "output_index", 0)
        delta = getattr(event, "delta", "")
        content_idx = output_to_content.get(out_idx)
        if content_idx is not None and delta:
            block = output.content[content_idx]
            if isinstance(block, TextContent):
                delta = _normalize_stream_delta(block.text, delta)
                if not delta:
                    return False
                block.text += delta
            stream.push(TextDeltaEvent(
                content_index=content_idx,
                delta=delta,
                partial=output,
            ))

    elif etype == "response.output_text.done":
        out_idx = getattr(event, "output_index", 0)
        content_idx = output_to_content.get(out_idx)
        if content_idx is not None:
            block = output.content[content_idx]
            if isinstance(block, TextContent):
                stream.push(TextEndEvent(
                    content_index=content_idx,
                    content=block,
                    partial=output,
                ))

    elif etype == "response.function_call_arguments.delta":
        out_idx = getattr(event, "output_index", 0)
        delta = getattr(event, "delta", "")
        content_idx = output_to_content.get(out_idx)
        if content_idx is not None and delta:
            tool_json_bufs[out_idx] = tool_json_bufs.get(out_idx, "") + delta
            stream.push(ToolCallDeltaEvent(
                content_index=content_idx,
                delta=delta,
                partial=output,
            ))

    elif etype == "response.function_call_arguments.done":
        out_idx = getattr(event, "output_index", 0)
        content_idx = output_to_content.get(out_idx)
        if content_idx is not None:
            block = output.content[content_idx]
            if isinstance(block, ToolCall):
                raw = (
                    getattr(event, "arguments", "")
                    or tool_json_bufs.get(out_idx, "")
                )
                try:
                    block.arguments = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    block.arguments = {}
                stream.push(ToolCallEndEvent(
                    content_index=content_idx,
                    content=block,
                    partial=output,
                ))

    elif etype == "response.reasoning_summary_text.delta":
        out_idx = getattr(event, "output_index", 0)
        delta = getattr(event, "delta", "")
        content_idx = output_to_content.get(out_idx)
        if content_idx is not None and delta:
            block = output.content[content_idx]
            if isinstance(block, ThinkingContent):
                delta = _normalize_stream_delta(block.thinking, delta)
                if not delta:
                    return False
                block.thinking += delta
            stream.push(ThinkingDeltaEvent(
                content_index=content_idx,
                delta=delta,
                partial=output,
            ))

    elif etype == "response.output_item.done":
        out_idx = getattr(event, "output_index", 0)
        item = getattr(event, "item", None)
        content_idx = output_to_content.get(out_idx)

        if item and getattr(item, "type", "") == "reasoning" and content_idx is not None:
            block = output.content[content_idx]
            if isinstance(block, ThinkingContent):
                if not block.thinking:
                    summary = getattr(item, "summary", None)
                    if summary:
                        texts = []
                        for summary_item in summary:
                            if getattr(summary_item, "type", "") == "summary_text":
                                texts.append(getattr(summary_item, "text", ""))
                        if texts:
                            block.thinking = "\n".join(texts)
                block.thinking_signature = _serialize_sdk_item(item)
                stream.push(ThinkingEndEvent(
                    content_index=content_idx,
                    content=block,
                    partial=output,
                ))

    elif etype == "response.completed":
        resp = getattr(event, "response", None)
        if resp:
            output.response_id = getattr(resp, "id", None)

            usage = getattr(resp, "usage", None)
            if usage:
                cached_tokens = 0
                details = getattr(usage, "input_tokens_details", None)
                if details:
                    cached_tokens = getattr(details, "cached_tokens", 0) or 0
                total_input = getattr(usage, "input_tokens", 0)
                output.usage.input = max(total_input - cached_tokens, 0)
                output.usage.output = getattr(usage, "output_tokens", 0)
                output.usage.cache_read = cached_tokens
                output.usage.total_tokens = (
                    getattr(usage, "total_tokens", 0)
                    or (output.usage.input + output.usage.output + output.usage.cache_read)
                )

            status = getattr(resp, "status", "completed")
            if status == "completed":
                resp_output = getattr(resp, "output", [])
                has_tools = any(
                    getattr(item, "type", "") == "function_call"
                    for item in resp_output
                )
                output.stop_reason = StopReason.TOOL_USE if has_tools else StopReason.STOP
            elif status == "incomplete":
                incomplete = getattr(resp, "incomplete_details", None)
                reason = getattr(incomplete, "reason", "") if incomplete else ""
                output.stop_reason = (
                    StopReason.LENGTH
                    if reason == "max_output_tokens"
                    else StopReason.ERROR
                )
            else:
                output.stop_reason = StopReason.ERROR
            return True

    return False


def get_provider_entry() -> ApiProviderEntry:
    return ApiProviderEntry(
        api="ollama-responses",
        stream=stream_ollama,  # type: ignore[arg-type]
        stream_simple=stream_simple_ollama,  # type: ignore[arg-type]
    )
