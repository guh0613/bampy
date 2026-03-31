"""OpenAI Responses API provider adapter.

Maps OpenAI Responses API streaming events -> bampy AssistantMessageEvent protocol.
"""

from __future__ import annotations

import json
from typing import Any

from bampy.ai.api_registry import ApiProviderEntry
from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    ImageContent,
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
    ThinkingLevel,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from bampy.ai.models import calculate_cost, supports_xhigh
from bampy.ai.providers._cancellation import spawn_provider_task
from bampy.ai.providers._transform import sanitize_tool_call_id


# ---------------------------------------------------------------------------
# Reasoning effort mapping
# ---------------------------------------------------------------------------

_EFFORT_MAP: dict[ThinkingLevel, str] = {
    ThinkingLevel.MINIMAL: "minimal",
    ThinkingLevel.LOW: "low",
    ThinkingLevel.MEDIUM: "medium",
    ThinkingLevel.HIGH: "high",
    ThinkingLevel.XHIGH: "xhigh",
}

def _resolve_reasoning_effort(
    model: Model,
    reasoning: ThinkingLevel | None,
) -> str | None:
    """Map simple reasoning levels to the provider's effort values."""
    if reasoning is None:
        return None
    if reasoning == ThinkingLevel.XHIGH and not supports_xhigh(model):
        return "high"
    return _EFFORT_MAP.get(reasoning, "medium")


def _serialize_sdk_item(item: Any) -> str | None:
    """Serialize an SDK response item so it can be round-tripped later."""
    if item is None:
        return None
    if hasattr(item, "model_dump"):
        return json.dumps(item.model_dump(exclude_none=True))
    if isinstance(item, dict):
        return json.dumps(item)
    return None


def _parse_reasoning_signature(signature: str | None) -> dict[str, Any] | None:
    """Decode a previously stored reasoning item."""
    if not signature:
        return None
    try:
        item = json.loads(signature)
    except json.JSONDecodeError:
        return None
    if isinstance(item, dict) and item.get("type") == "reasoning":
        return item
    return None


def _split_openai_tool_call_id(tool_call_id: str) -> tuple[str, str | None]:
    """Split the composite Responses API tool-call id into call + item ids."""
    if "|" in tool_call_id:
        call_id, item_id = tool_call_id.split("|", 1)
        return sanitize_tool_call_id(call_id), sanitize_tool_call_id(item_id)
    return sanitize_tool_call_id(tool_call_id), None


def _supports_multimodal_tool_results(model: Model) -> bool:
    """Whether tool results may include image blocks for this model/provider."""
    return model.provider == "openai" and "image" in model.input_types


# ---------------------------------------------------------------------------
# Message conversion (bampy -> OpenAI Responses API input format)
# ---------------------------------------------------------------------------

def _convert_messages(
    context: Context,
    *,
    allow_tool_result_images: bool = False,
) -> list[dict[str, Any]]:
    """Convert context to OpenAI Responses API input items."""
    from bampy.ai.types import AssistantMessage, ToolResultMessage, UserMessage

    items: list[dict[str, Any]] = []

    # System prompt (Responses API uses "developer" role, but "system" also works)
    if context.system_prompt:
        items.append({"role": "developer", "content": context.system_prompt})

    for msg in context.messages:
        if isinstance(msg, UserMessage):
            items.append({
                "role": "user",
                "content": _convert_user_content(msg.content),
            })

        elif isinstance(msg, AssistantMessage):
            _convert_assistant_items(msg, items)

        elif isinstance(msg, ToolResultMessage):
            items.append({
                "type": "function_call_output",
                "call_id": sanitize_tool_call_id(msg.tool_call_id),
                "output": _convert_tool_result_output(
                    msg.content,
                    allow_images=allow_tool_result_images,
                ),
            })

    return items


def _convert_user_content(content: str | list) -> str | list[dict[str, Any]]:
    """Convert user content to Responses API format."""
    if isinstance(content, str):
        return content

    parts: list[dict[str, Any]] = []
    for item in content:
        converted = _convert_content_block(item)
        if converted is not None:
            parts.append(converted)
    return parts if parts else ""


def _convert_assistant_items(
    msg: AssistantMessage, items: list[dict[str, Any]]
) -> None:
    """Convert an assistant message to Responses API input items.

    Text content -> role-based assistant message.
    Tool calls -> function_call input items.
    """
    text_parts: list[str] = []

    for block in msg.content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)
        elif isinstance(block, ThinkingContent):
            if text_parts:
                items.append({"role": "assistant", "content": "\n".join(text_parts)})
                text_parts.clear()
            reasoning_item = _parse_reasoning_signature(block.thinking_signature)
            if reasoning_item is not None:
                items.append(reasoning_item)
        elif isinstance(block, ToolCall):
            # Flush any pending text before emitting tool call items
            if text_parts:
                items.append({"role": "assistant", "content": "\n".join(text_parts)})
                text_parts.clear()
            call_id, item_id = _split_openai_tool_call_id(block.id)
            items.append({
                "type": "function_call",
                "call_id": call_id,
                **({"id": item_id} if item_id else {}),
                "name": block.name,
                "arguments": json.dumps(block.arguments),
            })

    # Remaining text
    if text_parts:
        items.append({"role": "assistant", "content": "\n".join(text_parts)})


def _convert_content_block(item: Any) -> dict[str, Any] | None:
    """Convert a text/image block to Responses API input content."""
    if isinstance(item, TextContent):
        return {"type": "input_text", "text": item.text}
    if isinstance(item, ImageContent):
        return {
            "type": "input_image",
            "image_url": f"data:{item.mime_type};base64,{item.data}",
        }
    return None


def _tool_result_to_string(content: list) -> str:
    """Flatten tool result content to a text fallback for Responses API."""
    parts: list[str] = []
    for item in content:
        if isinstance(item, TextContent):
            parts.append(item.text)
        elif isinstance(item, ImageContent):
            parts.append("[image]")
    return "\n".join(parts) if parts else ""


def _convert_tool_result_output(
    content: list,
    *,
    allow_images: bool,
) -> str | list[dict[str, Any]]:
    """Convert tool result content to Responses API output payload."""
    if not allow_images:
        return _tool_result_to_string(content)

    has_image = any(isinstance(item, ImageContent) for item in content)
    if not has_image:
        return _tool_result_to_string(content)

    blocks: list[dict[str, Any]] = []
    for item in content:
        converted = _convert_content_block(item)
        if converted is not None:
            blocks.append(converted)
    return blocks if blocks else ""


def _convert_tools(tools: list | None) -> list[dict[str, Any]] | None:
    """Convert bampy Tool definitions to Responses API tool format."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Public stream functions
# ---------------------------------------------------------------------------

def stream_openai(
    model: Model,
    context: Context,
    options: OpenAIOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream an OpenAI Responses API response with fine-grained events."""
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
            # Build client
            api_key = options.api_key if options else None
            base_url = model.base_url or None
            # OpenAI SDK expects base_url to include /v1 path
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
            if extra_headers:
                client_kwargs["default_headers"] = extra_headers

            client = openai_sdk.AsyncOpenAI(**client_kwargs)

            # Build params
            input_items = _convert_messages(
                context,
                allow_tool_result_images=_supports_multimodal_tool_results(model),
            )
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

            # Reasoning effort for reasoning models
            if (
                model.reasoning
                and options
                and isinstance(options, OpenAIOptions)
                and options.reasoning_effort
            ):
                params["reasoning"] = {"effort": options.reasoning_effort}
                params["include"] = ["reasoning.encrypted_content"]

            # Emit start
            event_stream.push(StartEvent(partial=output))

            # Track state: output_index -> content_index
            output_to_content: dict[int, int] = {}
            tool_json_bufs: dict[int, str] = {}

            # Stream via Responses API
            response = await client.responses.create(**params)
            async for event in response:
                _handle_stream_event(
                    event, output, event_stream,
                    output_to_content, tool_json_bufs,
                )

            # Usage cost
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


def stream_simple_openai(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Simplified streaming -- maps SimpleStreamOptions to OpenAIOptions."""
    openai_opts: OpenAIOptions | None = None

    if options is not None:
        reasoning_effort = (
            _resolve_reasoning_effort(model, options.reasoning)
            if model.reasoning
            else None
        )

        openai_opts = OpenAIOptions(
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            api_key=options.api_key,
            max_retry_delay_ms=options.max_retry_delay_ms,
            headers=options.headers,
            cancellation=options.cancellation,
            reasoning_effort=reasoning_effort,
        )

    return stream_openai(model, context, openai_opts)


# ---------------------------------------------------------------------------
# Responses API streaming event handler
# ---------------------------------------------------------------------------

def _handle_stream_event(
    event: Any,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    output_to_content: dict[int, int],
    tool_json_bufs: dict[int, str],
) -> None:
    """Map a single Responses API SSE event to bampy events."""
    etype = getattr(event, "type", "")

    # -- Output item added (message / function_call / reasoning) -----------
    if etype == "response.output_item.added":
        item = getattr(event, "item", None)
        out_idx = getattr(event, "output_index", 0)
        if item is None:
            return

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
                content_index=content_idx, content=tool_call, partial=output,
            ))

        elif item_type == "reasoning":
            thinking = ThinkingContent(thinking="")
            output.content.append(thinking)
            content_idx = len(output.content) - 1
            output_to_content[out_idx] = content_idx
            stream.push(ThinkingStartEvent(
                content_index=content_idx, content=thinking, partial=output,
            ))

        # item_type == "message" is handled by content_part.added below

    # -- Content part added (text output within a message item) ------------
    elif etype == "response.content_part.added":
        out_idx = getattr(event, "output_index", 0)
        part = getattr(event, "part", None)
        if part and getattr(part, "type", "") == "output_text":
            text_content = TextContent(text="")
            output.content.append(text_content)
            content_idx = len(output.content) - 1
            output_to_content[out_idx] = content_idx
            stream.push(TextStartEvent(
                content_index=content_idx, content=text_content, partial=output,
            ))

    # -- Text deltas -------------------------------------------------------
    elif etype == "response.output_text.delta":
        out_idx = getattr(event, "output_index", 0)
        delta = getattr(event, "delta", "")
        content_idx = output_to_content.get(out_idx)
        if content_idx is not None and delta:
            block = output.content[content_idx]
            if isinstance(block, TextContent):
                block.text += delta
            stream.push(TextDeltaEvent(
                content_index=content_idx, delta=delta, partial=output,
            ))

    elif etype == "response.output_text.done":
        out_idx = getattr(event, "output_index", 0)
        content_idx = output_to_content.get(out_idx)
        if content_idx is not None:
            block = output.content[content_idx]
            if isinstance(block, TextContent):
                stream.push(TextEndEvent(
                    content_index=content_idx, content=block, partial=output,
                ))

    # -- Function call argument deltas -------------------------------------
    elif etype == "response.function_call_arguments.delta":
        out_idx = getattr(event, "output_index", 0)
        delta = getattr(event, "delta", "")
        content_idx = output_to_content.get(out_idx)
        if content_idx is not None and delta:
            tool_json_bufs[out_idx] = tool_json_bufs.get(out_idx, "") + delta
            stream.push(ToolCallDeltaEvent(
                content_index=content_idx, delta=delta, partial=output,
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
                    content_index=content_idx, content=block, partial=output,
                ))

    # -- Reasoning summary deltas ------------------------------------------
    elif etype == "response.reasoning_summary_text.delta":
        out_idx = getattr(event, "output_index", 0)
        delta = getattr(event, "delta", "")
        content_idx = output_to_content.get(out_idx)
        if content_idx is not None and delta:
            block = output.content[content_idx]
            if isinstance(block, ThinkingContent):
                block.thinking += delta
            stream.push(ThinkingDeltaEvent(
                content_index=content_idx, delta=delta, partial=output,
            ))

    # -- Output item done (finalize reasoning / other items) ---------------
    elif etype == "response.output_item.done":
        out_idx = getattr(event, "output_index", 0)
        item = getattr(event, "item", None)
        content_idx = output_to_content.get(out_idx)

        if item and getattr(item, "type", "") == "reasoning" and content_idx is not None:
            block = output.content[content_idx]
            if isinstance(block, ThinkingContent):
                # If we didn't get streaming deltas, extract summary now
                if not block.thinking:
                    summary = getattr(item, "summary", None)
                    if summary:
                        texts = []
                        for s in summary:
                            if getattr(s, "type", "") == "summary_text":
                                texts.append(getattr(s, "text", ""))
                        if texts:
                            block.thinking = "\n".join(texts)
                block.thinking_signature = _serialize_sdk_item(item)
                stream.push(ThinkingEndEvent(
                    content_index=content_idx, content=block, partial=output,
                ))

    # -- Response completed (usage + stop reason) --------------------------
    elif etype == "response.completed":
        resp = getattr(event, "response", None)
        if resp:
            output.response_id = getattr(resp, "id", None)

            # Usage
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

            # Stop reason
            status = getattr(resp, "status", "completed")
            if status == "completed":
                resp_output = getattr(resp, "output", [])
                has_tools = any(
                    getattr(item, "type", "") == "function_call"
                    for item in resp_output
                )
                output.stop_reason = (
                    StopReason.TOOL_USE if has_tools else StopReason.STOP
                )
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emit_error(
    stream: AssistantMessageEventStream,
    model: Model,
    message: str,
) -> None:
    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason=StopReason.ERROR,
        error_message=message,
    )
    stream.push(ErrorEvent(reason=StopReason.ERROR, error=output))
    stream.end(output)


# ---------------------------------------------------------------------------
# Provider entry
# ---------------------------------------------------------------------------

def get_provider_entry() -> ApiProviderEntry:
    return ApiProviderEntry(
        api="openai-responses",
        stream=stream_openai,  # type: ignore[arg-type]
        stream_simple=stream_simple_openai,  # type: ignore[arg-type]
    )
