"""Anthropic Messages API provider adapter.

Maps Anthropic SDK streaming events → bampy AssistantMessageEvent protocol.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from bampy.ai.provider import ApiProviderEntry
from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AnthropicOptions,
    AnthropicThinkingAdaptive,
    AnthropicThinkingEnabled,
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamOptions,
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
    Usage,
    UsageCost,
)
from bampy.ai.models import calculate_cost
from bampy.ai.providers._transform import sanitize_tool_call_id


# ---------------------------------------------------------------------------
# Thinking level → Anthropic thinking config
# ---------------------------------------------------------------------------

_THINKING_BUDGETS: dict[ThinkingLevel, int] = {
    ThinkingLevel.MINIMAL: 1024,
    ThinkingLevel.LOW: 2048,
    ThinkingLevel.MEDIUM: 8192,
    ThinkingLevel.HIGH: 16384,
}

# Models that support adaptive thinking (effort-based)
_ADAPTIVE_MODELS = {"claude-opus-4", "claude-sonnet-4", "claude-haiku-4"}

_EFFORT_MAP: dict[ThinkingLevel, str] = {
    ThinkingLevel.MINIMAL: "low",
    ThinkingLevel.LOW: "low",
    ThinkingLevel.MEDIUM: "medium",
    ThinkingLevel.HIGH: "high",
}


def _resolve_thinking(
    model: Model,
    reasoning: ThinkingLevel | None,
    thinking_config: AnthropicThinkingEnabled | AnthropicThinkingAdaptive | None,
) -> dict[str, Any] | None:
    """Build the ``thinking`` parameter for the Anthropic API."""
    # Explicit config takes priority
    if thinking_config is not None:
        if isinstance(thinking_config, AnthropicThinkingAdaptive):
            return {"type": "adaptive", "effort": thinking_config.effort}
        return {"type": "enabled", "budget_tokens": thinking_config.budget_tokens}

    if reasoning is None:
        return None

    # Check if model supports adaptive thinking
    model_family = "-".join(model.id.split("-")[:3])  # e.g. "claude-opus-4"
    if model_family in _ADAPTIVE_MODELS:
        effort = _EFFORT_MAP.get(reasoning, "medium")
        return {"type": "adaptive", "effort": effort}

    # Budget-based thinking for older models
    budget = _THINKING_BUDGETS.get(reasoning, 8192)
    return {"type": "enabled", "budget_tokens": budget}


# ---------------------------------------------------------------------------
# Message conversion (bampy → Anthropic API format)
# ---------------------------------------------------------------------------

def _convert_messages(
    context: Context,
) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Convert context to Anthropic ``system`` and ``messages`` params."""
    from bampy.ai.types import AssistantMessage, ToolResultMessage, UserMessage

    system = context.system_prompt
    messages: list[dict[str, Any]] = []

    for msg in context.messages:
        if isinstance(msg, UserMessage):
            messages.append({
                "role": "user",
                "content": _convert_user_content(msg.content),
            })

        elif isinstance(msg, AssistantMessage):
            content_blocks = _convert_assistant_content(msg.content)
            if content_blocks:
                messages.append({
                    "role": "assistant",
                    "content": content_blocks,
                })

        elif isinstance(msg, ToolResultMessage):
            # Anthropic expects tool results as user messages
            _append_tool_result(messages, msg)

    return system, messages


def _convert_user_content(content: str | list) -> str | list[dict[str, Any]]:
    """Convert user message content to Anthropic format."""
    if isinstance(content, str):
        return content

    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, TextContent):
            blocks.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": item.mime_type,
                    "data": item.data,
                },
            })
    return blocks if blocks else ""


def _convert_assistant_content(content: list) -> list[dict[str, Any]]:
    """Convert assistant content blocks to Anthropic format."""
    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, TextContent):
            block: dict[str, Any] = {"type": "text", "text": item.text}
            if item.text_signature:
                block["citations"] = [{"type": "text_signature", "signature": item.text_signature}]
            blocks.append(block)

        elif isinstance(item, ThinkingContent):
            if item.redacted:
                blocks.append({"type": "redacted_thinking", "data": ""})
            elif item.thinking_signature:
                blocks.append({
                    "type": "thinking",
                    "thinking": item.thinking,
                    "signature": item.thinking_signature,
                })
            else:
                blocks.append({"type": "thinking", "thinking": item.thinking})

        elif isinstance(item, ToolCall):
            tool_id = sanitize_tool_call_id(item.id)
            blocks.append({
                "type": "tool_use",
                "id": tool_id,
                "name": item.name,
                "input": item.arguments,
            })
    return blocks


def _append_tool_result(messages: list[dict[str, Any]], msg: Any) -> None:
    """Append a tool result. Groups consecutive results into one user message."""
    result_block = {
        "type": "tool_result",
        "tool_use_id": sanitize_tool_call_id(msg.tool_call_id),
        "content": _convert_tool_result_content(msg.content),
    }
    if msg.is_error:
        result_block["is_error"] = True

    # Group with previous tool results in same user message
    if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
        last_content = messages[-1]["content"]
        if last_content and isinstance(last_content[-1], dict) and last_content[-1].get("type") == "tool_result":
            last_content.append(result_block)
            return

    messages.append({"role": "user", "content": [result_block]})


def _convert_tool_result_content(content: list) -> str | list[dict[str, Any]]:
    """Convert tool result content blocks."""
    if not content:
        return ""
    if len(content) == 1 and isinstance(content[0], TextContent):
        return content[0].text

    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, TextContent):
            blocks.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": item.mime_type,
                    "data": item.data,
                },
            })
    return blocks


def _convert_tools(tools: list | None) -> list[dict[str, Any]] | None:
    """Convert bampy Tool definitions to Anthropic tool format."""
    if not tools:
        return None
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------

_STOP_REASON_MAP: dict[str, StopReason] = {
    "end_turn": StopReason.STOP,
    "stop_sequence": StopReason.STOP,
    "max_tokens": StopReason.LENGTH,
    "tool_use": StopReason.TOOL_USE,
    "refusal": StopReason.ERROR,
}


# ---------------------------------------------------------------------------
# Public stream functions
# ---------------------------------------------------------------------------

def stream_anthropic(
    model: Model,
    context: Context,
    options: AnthropicOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream an Anthropic Messages API response with fine-grained events."""
    event_stream = AssistantMessageEventStream()

    async def _run() -> None:
        try:
            import anthropic as anthropic_sdk
        except ImportError as e:
            _emit_error(event_stream, model, f"anthropic SDK not installed: {e}")
            return

        output = AssistantMessage(
            api=model.api,
            provider=model.provider,
            model=model.id,
            content=[],
        )

        try:
            # Build client
            api_key = options.api_key if options else None
            base_url = model.base_url or None
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

            client = anthropic_sdk.AsyncAnthropic(**client_kwargs)

            # Build params
            system, messages = _convert_messages(context)
            max_tokens = (options.max_tokens if options and options.max_tokens else None) or model.max_tokens

            params: dict[str, Any] = {
                "model": model.id,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if system:
                params["system"] = system
            if options and options.temperature is not None:
                params["temperature"] = options.temperature

            tools = _convert_tools(context.tools)
            if tools:
                params["tools"] = tools

            # Thinking config
            thinking_config = options.thinking if options and isinstance(options, AnthropicOptions) else None
            thinking = _resolve_thinking(model, None, thinking_config)
            if thinking:
                params["thinking"] = thinking
                # Anthropic requires temperature to be unset or 1.0 with thinking
                params.pop("temperature", None)

            # Emit start
            event_stream.push(StartEvent(partial=output))

            # Track content block state
            block_types: dict[int, str] = {}  # index → "text" | "thinking" | "tool_use"
            tool_json_bufs: dict[int, str] = {}  # index → accumulated JSON string

            # Stream
            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    _handle_sse_event(event, output, event_stream, block_types, tool_json_bufs)

            # Update usage cost
            output.usage.cost = calculate_cost(model, output.usage)

            event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            event_stream.end(output)

        except Exception as e:
            output.stop_reason = StopReason.ERROR
            output.error_message = str(e)
            event_stream.push(ErrorEvent(reason=StopReason.ERROR, error=output))
            event_stream.end(output)

    asyncio.get_running_loop().create_task(_run())
    return event_stream


def stream_simple_anthropic(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Simplified streaming — maps SimpleStreamOptions to AnthropicOptions."""
    anthropic_opts: AnthropicOptions | None = None

    if options is not None:
        thinking_config = None
        if options.reasoning is not None and model.reasoning:
            thinking = _resolve_thinking(model, options.reasoning, None)
            if thinking:
                if thinking["type"] == "adaptive":
                    thinking_config = AnthropicThinkingAdaptive(effort=thinking["effort"])
                else:
                    thinking_config = AnthropicThinkingEnabled(budget_tokens=thinking["budget_tokens"])

        anthropic_opts = AnthropicOptions(
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            api_key=options.api_key,
            max_retry_delay_ms=options.max_retry_delay_ms,
            headers=options.headers,
            thinking=thinking_config,
        )

    return stream_anthropic(model, context, anthropic_opts)


# ---------------------------------------------------------------------------
# SSE event handling
# ---------------------------------------------------------------------------

def _handle_sse_event(
    event: Any,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    block_types: dict[int, str],
    tool_json_bufs: dict[int, str],
) -> None:
    """Map a single Anthropic SSE event to bampy events."""
    etype = getattr(event, "type", None)

    if etype == "message_start":
        # Extract initial usage
        msg = getattr(event, "message", None)
        if msg:
            usage = getattr(msg, "usage", None)
            if usage:
                output.usage.input = getattr(usage, "input_tokens", 0)
                output.usage.cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                output.usage.cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            output.response_id = getattr(msg, "id", None)

    elif etype == "content_block_start":
        index = getattr(event, "index", 0)
        block = getattr(event, "content_block", None)
        if block is None:
            return
        block_type = getattr(block, "type", "text")
        block_types[index] = block_type

        if block_type == "text":
            content = TextContent(text=getattr(block, "text", ""))
            output.content.append(content)
            stream.push(TextStartEvent(
                content_index=index, content=content, partial=output,
            ))

        elif block_type == "thinking":
            content = ThinkingContent(thinking=getattr(block, "thinking", ""))
            output.content.append(content)
            stream.push(ThinkingStartEvent(
                content_index=index, content=content, partial=output,
            ))

        elif block_type == "tool_use":
            tool_id = sanitize_tool_call_id(getattr(block, "id", ""))
            content = ToolCall(
                id=tool_id,
                name=getattr(block, "name", ""),
                arguments={},
            )
            output.content.append(content)
            tool_json_bufs[index] = ""
            stream.push(ToolCallStartEvent(
                content_index=index, content=content, partial=output,
            ))

        elif block_type == "redacted_thinking":
            content = ThinkingContent(thinking="", redacted=True)
            output.content.append(content)
            stream.push(ThinkingStartEvent(
                content_index=index, content=content, partial=output,
            ))

    elif etype == "content_block_delta":
        index = getattr(event, "index", 0)
        delta = getattr(event, "delta", None)
        if delta is None:
            return
        delta_type = getattr(delta, "type", "")
        block_type = block_types.get(index)

        if delta_type == "text_delta" and block_type == "text":
            text = getattr(delta, "text", "")
            # Update accumulated text
            if index < len(output.content):
                block = output.content[index]
                if isinstance(block, TextContent):
                    block.text += text
            stream.push(TextDeltaEvent(
                content_index=index, delta=text, partial=output,
            ))

        elif delta_type == "thinking_delta" and block_type in ("thinking", "redacted_thinking"):
            text = getattr(delta, "thinking", "")
            if index < len(output.content):
                block = output.content[index]
                if isinstance(block, ThinkingContent):
                    block.thinking += text
            stream.push(ThinkingDeltaEvent(
                content_index=index, delta=text, partial=output,
            ))

        elif delta_type == "input_json_delta" and block_type == "tool_use":
            json_chunk = getattr(delta, "partial_json", "")
            tool_json_bufs[index] = tool_json_bufs.get(index, "") + json_chunk
            stream.push(ToolCallDeltaEvent(
                content_index=index, delta=json_chunk, partial=output,
            ))

    elif etype == "content_block_stop":
        index = getattr(event, "index", 0)
        block_type = block_types.get(index)

        if index < len(output.content):
            content_block = output.content[index]

            # Parse final tool arguments
            if block_type == "tool_use" and isinstance(content_block, ToolCall):
                raw_json = tool_json_bufs.get(index, "")
                try:
                    content_block.arguments = json.loads(raw_json) if raw_json else {}
                except json.JSONDecodeError:
                    content_block.arguments = {}

                stream.push(ToolCallEndEvent(
                    content_index=index, content=content_block, partial=output,
                ))

            elif block_type == "text" and isinstance(content_block, TextContent):
                stream.push(TextEndEvent(
                    content_index=index, content=content_block, partial=output,
                ))

            elif block_type in ("thinking", "redacted_thinking") and isinstance(content_block, ThinkingContent):
                # Capture signature if present
                sig = getattr(getattr(event, "content_block", None), "signature", None)
                if sig:
                    content_block.thinking_signature = sig
                stream.push(ThinkingEndEvent(
                    content_index=index, content=content_block, partial=output,
                ))

    elif etype == "message_delta":
        delta = getattr(event, "delta", None)
        if delta:
            stop = getattr(delta, "stop_reason", None)
            if stop:
                output.stop_reason = _STOP_REASON_MAP.get(stop, StopReason.STOP)

        usage = getattr(event, "usage", None)
        if usage:
            output.usage.output = getattr(usage, "output_tokens", 0)
            output.usage.total_tokens = (
                output.usage.input + output.usage.output
                + output.usage.cache_read + output.usage.cache_write
            )


def _emit_error(
    stream: AssistantMessageEventStream,
    model: Model,
    message: str,
) -> None:
    """Emit an error event and end the stream."""
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
# Provider entry (used by registry)
# ---------------------------------------------------------------------------

def get_provider_entry() -> ApiProviderEntry:
    return ApiProviderEntry(
        api="anthropic-messages",
        stream=stream_anthropic,  # type: ignore[arg-type]
        stream_simple=stream_simple_anthropic,  # type: ignore[arg-type]
    )
