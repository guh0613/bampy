"""Google Gemini API provider adapter.

Maps Google GenAI SDK streaming events → bampy AssistantMessageEvent protocol.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

from bampy.ai.models import calculate_cost
from bampy.ai.provider import ApiProviderEntry
from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    GeminiOptions,
    ImageContent,
    Model,
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
    ToolCallEndEvent,
    ToolCallStartEvent,
)


# ---------------------------------------------------------------------------
# Thinking level → budget mapping
# ---------------------------------------------------------------------------

_THINKING_BUDGETS: dict[ThinkingLevel, int] = {
    ThinkingLevel.MINIMAL: 1024,
    ThinkingLevel.LOW: 2048,
    ThinkingLevel.MEDIUM: 8192,
    ThinkingLevel.HIGH: 16384,
}


# ---------------------------------------------------------------------------
# Message conversion (bampy → Gemini API format)
# ---------------------------------------------------------------------------

def _supports_multimodal_tool_result(model_id: str) -> bool:
    """Check if the model supports image data in function responses."""
    return model_id.startswith("gemini-3")


def _convert_messages(context: Context, *, model_id: str = "") -> list[Any]:
    """Convert context messages to Gemini contents format."""
    from google.genai import types

    from bampy.ai.types import AssistantMessage, ToolResultMessage, UserMessage

    multimodal_tr = _supports_multimodal_tool_result(model_id)
    contents: list[Any] = []

    for msg in context.messages:
        if isinstance(msg, UserMessage):
            parts = _convert_user_parts(msg.content)
            contents.append(types.Content(role="user", parts=parts))

        elif isinstance(msg, AssistantMessage):
            parts = _convert_assistant_parts(msg.content)
            if parts:
                contents.append(types.Content(role="model", parts=parts))

        elif isinstance(msg, ToolResultMessage):
            _append_tool_result(contents, msg, multimodal=multimodal_tr)

    return contents


def _convert_user_parts(content: str | list) -> list[Any]:
    """Convert user message content to Gemini Parts."""
    from google.genai import types

    if isinstance(content, str):
        return [types.Part(text=content)]

    parts: list[Any] = []
    for item in content:
        if isinstance(item, TextContent):
            parts.append(types.Part(text=item.text))
        elif isinstance(item, ImageContent):
            parts.append(types.Part(
                inline_data=types.Blob(
                    mime_type=item.mime_type,
                    data=base64.b64decode(item.data),
                ),
            ))
    return parts or [types.Part(text="")]


def _convert_assistant_parts(content: list) -> list[Any]:
    """Convert assistant content blocks to Gemini Parts."""
    from google.genai import types

    parts: list[Any] = []
    for item in content:
        if isinstance(item, TextContent):
            parts.append(types.Part(text=item.text))
        elif isinstance(item, ThinkingContent):
            # Gemini doesn't accept thinking blocks in input; skip
            pass
        elif isinstance(item, ToolCall):
            fc = types.FunctionCall(
                name=item.name,
                args=item.arguments,
                id=item.id,
            )
            part_kwargs: dict[str, Any] = {"function_call": fc}
            # Restore thought_signature for Gemini 3.x models
            if item.thought_signature is not None:
                part_kwargs["thought_signature"] = item.thought_signature
            parts.append(types.Part(**part_kwargs))
    return parts


def _append_tool_result(
    contents: list[Any], msg: Any, *, multimodal: bool = False,
) -> None:
    """Append a tool result as a user Content with function_response."""
    from google.genai import types

    # Separate text and image content
    texts: list[str] = []
    image_parts: list[Any] = []
    for item in msg.content:
        if isinstance(item, TextContent):
            texts.append(item.text)
        elif isinstance(item, ImageContent):
            if multimodal:
                image_parts.append(
                    types.FunctionResponsePart.from_bytes(
                        data=base64.b64decode(item.data),
                        mime_type=item.mime_type,
                    )
                )
            else:
                # Fallback: model doesn't support multimodal function responses
                texts.append("[image]")

    response_data = {"result": "\n".join(texts) if texts else ""}
    fr_kwargs: dict[str, Any] = {
        "name": msg.tool_name,
        "response": response_data,
    }
    if image_parts:
        fr_kwargs["parts"] = image_parts

    part = types.Part(
        function_response=types.FunctionResponse(**fr_kwargs),
    )

    # Group consecutive tool results into one user Content
    if contents and getattr(contents[-1], "role", None) == "user":
        last_parts = getattr(contents[-1], "parts", [])
        if last_parts and any(
            getattr(p, "function_response", None) is not None for p in last_parts
        ):
            last_parts.append(part)
            return

    contents.append(types.Content(role="user", parts=[part]))


def _convert_tools(tools: list | None) -> list[Any] | None:
    """Convert bampy Tool definitions to Gemini tool format."""
    if not tools:
        return None

    from google.genai import types

    declarations = []
    for t in tools:
        declarations.append(types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters_json_schema=t.parameters,
        ))
    return [types.Tool(function_declarations=declarations)]


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------

_STOP_REASON_MAP: dict[str, StopReason] = {
    "STOP": StopReason.STOP,
    "MAX_TOKENS": StopReason.LENGTH,
    "SAFETY": StopReason.ERROR,
    "RECITATION": StopReason.ERROR,
    "MALFORMED_FUNCTION_CALL": StopReason.ERROR,
    "OTHER": StopReason.ERROR,
}


# ---------------------------------------------------------------------------
# Public stream functions
# ---------------------------------------------------------------------------

def stream_gemini(
    model: Model,
    context: Context,
    options: GeminiOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a Gemini API response with fine-grained events."""
    event_stream = AssistantMessageEventStream()

    async def _run() -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            _emit_error(event_stream, model, f"google-genai SDK not installed: {e}")
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
            client_kwargs: dict[str, Any] = {}
            if api_key:
                client_kwargs["api_key"] = api_key

            # Support custom base_url and headers via http_options
            base_url = model.base_url or None
            extra_headers: dict[str, str] = {}
            if model.headers:
                extra_headers.update(model.headers)
            if options and options.headers:
                extra_headers.update(options.headers)
            if base_url or extra_headers:
                http_opts: dict[str, Any] = {}
                if base_url:
                    http_opts["base_url"] = base_url
                if extra_headers:
                    http_opts["headers"] = extra_headers
                client_kwargs["http_options"] = http_opts

            client = genai.Client(**client_kwargs)

            # Build config
            contents = _convert_messages(context, model_id=model.id)
            config_kwargs: dict[str, Any] = {}

            max_tokens = (
                (options.max_tokens if options and options.max_tokens else None)
                or model.max_tokens
            )
            config_kwargs["max_output_tokens"] = max_tokens

            if options and options.temperature is not None:
                config_kwargs["temperature"] = options.temperature

            if context.system_prompt:
                config_kwargs["system_instruction"] = context.system_prompt

            tools = _convert_tools(context.tools)
            if tools:
                config_kwargs["tools"] = tools
                config_kwargs["automatic_function_calling"] = (
                    types.AutomaticFunctionCallingConfig(disable=True)
                )

            # Thinking config
            if (
                options
                and isinstance(options, GeminiOptions)
                and options.thinking_budget is not None
            ):
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=options.thinking_budget,
                    include_thoughts=True,
                )
            elif model.reasoning:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=True,
                )

            config = types.GenerateContentConfig(**config_kwargs)

            # Emit start
            event_stream.push(StartEvent(partial=output))

            # Track streaming state
            current_block_type: str | None = None  # "text" | "thinking"
            content_index = -1

            # Stream
            async for chunk in await client.aio.models.generate_content_stream(
                model=model.id,
                contents=contents,
                config=config,
            ):
                # Capture response_id
                resp_id = getattr(chunk, "response_id", None)
                if resp_id:
                    output.response_id = resp_id

                if not chunk.candidates:
                    if chunk.usage_metadata:
                        _update_usage(output, chunk.usage_metadata)
                    continue

                candidate = chunk.candidates[0]

                # Check finish reason
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason is not None:
                    reason_name = (
                        getattr(finish_reason, "name", None)
                        or str(finish_reason)
                    )
                    output.stop_reason = _STOP_REASON_MAP.get(
                        reason_name, StopReason.STOP,
                    )

                content = getattr(candidate, "content", None)
                if not content:
                    if chunk.usage_metadata:
                        _update_usage(output, chunk.usage_metadata)
                    continue

                parts = getattr(content, "parts", None) or []
                for part in parts:
                    fc = getattr(part, "function_call", None)
                    is_thought = getattr(part, "thought", False)
                    text = getattr(part, "text", None)

                    if fc is not None:
                        # End previous block if any
                        if current_block_type is not None:
                            _end_current_block(
                                event_stream, output,
                                content_index, current_block_type,
                            )
                            current_block_type = None

                        # Emit tool call as a complete block
                        args = dict(fc.args) if fc.args else {}
                        call_id = (
                            getattr(fc, "id", None)
                            or f"gemini_call_{len(output.content)}"
                        )
                        sig = getattr(part, "thought_signature", None)
                        tool_call = ToolCall(
                            id=call_id,
                            name=fc.name,
                            arguments=args,
                            thought_signature=sig,
                        )
                        output.content.append(tool_call)
                        content_index = len(output.content) - 1
                        event_stream.push(ToolCallStartEvent(
                            content_index=content_index,
                            content=tool_call,
                            partial=output,
                        ))
                        event_stream.push(ToolCallEndEvent(
                            content_index=content_index,
                            content=tool_call,
                            partial=output,
                        ))

                    elif is_thought and text:
                        if current_block_type != "thinking":
                            # End previous block
                            if current_block_type is not None:
                                _end_current_block(
                                    event_stream, output,
                                    content_index, current_block_type,
                                )
                            # Start thinking block
                            thinking = ThinkingContent(thinking="")
                            output.content.append(thinking)
                            content_index = len(output.content) - 1
                            current_block_type = "thinking"
                            event_stream.push(ThinkingStartEvent(
                                content_index=content_index,
                                content=thinking,
                                partial=output,
                            ))

                        # Delta
                        block = output.content[content_index]
                        if isinstance(block, ThinkingContent):
                            block.thinking += text
                        event_stream.push(ThinkingDeltaEvent(
                            content_index=content_index,
                            delta=text,
                            partial=output,
                        ))

                    elif text:
                        if current_block_type != "text":
                            # End previous block
                            if current_block_type is not None:
                                _end_current_block(
                                    event_stream, output,
                                    content_index, current_block_type,
                                )
                            # Start text block
                            text_content = TextContent(text="")
                            output.content.append(text_content)
                            content_index = len(output.content) - 1
                            current_block_type = "text"
                            event_stream.push(TextStartEvent(
                                content_index=content_index,
                                content=text_content,
                                partial=output,
                            ))

                        # Delta
                        block = output.content[content_index]
                        if isinstance(block, TextContent):
                            block.text += text
                        event_stream.push(TextDeltaEvent(
                            content_index=content_index,
                            delta=text,
                            partial=output,
                        ))

                # Usage from chunk
                if chunk.usage_metadata:
                    _update_usage(output, chunk.usage_metadata)

            # End final block
            if current_block_type is not None:
                _end_current_block(
                    event_stream, output,
                    content_index, current_block_type,
                )

            # Override stop reason if tool calls present
            if any(isinstance(b, ToolCall) for b in output.content):
                output.stop_reason = StopReason.TOOL_USE

            # Calculate cost
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


def stream_simple_gemini(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Simplified streaming — maps SimpleStreamOptions to GeminiOptions."""
    gemini_opts: GeminiOptions | None = None

    if options is not None:
        thinking_budget = None
        if options.reasoning is not None and model.reasoning:
            thinking_budget = _THINKING_BUDGETS.get(options.reasoning, 8192)

        gemini_opts = GeminiOptions(
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            api_key=options.api_key,
            max_retry_delay_ms=options.max_retry_delay_ms,
            headers=options.headers,
            thinking_budget=thinking_budget,
        )

    return stream_gemini(model, context, gemini_opts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _end_current_block(
    stream: AssistantMessageEventStream,
    output: AssistantMessage,
    content_index: int,
    block_type: str,
) -> None:
    """Emit an end event for the current content block."""
    if content_index < 0 or content_index >= len(output.content):
        return
    block = output.content[content_index]
    if block_type == "text" and isinstance(block, TextContent):
        stream.push(TextEndEvent(
            content_index=content_index, content=block, partial=output,
        ))
    elif block_type == "thinking" and isinstance(block, ThinkingContent):
        stream.push(ThinkingEndEvent(
            content_index=content_index, content=block, partial=output,
        ))


def _update_usage(output: AssistantMessage, usage_meta: Any) -> None:
    """Update usage from Gemini usage metadata."""
    output.usage.input = getattr(usage_meta, "prompt_token_count", 0) or 0
    output.usage.output = getattr(usage_meta, "candidates_token_count", 0) or 0
    output.usage.cache_read = (
        getattr(usage_meta, "cached_content_token_count", 0) or 0
    )
    output.usage.total_tokens = (
        getattr(usage_meta, "total_token_count", 0) or 0
    )


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
        api="google-genai",
        stream=stream_gemini,  # type: ignore[arg-type]
        stream_simple=stream_simple_gemini,  # type: ignore[arg-type]
    )
