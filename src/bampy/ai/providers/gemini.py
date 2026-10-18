"""Google Gemini Developer API and Vertex AI provider adapter.

Maps Google GenAI SDK streaming events → bampy AssistantMessageEvent protocol.
"""

from __future__ import annotations

import base64
import re
import uuid
from contextlib import aclosing
from typing import Any

from bampy.ai.models import calculate_cost
from bampy.ai.providers._cancellation import spawn_provider_task
from bampy.ai.api_registry import ApiProviderEntry
from bampy.ai.providers._transform import sanitize_tool_call_id, transform_messages
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
    VertexAIOptions,
)


# ---------------------------------------------------------------------------
# Thinking level → budget mapping
# ---------------------------------------------------------------------------

_THINKING_BUDGETS: dict[ThinkingLevel, int] = {
    ThinkingLevel.MINIMAL: 1024,
    ThinkingLevel.LOW: 2048,
    ThinkingLevel.MEDIUM: 8192,
    ThinkingLevel.HIGH: 16384,
    ThinkingLevel.XHIGH: 16384,
    ThinkingLevel.MAX: 16384,
}
_GEMINI25_THINKING_LEVEL_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}
_THINKING_LEVELS: dict[ThinkingLevel, str] = {
    ThinkingLevel.MINIMAL: "minimal",
    ThinkingLevel.LOW: "low",
    ThinkingLevel.MEDIUM: "medium",
    ThinkingLevel.HIGH: "high",
    ThinkingLevel.XHIGH: "high",
    ThinkingLevel.MAX: "high",
}
_SKIP_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"


# ---------------------------------------------------------------------------
# Message conversion (bampy → Gemini API format)
# ---------------------------------------------------------------------------

def _gemini_version(model_id: str) -> tuple[int, int]:
    """Recognize model IDs as well as SDK-supported model resource names."""
    match = re.match(r"gemini-(\d+)(?:\.(\d+))?(?:-|$)", model_id.rsplit("/", 1)[-1])
    return (int(match[1]), int(match[2] or 0)) if match else (0, 0)


def _supports_multimodal_tool_result(model_id: str) -> bool:
    """Check if the model supports image data in function responses."""
    return _gemini_version(model_id) >= (3, 0)


def _uses_thinking_level(model_id: str) -> bool:
    """Whether a Gemini model should use thinking_level instead of budget."""
    return _gemini_version(model_id) >= (3, 0)


def _normalize_thinking_level(model_id: str, level: str) -> str:
    """Clamp a user-provided thinking level to the model family's support."""
    version = _gemini_version(model_id)
    if version == (3, 0) and "-pro" in model_id:
        return "low" if level in {"minimal", "low"} else "high"
    if level == "minimal" and (
        version >= (4, 0)
        or "-pro" in model_id
        or (version >= (3, 7) and "-flash" in model_id)
    ):
        return "low"
    return level


def _reasoning_to_thinking_level(
    model_id: str,
    reasoning: ThinkingLevel,
) -> str:
    """Map bampy's reasoning enum to the model's thinking_level values."""
    return _normalize_thinking_level(model_id, _THINKING_LEVELS[reasoning])


def _requires_tool_call_id(model_id: str) -> bool:
    """Whether function call / response ids must be explicitly preserved."""
    return (
        _gemini_version(model_id) >= (3, 5)
        or model_id.startswith("claude-")
        or model_id.startswith("gpt-oss-")
    )


def _normalize_gemini_tool_call_id(tool_call_id: str, model_id: str) -> str:
    """Normalize third-party IDs while preserving Gemini's opaque call IDs."""
    if not model_id.startswith(("claude-", "gpt-oss-")):
        return tool_call_id
    return sanitize_tool_call_id(tool_call_id)


def _convert_messages(model: Model, context: Context) -> list[Any]:
    """Convert context messages to Gemini contents format."""
    from google.genai import types

    from bampy.ai.types import AssistantMessage, ToolResultMessage, UserMessage

    multimodal_tr = _supports_multimodal_tool_result(model.id)
    contents: list[Any] = []
    transformed = transform_messages(
        context.messages,
        target=model,
        normalize_id=lambda tool_call_id, _source: _normalize_gemini_tool_call_id(
            tool_call_id, model.id
        ),
    )

    for msg in transformed:
        if isinstance(msg, UserMessage):
            parts = _convert_user_parts(msg.content)
            contents.append(types.Content(role="user", parts=parts))

        elif isinstance(msg, AssistantMessage):
            parts = _convert_assistant_parts(model, msg.content)
            if parts:
                contents.append(types.Content(role="model", parts=parts))

        elif isinstance(msg, ToolResultMessage):
            _append_tool_result(
                contents,
                msg,
                multimodal=multimodal_tr,
                model_id=model.id,
            )

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


def _convert_assistant_parts(model: Model, content: list) -> list[Any]:
    """Convert assistant content blocks to Gemini Parts."""
    from google.genai import types

    parts: list[Any] = []
    for item in content:
        if isinstance(item, TextContent):
            if not item.text and not item.text_signature:
                continue
            part_kwargs: dict[str, Any] = {"text": item.text}
            if isinstance(item.text_signature, bytes) and item.text_signature:
                part_kwargs["thought_signature"] = item.text_signature
            parts.append(types.Part(**part_kwargs))
        elif isinstance(item, ThinkingContent):
            if not item.thinking and not isinstance(item.thinking_signature, bytes):
                continue
            part_kwargs = {
                "thought": True,
                "text": item.thinking,
            }
            if isinstance(item.thinking_signature, bytes) and item.thinking_signature:
                part_kwargs["thought_signature"] = item.thinking_signature
            parts.append(types.Part(**part_kwargs))
        elif isinstance(item, ToolCall):
            fc_kwargs: dict[str, Any] = {
                "name": item.name,
                "args": item.arguments,
            }
            if _requires_tool_call_id(model.id):
                fc_kwargs["id"] = item.id
            fc = types.FunctionCall(**fc_kwargs)
            part_kwargs: dict[str, Any] = {"function_call": fc}
            effective_signature = item.thought_signature
            if effective_signature is None and _uses_thinking_level(model.id):
                effective_signature = _SKIP_THOUGHT_SIGNATURE
            if effective_signature is not None:
                part_kwargs["thought_signature"] = effective_signature
            parts.append(types.Part(**part_kwargs))
    return parts


def _append_tool_result(
    contents: list[Any],
    msg: Any,
    *,
    multimodal: bool = False,
    model_id: str = "",
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
    if _requires_tool_call_id(model_id):
        fr_kwargs["id"] = msg.tool_call_id
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


def _build_client_kwargs(model: Model, options: GeminiOptions | None) -> dict[str, Any]:
    """Select the channel explicitly, leaving credential discovery to the SDK."""
    vertex = model.api == "google-vertex"
    kwargs: dict[str, Any] = {"enterprise": vertex}
    if options and options.api_key:
        kwargs["api_key"] = options.api_key
    if isinstance(options, VertexAIOptions):
        if not vertex:
            raise ValueError("VertexAIOptions requires Model.api='google-vertex'")
        for field in ("project", "location", "credentials"):
            value = getattr(options, field)
            if value is not None:
                kwargs[field] = value

    http_options: dict[str, Any] = {}
    if model.base_url:
        # The SDK appends api_version itself. Accept legacy base URLs that
        # already end in /v1beta (or a gateway's /v1beta1) without duplicating it.
        version_suffix = re.search(r"/(v\d+(?:(?:alpha|beta)\d*)?)/?$", model.base_url)
        if version_suffix:
            http_options["base_url"] = model.base_url[:version_suffix.start()]
            http_options["api_version"] = version_suffix[1]
        else:
            http_options["base_url"] = model.base_url
    headers = {**(model.headers or {}), **(options.headers or {} if options else {})}
    if headers:
        http_options["headers"] = headers
    if isinstance(options, GeminiOptions):
        if options.api_version:
            http_options["api_version"] = options.api_version
        if options.timeout_ms is not None:
            http_options["timeout"] = options.timeout_ms
    if _gemini_version(model.id) >= (4, 0):
        # Each continuation hop can take up to 20 minutes.
        http_options.setdefault("timeout", 1_200_000)
        max_tokens = (
            options.max_tokens if options and options.max_tokens is not None else model.max_tokens
        )
        if vertex and max_tokens > 65_536 and http_options.get("api_version") == "v1":
            raise ValueError("Gemini 4 long decode on Vertex AI requires api_version='v1beta1'")
    if http_options:
        kwargs["http_options"] = http_options
    return kwargs


def _build_config(model: Model, context: Context, options: GeminiOptions | None) -> Any:
    from google.genai import types

    max_tokens = (
        options.max_tokens if options and options.max_tokens is not None else model.max_tokens
    )
    kwargs: dict[str, Any] = {
        "max_output_tokens": max_tokens,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    version = _gemini_version(model.id)
    if options and options.temperature is not None and version < (3, 5):
        kwargs["temperature"] = options.temperature
    if context.system_prompt:
        kwargs["system_instruction"] = context.system_prompt
    tools = _convert_tools(context.tools)
    if tools:
        kwargs["tools"] = tools
        if model.api == "google-vertex":
            # bampy exposes complete tool calls, not Vertex's PartialArg protocol.
            kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    stream_function_call_arguments=False,
                ),
            )

    thinking: dict[str, Any] = {}
    if isinstance(options, GeminiOptions):
        if version >= (3, 5) and options.thinking_budget is not None:
            raise ValueError(f"{model.id} requires thinking_level; use it instead of thinking_budget")
        if options.thinking_budget is not None and options.thinking_level is not None:
            raise ValueError("Set only one of thinking_level and thinking_budget")
        if options.thinking_level is not None:
            if _uses_thinking_level(model.id):
                thinking["thinking_level"] = _normalize_thinking_level(
                    model.id, options.thinking_level,
                )
            else:
                thinking["thinking_budget"] = _GEMINI25_THINKING_LEVEL_BUDGETS[
                    options.thinking_level
                ]
        elif options.thinking_budget is not None:
            thinking["thinking_budget"] = options.thinking_budget
    if thinking or model.reasoning:
        kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True, **thinking)
    return types.GenerateContentConfig(**kwargs)


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
    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        content=[],
    )

    async def _run() -> None:
        try:
            from google import genai
        except ImportError as e:
            _emit_error(event_stream, model, f"google-genai SDK not installed: {e}")
            return

        try:
            contents = _convert_messages(model, context)
            if _gemini_version(model.id) >= (3, 5) and contents and contents[-1].role == "model":
                raise ValueError(
                    f"{model.id} does not support assistant prefills; "
                    "end the conversation with a user message or tool result"
                )
            config = _build_config(model, context, options)

            event_stream.push(StartEvent(partial=output))
            state = _GeminiStreamState(event_stream, output)
            # Close both SDK transports on success, failure, and cancellation.
            with genai.Client(**_build_client_kwargs(model, options)) as client:
                async with client.aio as aclient:
                    response = await aclient.models.generate_content_stream(
                        model=model.id,
                        contents=contents,
                        config=config,
                    )
                    async with aclosing(response):
                        async for chunk in response:
                            state.process(chunk)
            state.finish()

            # Calculate cost
            output.usage.cost = calculate_cost(model, output.usage)

            if output.stop_reason == StopReason.ERROR:
                event_stream.push(ErrorEvent(reason=StopReason.ERROR, error=output))
            else:
                event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            event_stream.end(output)

        except Exception as e:
            output.stop_reason = StopReason.ERROR
            output.error_message = str(e)
            output.usage.cost = calculate_cost(model, output.usage)
            event_stream.push(ErrorEvent(reason=StopReason.ERROR, error=output))
            event_stream.end(output)

    spawn_provider_task(
        event_stream=event_stream,
        output=output,
        options=options,
        runner=_run,
    )
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
        thinking_level = None
        if options.reasoning is not None and model.reasoning:
            if _uses_thinking_level(model.id):
                thinking_level = _reasoning_to_thinking_level(
                    model.id,
                    options.reasoning,
                )
            else:
                thinking_budget = _THINKING_BUDGETS.get(options.reasoning, 8192)

        gemini_opts = GeminiOptions(
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            api_key=options.api_key,
            max_retry_delay_ms=options.max_retry_delay_ms,
            headers=options.headers,
            cancellation=options.cancellation,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
        )

    return stream_gemini(model, context, gemini_opts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _GeminiStreamState:
    """Translate one SDK stream, including automatically resumed decode hops."""

    def __init__(self, stream: AssistantMessageEventStream, output: AssistantMessage):
        self.stream = stream
        self.output = output
        self.current_block_type: str | None = None
        self.content_index = -1
        self.saw_finish = False

    def process(self, chunk: Any) -> None:
        if chunk.response_id:
            self.output.response_id = chunk.response_id
        # The SDK already accumulates usage across continuation hops.
        if chunk.usage_metadata:
            _update_usage(self.output, chunk.usage_metadata)
        feedback = chunk.prompt_feedback
        if feedback and feedback.block_reason not in (None, "BLOCKED_REASON_UNSPECIFIED"):
            self.saw_finish = True
            self.output.stop_reason = StopReason.ERROR
            self.output.error_message = (
                f"Gemini prompt blocked: {feedback.block_reason}"
                f" {feedback.block_reason_message or ''}"
            ).strip()
        if not chunk.candidates:
            return
        candidate = chunk.candidates[0]
        if candidate.finish_reason not in (None, "FINISH_REASON_UNSPECIFIED"):
            self.saw_finish = True
            reason = getattr(candidate.finish_reason, "name", None) or str(candidate.finish_reason)
            self.output.stop_reason = _STOP_REASON_MAP.get(reason, StopReason.ERROR)
            if self.output.stop_reason == StopReason.ERROR:
                self.output.error_message = (
                    f"Gemini generation stopped: {reason} {candidate.finish_message or ''}"
                ).strip()
        if candidate.content:
            for part in candidate.content.parts or []:
                self._process_part(part)

    def _close_block(self) -> None:
        if self.current_block_type is not None:
            _end_current_block(
                self.stream, self.output, self.content_index, self.current_block_type,
            )
            self.current_block_type = None

    def _process_part(self, part: Any) -> None:
        if part.function_call is not None:
            self._close_block()
            fc = part.function_call
            if fc.partial_args or fc.will_continue:
                raise ValueError("Received partial function arguments despite requesting complete calls")
            tool_call = ToolCall(
                id=fc.id or f"gemini_call_{uuid.uuid4().hex}",
                name=fc.name,
                arguments=dict(fc.args or {}),
                thought_signature=part.thought_signature,
            )
            self.output.content.append(tool_call)
            self.content_index = len(self.output.content) - 1
            self.stream.push(ToolCallStartEvent(
                content_index=self.content_index, content=tool_call, partial=self.output,
            ))
            self.stream.push(ToolCallEndEvent(
                content_index=self.content_index, content=tool_call, partial=self.output,
            ))
            return
        if part.text is None and part.thought_signature is None:
            return

        block_type = "thinking" if part.thought else "text"
        # A signature-bearing part must not be merged with neighboring parts.
        previous = self.output.content[self.content_index] if self.content_index >= 0 else None
        previous_signature = (
            previous.thinking_signature if isinstance(previous, ThinkingContent)
            else previous.text_signature if isinstance(previous, TextContent)
            else None
        )
        if (
            self.current_block_type != block_type
            or part.thought_signature is not None
            or previous_signature is not None
        ):
            self._close_block()
            block = (
                ThinkingContent(thinking="", thinking_signature=part.thought_signature)
                if part.thought else TextContent(text="", text_signature=part.thought_signature)
            )
            self.output.content.append(block)
            self.content_index = len(self.output.content) - 1
            self.current_block_type = block_type
            start_event = ThinkingStartEvent if part.thought else TextStartEvent
            self.stream.push(start_event(
                content_index=self.content_index, content=block, partial=self.output,
            ))
        block = self.output.content[self.content_index]
        text = part.text or ""
        if isinstance(block, ThinkingContent):
            block.thinking += text
            delta_event = ThinkingDeltaEvent
        else:
            assert isinstance(block, TextContent)
            block.text += text
            delta_event = TextDeltaEvent
        self.stream.push(delta_event(
            content_index=self.content_index, delta=text, partial=self.output,
        ))

    def finish(self) -> None:
        self._close_block()
        if not self.saw_finish:
            raise ValueError("Gemini stream ended without a finish reason")
        if self.output.stop_reason == StopReason.STOP and any(
            isinstance(block, ToolCall) for block in self.output.content
        ):
            self.output.stop_reason = StopReason.TOOL_USE


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
    """Normalize cumulative usage into disjoint billable token categories."""
    prompt = getattr(usage_meta, "prompt_token_count", 0) or 0
    cached = getattr(usage_meta, "cached_content_token_count", 0) or 0
    output.usage.input = max(0, prompt - cached)
    output.usage.output = (
        (getattr(usage_meta, "candidates_token_count", 0) or 0)
        + (getattr(usage_meta, "thoughts_token_count", 0) or 0)
    )
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


def get_vertex_provider_entry() -> ApiProviderEntry:
    return ApiProviderEntry(
        api="google-vertex",
        stream=stream_gemini,  # type: ignore[arg-type]
        stream_simple=stream_simple_gemini,  # type: ignore[arg-type]
    )
