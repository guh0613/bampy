"""OpenAI provider adapters.

Supports both the Responses API (``openai-responses``) and the Chat
Completions API (``openai-completions``).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from bampy.ai.api_registry import ApiProviderEntry
from bampy.ai.models import calculate_cost, supports_xhigh
from bampy.ai.providers._cancellation import spawn_provider_task
from bampy.ai.providers._transform import sanitize_tool_call_id, transform_messages
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
    ToolResultMessage,
    UserMessage,
)
from bampy.ai.validation import parse_partial_json


# ---------------------------------------------------------------------------
# Reasoning effort mapping
# ---------------------------------------------------------------------------

_EFFORT_MAP: dict[ThinkingLevel, str] = {
    ThinkingLevel.MINIMAL: "minimal",
    ThinkingLevel.LOW: "low",
    ThinkingLevel.MEDIUM: "medium",
    ThinkingLevel.HIGH: "high",
    ThinkingLevel.XHIGH: "xhigh",
    ThinkingLevel.MAX: "max",
}

_CHAT_REASONING_FIELDS = (
    "reasoning_content",
    "reasoning",
    "reasoning_text",
)
_DEFAULT_USER_AGENT = "bampy/1.0"


def _resolve_reasoning_effort(
    model: Model,
    reasoning: ThinkingLevel | None,
) -> str | None:
    """Map simple reasoning levels to the provider's effort values."""
    if reasoning is None:
        return None
    return _normalize_reasoning_effort(model, _EFFORT_MAP.get(reasoning, "medium"))


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
    if not isinstance(signature, str) or not signature:
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


def _normalize_responses_id_part(part: str) -> str:
    """Normalize a Responses API identifier segment."""
    sanitized = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in part)
    return sanitized[:64].rstrip("_")


def _build_foreign_responses_item_id(item_id: str) -> str:
    """Build a stable foreign item id compatible with Responses API."""
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:16]
    return f"fc_{digest}"


def _normalize_responses_tool_call_id(
    tool_call_id: str,
    source: AssistantMessage,
    model: Model,
) -> str:
    """Normalize a tool-call id for Responses API history replay."""
    if "|" not in tool_call_id:
        return _normalize_responses_id_part(tool_call_id)

    call_id, item_id = tool_call_id.split("|", 1)
    normalized_call_id = _normalize_responses_id_part(call_id)
    is_foreign_tool_call = source.provider != model.provider or source.api != model.api
    normalized_item_id = (
        _build_foreign_responses_item_id(item_id)
        if is_foreign_tool_call
        else _normalize_responses_id_part(item_id)
    )
    if not normalized_item_id.startswith("fc_"):
        normalized_item_id = _normalize_responses_id_part(f"fc_{normalized_item_id}")
    if not normalized_item_id:
        normalized_item_id = _build_foreign_responses_item_id(item_id)
    return f"{normalized_call_id}|{normalized_item_id}"


def _supports_multimodal_tool_results(model: Model) -> bool:
    """Whether tool results may include image blocks for this model/provider."""
    return model.provider == "openai" and "image" in model.input_types


def _create_openai_client(
    openai_sdk: Any,
    model: Model,
    options: OpenAIOptions | None,
) -> Any:
    """Create an AsyncOpenAI client from model and per-call options."""
    api_key = options.api_key if options else None
    base_url = model.base_url or None
    if base_url and not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    extra_headers: dict[str, str] = {}
    if model.headers:
        extra_headers.update(model.headers)
    if options and options.headers:
        extra_headers.update(options.headers)
    extra_headers.setdefault("User-Agent", _DEFAULT_USER_AGENT)
    if extra_headers:
        client_kwargs["default_headers"] = extra_headers

    return openai_sdk.AsyncOpenAI(**client_kwargs)


def _option_value(obj: Any, name: str) -> Any:
    """Read a field from either a pydantic model or plain dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is not None:
        return value
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(name)
    return None


def _chat_reasoning_fields(model: Model) -> tuple[str, ...]:
    """Return reasoning delta fields recognized by the chat-completions path."""
    compat = model.openai_chat_compat
    if compat and compat.stream_reasoning_fields:
        fields = list(dict.fromkeys([*compat.stream_reasoning_fields, *_CHAT_REASONING_FIELDS]))
        return tuple(fields)
    return _CHAT_REASONING_FIELDS


def _jsonable_reasoning_value(value: Any) -> Any:
    """Convert SDK objects inside non-standard reasoning fields to JSON values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {key: _jsonable_reasoning_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_reasoning_value(item) for item in value]
    return value


def _chat_reasoning_value_to_text(value: Any) -> str:
    """Store a reasoning delta in ThinkingContent without losing raw JSON blocks."""
    if isinstance(value, str):
        return value
    return json.dumps(
        _jsonable_reasoning_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _try_parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _reasoning_detail_key(item: Any) -> tuple[Any, Any, Any] | None:
    if not isinstance(item, dict) or not isinstance(item.get("text"), str):
        return None
    if "index" not in item:
        return None
    return (item.get("index"), item.get("type"), item.get("format"))


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _merge_reasoning_details_value(existing: Any, incoming: Any) -> Any:
    if existing is None:
        return incoming

    current_items = _as_list(existing)
    incoming_items = _as_list(incoming)
    result = [_jsonable_reasoning_value(item) for item in current_items]
    keyed_positions = {
        key: pos
        for pos, item in enumerate(result)
        if (key := _reasoning_detail_key(item)) is not None
    }

    for item in incoming_items:
        item = _jsonable_reasoning_value(item)
        key = _reasoning_detail_key(item)
        pos = keyed_positions.get(key) if key is not None else None
        if pos is None or not isinstance(result[pos], dict):
            if key is not None:
                keyed_positions[key] = len(result)
            result.append(item)
            continue
        result[pos] = {
            **result[pos],
            "text": result[pos]["text"] + item["text"],
        }

    if isinstance(existing, list) or isinstance(incoming, list) or len(result) != 1:
        return result
    return result[0]


def _append_chat_reasoning_value(existing: str, field: str, value: Any) -> str:
    """Append a reasoning delta while preserving structured reasoning_details."""
    if field != "reasoning_details":
        return existing + _chat_reasoning_value_to_text(value)

    incoming = _try_parse_json(value) if isinstance(value, str) else None
    if incoming is None:
        if isinstance(value, str):
            return existing + value
        incoming = _jsonable_reasoning_value(value)

    if not existing:
        return _chat_reasoning_value_to_text(incoming)

    current = _try_parse_json(existing)
    if current is None:
        return existing + _chat_reasoning_value_to_text(value)

    return _chat_reasoning_value_to_text(
        _merge_reasoning_details_value(current, incoming)
    )


def _chat_replay_value(field: str, text: str) -> Any:
    """Convert stored thinking text back to the provider-specific replay value."""
    if field == "reasoning_details":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _merge_chat_replay_value(existing: Any, incoming: Any) -> Any:
    if existing is None:
        return incoming
    if isinstance(existing, str) and isinstance(incoming, str):
        return "\n".join(part for part in (existing, incoming) if part)
    if isinstance(existing, list) and isinstance(incoming, list):
        return [*existing, *incoming]
    if isinstance(existing, list):
        return [*existing, incoming]
    if isinstance(incoming, list):
        return [existing, *incoming]
    return [existing, incoming]


def _chat_replay_payloads(
    model: Model,
    thinking_blocks: list[ThinkingContent],
) -> dict[str, Any]:
    """Build assistant reasoning fields used to replay prior chat thinking."""
    reasoning_fields = _chat_reasoning_fields(model)
    compat = model.openai_chat_compat
    fallback_field = compat.replay_thinking_field if compat else None
    payloads: dict[str, Any] = {}

    for block in thinking_blocks:
        signature = block.thinking_signature
        field = (
            signature
            if isinstance(signature, str) and signature in reasoning_fields
            else fallback_field
        )
        if not field:
            continue
        value = _chat_replay_value(field, block.thinking)
        if field == "reasoning_details":
            payloads[field] = _merge_reasoning_details_value(
                payloads.get(field),
                value,
            )
        else:
            payloads[field] = _merge_chat_replay_value(payloads.get(field), value)

    return payloads


def _normalize_reasoning_effort(model: Model, effort: str | None) -> str | None:
    """Normalize a public reasoning effort to the target backend's vocabulary."""
    if effort is None or effort == "none":
        return None

    compat = model.openai_chat_compat
    if compat and compat.reasoning_effort_map:
        return compat.reasoning_effort_map.get(effort, effort)

    if effort == "max":
        return "xhigh" if supports_xhigh(model) else "high"
    if effort == "xhigh" and not supports_xhigh(model):
        return "high"
    return effort


def _chat_supports_reasoning_effort(model: Model) -> bool:
    """Whether the chat-completions backend accepts ``reasoning_effort``."""
    compat = model.openai_chat_compat
    return compat.supports_reasoning_effort if compat is not None else True


def _chat_max_tokens_field(model: Model) -> str:
    """Return the max-token parameter name for the backend."""
    compat = model.openai_chat_compat
    return compat.max_tokens_field if compat is not None else "max_completion_tokens"


def _chat_system_role(model: Model) -> str:
    """Return the system prompt role accepted by the chat-completions backend."""
    compat = model.openai_chat_compat
    if compat and compat.system_role:
        return compat.system_role
    return "developer" if model.reasoning else "system"


def _chat_thinking_enabled(model: Model, options: OpenAIOptions | None) -> bool:
    """Whether provider-specific thinking mode should be enabled."""
    compat = model.openai_chat_compat
    if compat is None or compat.thinking_param == "none":
        return False

    if compat.thinking_param in {"kimi", "zai", "deepseek"}:
        if options and options.reasoning_effort == "none":
            return False
        if options and options.reasoning_effort is not None:
            return True
        return compat.thinking_default_enabled

    return False


def _validate_chat_tool_choice(
    model: Model,
    options: OpenAIOptions | None,
    *,
    has_tools: bool,
) -> None:
    """Validate tool-choice constraints for compat chat-completions backends."""
    compat = model.openai_chat_compat
    if compat is None or not compat.thinking_tool_choice or not has_tools:
        return
    if not _chat_thinking_enabled(model, options):
        return

    tool_choice = options.tool_choice if options and options.tool_choice is not None else "auto"
    if not isinstance(tool_choice, str) or tool_choice not in compat.thinking_tool_choice:
        allowed = ", ".join(sorted(compat.thinking_tool_choice))
        raise ValueError(
            f"{model.provider}/{model.id} only supports tool_choice in {{{allowed}}} when thinking is enabled"
        )


def _build_chat_completion_params(
    model: Model,
    context: Context,
    options: OpenAIOptions | None,
) -> dict[str, Any]:
    """Build one chat-completions request payload."""
    max_tokens = (
        options.max_tokens
        if options and options.max_tokens is not None
        else model.max_tokens
    )
    params: dict[str, Any] = {
        "model": model.id,
        "messages": _convert_chat_completion_messages(model, context),
        "stream": True,
        "stream_options": {"include_usage": True},
        _chat_max_tokens_field(model): max_tokens,
    }
    extra_body: dict[str, Any] = {}
    compat = model.openai_chat_compat

    thinking_enabled = _chat_thinking_enabled(model, options)
    if options and options.temperature is not None and not (
        thinking_enabled and compat and compat.thinking_param == "deepseek"
    ):
        params["temperature"] = options.temperature
    if model.reasoning and options and _chat_supports_reasoning_effort(model):
        reasoning_effort = _normalize_reasoning_effort(model, options.reasoning_effort)
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
    if options and options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice
    if options and options.response_format is not None:
        params["response_format"] = options.response_format
    if options and options.parallel_tool_calls is not None:
        params["parallel_tool_calls"] = options.parallel_tool_calls
    if options and options.prompt_cache_key:
        params["prompt_cache_key"] = options.prompt_cache_key
    if options and options.prompt_cache_retention:
        params["prompt_cache_retention"] = options.prompt_cache_retention
    if options and options.service_tier:
        params["service_tier"] = options.service_tier
    if options and options.verbosity:
        params["verbosity"] = options.verbosity
    if compat is None or compat.supports_store:
        params["store"] = options.store if options and options.store is not None else False

    tools = _convert_chat_completion_tools(context.tools)
    if tools:
        params["tools"] = tools

    _validate_chat_tool_choice(model, options, has_tools=bool(tools))
    if compat and compat.thinking_param in {"kimi", "zai", "deepseek"}:
        extra_body["thinking"] = {
            "type": "enabled" if thinking_enabled else "disabled"
        }

    if extra_body:
        params["extra_body"] = extra_body

    return params


# ---------------------------------------------------------------------------
# Message conversion (bampy -> OpenAI Responses API input format)
# ---------------------------------------------------------------------------

def _convert_messages(
    model: Model,
    context: Context,
    *,
    allow_tool_result_images: bool = False,
) -> list[dict[str, Any]]:
    """Convert context to OpenAI Responses API input items."""
    items: list[dict[str, Any]] = []

    if context.system_prompt:
        items.append({
            "role": "developer" if model.reasoning else "system",
            "content": context.system_prompt,
        })

    transformed = transform_messages(
        context.messages,
        target_model=model.id,
        target_provider=model.provider,
        target_api=model.api,
        normalize_id=lambda tool_call_id, source: _normalize_responses_tool_call_id(
            tool_call_id, source, model
        ),
    )

    for msg in transformed:
        if isinstance(msg, UserMessage):
            items.append({
                "role": "user",
                "content": _convert_user_content(msg.content),
            })
        elif isinstance(msg, AssistantMessage):
            _convert_assistant_items(model, msg, items)
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


def _convert_user_content(content: str | list[Any]) -> str | list[dict[str, Any]]:
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
    model: Model,
    msg: AssistantMessage,
    items: list[dict[str, Any]],
) -> None:
    """Convert an assistant message to Responses API input items."""
    text_parts: list[str] = []
    is_different_model = (
        msg.model != model.id
        and msg.provider == model.provider
        and msg.api == model.api
    )

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
            if text_parts:
                items.append({"role": "assistant", "content": "\n".join(text_parts)})
                text_parts.clear()
            call_id, item_id = _split_openai_tool_call_id(block.id)
            if is_different_model and item_id and item_id.startswith("fc_"):
                item_id = None
            items.append({
                "type": "function_call",
                "call_id": call_id,
                **({"id": item_id} if item_id else {}),
                "name": block.name,
                "arguments": json.dumps(block.arguments),
            })

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


def _tool_result_to_string(content: list[Any]) -> str:
    """Flatten tool result content to a string for Responses API."""
    parts: list[str] = []
    for item in content:
        if isinstance(item, TextContent):
            parts.append(item.text)
        elif isinstance(item, ImageContent):
            parts.append("[image]")
    return "\n".join(parts) if parts else ""


def _convert_tool_result_output(
    content: list[Any],
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


def _convert_tools(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Convert bampy Tool definitions to Responses API tool format."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools
    ]


# ---------------------------------------------------------------------------
# Message conversion (bampy -> OpenAI Chat Completions format)
# ---------------------------------------------------------------------------

def _normalize_chat_tool_call_id(tool_call_id: str, _source: AssistantMessage | None = None) -> str:
    """Normalize tool call ids for Chat Completions history replay."""
    return sanitize_tool_call_id(tool_call_id.split("|", 1)[0])


def _convert_chat_completion_user_content(content: str | list[Any]) -> str | list[dict[str, Any]]:
    """Convert user content to Chat Completions message parts."""
    if isinstance(content, str):
        return content

    parts: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, TextContent):
            parts.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{item.mime_type};base64,{item.data}",
                    "detail": "auto",
                },
            })
    return parts if parts else ""


def _tool_result_text(content: list[Any]) -> str:
    """Return only the text portion of a tool result payload."""
    return "\n".join(
        item.text
        for item in content
        if isinstance(item, TextContent)
    )


def _convert_chat_completion_messages(
    model: Model,
    context: Context,
) -> list[dict[str, Any]]:
    """Convert context to OpenAI Chat Completions messages."""
    messages: list[dict[str, Any]] = []

    if context.system_prompt:
        messages.append({"role": _chat_system_role(model), "content": context.system_prompt})

    transformed = transform_messages(
        context.messages,
        target_model=model.id,
        target_provider=model.provider,
        target_api=model.api,
        normalize_id=_normalize_chat_tool_call_id,
    )

    index = 0
    while index < len(transformed):
        msg = transformed[index]

        if isinstance(msg, UserMessage):
            payload = _convert_chat_completion_user_content(msg.content)
            if payload:
                messages.append({"role": "user", "content": payload})
            index += 1
            continue

        if isinstance(msg, AssistantMessage):
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": None,
            }

            text = "".join(
                block.text
                for block in msg.content
                if isinstance(block, TextContent) and block.text
            )
            if text:
                assistant["content"] = text

            thinking_blocks = [
                block
                for block in msg.content
                if isinstance(block, ThinkingContent) and block.thinking.strip()
            ]
            if thinking_blocks:
                assistant.update(_chat_replay_payloads(model, thinking_blocks))

            tool_calls = [
                {
                    "id": _normalize_chat_tool_call_id(block.id),
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.arguments),
                    },
                }
                for block in msg.content
                if isinstance(block, ToolCall)
            ]
            if tool_calls:
                assistant["tool_calls"] = tool_calls

            has_content = assistant["content"] not in (None, "")
            has_reasoning = any(field in assistant for field in _chat_reasoning_fields(model))
            if has_content or tool_calls or has_reasoning:
                messages.append(assistant)

            index += 1
            continue

        if isinstance(msg, ToolResultMessage):
            image_parts: list[dict[str, Any]] = []

            while index < len(transformed):
                current = transformed[index]
                if not isinstance(current, ToolResultMessage):
                    break

                text_result = _tool_result_text(current.content)
                messages.append({
                    "role": "tool",
                    "tool_call_id": _normalize_chat_tool_call_id(current.tool_call_id),
                    "content": text_result or "(see attached image)",
                })

                if "image" in model.input_types:
                    for block in current.content:
                        if isinstance(block, ImageContent):
                            image_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block.mime_type};base64,{block.data}",
                                    "detail": "auto",
                                },
                            })
                index += 1

            if image_parts:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Attached image(s) from tool result:"},
                        *image_parts,
                    ],
                })
            continue

        index += 1

    return messages


def _convert_chat_completion_tools(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Convert bampy Tool definitions to Chat Completions tools."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": False,
            },
        }
        for tool in tools
    ]


# ---------------------------------------------------------------------------
# Responses API stream
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
        except ImportError as exc:
            _emit_error(event_stream, model, f"openai SDK not installed: {exc}")
            return

        try:
            client = _create_openai_client(openai_sdk, model, options)
            max_tokens = (
                options.max_tokens
                if options and options.max_tokens is not None
                else model.max_tokens
            )
            params: dict[str, Any] = {
                "model": model.id,
                "input": _convert_messages(
                    model,
                    context,
                    allow_tool_result_images=_supports_multimodal_tool_results(model),
                ),
                "stream": True,
                "max_output_tokens": max_tokens,
            }

            if options and options.temperature is not None and not model.reasoning:
                params["temperature"] = options.temperature
            if options and options.prompt_cache_key:
                params["prompt_cache_key"] = options.prompt_cache_key
            if options and options.prompt_cache_retention:
                params["prompt_cache_retention"] = options.prompt_cache_retention
            if options and options.service_tier:
                params["service_tier"] = options.service_tier
            params["store"] = options.store if options and options.store is not None else False

            tools = _convert_tools(context.tools)
            if tools:
                params["tools"] = tools

            if model.reasoning:
                if options and options.reasoning_effort:
                    params["reasoning"] = {"effort": options.reasoning_effort}
                params["include"] = ["reasoning.encrypted_content"]

            event_stream.push(StartEvent(partial=output))

            output_to_content: dict[int, int] = {}
            tool_json_bufs: dict[int, str] = {}

            response = await client.responses.create(**params)
            async for event in response:
                _handle_stream_event(
                    event,
                    output,
                    event_stream,
                    output_to_content,
                    tool_json_bufs,
                )

            output.usage.cost = calculate_cost(model, output.usage)

            if output.stop_reason == StopReason.ERROR:
                raise RuntimeError(output.error_message or "OpenAI Responses request failed")

            event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            event_stream.end(output)

        except Exception as exc:
            output.stop_reason = StopReason.ERROR
            output.error_message = str(exc)
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


def _handle_stream_event(
    event: Any,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    output_to_content: dict[int, int],
    tool_json_bufs: dict[int, str],
) -> None:
    """Map a single Responses API SSE event to bampy events."""
    etype = getattr(event, "type", "")

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
            tool_call_id = (
                f"{call_id}|{sanitize_tool_call_id(item_id)}"
                if item_id
                else call_id
            )
            tool_call = ToolCall(
                id=tool_call_id,
                name=getattr(item, "name", ""),
                arguments={},
            )
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
                raw = getattr(event, "arguments", "") or tool_json_bufs.get(out_idx, "")
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
                    summary = getattr(item, "summary", None) or []
                    texts = [
                        getattr(part, "text", "")
                        for part in summary
                        if getattr(part, "type", "") == "summary_text"
                    ]
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
        if resp is None:
            return

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
            resp_output = getattr(resp, "output", []) or []
            has_tools = any(
                getattr(item, "type", "") == "function_call"
                for item in resp_output
            )
            if not has_tools:
                has_tools = any(isinstance(block, ToolCall) for block in output.content)
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
            error = getattr(resp, "error", None)
            output.error_message = getattr(error, "message", None) or output.error_message


# ---------------------------------------------------------------------------
# Chat Completions stream
# ---------------------------------------------------------------------------

def _start_text_block(
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
) -> int:
    text = TextContent(text="")
    output.content.append(text)
    content_idx = len(output.content) - 1
    stream.push(TextStartEvent(content_index=content_idx, content=text, partial=output))
    return content_idx


def _start_thinking_block(
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    *,
    signature: str | None = None,
) -> int:
    thinking = ThinkingContent(thinking="", thinking_signature=signature)
    output.content.append(thinking)
    content_idx = len(output.content) - 1
    stream.push(ThinkingStartEvent(
        content_index=content_idx,
        content=thinking,
        partial=output,
    ))
    return content_idx


def _end_text_block(
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    content_idx: int | None,
) -> None:
    if content_idx is None:
        return
    block = output.content[content_idx]
    if isinstance(block, TextContent):
        stream.push(TextEndEvent(
            content_index=content_idx,
            content=block,
            partial=output,
        ))


def _end_thinking_block(
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    content_idx: int | None,
) -> None:
    if content_idx is None:
        return
    block = output.content[content_idx]
    if isinstance(block, ThinkingContent):
        stream.push(ThinkingEndEvent(
            content_index=content_idx,
            content=block,
            partial=output,
        ))


def _start_tool_call_block(
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    *,
    tool_call_id: str,
    name: str = "",
) -> int:
    tool_call = ToolCall(id=tool_call_id, name=name, arguments={})
    output.content.append(tool_call)
    content_idx = len(output.content) - 1
    stream.push(ToolCallStartEvent(
        content_index=content_idx,
        content=tool_call,
        partial=output,
    ))
    return content_idx


def _end_tool_call_block(
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    content_idx: int,
) -> None:
    block = output.content[content_idx]
    if isinstance(block, ToolCall):
        stream.push(ToolCallEndEvent(
            content_index=content_idx,
            content=block,
            partial=output,
        ))


def _parse_chat_completion_usage(usage: Any) -> dict[str, int]:
    """Normalize chat-completion usage payloads."""
    prompt_details = _option_value(usage, "prompt_tokens_details")
    prompt_tokens = _option_value(usage, "prompt_tokens") or 0
    cache_hit_tokens = _option_value(usage, "prompt_cache_hit_tokens")
    cache_miss_tokens = _option_value(usage, "prompt_cache_miss_tokens")
    if cache_hit_tokens is not None or cache_miss_tokens is not None:
        cached_tokens = cache_hit_tokens or 0
        input_tokens = cache_miss_tokens if cache_miss_tokens is not None else max(prompt_tokens - cached_tokens, 0)
    else:
        cached_tokens = _option_value(prompt_details, "cached_tokens") or 0
        input_tokens = max(prompt_tokens - cached_tokens, 0)
    completion_tokens = _option_value(usage, "completion_tokens") or 0
    total_tokens = _option_value(usage, "total_tokens") or (
        input_tokens + completion_tokens + cached_tokens
    )
    return {
        "input": input_tokens,
        "output": completion_tokens,
        "cache_read": cached_tokens,
        "total_tokens": total_tokens,
    }


def _map_chat_completion_finish_reason(
    finish_reason: str | None,
) -> tuple[StopReason, str | None]:
    """Map a Chat Completions finish reason into the bampy enum."""
    if finish_reason in (None, "stop", "end"):
        return StopReason.STOP, None
    if finish_reason == "length":
        return StopReason.LENGTH, None
    if finish_reason in ("tool_calls", "function_call"):
        return StopReason.TOOL_USE, None
    if finish_reason == "content_filter":
        return StopReason.ERROR, "Response was filtered by the provider"
    if finish_reason == "insufficient_system_resource":
        return StopReason.ERROR, "Provider stopped due to insufficient system resources"
    return StopReason.ERROR, f"Unhandled finish_reason: {finish_reason}"


def _apply_chat_completion_delta(
    delta: Any,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    *,
    reasoning_fields: tuple[str, ...],
    active_text_index: int | None,
    active_thinking_index: int | None,
    tool_indexes: dict[int, int],
    tool_json_bufs: dict[int, str],
    current_scalar_kind: str | None,
) -> tuple[int | None, int | None, str | None]:
    """Apply one Chat Completions delta object."""
    content_delta = _option_value(delta, "content")
    refusal_delta = _option_value(delta, "refusal")
    text_delta = content_delta if content_delta is not None else refusal_delta

    if text_delta:
        if current_scalar_kind == "thinking":
            _end_thinking_block(output, stream, active_thinking_index)
            active_thinking_index = None
        if current_scalar_kind != "text" or active_text_index is None:
            active_text_index = _start_text_block(output, stream)
        current_scalar_kind = "text"
        text_block = output.content[active_text_index]
        if isinstance(text_block, TextContent):
            text_block.text += text_delta
        stream.push(TextDeltaEvent(
            content_index=active_text_index,
            delta=text_delta,
            partial=output,
        ))

    for field in reasoning_fields:
        reasoning_value = _option_value(delta, field)
        if (
            reasoning_value is None
            or reasoning_value == ""
            or reasoning_value == []
            or reasoning_value == {}
        ):
            continue
        reasoning_delta = _chat_reasoning_value_to_text(reasoning_value)

        if current_scalar_kind == "text":
            _end_text_block(output, stream, active_text_index)
            active_text_index = None

        reuse_current = False
        if active_thinking_index is not None:
            block = output.content[active_thinking_index]
            reuse_current = (
                isinstance(block, ThinkingContent)
                and block.thinking_signature == field
            )
        if current_scalar_kind != "thinking" or not reuse_current:
            if current_scalar_kind == "thinking":
                _end_thinking_block(output, stream, active_thinking_index)
            active_thinking_index = _start_thinking_block(
                output,
                stream,
                signature=field,
            )
        current_scalar_kind = "thinking"
        thinking_block = output.content[active_thinking_index]
        if isinstance(thinking_block, ThinkingContent):
            thinking_block.thinking = _append_chat_reasoning_value(
                thinking_block.thinking,
                field,
                reasoning_value,
            )
        stream.push(ThinkingDeltaEvent(
            content_index=active_thinking_index,
            delta=reasoning_delta,
            partial=output,
        ))

    raw_tool_calls = _option_value(delta, "tool_calls") or []
    deprecated_function_call = _option_value(delta, "function_call")
    if deprecated_function_call:
        raw_tool_calls = [
            {
                "index": 0,
                "function": {
                    "name": _option_value(deprecated_function_call, "name"),
                    "arguments": _option_value(deprecated_function_call, "arguments"),
                },
            },
        ]

    if raw_tool_calls:
        if current_scalar_kind == "text":
            _end_text_block(output, stream, active_text_index)
            active_text_index = None
        elif current_scalar_kind == "thinking":
            _end_thinking_block(output, stream, active_thinking_index)
            active_thinking_index = None
        current_scalar_kind = None

    for raw_tool_call in raw_tool_calls:
        tool_idx = _option_value(raw_tool_call, "index")
        if tool_idx is None:
            tool_idx = 0
        function = _option_value(raw_tool_call, "function") or {}
        raw_tool_call_id = _option_value(raw_tool_call, "id")
        tool_call_id = (
            sanitize_tool_call_id(raw_tool_call_id)
            if raw_tool_call_id
            else None
        )
        tool_name = _option_value(function, "name") or ""

        if tool_idx not in tool_indexes:
            tool_indexes[tool_idx] = _start_tool_call_block(
                output,
                stream,
                tool_call_id=tool_call_id or f"call_{tool_idx}",
                name=tool_name,
            )
            tool_json_bufs[tool_idx] = ""

        content_idx = tool_indexes[tool_idx]
        tool_block = output.content[content_idx]
        if not isinstance(tool_block, ToolCall):
            continue

        if tool_call_id:
            tool_block.id = tool_call_id
        if tool_name:
            tool_block.name = tool_name

        arguments_delta = _option_value(function, "arguments") or ""
        if arguments_delta:
            tool_json_bufs[tool_idx] = tool_json_bufs.get(tool_idx, "") + arguments_delta
            tool_block.arguments = parse_partial_json(tool_json_bufs[tool_idx])
        stream.push(ToolCallDeltaEvent(
            content_index=content_idx,
            delta=arguments_delta,
            partial=output,
        ))

    return active_text_index, active_thinking_index, current_scalar_kind


def stream_openai_completions(
    model: Model,
    context: Context,
    options: OpenAIOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream an OpenAI Chat Completions response."""
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
        except ImportError as exc:
            _emit_error(event_stream, model, f"openai SDK not installed: {exc}")
            return

        try:
            client = _create_openai_client(openai_sdk, model, options)
            params = _build_chat_completion_params(model, context, options)

            event_stream.push(StartEvent(partial=output))

            active_text_index: int | None = None
            active_thinking_index: int | None = None
            current_scalar_kind: str | None = None
            tool_indexes: dict[int, int] = {}
            tool_json_bufs: dict[int, str] = {}
            closed_tool_indexes: set[int] = set()
            reasoning_fields = _chat_reasoning_fields(model)

            response = await client.chat.completions.create(**params)
            async for chunk in response:
                output.response_id = getattr(chunk, "id", None) or output.response_id

                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    usage = _parse_chat_completion_usage(chunk_usage)
                    output.usage.input = usage["input"]
                    output.usage.output = usage["output"]
                    output.usage.cache_read = usage["cache_read"]
                    output.usage.total_tokens = usage["total_tokens"]

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                choice = choices[0]
                choice_usage = _option_value(choice, "usage")
                if choice_usage and not chunk_usage:
                    usage = _parse_chat_completion_usage(choice_usage)
                    output.usage.input = usage["input"]
                    output.usage.output = usage["output"]
                    output.usage.cache_read = usage["cache_read"]
                    output.usage.total_tokens = usage["total_tokens"]

                finish_reason = _option_value(choice, "finish_reason")
                if finish_reason:
                    output.stop_reason, output.error_message = _map_chat_completion_finish_reason(
                        finish_reason
                    )

                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                active_text_index, active_thinking_index, current_scalar_kind = _apply_chat_completion_delta(
                    delta,
                    output,
                    event_stream,
                    reasoning_fields=reasoning_fields,
                    active_text_index=active_text_index,
                    active_thinking_index=active_thinking_index,
                    tool_indexes=tool_indexes,
                    tool_json_bufs=tool_json_bufs,
                    current_scalar_kind=current_scalar_kind,
                )

            if current_scalar_kind == "text":
                _end_text_block(output, event_stream, active_text_index)
            elif current_scalar_kind == "thinking":
                _end_thinking_block(output, event_stream, active_thinking_index)

            for tool_idx, content_idx in sorted(tool_indexes.items(), key=lambda item: item[1]):
                if tool_idx in closed_tool_indexes:
                    continue
                block = output.content[content_idx]
                if isinstance(block, ToolCall):
                    block.arguments = parse_partial_json(tool_json_bufs.get(tool_idx, ""))
                _end_tool_call_block(output, event_stream, content_idx)
                closed_tool_indexes.add(tool_idx)

            if output.stop_reason == StopReason.STOP and any(
                isinstance(block, ToolCall) for block in output.content
            ):
                output.stop_reason = StopReason.TOOL_USE

            output.usage.cost = calculate_cost(model, output.usage)

            if output.stop_reason == StopReason.ERROR:
                raise RuntimeError(output.error_message or "OpenAI Chat Completions request failed")

            event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            event_stream.end(output)

        except Exception as exc:
            output.stop_reason = StopReason.ERROR
            output.error_message = str(exc)
            event_stream.push(ErrorEvent(reason=StopReason.ERROR, error=output))
            event_stream.end(output)

    spawn_provider_task(
        event_stream=event_stream,
        output=output,
        options=options,
        runner=_run,
    )
    return event_stream


def stream_simple_openai_completions(
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

    return stream_openai_completions(model, context, openai_opts)


# ---------------------------------------------------------------------------
# Shared helpers
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
# Provider entries
# ---------------------------------------------------------------------------

def get_provider_entry() -> ApiProviderEntry:
    return ApiProviderEntry(
        api="openai-responses",
        stream=stream_openai,  # type: ignore[arg-type]
        stream_simple=stream_simple_openai,  # type: ignore[arg-type]
    )


def get_completions_provider_entry() -> ApiProviderEntry:
    return ApiProviderEntry(
        api="openai-completions",
        stream=stream_openai_completions,  # type: ignore[arg-type]
        stream_simple=stream_simple_openai_completions,  # type: ignore[arg-type]
    )
