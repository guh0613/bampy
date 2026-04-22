"""Cross-provider message transformation utilities."""

from __future__ import annotations

import re
from typing import Callable

from bampy.ai.types import (
    AssistantMessage,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

_MAX_ID_LEN = 64
NormalizeId = Callable[[str, AssistantMessage], str]


def sanitize_tool_call_id(tool_call_id: str) -> str:
    """Ensure a tool-call ID matches common provider constraints."""
    # Strip any compound id format (e.g. "callId|itemId" from OpenAI Responses)
    if "|" in tool_call_id:
        tool_call_id = tool_call_id.split("|")[0]
    # Replace invalid chars
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)
    return sanitized[:_MAX_ID_LEN]


def transform_messages(
    messages: list[Message],
    *,
    target_model: str | None = None,
    target_provider: str | None = None,
    target_api: str | None = None,
    normalize_id: NormalizeId | None = None,
) -> list[Message]:
    """Transform a message list for cross-provider compatibility.
    """
    if normalize_id is None:
        normalize_id = lambda tool_call_id, _source: sanitize_tool_call_id(tool_call_id)

    # Pass 1: per-message transforms
    transformed: list[Message] = []
    id_map: dict[str, str] = {}  # old_id → new_id

    for msg in messages:
        if isinstance(msg, UserMessage):
            transformed.append(msg)

        elif isinstance(msg, AssistantMessage):
            new_content = _transform_assistant_content(
                msg, target_model, target_provider, target_api, normalize_id, id_map
            )
            new_msg = msg.model_copy(update={"content": new_content})
            transformed.append(new_msg)

        elif isinstance(msg, ToolResultMessage):
            new_id = id_map.get(msg.tool_call_id)
            if new_id is not None and new_id != msg.tool_call_id:
                transformed.append(msg.model_copy(update={"tool_call_id": new_id}))
            else:
                transformed.append(msg)

    # Pass 2: insert synthetic tool results for orphaned tool calls
    return _insert_synthetic_results(transformed)


def _transform_assistant_content(
    msg: AssistantMessage,
    target_model: str | None,
    target_provider: str | None,
    target_api: str | None,
    normalize_id: Callable[[str], str],
    id_map: dict[str, str],
) -> list:
    """Transform content blocks of an assistant message."""
    same_model = (
        target_model is not None
        and target_model == msg.model
        and target_provider == msg.provider
        and target_api == msg.api
    )
    new_content = []

    for block in msg.content:
        if isinstance(block, ThinkingContent):
            if block.redacted:
                if same_model:
                    new_content.append(block)
                continue
            if not block.thinking.strip() and not same_model:
                continue
            if same_model:
                new_content.append(block)
            else:
                new_content.append(TextContent(text=block.thinking))

        elif isinstance(block, ToolCall):
            if same_model:
                new_content.append(block)
                continue

            new_id = normalize_id(block.id, msg)
            id_map[block.id] = new_id
            block_update = {"id": new_id}
            if block.thought_signature is not None:
                block_update["thought_signature"] = None
            new_content.append(block.model_copy(update=block_update))

        else:
            if same_model:
                new_content.append(block)
            else:
                new_content.append(TextContent(text=block.text))

    return new_content


def _insert_synthetic_results(messages: list[Message]) -> list[Message]:
    """Insert synthetic ToolResultMessages for any tool calls that lack a result."""
    result: list[Message] = []
    pending_calls: list[ToolCall] = []
    resolved_ids: set[str] = set()

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            if msg.stop_reason in ("error", "aborted"):
                continue

            # Before adding a new assistant message, flush any orphaned calls
            for tc in pending_calls:
                if tc.id not in resolved_ids:
                    result.append(
                        ToolResultMessage(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            content=[TextContent(text="No result provided")],
                            is_error=True,
                        )
                    )
            pending_calls.clear()
            resolved_ids.clear()

            # Track new tool calls
            for block in msg.content:
                if isinstance(block, ToolCall):
                    pending_calls.append(block)
            result.append(msg)

        elif isinstance(msg, ToolResultMessage):
            # Mark this call as resolved
            resolved_ids.add(msg.tool_call_id)
            pending_calls = [tc for tc in pending_calls if tc.id != msg.tool_call_id]
            result.append(msg)

        elif isinstance(msg, UserMessage):
            # User message interrupts: flush orphaned calls
            for tc in pending_calls:
                if tc.id not in resolved_ids:
                    result.append(
                        ToolResultMessage(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            content=[TextContent(text="No result provided")],
                            is_error=True,
                        )
                    )
            pending_calls.clear()
            resolved_ids.clear()
            result.append(msg)

    # Trailing orphans
    for tc in pending_calls:
        if tc.id not in resolved_ids:
            result.append(
                ToolResultMessage(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    content=[TextContent(text="No result provided")],
                    is_error=True,
                )
            )

    return result
