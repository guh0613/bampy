"""Cross-provider message transformation utilities.

Handles thinking block conversion, tool-call ID normalisation, and synthetic tool-result insertion
for orphaned tool calls.
"""

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

# Anthropic tool-call ID pattern
_ANTHROPIC_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_MAX_ID_LEN = 64


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
    normalize_id: Callable[[str], str] | None = None,
) -> list[Message]:
    """Transform a message list for cross-provider compatibility.
    """
    if normalize_id is None:
        normalize_id = sanitize_tool_call_id

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
            new_id = id_map.get(msg.tool_call_id, normalize_id(msg.tool_call_id))
            new_msg = msg.model_copy(update={"tool_call_id": new_id})
            transformed.append(new_msg)

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
            if same_model:
                # Keep thinking blocks for same model (with signature)
                new_content.append(block)
            elif block.redacted:
                # Skip redacted thinking for different models
                continue
            else:
                # Convert thinking to text for different models
                new_content.append(
                    TextContent(text=f"<thinking>\n{block.thinking}\n</thinking>")
                )

        elif isinstance(block, ToolCall):
            new_id = normalize_id(block.id)
            id_map[block.id] = new_id
            new_content.append(block.model_copy(update={"id": new_id}))

        else:
            new_content.append(block)

    return new_content


def _insert_synthetic_results(messages: list[Message]) -> list[Message]:
    """Insert synthetic ToolResultMessages for any tool calls that lack a result."""
    result: list[Message] = []
    pending_calls: list[ToolCall] = []

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            # Before adding a new assistant message, flush any orphaned calls
            for tc in pending_calls:
                result.append(
                    ToolResultMessage(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=[TextContent(text="[Tool call interrupted — no result available]")],
                        is_error=True,
                    )
                )
            pending_calls.clear()

            # Track new tool calls
            for block in msg.content:
                if isinstance(block, ToolCall):
                    pending_calls.append(block)
            result.append(msg)

        elif isinstance(msg, ToolResultMessage):
            # Mark this call as resolved
            pending_calls = [tc for tc in pending_calls if tc.id != msg.tool_call_id]
            result.append(msg)

        elif isinstance(msg, UserMessage):
            # User message interrupts: flush orphaned calls
            for tc in pending_calls:
                result.append(
                    ToolResultMessage(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=[TextContent(text="[Tool call interrupted by user]")],
                        is_error=True,
                    )
                )
            pending_calls.clear()
            result.append(msg)

    # Trailing orphans
    for tc in pending_calls:
        result.append(
            ToolResultMessage(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=[TextContent(text="[Tool call interrupted — no result available]")],
                is_error=True,
            )
        )

    return result
