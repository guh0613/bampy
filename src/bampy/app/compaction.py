"""Context compaction for long sessions.

Pure functions for compaction logic.  The :class:`SessionManager` handles I/O;
after compaction the session is reloaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from bampy.ai.types import Message, StopReason, TextContent, Usage

from .messages import (
    convert_app_messages_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)
from .session import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomMessageEntry,
    SessionEntry,
    SessionMessageEntry,
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(message: Any) -> int:
    """Estimate token count using a chars/4 heuristic (conservative)."""
    chars = 0
    role = getattr(message, "role", None)
    image_token_cost = 4800

    if role in ("user", "custom"):
        content = getattr(message, "content", "")
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    chars += len(text)
                block_type = getattr(block, "type", None)
                if block_type is None and isinstance(block, dict):
                    block_type = block.get("type")
                if block_type == "image":
                    chars += image_token_cost
        return math.ceil(chars / 4)

    if role == "assistant":
        content = getattr(message, "content", [])
        if isinstance(content, list):
            for block in content:
                if hasattr(block, "text"):
                    chars += len(block.text)
                elif hasattr(block, "thinking"):
                    chars += len(block.thinking)
                elif hasattr(block, "name"):
                    import json
                    chars += len(getattr(block, "name", ""))
                    args = getattr(block, "arguments", {})
                    chars += len(json.dumps(args) if isinstance(args, dict) else str(args))
                elif isinstance(block, dict):
                    chars += len(block.get("text", ""))
                    chars += len(block.get("thinking", ""))
                    if "name" in block:
                        import json
                        chars += len(block.get("name", ""))
                        chars += len(json.dumps(block.get("arguments", {})))
        return math.ceil(chars / 4)

    if role == "tool_result":
        content = getattr(message, "content", [])
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "")
                if text:
                    chars += len(text)
                if (getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else "")) == "image":
                    chars += image_token_cost
        return math.ceil(chars / 4)

    if role in ("compaction_summary", "branch_summary"):
        summary = getattr(message, "summary", "")
        return math.ceil(len(summary) / 4)

    # dict-based messages (from session)
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    chars += len(block.get("text", ""))
                    chars += len(block.get("thinking", ""))
                    if "name" in block:
                        import json
                        chars += len(block.get("name", ""))
                        chars += len(json.dumps(block.get("arguments", {})))
                    if block.get("type") == "image":
                        chars += image_token_cost
        return math.ceil(chars / 4)

    return 0


def calculate_context_tokens(usage: Usage) -> int:
    return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)


# ---------------------------------------------------------------------------
# Context estimation from messages
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ContextUsageEstimate:
    tokens: int = 0
    usage_tokens: int = 0
    trailing_tokens: int = 0
    last_usage_index: int | None = None


def _get_assistant_usage(msg: Any) -> Usage | None:
    """Get usage from an assistant message, skipping error/aborted."""
    if getattr(msg, "role", None) != "assistant":
        return None
    usage = getattr(msg, "usage", None)
    if usage is None:
        return None
    stop_reason = getattr(msg, "stop_reason", None)
    if stop_reason in (StopReason.ABORTED, StopReason.ERROR, "aborted", "error"):
        return None
    if isinstance(usage, dict):
        return Usage(**usage)
    return usage


def estimate_context_tokens(messages: list[Any]) -> ContextUsageEstimate:
    """Estimate context tokens using the last assistant usage + trailing estimate."""
    # Find last usage
    last_usage: Usage | None = None
    last_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        u = _get_assistant_usage(messages[i])
        if u is not None:
            last_usage = u
            last_idx = i
            break

    if last_usage is None:
        total = sum(estimate_tokens(m) for m in messages)
        return ContextUsageEstimate(tokens=total, trailing_tokens=total)

    usage_tokens = calculate_context_tokens(last_usage)
    trailing = sum(estimate_tokens(messages[i]) for i in range(last_idx + 1, len(messages)))
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing,
        last_usage_index=last_idx,
    )


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


# ---------------------------------------------------------------------------
# Cut point detection
# ---------------------------------------------------------------------------

def _get_message_from_entry(entry: SessionEntry) -> Any | None:
    if isinstance(entry, SessionMessageEntry):
        return entry.message
    if isinstance(entry, CustomMessageEntry):
        return create_custom_message(
            entry.custom_type,
            entry.content if isinstance(entry.content, str) else entry.content,
            entry.display,
            entry.details,
            entry.timestamp,
        )
    if isinstance(entry, BranchSummaryEntry):
        return create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)
    if isinstance(entry, CompactionEntry):
        return create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp)
    return None


def _find_valid_cut_points(entries: list[SessionEntry], start: int, end: int) -> list[int]:
    """Find indices where we can safely cut (user/assistant/custom, not tool_result)."""
    cut_points: list[int] = []
    for i in range(start, end):
        entry = entries[i]
        if isinstance(entry, SessionMessageEntry):
            msg = entry.message
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role in ("user", "assistant", "custom", "compaction_summary", "branch_summary"):
                cut_points.append(i)
        elif isinstance(entry, (BranchSummaryEntry, CustomMessageEntry)):
            cut_points.append(i)
    return cut_points


@dataclass(slots=True)
class CutPointResult:
    first_kept_entry_index: int = 0
    turn_start_index: int = -1
    is_split_turn: bool = False


def _find_turn_start(entries: list[SessionEntry], entry_index: int, start: int) -> int:
    for i in range(entry_index, start - 1, -1):
        entry = entries[i]
        if isinstance(entry, (BranchSummaryEntry, CustomMessageEntry)):
            return i
        if isinstance(entry, SessionMessageEntry):
            msg = entry.message
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "user":
                return i
    return -1


def find_cut_point(
    entries: list[SessionEntry],
    start: int,
    end: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    cut_points = _find_valid_cut_points(entries, start, end)
    if not cut_points:
        return CutPointResult(first_kept_entry_index=start)

    accumulated = 0
    cut_index = cut_points[0]

    for i in range(end - 1, start - 1, -1):
        entry = entries[i]
        msg = _get_message_from_entry(entry)
        if msg is None:
            continue
        accumulated += estimate_tokens(msg)
        if accumulated >= keep_recent_tokens:
            for c in cut_points:
                if c >= i:
                    cut_index = c
                    break
            break

    # Check if splitting a turn
    cut_entry = entries[cut_index]
    is_user = isinstance(cut_entry, SessionMessageEntry) and (
        (isinstance(cut_entry.message, dict) and cut_entry.message.get("role") == "user")
        or getattr(cut_entry.message, "role", None) == "user"
    )
    turn_start = -1 if is_user else _find_turn_start(entries, cut_index, start)

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start,
        is_split_turn=not is_user and turn_start != -1,
    )


# ---------------------------------------------------------------------------
# Compaction preparation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CompactionPreparation:
    first_kept_entry_id: str = ""
    messages_to_summarize: list[Any] = field(default_factory=list)
    turn_prefix_messages: list[Any] = field(default_factory=list)
    is_split_turn: bool = False
    tokens_before: int = 0
    previous_summary: str | None = None
    settings: CompactionSettings = field(default_factory=CompactionSettings)


def prepare_compaction(
    path_entries: list[SessionEntry],
    settings: CompactionSettings | None = None,
) -> CompactionPreparation | None:
    """Pre-calculate compaction data from session entries on the current branch."""
    if settings is None:
        settings = DEFAULT_COMPACTION_SETTINGS

    if path_entries and isinstance(path_entries[-1], CompactionEntry):
        return None

    # Find previous compaction
    prev_compaction_idx = -1
    for i in range(len(path_entries) - 1, -1, -1):
        if isinstance(path_entries[i], CompactionEntry):
            prev_compaction_idx = i
            break

    boundary_start = prev_compaction_idx + 1
    boundary_end = len(path_entries)

    # Estimate tokens
    usage_start = max(prev_compaction_idx, 0)
    usage_messages: list[Any] = []
    for i in range(usage_start, boundary_end):
        msg = _get_message_from_entry(path_entries[i])
        if msg is not None:
            usage_messages.append(msg)
    tokens_before = estimate_context_tokens(usage_messages).tokens

    cut = find_cut_point(path_entries, boundary_start, boundary_end, settings.keep_recent_tokens)

    first_kept_entry = path_entries[cut.first_kept_entry_index]
    if not first_kept_entry.id:
        return None

    history_end = cut.turn_start_index if cut.is_split_turn else cut.first_kept_entry_index

    messages_to_summarize: list[Any] = []
    for i in range(boundary_start, history_end):
        msg = _get_message_from_entry(path_entries[i])
        if msg is not None:
            messages_to_summarize.append(msg)

    turn_prefix_messages: list[Any] = []
    if cut.is_split_turn:
        for i in range(cut.turn_start_index, cut.first_kept_entry_index):
            msg = _get_message_from_entry(path_entries[i])
            if msg is not None:
                turn_prefix_messages.append(msg)

    previous_summary: str | None = None
    if prev_compaction_idx >= 0:
        prev = path_entries[prev_compaction_idx]
        if isinstance(prev, CompactionEntry):
            previous_summary = prev.summary

    return CompactionPreparation(
        first_kept_entry_id=first_kept_entry.id,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        is_split_turn=cut.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Compaction result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CompactionResult:
    summary: str = ""
    first_kept_entry_id: str = ""
    tokens_before: int = 0
    details: Any = None


# ---------------------------------------------------------------------------
# Summarisation prompts
# ---------------------------------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = "You are a precise summarization assistant. Create structured context checkpoints."

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished

Use the same structured format. Keep each section concise."""

TURN_PREFIX_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""

TOOL_RESULT_MAX_CHARS = 2_000


# ---------------------------------------------------------------------------
# LLM-based summarisation
# ---------------------------------------------------------------------------

def _serialize_conversation(messages: list[Message]) -> str:
    """Serialize LLM messages to a readable text format for the summarization prompt."""
    parts: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif hasattr(block, "thinking"):
                    text_parts.append(f"[thinking] {block.thinking}")
                elif hasattr(block, "name"):
                    text_parts.append(f"[tool_call: {block.name}({block.arguments})]")
                elif isinstance(block, dict):
                    if "text" in block:
                        text_parts.append(block["text"])
            text = "\n".join(text_parts)
        else:
            text = str(content)
        if role == "tool_result":
            text = _truncate_for_summary(text, TOOL_RESULT_MAX_CHARS)
        parts.append(f"[{role}]\n{text}")
    return "\n\n".join(parts)


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


async def generate_summary(
    current_messages: list[Any],
    model: Any,
    reserve_tokens: int,
    api_key: str,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    cancellation: Any | None = None,
) -> str:
    """Generate an LLM-based summary of conversation messages.

    Uses :func:`bampy.ai.complete_simple` to produce the summary.
    """
    from bampy.ai.stream import complete_simple
    from bampy.ai.types import Context, SimpleStreamOptions, UserMessage

    max_tokens = int(0.8 * reserve_tokens)

    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    llm_messages = convert_app_messages_to_llm(current_messages)
    conversation_text = _serialize_conversation(llm_messages)

    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    ctx = Context(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[UserMessage(content=[TextContent(text=prompt_text)])],
    )
    options = SimpleStreamOptions(
        max_tokens=max_tokens,
        api_key=api_key,
        cancellation=cancellation,
    )

    response = await complete_simple(model, ctx, options)

    if response.stop_reason == StopReason.ERROR:
        raise RuntimeError(f"Summarization failed: {response.error_message or 'Unknown error'}")

    return "\n".join(
        block.text for block in response.content
        if hasattr(block, "text") and hasattr(block, "type") and block.type == "text"
    )


async def generate_turn_prefix_summary(
    current_messages: list[Any],
    model: Any,
    reserve_tokens: int,
    api_key: str,
    cancellation: Any | None = None,
) -> str:
    """Generate a compact summary for the prefix of a split turn."""
    from bampy.ai.stream import complete_simple
    from bampy.ai.types import Context, SimpleStreamOptions, UserMessage

    llm_messages = convert_app_messages_to_llm(current_messages)
    conversation_text = _serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_PROMPT}"
    ctx = Context(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[UserMessage(content=[TextContent(text=prompt_text)])],
    )
    options = SimpleStreamOptions(
        max_tokens=int(0.5 * reserve_tokens),
        api_key=api_key,
        cancellation=cancellation,
    )

    response = await complete_simple(model, ctx, options)
    if response.stop_reason == StopReason.ERROR:
        raise RuntimeError(
            f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}"
        )

    return "\n".join(
        block.text for block in response.content
        if hasattr(block, "text") and hasattr(block, "type") and block.type == "text"
    )


async def compact(
    preparation: CompactionPreparation,
    model: Any,
    api_key: str,
    custom_instructions: str | None = None,
    cancellation: Any | None = None,
) -> CompactionResult:
    """Generate summaries for compaction using prepared data."""
    summary: str

    if preparation.is_split_turn and preparation.turn_prefix_messages:
        import asyncio

        async def _no_prior_history() -> str:
            return "No prior history."

        history_coro = (
            generate_summary(
                preparation.messages_to_summarize,
                model,
                preparation.settings.reserve_tokens,
                api_key,
                custom_instructions,
                preparation.previous_summary,
                cancellation,
            )
            if preparation.messages_to_summarize
            else _no_prior_history()
        )
        prefix_coro = generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            model,
            preparation.settings.reserve_tokens,
            api_key,
            cancellation=cancellation,
        )
        history_result, prefix_result = await asyncio.gather(history_coro, prefix_coro)
        summary = f"{history_result}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_result}"
    else:
        summary = await generate_summary(
            preparation.messages_to_summarize,
            model,
            preparation.settings.reserve_tokens,
            api_key,
            custom_instructions,
            preparation.previous_summary,
            cancellation,
        )

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
    )
