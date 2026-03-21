"""App-layer custom message types and LLM conversion.

Defines message types specific to the application layer (compaction summaries,
branch summaries, extension-injected messages) and provides conversion to
standard LLM messages for the agent layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bampy.ai.types import ImageContent, TextContent, UserMessage, time_ms

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted "
    "into the following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

BRANCH_SUMMARY_PREFIX = (
    "The following is a summary of a branch that this conversation "
    "came back from:\n\n<summary>\n"
)
BRANCH_SUMMARY_SUFFIX = "</summary>"


@dataclass(slots=True)
class CompactionSummaryMessage:
    """Injected after context compaction to carry the summary."""

    role: str = field(default="compaction_summary", init=False)
    summary: str = ""
    tokens_before: int = 0
    timestamp: float = field(default_factory=time_ms)


@dataclass(slots=True)
class BranchSummaryMessage:
    """Injected when branching to summarise the abandoned path."""

    role: str = field(default="branch_summary", init=False)
    summary: str = ""
    from_id: str = ""
    timestamp: float = field(default_factory=time_ms)


@dataclass(slots=True)
class CustomMessage:
    """Extension-injected message that participates in LLM context."""

    role: str = field(default="custom", init=False)
    custom_type: str = ""
    content: str | list[TextContent | ImageContent] = ""
    display: bool = True
    details: Any = None
    timestamp: float = field(default_factory=time_ms)


def create_compaction_summary_message(
    summary: str,
    tokens_before: int,
    timestamp: str | float | None = None,
) -> CompactionSummaryMessage:
    ts = _parse_timestamp(timestamp)
    return CompactionSummaryMessage(summary=summary, tokens_before=tokens_before, timestamp=ts)


def create_branch_summary_message(
    summary: str,
    from_id: str,
    timestamp: str | float | None = None,
) -> BranchSummaryMessage:
    ts = _parse_timestamp(timestamp)
    return BranchSummaryMessage(summary=summary, from_id=from_id, timestamp=ts)


def create_custom_message(
    custom_type: str,
    content: str | list[TextContent | ImageContent],
    display: bool = True,
    details: Any = None,
    timestamp: str | float | None = None,
) -> CustomMessage:
    ts = _parse_timestamp(timestamp)
    return CustomMessage(
        custom_type=custom_type,
        content=content,
        display=display,
        details=details,
        timestamp=ts,
    )


def _parse_timestamp(timestamp: str | float | None) -> float:
    if timestamp is None:
        return time_ms()
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.timestamp() * 1000
    except (ValueError, AttributeError):
        return time_ms()


# ---------------------------------------------------------------------------
# LLM conversion
# ---------------------------------------------------------------------------

def convert_compaction_summary_to_llm(msg: CompactionSummaryMessage) -> UserMessage:
    text = COMPACTION_SUMMARY_PREFIX + msg.summary + COMPACTION_SUMMARY_SUFFIX
    return UserMessage(content=[TextContent(text=text)], timestamp=msg.timestamp)


def convert_branch_summary_to_llm(msg: BranchSummaryMessage) -> UserMessage:
    text = BRANCH_SUMMARY_PREFIX + msg.summary + BRANCH_SUMMARY_SUFFIX
    return UserMessage(content=[TextContent(text=text)], timestamp=msg.timestamp)


def convert_custom_to_llm(msg: CustomMessage) -> UserMessage:
    if isinstance(msg.content, str):
        content: str | list[TextContent | ImageContent] = [TextContent(text=msg.content)]
    else:
        content = msg.content
    return UserMessage(content=content, timestamp=msg.timestamp)


def convert_app_messages_to_llm(messages: list[Any]) -> list[Any]:
    """Convert a list of agent messages, handling app-layer custom types.

    Standard LLM messages pass through; app-layer messages are converted to
    ``UserMessage``.  Returns a new list.
    """
    from bampy.agent.messages import convert_message_to_llm

    result: list[Any] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role == "compaction_summary":
            result.append(convert_compaction_summary_to_llm(msg))
        elif role == "branch_summary":
            result.append(convert_branch_summary_to_llm(msg))
        elif role == "custom":
            result.append(convert_custom_to_llm(msg))
        else:
            result.extend(convert_message_to_llm(msg))
    return result


def register_app_message_converters() -> None:
    """Register converters for app-layer message types in the agent message registry."""
    from bampy.agent.messages import register_message_converter

    register_message_converter(
        "compaction_summary",
        lambda msg: convert_compaction_summary_to_llm(msg),
        source_id="bampy.app",
    )
    register_message_converter(
        "branch_summary",
        lambda msg: convert_branch_summary_to_llm(msg),
        source_id="bampy.app",
    )
    register_message_converter(
        "custom",
        lambda msg: convert_custom_to_llm(msg),
        source_id="bampy.app",
    )
