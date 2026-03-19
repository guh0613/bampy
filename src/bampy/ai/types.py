"""Core types for the bampy AI layer.

Defines message types, content blocks, model metadata, stream events,
and options.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def time_ms() -> float:
    """Current time in milliseconds (like JS Date.now())."""
    return time.time() * 1000


# ---------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = None


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = None
    redacted: bool = False


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str  # base64-encoded
    mime_type: str


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: bytes | None = None  # Gemini: opaque token echoed back


# Discriminated unions for content blocks
UserContentBlock = Annotated[
    Union[TextContent, ImageContent],
    Field(discriminator="type"),
]

AssistantContentBlock = Annotated[
    Union[TextContent, ThinkingContent, ToolCall],
    Field(discriminator="type"),
]

ToolResultContentBlock = Annotated[
    Union[TextContent, ImageContent],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

class UsageCost(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: UsageCost = Field(default_factory=UsageCost)


class StopReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_USE = "tool_use"
    ERROR = "error"
    ABORTED = "aborted"


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[UserContentBlock]
    timestamp: float = Field(default_factory=time_ms)


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContentBlock] = Field(default_factory=list)
    api: str = ""
    provider: str = ""
    model: str = ""
    response_id: str | None = None
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = StopReason.STOP
    error_message: str | None = None
    timestamp: float = Field(default_factory=time_ms)


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[ToolResultContentBlock] = Field(default_factory=list)
    details: Any = None
    is_error: bool = False
    timestamp: float = Field(default_factory=time_ms)


Message = Annotated[
    Union[UserMessage, AssistantMessage, ToolResultMessage],
    Field(discriminator="role"),
]


# ---------------------------------------------------------------------------
# Tool definition (LLM-level)
# ---------------------------------------------------------------------------

class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

class ModelCost(BaseModel):
    input: float = 0.0       # $ per million tokens
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


class ThinkingLevel(StrEnum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Model(BaseModel):
    id: str
    name: str
    api: str              # API type, e.g. "anthropic-messages", "openai-responses"
    provider: str         # e.g. "anthropic", "openai"
    base_url: str = ""
    reasoning: bool = False
    input_types: list[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    context_window: int = 128_000
    max_tokens: int = 4096
    cost: ModelCost = Field(default_factory=ModelCost)
    headers: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# LLM call context
# ---------------------------------------------------------------------------

class Context(BaseModel):
    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] | None = None


# ---------------------------------------------------------------------------
# Stream options
# ---------------------------------------------------------------------------

class StreamOptions(BaseModel):
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    max_retry_delay_ms: int = 60_000
    headers: dict[str, str] | None = None


class SimpleStreamOptions(StreamOptions):
    reasoning: ThinkingLevel | None = None


# Provider-specific options

class AnthropicThinkingEnabled(BaseModel):
    type: Literal["enabled"] = "enabled"
    budget_tokens: int = 8192


class AnthropicThinkingAdaptive(BaseModel):
    type: Literal["adaptive"] = "adaptive"
    effort: Literal["low", "medium", "high"] = "medium"


AnthropicThinkingConfig = AnthropicThinkingEnabled | AnthropicThinkingAdaptive


class AnthropicOptions(StreamOptions):
    thinking: AnthropicThinkingConfig | None = None
    cache_retention: Literal["short", "long"] | None = None


class OpenAIOptions(StreamOptions):
    reasoning_effort: Literal["low", "medium", "high"] | None = None


class GeminiOptions(StreamOptions):
    thinking_budget: int | None = None


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------

class StartEvent(BaseModel):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class TextStartEvent(BaseModel):
    type: Literal["text_start"] = "text_start"
    content_index: int
    content: TextContent
    partial: AssistantMessage


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class TextEndEvent(BaseModel):
    type: Literal["text_end"] = "text_end"
    content_index: int
    content: TextContent
    partial: AssistantMessage


class ThinkingStartEvent(BaseModel):
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int
    content: ThinkingContent
    partial: AssistantMessage


class ThinkingDeltaEvent(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ThinkingEndEvent(BaseModel):
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: ThinkingContent
    partial: AssistantMessage


class ToolCallStartEvent(BaseModel):
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int
    content: ToolCall
    partial: AssistantMessage


class ToolCallDeltaEvent(BaseModel):
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(BaseModel):
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    content: ToolCall
    partial: AssistantMessage


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    reason: StopReason
    message: AssistantMessage


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    reason: StopReason
    error: AssistantMessage


AssistantMessageEvent = Annotated[
    Union[
        StartEvent,
        TextStartEvent, TextDeltaEvent, TextEndEvent,
        ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
        ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
        DoneEvent, ErrorEvent,
    ],
    Field(discriminator="type"),
]
