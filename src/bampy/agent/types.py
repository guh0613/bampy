"""Core types for the bampy agent runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel

from bampy.agent.cancellation import CancellationToken
from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Message,
    Model,
    SimpleStreamOptions,
    ThinkingLevel,
    Tool,
    ToolResultContentBlock,
    ToolResultMessage,
)
from bampy.ai.validation import schema_from_model


@runtime_checkable
class CustomAgentMessage(Protocol):
    """Protocol for custom non-LLM messages stored in agent state."""

    role: str
    timestamp: float


AgentMessage: TypeAlias = Message | CustomAgentMessage | Mapping[str, Any]


class StreamFn(Protocol):
    """Stream function used by the agent loop."""

    def __call__(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream | Awaitable[AssistantMessageEventStream]:
        ...


class ToolExecutionMode(StrEnum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class AgentThinkingLevel(StrEnum):
    """Agent-facing reasoning level.

    ``off`` is handled at the agent layer and converted to ``None`` before the
    request reaches ``bampy.ai``.
    """

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"

    def to_ai_reasoning(self) -> ThinkingLevel | None:
        if self is AgentThinkingLevel.OFF:
            return None
        return ThinkingLevel(self.value)


ToolParameters: TypeAlias = dict[str, Any] | type[BaseModel]


@dataclass(slots=True)
class AgentToolResult:
    """Final or partial tool output emitted by the agent runtime."""

    content: list[ToolResultContentBlock] = field(default_factory=list)
    details: Any = None


AgentToolUpdateCallback: TypeAlias = Callable[
    [AgentToolResult],
    Awaitable[None] | None,
]


@runtime_checkable
class AgentTool(Protocol):
    """Tool contract for the layer2 agent runtime."""

    name: str
    label: str
    description: str
    parameters: ToolParameters

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        ...


@dataclass(slots=True)
class AgentContext:
    """Conversation state used by the agent loop."""

    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)


@dataclass(slots=True)
class BeforeToolCallResult:
    """Decision returned by ``before_tool_call`` hooks."""

    block: bool = False
    reason: str | None = None


_UNSET = object()


@dataclass(slots=True)
class AfterToolCallResult:
    """Partial override returned by ``after_tool_call`` hooks."""

    content: list[ToolResultContentBlock] | object = _UNSET
    details: Any = _UNSET
    is_error: bool | object = _UNSET


@dataclass(slots=True)
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: Any
    args: Any
    context: AgentContext


@dataclass(slots=True)
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: Any
    args: Any
    result: AgentToolResult
    is_error: bool
    context: AgentContext


ConvertToLlmFn: TypeAlias = Callable[
    [list[AgentMessage]],
    list[Message] | Awaitable[list[Message]],
]
TransformContextFn: TypeAlias = Callable[
    [list[AgentMessage], CancellationToken | None],
    list[AgentMessage] | Awaitable[list[AgentMessage]],
]
MessageSupplierFn: TypeAlias = Callable[
    [],
    list[AgentMessage] | Awaitable[list[AgentMessage]],
]
ApiKeyResolverFn: TypeAlias = Callable[
    [str],
    str | None | Awaitable[str | None],
]
BeforeToolCallHook: TypeAlias = Callable[
    [BeforeToolCallContext, CancellationToken | None],
    BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
]
AfterToolCallHook: TypeAlias = Callable[
    [AfterToolCallContext, CancellationToken | None],
    AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
]


@dataclass(slots=True)
class AgentLoopConfig:
    """Configuration passed into the low-level agent loop."""

    model: Model
    convert_to_llm: ConvertToLlmFn
    stream_options: SimpleStreamOptions = field(default_factory=SimpleStreamOptions)
    transform_context: TransformContextFn | None = None
    get_api_key: ApiKeyResolverFn | None = None
    get_steering_messages: MessageSupplierFn | None = None
    get_follow_up_messages: MessageSupplierFn | None = None
    tool_execution: ToolExecutionMode = ToolExecutionMode.PARALLEL
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None
    max_turns: int = 50


@dataclass(slots=True)
class AgentState:
    """Mutable agent state exposed by ``Agent.state``."""

    system_prompt: str
    model: Model
    thinking_level: AgentThinkingLevel = AgentThinkingLevel.OFF
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    stream_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None


@dataclass(slots=True)
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass(slots=True)
class AgentEndEvent:
    messages: list[AgentMessage]
    type: Literal["agent_end"] = "agent_end"


@dataclass(slots=True)
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"


@dataclass(slots=True)
class TurnEndEvent:
    message: AgentMessage
    tool_results: list[ToolResultMessage]
    type: Literal["turn_end"] = "turn_end"


@dataclass(slots=True)
class MessageStartEvent:
    message: AgentMessage
    type: Literal["message_start"] = "message_start"


@dataclass(slots=True)
class MessageUpdateEvent:
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent
    type: Literal["message_update"] = "message_update"


@dataclass(slots=True)
class MessageEndEvent:
    message: AgentMessage
    type: Literal["message_end"] = "message_end"


@dataclass(slots=True)
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass(slots=True)
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: AgentToolResult
    type: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass(slots=True)
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: AgentToolResult
    is_error: bool
    type: Literal["tool_execution_end"] = "tool_execution_end"


AgentEvent: TypeAlias = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)


def tool_schema(parameters: ToolParameters) -> dict[str, Any]:
    """Resolve an agent-tool parameter definition into JSON schema."""
    if isinstance(parameters, dict):
        return parameters
    if isinstance(parameters, type) and issubclass(parameters, BaseModel):
        return schema_from_model(parameters)
    if hasattr(parameters, "model_json_schema"):
        schema = parameters.model_json_schema()
        if isinstance(schema, dict):
            return schema
    raise TypeError(f"Unsupported tool parameter definition: {type(parameters)!r}")


def to_ai_tool(tool: AgentTool) -> Tool:
    """Convert a layer2 tool into the LLM-facing tool definition."""
    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=tool_schema(tool.parameters),
    )
