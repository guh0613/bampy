"""Extension system: types, API, runner, and context.

Extensions are Python modules that export a ``setup(api)`` factory function.
During setup they register event handlers, tools, and commands via the
:class:`ExtensionAPI`.  The :class:`ExtensionRunner` manages lifecycle and
event dispatch.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import (
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    ToolParameters,
)
from bampy.ai.types import (
    AssistantMessageEvent,
    ImageContent,
    Model,
    TextContent,
    ToolResultMessage,
    UserMessage,
)

from .session import SessionManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extension event types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SessionStartEvent:
    type: str = field(default="session_start", init=False)


@dataclass(slots=True)
class SessionShutdownEvent:
    type: str = field(default="session_shutdown", init=False)


@dataclass(slots=True)
class SessionCompactEvent:
    type: str = field(default="session_compact", init=False)
    from_extension: bool = False


@dataclass(slots=True)
class ContextEvent:
    """Fired before each LLM call. Handlers may modify ``messages``."""

    type: str = field(default="context", init=False)
    messages: list[AgentMessage] = field(default_factory=list)


@dataclass(slots=True)
class BeforeAgentStartEvent:
    type: str = field(default="before_agent_start", init=False)
    prompt: str = ""
    images: list[ImageContent] | None = None
    system_prompt: str = ""


@dataclass(slots=True)
class AgentStartEvent:
    type: str = field(default="agent_start", init=False)


@dataclass(slots=True)
class AgentEndEvent:
    type: str = field(default="agent_end", init=False)
    messages: list[AgentMessage] = field(default_factory=list)


@dataclass(slots=True)
class TurnStartEvent:
    type: str = field(default="turn_start", init=False)
    turn_index: int = 0


@dataclass(slots=True)
class TurnEndEvent:
    type: str = field(default="turn_end", init=False)
    turn_index: int = 0
    message: AgentMessage | None = None
    tool_results: list[ToolResultMessage] = field(default_factory=list)


@dataclass(slots=True)
class MessageStartEvent:
    type: str = field(default="message_start", init=False)
    message: AgentMessage | None = None


@dataclass(slots=True)
class MessageUpdateEvent:
    type: str = field(default="message_update", init=False)
    message: AgentMessage | None = None
    assistant_message_event: AssistantMessageEvent | None = None


@dataclass(slots=True)
class MessageEndEvent:
    type: str = field(default="message_end", init=False)
    message: AgentMessage | None = None


@dataclass(slots=True)
class ToolExecutionStartEvent:
    type: str = field(default="tool_execution_start", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None


@dataclass(slots=True)
class ToolExecutionUpdateEvent:
    type: str = field(default="tool_execution_update", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None
    partial_result: Any = None


@dataclass(slots=True)
class ToolExecutionEndEvent:
    type: str = field(default="tool_execution_end", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    result: Any = None
    is_error: bool = False


@dataclass(slots=True)
class ToolCallEvent:
    """Fired before a tool executes. Handlers can block via ``ToolCallEventResult``."""

    type: str = field(default="tool_call", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResultEvent:
    """Fired after a tool executes. Handlers can modify the result."""

    type: str = field(default="tool_result", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    content: list[TextContent | ImageContent] = field(default_factory=list)
    details: Any = None
    is_error: bool = False


@dataclass(slots=True)
class InputEvent:
    """Fired when user input is received before agent processing."""

    type: str = field(default="input", init=False)
    text: str = ""
    images: list[ImageContent] | None = None
    source: str = "interactive"


ExtensionEvent: TypeAlias = (
    SessionStartEvent
    | SessionShutdownEvent
    | SessionCompactEvent
    | ContextEvent
    | BeforeAgentStartEvent
    | AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | ToolCallEvent
    | ToolResultEvent
    | InputEvent
)


# ---------------------------------------------------------------------------
# Event result types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ContextEventResult:
    messages: list[AgentMessage] | None = None


@dataclass(slots=True)
class ToolCallEventResult:
    block: bool = False
    reason: str | None = None


@dataclass(slots=True)
class ToolResultEventResult:
    content: list[TextContent | ImageContent] | object = field(default_factory=lambda: _UNSET)
    details: Any = field(default_factory=lambda: _UNSET)
    is_error: bool | object = field(default_factory=lambda: _UNSET)


@dataclass(slots=True)
class BeforeAgentStartEventResult:
    system_prompt: str | None = None


@dataclass(slots=True)
class InputEventResult:
    action: Literal["continue", "transform", "handled"] = "continue"
    text: str = ""
    images: list[ImageContent] | None = None


# ---------------------------------------------------------------------------
# Extension context
# ---------------------------------------------------------------------------

@dataclass
class ExtensionContext:
    """Context passed to extension event handlers."""

    cwd: str = ""
    session_manager: SessionManager | None = None
    model: Model | None = None

    def is_idle(self) -> bool:
        return self._is_idle() if self._is_idle else True

    def abort(self) -> None:
        if self._abort:
            self._abort()

    def has_pending_messages(self) -> bool:
        return self._has_pending_messages() if self._has_pending_messages else False

    def get_system_prompt(self) -> str:
        return self._get_system_prompt() if self._get_system_prompt else ""

    # Internal action references (set by runner)
    _is_idle: Callable[[], bool] | None = field(default=None, repr=False)
    _abort: Callable[[], None] | None = field(default=None, repr=False)
    _has_pending_messages: Callable[[], bool] | None = field(default=None, repr=False)
    _get_system_prompt: Callable[[], str] | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Tool definition for extensions
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ToolDefinition:
    """Tool registered by an extension via ``api.register_tool()``."""

    name: str
    label: str
    description: str
    parameters: ToolParameters
    execute: Callable[..., AgentToolResult | Awaitable[AgentToolResult]]
    prompt_snippet: str | None = None
    prompt_guidelines: list[str] | None = None


@dataclass(slots=True)
class RegisteredTool:
    definition: ToolDefinition
    extension_path: str = ""


# ---------------------------------------------------------------------------
# Registered command
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RegisteredCommand:
    name: str
    description: str = ""
    handler: Callable[..., Awaitable[None]] | None = None


# ---------------------------------------------------------------------------
# Loaded extension
# ---------------------------------------------------------------------------

@dataclass
class Extension:
    """A loaded extension with all its registered items."""

    path: str
    resolved_path: str = ""
    handlers: dict[str, list[Callable[..., Any]]] = field(default_factory=dict)
    tools: dict[str, RegisteredTool] = field(default_factory=dict)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    api: ExtensionAPI | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Extension API (passed to extension factories)
# ---------------------------------------------------------------------------

ExtensionHandler: TypeAlias = Callable[..., Any]
ExtensionFactory: TypeAlias = Callable[["ExtensionAPI"], None | Awaitable[None]]


class ExtensionAPI:
    """API passed to extension ``setup()`` functions for registration."""

    def __init__(self, extension_path: str) -> None:
        self._extension_path = extension_path
        self._handlers: dict[str, list[ExtensionHandler]] = {}
        self._tools: dict[str, RegisteredTool] = {}
        self._commands: dict[str, RegisteredCommand] = {}
        # Action stubs (replaced by runner)
        self._send_message: Callable[..., None] | None = None
        self._send_user_message: Callable[..., None] | None = None
        self._append_entry: Callable[..., None] | None = None

    def on(self, event: str, handler: ExtensionHandler) -> None:
        """Subscribe to an event."""
        self._handlers.setdefault(event, []).append(handler)

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool that the LLM can call."""
        self._tools[tool.name] = RegisteredTool(
            definition=tool,
            extension_path=self._extension_path,
        )

    def register_command(
        self,
        name: str,
        *,
        description: str = "",
        handler: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        """Register a slash command."""
        self._commands[name] = RegisteredCommand(
            name=name,
            description=description,
            handler=handler,
        )

    def send_message(
        self,
        custom_type: str,
        content: str | list[TextContent | ImageContent],
        *,
        display: bool = True,
        details: Any = None,
        trigger_turn: bool = False,
    ) -> None:
        """Send a custom message to the session."""
        if self._send_message:
            self._send_message(custom_type, content, display=display, details=details, trigger_turn=trigger_turn)

    def send_user_message(
        self,
        content: str | list[TextContent | ImageContent],
    ) -> None:
        """Send a user message to the agent (always triggers a turn)."""
        if self._send_user_message:
            self._send_user_message(content)

    def append_entry(self, custom_type: str, data: Any = None) -> None:
        """Append a custom entry to the session (not sent to LLM)."""
        if self._append_entry:
            self._append_entry(custom_type, data)

    def _build_extension(self) -> Extension:
        return Extension(
            path=self._extension_path,
            resolved_path=self._extension_path,
            handlers=dict(self._handlers),
            tools=dict(self._tools),
            commands=dict(self._commands),
            api=self,
        )


# ---------------------------------------------------------------------------
# Extension Runner
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExtensionError:
    extension_path: str
    event: str
    error: str
    stack: str | None = None


class ExtensionRunner:
    """Manages extension lifecycle and event dispatch."""

    def __init__(self) -> None:
        self._extensions: list[Extension] = []
        self._errors: list[ExtensionError] = []
        self._context_actions: dict[str, Callable[..., Any]] = {}
        self._session_manager: SessionManager | None = None
        self._model: Model | None = None
        self._send_message_action: Callable[..., None] | None = None
        self._send_user_message_action: Callable[..., None] | None = None
        self._append_entry_action: Callable[..., None] | None = None

    def set_extensions(self, extensions: list[Extension]) -> None:
        self._extensions = list(extensions)
        self._apply_api_actions()

    def set_session_manager(self, sm: SessionManager) -> None:
        self._session_manager = sm
        self._apply_api_actions()

    def set_model(self, model: Model | None) -> None:
        self._model = model

    def set_context_actions(self, actions: dict[str, Callable[..., Any]]) -> None:
        """Set action implementations for ExtensionContext."""
        self._context_actions = dict(actions)

    def bind_api_actions(
        self,
        *,
        send_message: Callable[..., None] | None = None,
        send_user_message: Callable[..., None] | None = None,
        append_entry: Callable[..., None] | None = None,
    ) -> None:
        """Bind runtime actions used by ExtensionAPI helper methods."""
        self._send_message_action = send_message
        self._send_user_message_action = send_user_message
        self._append_entry_action = append_entry
        self._apply_api_actions()

    @property
    def extensions(self) -> list[Extension]:
        return list(self._extensions)

    @property
    def errors(self) -> list[ExtensionError]:
        return list(self._errors)

    def create_context(self) -> ExtensionContext:
        return ExtensionContext(
            cwd=self._session_manager.cwd if self._session_manager else ".",
            session_manager=self._session_manager,
            model=self._model,
            _is_idle=self._context_actions.get("is_idle"),
            _abort=self._context_actions.get("abort"),
            _has_pending_messages=self._context_actions.get("has_pending_messages"),
            _get_system_prompt=self._context_actions.get("get_system_prompt"),
        )

    def get_all_registered_tools(self) -> list[RegisteredTool]:
        """Collect all tools registered by extensions (first-registration-wins)."""
        seen: set[str] = set()
        tools: list[RegisteredTool] = []
        for ext in self._extensions:
            for name, tool in ext.tools.items():
                if name not in seen:
                    seen.add(name)
                    tools.append(tool)
        return tools

    def get_all_commands(self) -> list[RegisteredCommand]:
        seen: set[str] = set()
        commands: list[RegisteredCommand] = []
        for ext in self._extensions:
            for name, cmd in ext.commands.items():
                if name not in seen:
                    seen.add(name)
                    commands.append(cmd)
        return commands

    # -----------------------------------------------------------------------
    # Event emission
    # -----------------------------------------------------------------------

    async def emit(self, event: ExtensionEvent) -> None:
        """Emit an event to all registered handlers."""
        event_type = event.type
        ctx = self.create_context()
        for ext in self._extensions:
            handlers = ext.handlers.get(event_type, [])
            for handler in handlers:
                try:
                    result = handler(event, ctx)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    self._record_error(ext.path, event_type, exc)

    async def emit_tool_call(self, event: ToolCallEvent) -> ToolCallEventResult | None:
        """Emit a tool_call event; return block result if any handler blocks."""
        ctx = self.create_context()
        for ext in self._extensions:
            handlers = ext.handlers.get("tool_call", [])
            for handler in handlers:
                try:
                    result = handler(event, ctx)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, ToolCallEventResult) and result.block:
                        return result
                except Exception as exc:
                    self._record_error(ext.path, "tool_call", exc)
        return None

    async def emit_tool_result(self, event: ToolResultEvent) -> ToolResultEventResult | None:
        """Emit a tool_result event and chain patches across handlers."""
        ctx = self.create_context()
        current_event = ToolResultEvent(
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            input=dict(event.input),
            content=list(event.content),
            details=event.details,
            is_error=event.is_error,
        )
        modified = False
        for ext in self._extensions:
            handlers = ext.handlers.get("tool_result", [])
            for handler in handlers:
                try:
                    result = handler(current_event, ctx)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, ToolResultEventResult):
                        if result.content is not _UNSET:
                            current_event.content = list(result.content)
                            modified = True
                        if result.details is not _UNSET:
                            current_event.details = result.details
                            modified = True
                        if result.is_error is not _UNSET:
                            current_event.is_error = bool(result.is_error)
                            modified = True
                except Exception as exc:
                    self._record_error(ext.path, "tool_result", exc)
        if not modified:
            return None
        return ToolResultEventResult(
            content=list(current_event.content),
            details=current_event.details,
            is_error=current_event.is_error,
        )

    async def emit_context(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        """Emit a context event; allow handlers to transform messages."""
        event = ContextEvent(messages=list(messages))
        ctx = self.create_context()
        for ext in self._extensions:
            handlers = ext.handlers.get("context", [])
            for handler in handlers:
                try:
                    result = handler(event, ctx)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, ContextEventResult) and result.messages is not None:
                        event.messages = result.messages
                except Exception as exc:
                    self._record_error(ext.path, "context", exc)
        return event.messages

    async def emit_before_agent_start(self, event: BeforeAgentStartEvent) -> str:
        """Emit before_agent_start; return (potentially modified) system prompt."""
        ctx = self.create_context()
        prompt = event.system_prompt
        for ext in self._extensions:
            handlers = ext.handlers.get("before_agent_start", [])
            for handler in handlers:
                try:
                    result = handler(event, ctx)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, BeforeAgentStartEventResult) and result.system_prompt is not None:
                        prompt = result.system_prompt
                        event.system_prompt = prompt
                except Exception as exc:
                    self._record_error(ext.path, "before_agent_start", exc)
        return prompt

    async def emit_input(self, event: InputEvent) -> InputEventResult | None:
        """Emit input event; return transform/handled result if any."""
        ctx = self.create_context()
        for ext in self._extensions:
            handlers = ext.handlers.get("input", [])
            for handler in handlers:
                try:
                    result = handler(event, ctx)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, InputEventResult) and result.action != "continue":
                        return result
                except Exception as exc:
                    self._record_error(ext.path, "input", exc)
        return None

    def _record_error(self, ext_path: str, event: str, exc: Exception) -> None:
        import traceback

        self._errors.append(
            ExtensionError(
                extension_path=ext_path,
                event=event,
                error=str(exc),
                stack=traceback.format_exc(),
            )
        )
        logger.warning("Extension error in %s [%s]: %s", ext_path, event, exc)

    def _apply_api_actions(self) -> None:
        for ext in self._extensions:
            if ext.api is None:
                continue
            ext.api._send_message = self._resolve_send_message_action()
            ext.api._send_user_message = self._resolve_send_user_message_action()
            ext.api._append_entry = self._resolve_append_entry_action()

    def _resolve_send_message_action(self) -> Callable[..., None] | None:
        if self._send_message_action is not None:
            return self._send_message_action
        if self._session_manager is None:
            return None

        def _default_send_message(
            custom_type: str,
            content: str | list[TextContent | ImageContent],
            *,
            display: bool = True,
            details: Any = None,
            trigger_turn: bool = False,
        ) -> None:
            self._session_manager.append_custom_message_entry(
                custom_type,
                content,
                display=display,
                details=details,
            )
            if trigger_turn:
                logger.warning(
                    "Extension requested trigger_turn for custom message %s, but no agent callback is bound",
                    custom_type,
                )

        return _default_send_message

    def _resolve_send_user_message_action(self) -> Callable[..., None] | None:
        if self._send_user_message_action is not None:
            return self._send_user_message_action
        if self._session_manager is None:
            return None

        def _default_send_user_message(
            content: str | list[TextContent | ImageContent],
        ) -> None:
            self._session_manager.append_message(UserMessage(content=content))

        return _default_send_user_message

    def _resolve_append_entry_action(self) -> Callable[..., None] | None:
        if self._append_entry_action is not None:
            return self._append_entry_action
        if self._session_manager is None:
            return None

        def _default_append_entry(custom_type: str, data: Any = None) -> None:
            self._session_manager.append_custom_entry(custom_type, data)

        return _default_append_entry


_UNSET = object()


# ---------------------------------------------------------------------------
# Wrap extension tool into AgentTool
# ---------------------------------------------------------------------------

def wrap_registered_tool(
    registered: RegisteredTool,
    runner: ExtensionRunner,
) -> AgentTool:
    """Adapt a :class:`RegisteredTool` into the layer-2 :class:`AgentTool` protocol."""
    defn = registered.definition

    class _WrappedTool:
        __slots__ = ("name", "label", "description", "parameters")

        def __init__(self) -> None:
            self.name = defn.name
            self.label = defn.label
            self.description = defn.description
            self.parameters = defn.parameters

        async def execute(
            self,
            tool_call_id: str,
            params: Any,
            cancellation: CancellationToken | None = None,
            on_update: AgentToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            ctx = runner.create_context()
            result = defn.execute(tool_call_id, params, cancellation, on_update, ctx)
            if inspect.isawaitable(result):
                return await result
            return result

    return _WrappedTool()  # type: ignore[return-value]
