"""App-layer AgentSession/runtime orchestration."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypeAlias

from bampy.agent import (
    AfterToolCallContext,
    AfterToolCallResult,
    Agent,
    AgentEvent,
    AgentMessage,
    AgentThinkingLevel,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolExecutionMode,
    default_convert_to_llm,
    message_role,
    message_timestamp,
)
from bampy.agent.messages import clone_message
from bampy.agent.types import (
    AfterToolCallHook,
    _UNSET as AGENT_UNSET,
    ApiKeyResolverFn,
    BeforeToolCallHook,
    ConvertToLlmFn,
    StreamFn,
    TransformContextFn,
)
from bampy.ai import ImageContent, Model, SimpleStreamOptions, UserMessage, get_model
from bampy.ai.types import AssistantMessage, StopReason, TextContent

from .compaction import (
    CompactionResult,
    CompactionSettings,
    DEFAULT_COMPACTION_SETTINGS,
    compact,
    estimate_context_tokens,
    prepare_compaction,
    should_compact,
)
from .extension import (
    AgentEndEvent,
    AgentStartEvent,
    BeforeAgentStartEvent,
    Extension,
    ExtensionRunner,
    InputEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    SessionCompactEvent,
    SessionShutdownEvent,
    SessionStartEvent,
    ToolCallEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
    wrap_registered_tool,
)
from .loader import LoadExtensionsResult, load_extensions
from .messages import create_custom_message, register_app_message_converters
from .session import CompactionEntry, SessionManager
from .skills import LoadSkillsResult, Skill, load_skills
from .system_prompt import ContextFile, BuildSystemPromptOptions, build_system_prompt, load_context_files
from .tools import create_coding_tools


def _default_model() -> Model:
    model = get_model("gemini-2.5-flash-lite", "google")
    if model is not None:
        return model
    model = get_model("gpt-4.1-mini", "openai")
    if model is not None:
        return model
    raise RuntimeError("No built-in fallback model is available")


def _maybe_model_from_session(data: dict[str, str] | None) -> Model | None:
    if not data:
        return None
    model_id = data.get("model_id", "").strip()
    provider = data.get("provider", "").strip() or None
    if not model_id:
        return None
    return get_model(model_id, provider)


def _same_model_identity(model: Model | None, data: dict[str, str] | None) -> bool:
    if model is None or not data:
        return False
    return (
        model.provider == data.get("provider", "").strip()
        and model.id == data.get("model_id", "").strip()
    )


def _parse_iso_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000
    except ValueError:
        return None


def _message_text(message: AgentMessage) -> str:
    role = message_role(message)
    if role != "user":
        return ""

    content = getattr(message, "content", None)
    if isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                parts.append(str(text))
        return "".join(parts)
    return ""


def _message_images(message: AgentMessage) -> list[ImageContent] | None:
    content = getattr(message, "content", None)
    if isinstance(message, dict):
        content = message.get("content")
    if not isinstance(content, list):
        return None

    images: list[ImageContent] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "image":
            continue
        if isinstance(block, ImageContent):
            images.append(block)
        elif isinstance(block, dict):
            images.append(ImageContent(**block))
    return images or None


def _as_tool_mapping(
    tools: list[AgentTool] | dict[str, AgentTool] | None,
    cwd: str,
) -> dict[str, AgentTool]:
    if tools is None:
        return {tool.name: tool for tool in create_coding_tools(cwd)}
    if isinstance(tools, dict):
        return dict(tools)
    return {tool.name: tool for tool in tools}


async def _maybe_await(value: Any) -> Any:
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


@dataclass(slots=True)
class AutoCompactionStartEvent:
    type: Literal["auto_compaction_start"] = "auto_compaction_start"
    reason: Literal["threshold"] = "threshold"


@dataclass(slots=True)
class AutoCompactionEndEvent:
    type: Literal["auto_compaction_end"] = "auto_compaction_end"
    result: CompactionResult | None = None
    aborted: bool = False
    error_message: str | None = None


AgentSessionEvent: TypeAlias = AgentEvent | AutoCompactionStartEvent | AutoCompactionEndEvent
AgentSessionListener: TypeAlias = Any


@dataclass(slots=True)
class CreateAgentSessionResult:
    session: "AgentSession"
    extensions: LoadExtensionsResult
    skills: LoadSkillsResult


class AgentSession:
    """High-level app runtime that wires session, extensions, tools, and compaction."""

    def __init__(
        self,
        *,
        cwd: str | None = None,
        model: Model | None = None,
        thinking_level: AgentThinkingLevel | str | None = None,
        steering_mode: str = "one-at-a-time",
        follow_up_mode: str = "one-at-a-time",
        tools: list[AgentTool] | dict[str, AgentTool] | None = None,
        active_tool_names: list[str] | None = None,
        session_manager: SessionManager | None = None,
        extension_runner: ExtensionRunner | None = None,
        extensions: list[Extension] | None = None,
        context_files: list[ContextFile] | None = None,
        skills: list[Skill] | None = None,
        custom_system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        stream_options: SimpleStreamOptions | None = None,
        get_api_key: ApiKeyResolverFn | None = None,
        stream_fn: StreamFn | None = None,
        convert_to_llm: ConvertToLlmFn | None = None,
        transform_context: TransformContextFn | None = None,
        tool_execution: ToolExecutionMode | str = ToolExecutionMode.PARALLEL,
        before_tool_call: BeforeToolCallHook | None = None,
        after_tool_call: AfterToolCallHook | None = None,
        max_turns: int = 50,
        compaction_settings: CompactionSettings | None = None,
        auto_compaction: bool = True,
        summarization_model: Model | None = None,
        summarization_api_key: str | None = None,
        summarization_custom_instructions: str | None = None,
    ) -> None:
        register_app_message_converters()

        self._cwd = os.path.abspath(cwd or session_manager.cwd if session_manager else os.getcwd())
        self.session_manager = session_manager or SessionManager.create(self._cwd)
        self._context_files = list(context_files or [])
        self._skills = list(skills or [])
        self._custom_system_prompt = custom_system_prompt
        self._append_system_prompt = append_system_prompt
        self._compaction_settings = compaction_settings or DEFAULT_COMPACTION_SETTINGS
        self._auto_compaction = auto_compaction
        self._summarization_model = summarization_model
        self._summarization_api_key = summarization_api_key
        self._summarization_custom_instructions = summarization_custom_instructions
        self._stream_options = stream_options.model_copy(deep=True) if stream_options else SimpleStreamOptions()
        self._get_api_key = get_api_key
        self._stream_fn = stream_fn
        self._user_transform_context = transform_context
        self._user_before_tool_call = before_tool_call
        self._user_after_tool_call = after_tool_call
        self._tool_execution = ToolExecutionMode(tool_execution)
        self._max_turns = max_turns
        self._listeners: list[AgentSessionListener] = []
        self._turn_index = 0
        self._started = False
        self._closed = False
        self._is_compacting = False
        self._event_tail: asyncio.Task[None] | None = None
        self._last_assistant_message: AssistantMessage | None = None

        session_context = self.session_manager.build_session_context()
        restored_model = model or _maybe_model_from_session(session_context.model) or _default_model()
        restored_thinking = AgentThinkingLevel(
            thinking_level or session_context.thinking_level or AgentThinkingLevel.OFF
        )

        self._base_tool_registry = _as_tool_mapping(tools, self._cwd)
        self.extension_runner = extension_runner or ExtensionRunner()
        if extensions is not None:
            self.extension_runner.set_extensions(extensions)
        self.extension_runner.set_session_manager(self.session_manager)
        self.extension_runner.set_model(restored_model)

        self._tool_registry: dict[str, AgentTool] = {}
        self._tool_snippets: dict[str, str] = {}
        self._prompt_guidelines: list[str] = []
        self._active_tool_names: list[str] = []
        self._refresh_tools(active_tool_names)

        self._base_system_prompt = self._build_base_system_prompt()
        self.agent = Agent(
            initial_state={
                "model": restored_model,
                "system_prompt": self._base_system_prompt,
                "thinking_level": restored_thinking,
                "tools": [self._tool_registry[name] for name in self._active_tool_names],
                "messages": list(session_context.messages),
            },
            convert_to_llm=convert_to_llm or default_convert_to_llm,
            transform_context=self._transform_context,
            steering_mode=steering_mode,
            follow_up_mode=follow_up_mode,
            stream_fn=self._stream_fn,
            stream_options=self._stream_options,
            get_api_key=self._get_api_key,
            tool_execution=self._tool_execution,
            max_turns=self._max_turns,
        )
        self.agent.set_before_tool_call(self._before_tool_call_hook)
        self.agent.set_after_tool_call(self._after_tool_call_hook)

        self.extension_runner.set_context_actions(
            {
                "is_idle": lambda: not self.agent.state.is_streaming,
                "abort": self.agent.abort,
                "has_pending_messages": self.agent.has_queued_messages,
                "get_system_prompt": lambda: self.agent.state.system_prompt,
            }
        )
        self.extension_runner.bind_api_actions(
            send_message=self._extension_send_message,
            send_user_message=self._extension_send_user_message,
            append_entry=self._extension_append_entry,
        )
        self._unsubscribe_agent = self.agent.subscribe(self._handle_agent_event)
        self._sync_session_settings()

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def model(self) -> Model:
        return self.agent.state.model

    @property
    def thinking_level(self) -> AgentThinkingLevel:
        return self.agent.state.thinking_level

    @property
    def steering_mode(self) -> str:
        return self.agent.get_steering_mode()

    @property
    def follow_up_mode(self) -> str:
        return self.agent.get_follow_up_mode()

    @property
    def messages(self) -> list[AgentMessage]:
        return self.agent.state.messages

    @property
    def system_prompt(self) -> str:
        return self.agent.state.system_prompt

    @property
    def active_tool_names(self) -> list[str]:
        return list(self._active_tool_names)

    @property
    def extension_errors(self) -> list[Any]:
        return self.extension_runner.errors

    def subscribe(self, listener: AgentSessionListener):
        self._listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        await self.extension_runner.emit(SessionStartEvent())

    async def close(self) -> None:
        if self._closed:
            return
        await self.wait_for_idle()
        self._closed = True
        await self.extension_runner.emit(SessionShutdownEvent())
        self._unsubscribe_agent()

    async def wait_for_idle(self) -> None:
        await self.agent.wait_for_idle()
        await self._drain_event_queue()

    def set_model(self, model: Model) -> None:
        self.agent.set_model(model)
        self.extension_runner.set_model(model)
        self._sync_session_settings()

    def set_thinking_level(self, thinking_level: AgentThinkingLevel | str) -> None:
        self.agent.set_thinking_level(thinking_level)
        self._sync_session_settings()

    def set_steering_mode(self, mode: str) -> None:
        self.agent.set_steering_mode(mode)

    def get_steering_mode(self) -> str:
        return self.agent.get_steering_mode()

    def set_follow_up_mode(self, mode: str) -> None:
        self.agent.set_follow_up_mode(mode)

    def get_follow_up_mode(self) -> str:
        return self.agent.get_follow_up_mode()

    def get_all_tools(self) -> list[AgentTool]:
        return [self._tool_registry[name] for name in self._tool_registry]

    def set_active_tools(self, tool_names: list[str]) -> None:
        self._refresh_tools(tool_names)
        self.agent.set_tools([self._tool_registry[name] for name in self._active_tool_names])
        self._base_system_prompt = self._build_base_system_prompt()
        self.agent.set_system_prompt(self._base_system_prompt)

    def reload_session_context(self) -> None:
        context = self.session_manager.build_session_context()
        self.agent.replace_messages(list(context.messages))

        session_model = (
            self.agent.state.model
            if _same_model_identity(self.agent.state.model, context.model)
            else _maybe_model_from_session(context.model)
        )
        if session_model is not None:
            self.agent.set_model(session_model)
            self.extension_runner.set_model(session_model)

        self.agent.set_thinking_level(context.thinking_level)
        self._sync_session_settings()

    async def prompt(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
        *,
        source: str = "interactive",
    ) -> None:
        await self.start()
        await self.wait_for_idle()
        await self._maybe_auto_compact()

        messages = self._normalize_prompt_input(input, images)
        messages = await self._apply_input_transform(messages, source)
        if not messages:
            return

        await self._prepare_agent_start(messages)
        await self.agent.prompt(messages)
        await self._drain_event_queue()

    async def continue_(self) -> None:
        await self.start()
        await self.wait_for_idle()
        await self._maybe_auto_compact()
        await self._prepare_agent_start([])
        await self.agent.continue_()
        await self._drain_event_queue()

    resume = continue_

    def steer(self, message: AgentMessage) -> None:
        self.agent.steer(message)

    def follow_up(self, message: AgentMessage) -> None:
        self.agent.follow_up(message)

    def clear_steering_queue(self) -> None:
        self.agent.clear_steering_queue()

    def clear_follow_up_queue(self) -> None:
        self.agent.clear_follow_up_queue()

    def clear_all_queues(self) -> None:
        self.agent.clear_all_queues()

    def has_queued_messages(self) -> bool:
        return self.agent.has_queued_messages()

    async def compact(self) -> CompactionResult | None:
        await self.start()
        await self.wait_for_idle()
        return await self._run_compaction(auto=False)

    def _normalize_prompt_input(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None,
    ) -> list[AgentMessage]:
        if isinstance(input, list):
            return list(input)
        if isinstance(input, str):
            if images:
                return [UserMessage(content=[TextContent(text=input), *images])]
            return [UserMessage(content=input)]
        return [input]

    async def _apply_input_transform(
        self,
        messages: list[AgentMessage],
        source: str,
    ) -> list[AgentMessage]:
        if len(messages) != 1 or message_role(messages[0]) != "user":
            return messages

        input_message = messages[0]
        event = InputEvent(
            text=_message_text(input_message),
            images=_message_images(input_message),
            source=source,
        )
        result = await self.extension_runner.emit_input(event)
        if result is None or result.action == "continue":
            return messages
        if result.action == "handled":
            return []
        return [UserMessage(content=[TextContent(text=result.text), *(result.images or [])])]

    async def _prepare_agent_start(self, messages: list[AgentMessage]) -> None:
        prompt_text = "\n".join(filter(None, (_message_text(message) for message in messages)))
        prompt_images = _message_images(messages[0]) if len(messages) == 1 else None
        self._base_system_prompt = self._build_base_system_prompt()
        system_prompt = await self.extension_runner.emit_before_agent_start(
            BeforeAgentStartEvent(
                prompt=prompt_text,
                images=prompt_images,
                system_prompt=self._base_system_prompt,
            )
        )
        self.agent.set_system_prompt(system_prompt)

    async def _transform_context(
        self,
        messages: list[AgentMessage],
        cancellation: Any | None = None,
    ) -> list[AgentMessage]:
        transformed = await self.extension_runner.emit_context(list(messages))
        if self._user_transform_context is not None:
            maybe_messages = await _maybe_await(self._user_transform_context(transformed, cancellation))
            if maybe_messages is not None:
                transformed = list(maybe_messages)
        return transformed

    async def _before_tool_call_hook(
        self,
        context: BeforeToolCallContext,
        cancellation: Any | None = None,
    ) -> BeforeToolCallResult | None:
        await self._drain_event_queue()
        extension_result = await self.extension_runner.emit_tool_call(
            ToolCallEvent(
                tool_call_id=context.tool_call.id,
                tool_name=context.tool_call.name,
                input=dict(context.args),
            )
        )
        if extension_result is not None and extension_result.block:
            return BeforeToolCallResult(block=True, reason=extension_result.reason)

        if self._user_before_tool_call is None:
            return None
        return await _maybe_await(self._user_before_tool_call(context, cancellation))

    async def _after_tool_call_hook(
        self,
        context: AfterToolCallContext,
        cancellation: Any | None = None,
    ) -> AfterToolCallResult | None:
        hook_result = await self.extension_runner.emit_tool_result(
            ToolResultEvent(
                tool_call_id=context.tool_call.id,
                tool_name=context.tool_call.name,
                input=dict(context.args),
                content=list(context.result.content),
                details=context.result.details,
                is_error=context.is_error,
            )
        )

        final_result = AfterToolCallResult()
        modified = False
        if hook_result is not None:
            final_result.content = list(hook_result.content)
            final_result.details = hook_result.details
            final_result.is_error = hook_result.is_error
            modified = True

        if self._user_after_tool_call is not None:
            user_result = await _maybe_await(self._user_after_tool_call(context, cancellation))
            if user_result is not None:
                final_result = user_result if not modified else self._merge_after_tool_results(final_result, user_result)
                modified = True

        return final_result if modified else None

    def _merge_after_tool_results(
        self,
        left: AfterToolCallResult,
        right: AfterToolCallResult,
    ) -> AfterToolCallResult:
        result = AfterToolCallResult()
        result.content = right.content if right.content is not AGENT_UNSET else left.content
        result.details = right.details if right.details is not AGENT_UNSET else left.details
        result.is_error = right.is_error if right.is_error is not AGENT_UNSET else left.is_error
        return result

    def _handle_agent_event(self, event: AgentEvent) -> None:
        loop = asyncio.get_running_loop()
        previous = self._event_tail

        async def _run() -> None:
            if previous is not None:
                try:
                    await previous
                except Exception:
                    pass
            await self._process_agent_event(event)

        self._event_tail = loop.create_task(_run())

    async def _process_agent_event(self, event: AgentEvent) -> None:
        await self._emit_extension_event(event)
        self._persist_event_message(event)

        if event.type == "agent_start":
            self._turn_index = 0
        elif event.type == "turn_end":
            self._turn_index += 1
        elif event.type == "message_end" and isinstance(event.message, AssistantMessage):
            self._last_assistant_message = event.message
        elif event.type == "agent_end" and self._auto_compaction and self._last_assistant_message is not None:
            assistant_message = self._last_assistant_message
            self._last_assistant_message = None
            await self._check_auto_compaction(assistant_message)

        self._emit(event)

    async def _emit_extension_event(self, event: AgentEvent) -> None:
        if event.type == "agent_start":
            await self.extension_runner.emit(AgentStartEvent())
            return
        if event.type == "agent_end":
            await self.extension_runner.emit(AgentEndEvent(messages=[clone_message(msg) for msg in event.messages]))
            return
        if event.type == "turn_start":
            await self.extension_runner.emit(TurnStartEvent(turn_index=self._turn_index))
            return
        if event.type == "turn_end":
            await self.extension_runner.emit(
                TurnEndEvent(
                    turn_index=self._turn_index,
                    message=clone_message(event.message),
                    tool_results=[result.model_copy(deep=True) for result in event.tool_results],
                )
            )
            return
        if event.type == "message_start":
            await self.extension_runner.emit(MessageStartEvent(message=clone_message(event.message)))
            return
        if event.type == "message_update":
            await self.extension_runner.emit(
                MessageUpdateEvent(
                    message=clone_message(event.message),
                    assistant_message_event=event.assistant_message_event.model_copy(deep=True),
                )
            )
            return
        if event.type == "message_end":
            await self.extension_runner.emit(MessageEndEvent(message=clone_message(event.message)))
            return
        if event.type == "tool_execution_start":
            await self.extension_runner.emit(
                ToolExecutionStartEvent(
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    args=event.args,
                )
            )
            return
        if event.type == "tool_execution_update":
            await self.extension_runner.emit(
                ToolExecutionUpdateEvent(
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    args=event.args,
                    partial_result=event.partial_result,
                )
            )
            return
        if event.type == "tool_execution_end":
            await self.extension_runner.emit(
                ToolExecutionEndEvent(
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    result=event.result,
                    is_error=event.is_error,
                )
            )

    def _persist_event_message(self, event: AgentEvent) -> None:
        if event.type != "message_end":
            return
        role = message_role(event.message)
        if role in {"user", "assistant", "tool_result"}:
            self.session_manager.append_message(event.message)
            return
        if role != "custom":
            return

        self.session_manager.append_custom_message_entry(
            getattr(event.message, "custom_type", getattr(event.message, "customType", "")),
            getattr(event.message, "content", ""),
            display=bool(getattr(event.message, "display", True)),
            details=getattr(event.message, "details", None),
        )

    async def _check_auto_compaction(self, assistant_message: AssistantMessage) -> None:
        if assistant_message.stop_reason in (StopReason.ABORTED, "aborted"):
            return
        await self._maybe_auto_compact()

    async def _maybe_auto_compact(self) -> None:
        if self._is_compacting:
            return
        if not self.agent.state.messages:
            return

        context_tokens = self._estimate_current_context_tokens()
        if not should_compact(
            context_tokens,
            self.agent.state.model.context_window,
            self._compaction_settings,
        ):
            return

        await self._run_compaction(auto=True)

    def _estimate_current_context_tokens(self) -> int:
        latest_compaction_ts = self._latest_compaction_timestamp()
        sanitized_messages: list[Any] = []
        for message in self.agent.state.messages:
            if message_role(message) != "assistant":
                sanitized_messages.append(message)
                continue

            timestamp = message_timestamp(message)
            if latest_compaction_ts is None or timestamp is None or timestamp > latest_compaction_ts:
                sanitized_messages.append(message)
                continue

            if hasattr(message, "model_copy"):
                sanitized_messages.append(message.model_copy(update={"usage": None}, deep=True))
                continue

            if isinstance(message, dict):
                cloned = dict(message)
                cloned["usage"] = None
                sanitized_messages.append(cloned)
                continue

            sanitized_messages.append(message)

        return estimate_context_tokens(sanitized_messages).tokens

    def _latest_compaction_timestamp(self) -> float | None:
        latest: float | None = None
        for entry in self.session_manager.get_branch():
            if not isinstance(entry, CompactionEntry):
                continue
            timestamp = _parse_iso_timestamp(entry.timestamp)
            if timestamp is None:
                continue
            latest = timestamp if latest is None else max(latest, timestamp)
        return latest

    async def _run_compaction(self, *, auto: bool) -> CompactionResult | None:
        preparation = prepare_compaction(
            self.session_manager.get_branch(),
            self._compaction_settings,
        )
        if preparation is None:
            return None

        summarization_model = self._summarization_model or self.agent.state.model
        api_key = await self._resolve_summarization_api_key(summarization_model)
        self._is_compacting = True
        if auto:
            self._emit(AutoCompactionStartEvent())

        try:
            result = await compact(
                preparation,
                summarization_model,
                api_key=api_key,
                custom_instructions=self._summarization_custom_instructions,
            )
            self.session_manager.append_compaction(
                result.summary,
                result.first_kept_entry_id,
                result.tokens_before,
                result.details,
            )
            self.reload_session_context()
            await self.extension_runner.emit(SessionCompactEvent(from_extension=False))
            if auto:
                self._emit(AutoCompactionEndEvent(result=result))
            return result
        except Exception as exc:
            if auto:
                self._emit(AutoCompactionEndEvent(aborted=True, error_message=str(exc)))
            raise
        finally:
            self._is_compacting = False

    async def _resolve_summarization_api_key(self, model: Model) -> str | None:
        if self._summarization_api_key is not None:
            return self._summarization_api_key
        if self._get_api_key is not None:
            resolved = await _maybe_await(self._get_api_key(model.provider))
            if resolved is not None:
                return resolved
        return self._stream_options.api_key

    async def _drain_event_queue(self) -> None:
        task = self._event_tail
        if task is not None:
            await task

    def _refresh_tools(self, active_tool_names: list[str] | None) -> None:
        registry = dict(self._base_tool_registry)
        tool_snippets: dict[str, str] = {}
        guidelines_by_tool: dict[str, list[str]] = {}

        for registered in self.extension_runner.get_all_registered_tools():
            definition = registered.definition
            registry[definition.name] = wrap_registered_tool(registered, self.extension_runner)
            if definition.prompt_snippet:
                tool_snippets[definition.name] = definition.prompt_snippet
            cleaned_guidelines = [
                guideline.strip()
                for guideline in (definition.prompt_guidelines or [])
                if guideline.strip()
            ]
            if cleaned_guidelines:
                guidelines_by_tool[definition.name] = cleaned_guidelines

        if active_tool_names is None:
            resolved_active = list(self._base_tool_registry)
            for name in registry:
                if name not in resolved_active:
                    resolved_active.append(name)
        else:
            resolved_active = [name for name in active_tool_names if name in registry]

        prompt_guidelines: list[str] = []
        for name in resolved_active:
            for guideline in guidelines_by_tool.get(name, []):
                if guideline not in prompt_guidelines:
                    prompt_guidelines.append(guideline)

        self._tool_registry = registry
        self._tool_snippets = tool_snippets
        self._prompt_guidelines = prompt_guidelines
        self._active_tool_names = resolved_active

    def _build_base_system_prompt(self) -> str:
        return build_system_prompt(
            BuildSystemPromptOptions(
                custom_prompt=self._custom_system_prompt,
                selected_tools=list(self._active_tool_names),
                tool_snippets=dict(self._tool_snippets),
                prompt_guidelines=list(self._prompt_guidelines),
                append_system_prompt=self._append_system_prompt,
                cwd=self._cwd,
                context_files=list(self._context_files),
                skills=list(self._skills),
            )
        )

    def _sync_session_settings(self) -> None:
        current_context = self.session_manager.build_session_context()
        current_model = current_context.model
        desired_model = {
            "provider": self.agent.state.model.provider,
            "model_id": self.agent.state.model.id,
        }
        if current_model != desired_model:
            self.session_manager.append_model_change(
                self.agent.state.model.provider,
                self.agent.state.model.id,
            )

        desired_thinking = str(self.agent.state.thinking_level)
        if current_context.thinking_level != desired_thinking:
            self.session_manager.append_thinking_level_change(desired_thinking)

    def _extension_send_message(
        self,
        custom_type: str,
        content: str | list[TextContent | ImageContent],
        *,
        display: bool = True,
        details: Any = None,
        trigger_turn: bool = False,
    ) -> None:
        message = create_custom_message(
            custom_type=custom_type,
            content=content,
            display=display,
            details=details,
        )
        if trigger_turn:
            self._schedule_prompt(message)
            return

        self.agent.append_message(message)
        self.session_manager.append_custom_message_entry(
            custom_type,
            content,
            display=display,
            details=details,
        )

    def _extension_send_user_message(
        self,
        content: str | list[TextContent | ImageContent],
    ) -> None:
        self._schedule_prompt(UserMessage(content=content))

    def _extension_append_entry(self, custom_type: str, data: Any = None) -> None:
        self.session_manager.append_custom_entry(custom_type, data)

    def _schedule_prompt(self, message: AgentMessage) -> None:
        loop = asyncio.get_running_loop()
        if self.agent.state.is_streaming:
            self.agent.follow_up(message)
            return
        loop.create_task(self.prompt(message))

    def _emit(self, event: AgentSessionEvent) -> None:
        for listener in tuple(self._listeners):
            listener(event)


async def create_agent_session(
    *,
    cwd: str | None = None,
    model: Model | None = None,
    thinking_level: AgentThinkingLevel | str | None = None,
    steering_mode: str = "one-at-a-time",
    follow_up_mode: str = "one-at-a-time",
    tools: list[AgentTool] | dict[str, AgentTool] | None = None,
    active_tool_names: list[str] | None = None,
    session_manager: SessionManager | None = None,
    extension_paths: list[str] | None = None,
    extension_modules: list[str] | None = None,
    discover_extensions: bool = True,
    skill_paths: list[str] | None = None,
    include_default_skills: bool = True,
    context_filenames: list[str] | None = None,
    custom_system_prompt: str | None = None,
    append_system_prompt: str | None = None,
    stream_options: SimpleStreamOptions | None = None,
    get_api_key: ApiKeyResolverFn | None = None,
    stream_fn: StreamFn | None = None,
    convert_to_llm: ConvertToLlmFn | None = None,
    transform_context: TransformContextFn | None = None,
    tool_execution: ToolExecutionMode | str = ToolExecutionMode.PARALLEL,
    before_tool_call: BeforeToolCallHook | None = None,
    after_tool_call: AfterToolCallHook | None = None,
    max_turns: int = 50,
    compaction_settings: CompactionSettings | None = None,
    auto_compaction: bool = True,
    summarization_model: Model | None = None,
    summarization_api_key: str | None = None,
    summarization_custom_instructions: str | None = None,
) -> CreateAgentSessionResult:
    resolved_cwd = os.path.abspath(cwd or os.getcwd())
    loaded_context_files = load_context_files(resolved_cwd, context_filenames)
    loaded_skills = load_skills(
        cwd=resolved_cwd,
        skill_paths=skill_paths,
        include_defaults=include_default_skills,
    )
    loaded_extensions = await load_extensions(
        paths=extension_paths,
        modules=extension_modules,
        cwd=resolved_cwd,
        discover=discover_extensions,
    )

    session = AgentSession(
        cwd=resolved_cwd,
        model=model,
        thinking_level=thinking_level,
        steering_mode=steering_mode,
        follow_up_mode=follow_up_mode,
        tools=tools,
        active_tool_names=active_tool_names,
        session_manager=session_manager,
        extensions=loaded_extensions.extensions,
        context_files=loaded_context_files,
        skills=loaded_skills.skills,
        custom_system_prompt=custom_system_prompt,
        append_system_prompt=append_system_prompt,
        stream_options=stream_options,
        get_api_key=get_api_key,
        stream_fn=stream_fn,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        tool_execution=tool_execution,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        max_turns=max_turns,
        compaction_settings=compaction_settings,
        auto_compaction=auto_compaction,
        summarization_model=summarization_model,
        summarization_api_key=summarization_api_key,
        summarization_custom_instructions=summarization_custom_instructions,
    )
    await session.start()
    return CreateAgentSessionResult(
        session=session,
        extensions=loaded_extensions,
        skills=loaded_skills,
    )
