"""Tests for bampy.app.runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path

from bampy.agent import AgentToolResult
from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    Usage,
)
from bampy.app.compaction import CompactionResult, CompactionSettings
from bampy.app.extension import (
    BeforeAgentStartEventResult,
    ContextEventResult,
    ExtensionAPI,
    ToolDefinition,
    ToolResultEventResult,
)
from bampy.app.runtime import AgentSession, create_agent_session
from bampy.app.session import CompactionEntry, InMemoryBackend, SessionManager, SessionMessageEntry


def create_model(
    *,
    context_window: int = 128_000,
) -> Model:
    return Model(
        id="mock-model",
        name="Mock Model",
        api="mock-api",
        provider="mock",
        context_window=context_window,
        max_tokens=4096,
    )


def done_stream(message: AssistantMessage) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason=message.stop_reason, message=message))
    return stream


class TestAgentSession:
    async def test_runtime_wires_extensions_tools_prompt_and_session_persistence(self):
        seen_prompts: list[str] = []
        seen_context_lengths: list[int] = []
        api = ExtensionAPI("ext.py")

        async def run_tool(*_args) -> AgentToolResult:
            return AgentToolResult(content=[TextContent(text="tool output")], details={"tool": "echo"})

        def on_context(event, _ctx):
            return ContextEventResult(
                messages=[*event.messages, {"role": "user", "content": "context tail"}]
            )

        def on_before_agent_start(event, _ctx):
            return BeforeAgentStartEventResult(system_prompt=event.system_prompt + "\nEXTENSION READY")

        def on_tool_result(event, _ctx):
            return ToolResultEventResult(
                content=[*event.content, TextContent(text="patched")],
                details={"patched": True},
            )

        api.on("context", on_context)
        api.on("before_agent_start", on_before_agent_start)
        api.on("tool_result", on_tool_result)
        api.register_tool(
            ToolDefinition(
                name="echo",
                label="Echo",
                description="Echo test tool",
                parameters={"type": "object", "properties": {"value": {"type": "string"}}},
                execute=run_tool,
                prompt_snippet="Use echo when the user asks for test echo behavior.",
            )
        )

        call_index = {"value": 0}

        def stream_fn(model, context, options):
            del model, options
            seen_prompts.append(context.system_prompt or "")
            seen_context_lengths.append(len(context.messages))

            if call_index["value"] == 0:
                call_index["value"] += 1
                return done_stream(
                    AssistantMessage(
                        api="mock-api",
                        provider="mock",
                        model="mock-model",
                        content=[ToolCall(id="call-1", name="echo", arguments={"value": "hi"})],
                        stop_reason=StopReason.TOOL_USE,
                    )
                )

            tool_result = next(
                message
                for message in reversed(context.messages)
                if isinstance(message, ToolResultMessage)
            )
            assert isinstance(tool_result, ToolResultMessage)
            assert [block.text for block in tool_result.content] == ["tool output", "patched"]
            call_index["value"] += 1
            return done_stream(
                AssistantMessage(
                    api="mock-api",
                    provider="mock",
                    model="mock-model",
                    content=[TextContent(text="final answer")],
                    stop_reason=StopReason.STOP,
                )
            )

        session = AgentSession(
            cwd="/repo",
            model=create_model(),
            session_manager=SessionManager.in_memory("/repo"),
            extensions=[api._build_extension()],
            stream_fn=stream_fn,
        )

        await session.prompt("hello")

        assert "Use echo when the user asks for test echo behavior." in seen_prompts[0]
        assert seen_prompts[0].endswith("EXTENSION READY")
        assert seen_context_lengths == [2, 4]
        assert [message.role for message in session.messages] == [
            "user",
            "assistant",
            "tool_result",
            "assistant",
        ]
        assert [block.text for block in session.messages[2].content] == ["tool output", "patched"]

        entries = [
            entry
            for entry in session.session_manager.get_entries()
            if isinstance(entry, SessionMessageEntry)
        ]
        assert [entry.message["role"] for entry in entries] == [
            "user",
            "assistant",
            "tool_result",
            "assistant",
        ]

    async def test_runtime_persists_terminal_error_assistant_message(self):
        backend = InMemoryBackend()
        session = AgentSession(
            cwd="/repo",
            model=create_model(),
            session_manager=SessionManager("/repo", backend=backend, persist=True),
            stream_fn=lambda model, context, options: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        await session.prompt("hello")

        entries = [
            entry
            for entry in session.session_manager.get_entries()
            if isinstance(entry, SessionMessageEntry)
        ]
        assert [entry.message["role"] for entry in entries] == ["user", "assistant"]
        assert entries[-1].message["stop_reason"] == StopReason.ERROR
        assert entries[-1].message["error_message"] == "boom"

        raw_entries = backend.read_all()
        assert raw_entries[0]["type"] == "session"
        assert [entry["type"] for entry in raw_entries[-2:]] == ["message", "message"]
        assert raw_entries[-1]["message"]["stop_reason"] == StopReason.ERROR

    async def test_runtime_persists_terminal_aborted_assistant_message(self):
        backend = InMemoryBackend()

        def stream_fn(model, context, options):
            del context
            stream = AssistantMessageEventStream()

            async def runner():
                while not options.cancellation.cancelled:
                    await asyncio.sleep(0.01)
                message = AssistantMessage(
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    content=[TextContent(text="partial result")],
                    stop_reason=StopReason.ABORTED,
                    error_message=options.cancellation.reason,
                )
                stream.push(ErrorEvent(reason=StopReason.ABORTED, error=message))

            asyncio.create_task(runner())
            return stream

        session = AgentSession(
            cwd="/repo",
            model=create_model(),
            session_manager=SessionManager("/repo", backend=backend, persist=True),
            stream_fn=stream_fn,
        )

        prompt_task = asyncio.create_task(session.prompt("hello"))
        await asyncio.sleep(0.03)
        session.abort("user aborted")
        await prompt_task

        entries = [
            entry
            for entry in session.session_manager.get_entries()
            if isinstance(entry, SessionMessageEntry)
        ]
        assert [entry.message["role"] for entry in entries] == ["user", "assistant"]
        assert entries[-1].message["stop_reason"] == StopReason.ABORTED
        assert entries[-1].message["error_message"] == "user aborted"

        raw_entries = backend.read_all()
        assert raw_entries[0]["type"] == "session"
        assert [entry["type"] for entry in raw_entries[-2:]] == ["message", "message"]
        assert raw_entries[-1]["message"]["stop_reason"] == StopReason.ABORTED

    async def test_create_agent_session_loads_extensions_skills_and_context_files(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("runtime guidance", encoding="utf-8")
        skill_dir = tmp_path / ".bampy" / "skills" / "skill-demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: skill-demo\ndescription: Demo skill\n---\n\n# Demo\n",
            encoding="utf-8",
        )

        ext_dir = tmp_path / ".bampy" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "demo_ext.py").write_text(
            "def setup(api):\n"
            "    def on_start(event, ctx):\n"
            "        api.send_message('note', 'loaded from extension')\n"
            "    api.on('session_start', on_start)\n",
            encoding="utf-8",
        )

        result = await create_agent_session(
            cwd=str(tmp_path),
            model=create_model(),
            session_manager=SessionManager.in_memory(str(tmp_path)),
            stream_fn=lambda model, context, options: done_stream(
                AssistantMessage(
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    content=[TextContent(text="ok")],
                )
            ),
        )

        session = result.session

        assert len(result.extensions.extensions) == 1
        assert len(result.extensions.errors) == 0
        assert [skill.name for skill in result.skills.skills] == ["skill-demo"]
        assert "runtime guidance" in session.system_prompt
        assert "skill-demo" in session.system_prompt
        assert session.messages[-1].role == "custom"
        assert session.messages[-1].content == "loaded from extension"

    async def test_auto_compaction_triggers_and_reloads_session_context(self, monkeypatch):
        async def fake_compact(
            preparation,
            model,
            api_key,
            custom_instructions=None,
            cancellation=None,
        ):
            del model, api_key, custom_instructions, cancellation
            return CompactionResult(
                summary="summary text",
                first_kept_entry_id=preparation.first_kept_entry_id,
                tokens_before=preparation.tokens_before,
            )

        monkeypatch.setattr("bampy.app.runtime.compact", fake_compact)

        events: list[str] = []
        session = AgentSession(
            cwd="/repo",
            model=create_model(context_window=100),
            session_manager=SessionManager.in_memory("/repo"),
            stream_fn=lambda model, context, options: done_stream(
                AssistantMessage(
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    content=[TextContent(text="big answer")],
                    usage=Usage(total_tokens=95),
                    stop_reason=StopReason.STOP,
                )
            ),
            compaction_settings=CompactionSettings(
                enabled=True,
                reserve_tokens=10,
                keep_recent_tokens=1,
            ),
            auto_compaction=True,
        )
        session.subscribe(lambda event: events.append(event.type))

        await session.prompt("trigger compaction")

        assert "auto_compaction_start" in events
        assert "auto_compaction_end" in events
        assert session.messages[0].role == "compaction_summary"
        assert any(isinstance(entry, CompactionEntry) for entry in session.session_manager.get_entries())

    def test_reload_session_context_preserves_current_model_overrides(self):
        session_manager = SessionManager.in_memory("/repo")
        overridden_model = create_model().model_copy(
            update={"base_url": "https://proxy.example.test"}
        )
        session = AgentSession(
            cwd="/repo",
            model=overridden_model,
            session_manager=session_manager,
            stream_fn=lambda model, context, options: done_stream(
                AssistantMessage(
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    content=[TextContent(text="ok")],
                )
            ),
        )

        session_manager.append_message(
            AssistantMessage(
                api=overridden_model.api,
                provider=overridden_model.provider,
                model=overridden_model.id,
                content=[TextContent(text="persisted")],
            )
        )

        session.reload_session_context()

        assert session.model.base_url == "https://proxy.example.test"

    async def test_session_exposes_steer_and_follow_up_queue_controls(self):
        session_manager = SessionManager.in_memory("/repo")
        session_manager.append_message(UserMessage(content="Initial"))
        session_manager.append_message(
            AssistantMessage(
                api="mock-api",
                provider="mock",
                model="mock-model",
                content=[TextContent(text="Initial response")],
                stop_reason=StopReason.STOP,
            )
        )

        response_count = {"value": 0}

        def stream_fn(model, context, options):
            del context, options
            response_count["value"] += 1
            return done_stream(
                AssistantMessage(
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    content=[TextContent(text=f"Processed {response_count['value']}")],
                    stop_reason=StopReason.STOP,
                )
            )

        session = AgentSession(
            cwd="/repo",
            model=create_model(),
            session_manager=session_manager,
            stream_fn=stream_fn,
        )

        assert session.steering_mode == "one-at-a-time"
        assert session.follow_up_mode == "one-at-a-time"

        session.set_steering_mode("all")
        session.set_follow_up_mode("all")

        assert session.get_steering_mode() == "all"
        assert session.get_follow_up_mode() == "all"

        session.steer(UserMessage(content="Steering A"))
        session.follow_up(UserMessage(content="Follow-up A"))
        assert session.has_queued_messages() is True

        session.clear_steering_queue()
        assert session.has_queued_messages() is True

        session.steer(UserMessage(content="Steering B"))
        session.follow_up(UserMessage(content="Follow-up B"))
        session.clear_follow_up_queue()
        assert session.has_queued_messages() is True

        session.follow_up(UserMessage(content="Follow-up C"))
        await session.continue_()

        assert response_count["value"] == 2
        assert session.has_queued_messages() is False
        assert [
            message.role if hasattr(message, "role") else message["role"]
            for message in session.messages
        ] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        steering_message = session.messages[2]
        follow_up_message = session.messages[4]
        assert (
            steering_message.content
            if hasattr(steering_message, "content")
            else steering_message["content"]
        ) == "Steering B"
        assert (
            follow_up_message.content
            if hasattr(follow_up_message, "content")
            else follow_up_message["content"]
        ) == "Follow-up C"

    async def test_create_agent_session_passes_queue_modes_to_agent(self, tmp_path: Path):
        result = await create_agent_session(
            cwd=str(tmp_path),
            model=create_model(),
            session_manager=SessionManager.in_memory(str(tmp_path)),
            discover_extensions=False,
            include_default_skills=False,
            steering_mode="all",
            follow_up_mode="all",
            stream_fn=lambda model, context, options: done_stream(
                AssistantMessage(
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    content=[TextContent(text="ok")],
                    stop_reason=StopReason.STOP,
                )
            ),
        )

        try:
            assert result.session.steering_mode == "all"
            assert result.session.follow_up_mode == "all"
        finally:
            await result.session.close()
