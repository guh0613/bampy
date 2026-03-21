"""Tests for bampy.app.extension."""

from __future__ import annotations

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult
from bampy.ai.types import Model, TextContent, UserMessage
from bampy.app.extension import (
    BeforeAgentStartEvent,
    BeforeAgentStartEventResult,
    ContextEventResult,
    Extension,
    ExtensionAPI,
    ExtensionRunner,
    InputEvent,
    InputEventResult,
    RegisteredCommand,
    ToolCallEvent,
    ToolCallEventResult,
    ToolDefinition,
    ToolResultEvent,
    ToolResultEventResult,
    wrap_registered_tool,
)
from bampy.app.session import SessionManager


class TestExtensionAPI:
    def test_build_extension_and_action_stubs(self):
        api = ExtensionAPI("ext.py")
        calls: dict[str, object] = {}

        api.on("session_start", lambda event, ctx: None)
        api.register_command("hello", description="wave")

        async def run_tool(*_args) -> AgentToolResult:
            return AgentToolResult(content=[TextContent(text="ok")])

        api.register_tool(
            ToolDefinition(
                name="echo",
                label="Echo",
                description="Echo input",
                parameters={"type": "object"},
                execute=run_tool,
            )
        )

        api._send_message = lambda *args, **kwargs: calls.setdefault("message", (args, kwargs))
        api._send_user_message = lambda content: calls.setdefault("user_message", content)
        api._append_entry = lambda custom_type, data=None: calls.setdefault(
            "entry", (custom_type, data)
        )

        api.send_message("note", "hello", trigger_turn=True)
        api.send_user_message("from user")
        api.append_entry("meta", {"x": 1})

        ext = api._build_extension()

        assert ext.path == "ext.py"
        assert "session_start" in ext.handlers
        assert isinstance(ext.commands["hello"], RegisteredCommand)
        assert "echo" in ext.tools
        assert ext.api is api
        assert calls["message"][0] == ("note", "hello")
        assert calls["message"][1]["trigger_turn"] is True
        assert calls["user_message"] == "from user"
        assert calls["entry"] == ("meta", {"x": 1})


class TestExtensionRunner:
    def _runner_with_context(self) -> ExtensionRunner:
        runner = ExtensionRunner()
        runner.set_session_manager(SessionManager.in_memory("/repo"))
        runner.set_model(
            Model(id="gpt-5", name="GPT-5", api="openai-responses", provider="openai")
        )
        runner.set_context_actions(
            {
                "is_idle": lambda: False,
                "abort": lambda: None,
                "has_pending_messages": lambda: True,
                "get_system_prompt": lambda: "base prompt",
            }
        )
        return runner

    async def test_emit_records_errors_and_continues(self):
        seen: list[str] = []
        runner = self._runner_with_context()

        async def ok_handler(event, ctx):
            seen.append(f"{event.type}:{ctx.cwd}")

        def bad_handler(_event, _ctx):
            raise RuntimeError("boom")

        runner.set_extensions(
            [
                Extension(path="ext-a", handlers={"agent_start": [bad_handler]}),
                Extension(path="ext-b", handlers={"agent_start": [ok_handler]}),
            ]
        )

        from bampy.app.extension import AgentStartEvent

        await runner.emit(AgentStartEvent())
        ctx = runner.create_context()

        assert seen == ["agent_start:/repo"]
        assert runner.errors[0].extension_path == "ext-a"
        assert runner.errors[0].event == "agent_start"
        assert ctx.is_idle() is False
        assert ctx.has_pending_messages() is True
        assert ctx.get_system_prompt() == "base prompt"

    async def test_emit_context_chains_message_mutations(self):
        runner = self._runner_with_context()

        def first(event, _ctx):
            assert event.messages[0].content == "start"
            return ContextEventResult(messages=[UserMessage(content="after first")])

        def second(event, _ctx):
            assert event.messages[0].content == "after first"
            return ContextEventResult(messages=[UserMessage(content="after second")])

        runner.set_extensions(
            [
                Extension(path="ext-a", handlers={"context": [first]}),
                Extension(path="ext-b", handlers={"context": [second]}),
            ]
        )

        messages = await runner.emit_context([UserMessage(content="start")])

        assert [message.content for message in messages] == ["after second"]

    async def test_emit_before_agent_start_chains_prompt_updates(self):
        seen_prompts: list[str] = []
        runner = self._runner_with_context()

        def first(event, _ctx):
            seen_prompts.append(event.system_prompt)
            return BeforeAgentStartEventResult(system_prompt=event.system_prompt + " A")

        def second(event, _ctx):
            seen_prompts.append(event.system_prompt)
            return BeforeAgentStartEventResult(system_prompt=event.system_prompt + " B")

        runner.set_extensions(
            [
                Extension(path="ext-a", handlers={"before_agent_start": [first]}),
                Extension(path="ext-b", handlers={"before_agent_start": [second]}),
            ]
        )

        prompt = await runner.emit_before_agent_start(
            BeforeAgentStartEvent(prompt="hi", system_prompt="base")
        )

        assert seen_prompts == ["base", "base A"]
        assert prompt == "base A B"

    async def test_emit_tool_call_and_input_short_circuit(self):
        runner = self._runner_with_context()

        def allow(_event, _ctx):
            return ToolCallEventResult(block=False)

        def block(_event, _ctx):
            return ToolCallEventResult(block=True, reason="denied")

        def transform_input(_event, _ctx):
            return InputEventResult(action="transform", text="rewritten")

        runner.set_extensions(
            [
                Extension(path="ext-a", handlers={"tool_call": [allow]}),
                Extension(path="ext-b", handlers={"tool_call": [block], "input": [transform_input]}),
            ]
        )

        tool_result = await runner.emit_tool_call(
            ToolCallEvent(tool_call_id="call-1", tool_name="search", input={"q": "x"})
        )
        input_result = await runner.emit_input(InputEvent(text="hello"))

        assert tool_result is not None
        assert tool_result.block is True
        assert tool_result.reason == "denied"
        assert input_result is not None
        assert input_result.action == "transform"
        assert input_result.text == "rewritten"

    async def test_emit_tool_result_chains_mutations(self):
        runner = self._runner_with_context()
        seen_content: list[str] = []

        def first(event, _ctx):
            seen_content.extend(block.text for block in event.content)
            return ToolResultEventResult(
                content=event.content + [TextContent(text="first")],
                details={"source": "ext-a"},
            )

        def second(event, _ctx):
            seen_content.extend(block.text for block in event.content)
            return ToolResultEventResult(is_error=True)

        runner.set_extensions(
            [
                Extension(path="ext-a", handlers={"tool_result": [first]}),
                Extension(path="ext-b", handlers={"tool_result": [second]}),
            ]
        )

        result = await runner.emit_tool_result(
            ToolResultEvent(
                tool_call_id="call-2",
                tool_name="search",
                content=[TextContent(text="base")],
            )
        )

        assert result is not None
        assert seen_content == ["base", "base", "first"]
        assert [block.text for block in result.content] == ["base", "first"]
        assert result.details == {"source": "ext-a"}
        assert result.is_error is True

    async def test_loaded_extension_api_helpers_bind_to_runner_actions(self):
        runner = self._runner_with_context()
        api = ExtensionAPI("ext.py")

        def on_start(_event, _ctx):
            api.append_entry("meta", {"x": 1})
            api.send_message("note", "hello extension")
            api.send_user_message("queued user")

        api.on("session_start", on_start)
        runner.set_extensions([api._build_extension()])

        from bampy.app.extension import SessionStartEvent

        await runner.emit(SessionStartEvent())

        entries = runner.create_context().session_manager.get_entries()
        assert [entry.type for entry in entries] == ["custom", "custom_message", "message"]
        assert entries[0].custom_type == "meta"
        assert entries[1].custom_type == "note"
        assert entries[2].message["role"] == "user"
        assert entries[2].message["content"] == "queued user"

    async def test_get_all_registered_items_prefers_first_registration(self):
        async def run_tool(*_args) -> AgentToolResult:
            return AgentToolResult(content=[TextContent(text="ok")])

        runner = self._runner_with_context()
        runner.set_extensions(
            [
                Extension(
                    path="ext-a",
                    tools={
                        "dup": wrap_tool_definition("dup", "A", run_tool),
                    },
                    commands={
                        "hello": RegisteredCommand(name="hello", description="a"),
                    },
                ),
                Extension(
                    path="ext-b",
                    tools={
                        "dup": wrap_tool_definition("dup", "B", run_tool),
                    },
                    commands={
                        "hello": RegisteredCommand(name="hello", description="b"),
                    },
                ),
            ]
        )

        tools = runner.get_all_registered_tools()
        commands = runner.get_all_commands()

        assert len(tools) == 1
        assert tools[0].definition.label == "A"
        assert len(commands) == 1
        assert commands[0].description == "a"


def wrap_tool_definition(name: str, label: str, execute):
    from bampy.app.extension import RegisteredTool

    return RegisteredTool(
        definition=ToolDefinition(
            name=name,
            label=label,
            description=f"{label} tool",
            parameters={"type": "object"},
            execute=execute,
        ),
        extension_path="ext.py",
    )


class TestWrapRegisteredTool:
    async def test_wrap_registered_tool_passes_context_and_supports_sync_execute(self):
        seen: dict[str, object] = {}
        runner = ExtensionRunner()
        runner.set_session_manager(SessionManager.in_memory("/repo"))
        runner.set_model(
            Model(id="gpt-5", name="GPT-5", api="openai-responses", provider="openai")
        )

        def execute(
            tool_call_id: str,
            params: dict[str, object],
            cancellation: CancellationToken | None,
            on_update,
            ctx,
        ) -> AgentToolResult:
            seen["tool_call_id"] = tool_call_id
            seen["params"] = params
            seen["cwd"] = ctx.cwd
            seen["model"] = ctx.model.id if ctx.model else None
            seen["cancelled"] = cancellation is not None
            seen["on_update"] = on_update is not None
            return AgentToolResult(content=[TextContent(text="done")])

        wrapped = wrap_registered_tool(
            wrap_tool_definition("echo", "Echo", execute),
            runner,
        )

        result = await wrapped.execute(
            "call-3",
            {"value": 1},
            cancellation=CancellationToken(),
            on_update=lambda _result: None,
        )

        assert result.content[0].text == "done"
        assert seen == {
            "tool_call_id": "call-3",
            "params": {"value": 1},
            "cwd": "/repo",
            "model": "gpt-5",
            "cancelled": True,
            "on_update": True,
        }
