"""Tests for bampy.app.tools."""

from __future__ import annotations

from pydantic import BaseModel

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult
from bampy.ai.types import TextContent
from bampy.app.tools import ToolFromFunction, tool


class TestToolFromFunction:
    async def test_execute_injects_special_arguments_and_wraps_string_result(self):
        seen: dict[str, object] = {}
        updates: list[str] = []
        cancellation = CancellationToken()

        async def greet(
            name: str,
            tool_call_id: str,
            cancellation: CancellationToken | None = None,
            on_update=None,
        ) -> str:
            seen["tool_call_id"] = tool_call_id
            seen["cancelled"] = cancellation is not None
            await on_update(AgentToolResult(content=[TextContent(text=f"partial:{name}")]))
            return f"hello {name}"

        async def on_update(result: AgentToolResult) -> None:
            updates.append(result.content[0].text)

        wrapped = ToolFromFunction(greet, description="Greets a user")
        result = await wrapped.execute(
            "call-1",
            {"name": "Ada"},
            cancellation=cancellation,
            on_update=on_update,
        )

        assert wrapped.description == "Greets a user"
        assert result.content[0].text == "hello Ada"
        assert updates == ["partial:Ada"]
        assert seen == {"tool_call_id": "call-1", "cancelled": True}

    async def test_execute_accepts_pydantic_params_and_list_results(self):
        class Params(BaseModel):
            value: str

        async def blocks(value: str) -> list[TextContent]:
            return [TextContent(text=value), TextContent(text=value.upper())]

        wrapped = ToolFromFunction(blocks)
        result = await wrapped.execute("call-2", Params(value="abc"))

        assert [block.text for block in result.content] == ["abc", "ABC"]

    async def test_execute_returns_agent_tool_result_unchanged(self):
        expected = AgentToolResult(content=[TextContent(text="ready")], details={"ok": True})

        async def direct() -> AgentToolResult:
            return expected

        wrapped = ToolFromFunction(direct)
        result = await wrapped.execute("call-3", {})

        assert result is expected

    def test_generated_parameter_model_skips_internal_arguments(self):
        async def sample(name: str, tool_call_id: str, cancellation=None, on_update=None) -> str:
            return name

        wrapped = ToolFromFunction(sample)

        assert set(wrapped.parameters.model_fields) == {"name"}

    def test_generated_parameter_model_keeps_non_injected_ctx_argument(self):
        async def sample(name: str, ctx: str) -> str:
            return f"{name}:{ctx}"

        wrapped = ToolFromFunction(sample)

        assert set(wrapped.parameters.model_fields) == {"name", "ctx"}

    async def test_execute_rejects_invalid_content_blocks(self):
        async def bad_blocks() -> list[int]:
            return [1, 2, 3]

        wrapped = ToolFromFunction(bad_blocks)

        try:
            await wrapped.execute("call-4", {})
        except TypeError as exc:
            assert "Unsupported tool content block" in str(exc)
        else:
            raise AssertionError("Expected execute() to reject invalid tool content")


class TestToolDecorator:
    async def test_tool_decorator_preserves_metadata_and_supports_sync_functions(self):
        @tool(name="sum_values", label="Sum", description="Add two integers")
        def sum_values(a: int, b: int) -> str:
            return str(a + b)

        result = await sum_values.execute("call-4", {"a": 2, "b": 3})

        assert sum_values.name == "sum_values"
        assert sum_values.label == "Sum"
        assert sum_values.description == "Add two integers"
        assert result.content[0].text == "5"
