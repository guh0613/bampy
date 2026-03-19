"""Tests for bampy.ai.providers.gemini — unit + live integration tests.

Live tests require .env.dev at project root (or environment variables):
  GEMINI_API_KEY  — Google AI API key
  GEMINI_BASE_URL — (optional) custom proxy base URL
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bampy.ai.models import get_model
from bampy.ai.types import (
    AssistantMessage,
    Context,
    GeminiOptions,
    ImageContent,
    Model,
    SimpleStreamOptions,
    StopReason,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Load .env.dev if present
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.dev"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_BASE_URL = os.environ.get("GEMINI_BASE_URL", "")

# Default model for live tests
_TEST_MODEL = "gemini-3.1-flash-lite-preview"

live = pytest.mark.skipif(not _API_KEY, reason="GEMINI_API_KEY not set")


def _model(model_id: str = _TEST_MODEL) -> Model:
    m = get_model(model_id, provider="google")
    assert m is not None, f"Model {model_id} not found in registry"
    if _BASE_URL:
        m = m.model_copy(update={"base_url": _BASE_URL})
    return m


def _opts(**kw) -> GeminiOptions:
    return GeminiOptions(api_key=_API_KEY, **kw)


# ---------------------------------------------------------------------------
# Unit tests — message / tool conversion (no API call)
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_user_text(self):
        from bampy.ai.providers.gemini import _convert_messages

        ctx = Context(messages=[UserMessage(content="hello")])
        contents = _convert_messages(ctx)
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "hello"

    def test_assistant_text_and_tool_call(self):
        from bampy.ai.providers.gemini import _convert_messages

        ctx = Context(messages=[
            AssistantMessage(content=[
                TextContent(text="Let me check."),
                ToolCall(id="c1", name="search", arguments={"q": "test"}),
            ]),
        ])
        contents = _convert_messages(ctx)
        assert len(contents) == 1
        assert contents[0].role == "model"
        parts = contents[0].parts
        assert parts[0].text == "Let me check."
        assert parts[1].function_call.name == "search"

    def test_tool_result_grouped(self):
        from bampy.ai.providers.gemini import _convert_messages

        ctx = Context(messages=[
            ToolResultMessage(
                tool_call_id="c1", tool_name="a",
                content=[TextContent(text="res1")],
            ),
            ToolResultMessage(
                tool_call_id="c2", tool_name="b",
                content=[TextContent(text="res2")],
            ),
        ])
        contents = _convert_messages(ctx)
        # Should be grouped into one user Content
        assert len(contents) == 1
        assert len(contents[0].parts) == 2

    def test_thinking_skipped_in_conversion(self):
        from bampy.ai.providers.gemini import _convert_messages

        ctx = Context(messages=[
            AssistantMessage(content=[
                ThinkingContent(thinking="deep thought"),
                TextContent(text="The answer."),
            ]),
        ])
        contents = _convert_messages(ctx)
        parts = contents[0].parts
        # Only text part, thinking skipped
        assert len(parts) == 1
        assert parts[0].text == "The answer."

    def test_image_content(self):
        from bampy.ai.providers.gemini import _convert_messages

        ctx = Context(messages=[
            UserMessage(content=[
                TextContent(text="What is this?"),
                ImageContent(data="aGVsbG8=", mime_type="image/png"),
            ]),
        ])
        contents = _convert_messages(ctx)
        parts = contents[0].parts
        assert len(parts) == 2
        assert parts[1].inline_data.mime_type == "image/png"

    def test_tool_result_with_image_gemini3(self):
        """Gemini 3.x: ImageContent → FunctionResponse.parts with inline_data."""
        from bampy.ai.providers.gemini import _append_tool_result

        msg = ToolResultMessage(
            tool_call_id="c1",
            tool_name="screenshot",
            content=[
                TextContent(text="Captured."),
                ImageContent(data="aGVsbG8=", mime_type="image/png"),
            ],
        )
        contents: list = []
        _append_tool_result(contents, msg, multimodal=True)

        fr = contents[0].parts[0].function_response
        assert fr.name == "screenshot"
        assert fr.response == {"result": "Captured."}
        assert fr.parts is not None
        assert len(fr.parts) == 1
        assert fr.parts[0].inline_data is not None
        assert fr.parts[0].inline_data.mime_type == "image/png"

    def test_tool_result_with_image_gemini2_fallback(self):
        """Gemini 2.x: ImageContent → text placeholder '[image]', no parts."""
        from bampy.ai.providers.gemini import _append_tool_result

        msg = ToolResultMessage(
            tool_call_id="c1",
            tool_name="screenshot",
            content=[
                TextContent(text="Captured."),
                ImageContent(data="aGVsbG8=", mime_type="image/png"),
            ],
        )
        contents: list = []
        _append_tool_result(contents, msg, multimodal=False)

        fr = contents[0].parts[0].function_response
        assert fr.response == {"result": "Captured.\n[image]"}
        assert fr.parts is None

    def test_convert_messages_multimodal_by_model_id(self):
        """_convert_messages enables multimodal tool result for gemini-3.x."""
        from bampy.ai.providers.gemini import _convert_messages

        ctx = Context(messages=[
            ToolResultMessage(
                tool_call_id="c1", tool_name="shot",
                content=[
                    TextContent(text="ok"),
                    ImageContent(data="aGVsbG8=", mime_type="image/png"),
                ],
            ),
        ])
        # gemini-3 model → inline_data
        c3 = _convert_messages(ctx, model_id="gemini-3-flash-preview")
        fr3 = c3[0].parts[0].function_response
        assert fr3.parts is not None

        # gemini-2 model → fallback text
        c2 = _convert_messages(ctx, model_id="gemini-2.5-flash")
        fr2 = c2[0].parts[0].function_response
        assert fr2.parts is None
        assert "[image]" in fr2.response["result"]

    def test_tool_result_text_only_no_image_parts(self):
        """Text-only tool results should not have FunctionResponse.parts."""
        from bampy.ai.providers.gemini import _append_tool_result

        msg = ToolResultMessage(
            tool_call_id="c2",
            tool_name="info",
            content=[TextContent(text="Some info")],
        )
        contents: list = []
        _append_tool_result(contents, msg)

        fr = contents[0].parts[0].function_response
        assert fr.response == {"result": "Some info"}
        assert fr.parts is None


class TestToolConversion:
    def test_convert_tools(self):
        from bampy.ai.providers.gemini import _convert_tools

        tools = [
            Tool(
                name="get_weather",
                description="Get weather info",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ]
        result = _convert_tools(tools)
        assert result is not None
        assert len(result) == 1
        decls = result[0].function_declarations
        assert len(decls) == 1
        assert decls[0].name == "get_weather"

    def test_convert_tools_none(self):
        from bampy.ai.providers.gemini import _convert_tools

        assert _convert_tools(None) is None
        assert _convert_tools([]) is None


class TestGeminiOptions:
    def test_default(self):
        opts = GeminiOptions()
        assert opts.thinking_budget is None
        assert opts.temperature is None

    def test_with_budget(self):
        opts = GeminiOptions(thinking_budget=4096, temperature=0.5)
        assert opts.thinking_budget == 4096
        assert opts.temperature == 0.5


# ---------------------------------------------------------------------------
# Live integration tests — require GEMINI_API_KEY
# ---------------------------------------------------------------------------


@live
class TestLiveStreaming:
    """Integration tests that hit the real Gemini API."""

    async def test_basic_text_stream(self):
        """Simple text completion with streaming events."""
        from bampy.ai.providers.gemini import stream_gemini

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Say exactly: Hello bampy")],
        )
        event_stream = stream_gemini(model, ctx, _opts())
        result: AssistantMessage = await event_stream.result()

        assert result.stop_reason == StopReason.STOP
        assert result.usage.input > 0
        assert result.usage.output > 0
        text = "".join(
            b.text for b in result.content if isinstance(b, TextContent)
        )
        assert "hello" in text.lower() or "bampy" in text.lower()

    async def test_stream_events_order(self):
        """Verify event ordering: start → text_start → deltas → text_end → done."""
        from bampy.ai.providers.gemini import stream_gemini

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Count from 1 to 3")],
        )
        event_stream = stream_gemini(model, ctx, _opts())

        event_types: list[str] = []
        async for event in event_stream:
            event_types.append(event.type)

        assert event_types[0] == "start"
        assert "text_start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types
        assert event_types[-1] == "done"

    async def test_stream_simple(self):
        """stream_simple_gemini maps SimpleStreamOptions correctly."""
        from bampy.ai.providers.gemini import stream_simple_gemini

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Reply with one word: OK")],
        )
        opts = SimpleStreamOptions(
            api_key=_API_KEY,
            reasoning=ThinkingLevel.LOW,
        )
        event_stream = stream_simple_gemini(model, ctx, opts)
        result = await event_stream.result()

        assert result.stop_reason == StopReason.STOP
        assert any(isinstance(b, TextContent) for b in result.content)

    async def test_system_prompt(self):
        """System prompt is forwarded correctly."""
        from bampy.ai.providers.gemini import stream_gemini

        model = _model()
        ctx = Context(
            system_prompt="You must always reply in uppercase only.",
            messages=[UserMessage(content="Say hello")],
        )
        event_stream = stream_gemini(model, ctx, _opts())
        result = await event_stream.result()

        text = "".join(
            b.text for b in result.content if isinstance(b, TextContent)
        )
        # Most of the text should be uppercase
        upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text.replace(" ", "")), 1)
        assert upper_ratio > 0.5, f"Expected uppercase reply, got: {text!r}"


@live
class TestLiveThinking:
    """Tests for thinking/reasoning support (uses a reasoning-capable model)."""

    async def test_thinking_content_returned(self):
        """Reasoning model should return ThinkingContent when budget is set."""
        from bampy.ai.providers.gemini import stream_gemini

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="What is 127 * 389? Think step by step.")],
        )
        opts = _opts(thinking_budget=4096)
        event_stream = stream_gemini(model, ctx, opts)
        result = await event_stream.result()

        assert result.stop_reason == StopReason.STOP
        has_thinking = any(
            isinstance(b, ThinkingContent) for b in result.content
        )
        has_text = any(isinstance(b, TextContent) for b in result.content)
        assert has_text, "Should have text response"
        # Thinking content is expected but not guaranteed by all models
        if has_thinking:
            thinking_text = "".join(
                b.thinking for b in result.content if isinstance(b, ThinkingContent)
            )
            assert len(thinking_text) > 0

    async def test_thinking_events_emitted(self):
        """Verify thinking_start/delta/end events appear in stream."""
        from bampy.ai.providers.gemini import stream_gemini

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Solve: 2^10 + 3^5. Show reasoning.")],
        )
        opts = _opts(thinking_budget=4096)
        event_stream = stream_gemini(model, ctx, opts)

        event_types: list[str] = []
        async for event in event_stream:
            event_types.append(event.type)

        assert "text_start" in event_types
        assert "done" in event_types
        # Thinking events are expected but not guaranteed
        if "thinking_start" in event_types:
            assert "thinking_delta" in event_types
            assert "thinking_end" in event_types


@live
class TestLiveToolCalling:
    """Tests for function calling / tool use."""

    _WEATHER_TOOL = Tool(
        name="get_weather",
        description="Get the current weather for a location.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. Tokyo",
                },
            },
            "required": ["location"],
        },
    )

    async def test_tool_call_generated(self):
        """Model should generate a tool call when tools are provided."""
        from bampy.ai.providers.gemini import stream_gemini

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="What is the weather in Tokyo?")],
            tools=[self._WEATHER_TOOL],
        )
        event_stream = stream_gemini(model, ctx, _opts())
        result = await event_stream.result()

        assert result.stop_reason == StopReason.TOOL_USE
        tool_calls = [b for b in result.content if isinstance(b, ToolCall)]
        assert len(tool_calls) >= 1
        tc = tool_calls[0]
        assert tc.name == "get_weather"
        assert "location" in tc.arguments

    async def test_tool_call_events(self):
        """Verify toolcall_start and toolcall_end events are emitted."""
        from bampy.ai.providers.gemini import stream_gemini

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="What is the weather in Paris?")],
            tools=[self._WEATHER_TOOL],
        )
        event_stream = stream_gemini(model, ctx, _opts())

        event_types: list[str] = []
        async for event in event_stream:
            event_types.append(event.type)

        assert "toolcall_start" in event_types
        assert "toolcall_end" in event_types

    async def test_multi_turn_with_tool_result(self):
        """Full tool-use loop: user → model(tool_call) → tool_result → model(text)."""
        from bampy.ai.providers.gemini import stream_gemini

        model = _model()

        # Turn 1: user asks, model calls tool
        ctx1 = Context(
            messages=[UserMessage(content="What is the weather in London?")],
            tools=[self._WEATHER_TOOL],
        )
        result1 = await stream_gemini(model, ctx1, _opts()).result()
        assert result1.stop_reason == StopReason.TOOL_USE

        tool_calls = [b for b in result1.content if isinstance(b, ToolCall)]
        assert len(tool_calls) >= 1
        tc = tool_calls[0]

        # Turn 2: provide tool result, get final answer
        ctx2 = Context(
            messages=[
                UserMessage(content="What is the weather in London?"),
                result1,
                ToolResultMessage(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    content=[TextContent(text="Cloudy, 15°C, light rain")],
                ),
            ],
            tools=[self._WEATHER_TOOL],
        )
        result2 = await stream_gemini(model, ctx2, _opts()).result()

        assert result2.stop_reason == StopReason.STOP
        text = "".join(
            b.text for b in result2.content if isinstance(b, TextContent)
        )
        # Should mention the weather info we provided
        assert any(w in text.lower() for w in ["cloud", "rain", "15", "london"])


@live
class TestLiveUsage:
    """Tests for usage/token tracking."""

    async def test_usage_populated(self):
        """Usage metadata should have non-zero token counts."""
        from bampy.ai.providers.gemini import stream_gemini

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Hi")],
        )
        result = await stream_gemini(model, ctx, _opts()).result()

        assert result.usage.input > 0
        assert result.usage.output > 0
        assert result.usage.total_tokens > 0

    async def test_cost_calculated(self):
        """Cost should be computed from usage and model pricing."""
        from bampy.ai.providers.gemini import stream_gemini

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Hi")],
        )
        result = await stream_gemini(model, ctx, _opts()).result()

        assert result.usage.cost.total > 0
        assert result.usage.cost.input > 0
        assert result.usage.cost.output > 0


@live
class TestLiveErrorHandling:
    """Tests for error scenarios."""

    async def test_invalid_api_key(self):
        """Invalid API key should yield an error event."""
        from bampy.ai.providers.gemini import stream_gemini

        model = _model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        opts = GeminiOptions(api_key="invalid-key-12345")
        result = await stream_gemini(model, ctx, opts).result()

        assert result.stop_reason == StopReason.ERROR
        assert result.error_message is not None

    async def test_missing_sdk_handled(self):
        """Module loads without requiring SDK at import time."""
        import bampy.ai.providers.gemini as mod

        assert hasattr(mod, "stream_gemini")
        assert hasattr(mod, "get_provider_entry")
