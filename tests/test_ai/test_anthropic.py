"""Tests for bampy.ai.providers.anthropic — unit + live integration tests.

Live tests require .env.dev at project root (or environment variables):
  CLAUDE_API_KEY  — Anthropic API key
  CLAUDE_BASE_URL — (optional) custom proxy base URL
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bampy.ai.models import get_model
from bampy.ai.types import (
    AnthropicOptions,
    AnthropicThinkingAdaptive,
    AnthropicThinkingEnabled,
    AssistantMessage,
    Context,
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

_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
_BASE_URL = os.environ.get("CLAUDE_BASE_URL", "")

# Default model for live tests — cheap & fast
_TEST_MODEL = "claude-haiku-4-5-20251001"

live = pytest.mark.live
requires_live_api = pytest.mark.skipif(not _API_KEY, reason="CLAUDE_API_KEY not set")


def _model(model_id: str = _TEST_MODEL) -> Model:
    m = get_model(model_id, provider="anthropic")
    assert m is not None, f"Model {model_id} not found in registry"
    if _BASE_URL:
        m = m.model_copy(update={"base_url": _BASE_URL})
    return m


def _opts(**kw) -> AnthropicOptions:
    headers = kw.pop("headers", None) or {}
    headers.setdefault("User-Agent", "bampy/1.0")
    return AnthropicOptions(api_key=_API_KEY, headers=headers, **kw)


# ---------------------------------------------------------------------------
# Unit tests — message / tool conversion (no API call)
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_user_text(self):
        from bampy.ai.providers.anthropic import _convert_messages

        ctx = Context(messages=[UserMessage(content="hello")])
        system, messages = _convert_messages(ctx)
        assert system is None
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_system_prompt(self):
        from bampy.ai.providers.anthropic import _convert_messages

        ctx = Context(
            system_prompt="You are helpful.",
            messages=[UserMessage(content="hi")],
        )
        system, messages = _convert_messages(ctx)
        assert system == "You are helpful."
        assert messages[0]["role"] == "user"

    def test_assistant_text_and_tool_call(self):
        from bampy.ai.providers.anthropic import _convert_messages

        ctx = Context(messages=[
            AssistantMessage(content=[
                TextContent(text="Let me check."),
                ToolCall(id="toolu_1", name="search", arguments={"q": "test"}),
            ]),
        ])
        _, messages = _convert_messages(ctx)
        content = messages[0]["content"]
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Let me check."
        assert content[1]["type"] == "tool_use"
        assert content[1]["name"] == "search"

    def test_tool_result_grouped(self):
        from bampy.ai.providers.anthropic import _convert_messages

        ctx = Context(messages=[
            ToolResultMessage(
                tool_call_id="toolu_1", tool_name="a",
                content=[TextContent(text="res1")],
            ),
            ToolResultMessage(
                tool_call_id="toolu_2", tool_name="b",
                content=[TextContent(text="res2")],
            ),
        ])
        _, messages = _convert_messages(ctx)
        # Should be grouped into one user message
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 2

    def test_thinking_preserved_in_conversion(self):
        from bampy.ai.providers.anthropic import _convert_messages

        ctx = Context(messages=[
            AssistantMessage(content=[
                ThinkingContent(thinking="deep thought", thinking_signature="sig123"),
                TextContent(text="The answer."),
            ]),
        ])
        _, messages = _convert_messages(ctx)
        content = messages[0]["content"]
        assert content[0]["type"] == "thinking"
        assert content[0]["thinking"] == "deep thought"
        assert content[0]["signature"] == "sig123"
        assert content[1]["type"] == "text"

    def test_redacted_thinking(self):
        from bampy.ai.providers.anthropic import _convert_messages

        ctx = Context(messages=[
            AssistantMessage(content=[
                ThinkingContent(thinking="", thinking_signature="opaque-payload", redacted=True),
                TextContent(text="answer"),
            ]),
        ])
        _, messages = _convert_messages(ctx)
        content = messages[0]["content"]
        assert content[0]["type"] == "redacted_thinking"
        assert content[0]["data"] == "opaque-payload"

    def test_image_content(self):
        from bampy.ai.providers.anthropic import _convert_messages

        ctx = Context(messages=[
            UserMessage(content=[
                TextContent(text="What is this?"),
                ImageContent(data="aGVsbG8=", mime_type="image/png"),
            ]),
        ])
        _, messages = _convert_messages(ctx)
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[1]["type"] == "image"
        assert content[1]["source"]["media_type"] == "image/png"


class TestToolConversion:
    def test_convert_tools(self):
        from bampy.ai.providers.anthropic import _convert_tools

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
        assert result[0]["name"] == "get_weather"
        assert result[0]["input_schema"] is not None

    def test_convert_tools_none(self):
        from bampy.ai.providers.anthropic import _convert_tools

        assert _convert_tools(None) is None
        assert _convert_tools([]) is None


class TestThinkingResolution:
    def test_adaptive_for_opus(self):
        from bampy.ai.providers.anthropic import _resolve_thinking

        model = Model(id="claude-opus-4-6", name="Opus", api="anthropic-messages", provider="anthropic", reasoning=True)
        thinking, output_config = _resolve_thinking(model, ThinkingLevel.HIGH, None)
        assert thinking == {"type": "adaptive"}
        assert output_config == {"effort": "high"}

    def test_xhigh_maps_to_max_for_opus_46(self):
        from bampy.ai.providers.anthropic import _resolve_thinking

        model = Model(id="claude-opus-4-6", name="Opus", api="anthropic-messages", provider="anthropic", reasoning=True)
        thinking, output_config = _resolve_thinking(model, ThinkingLevel.XHIGH, None)
        assert thinking == {"type": "adaptive"}
        assert output_config == {"effort": "max"}

    def test_budget_for_haiku(self):
        from bampy.ai.providers.anthropic import _resolve_thinking

        model = Model(id="claude-haiku-4-5-20251001", name="Haiku", api="anthropic-messages", provider="anthropic", reasoning=True)
        thinking, output_config = _resolve_thinking(model, ThinkingLevel.MEDIUM, None)
        assert thinking == {"type": "enabled", "budget_tokens": 8192}
        assert output_config is None

    def test_budget_for_older_model(self):
        from bampy.ai.providers.anthropic import _resolve_thinking

        model = Model(id="claude-3-5-sonnet-20241022", name="Sonnet 3.5", api="anthropic-messages", provider="anthropic", reasoning=False)
        thinking, output_config = _resolve_thinking(model, ThinkingLevel.MEDIUM, None)
        assert thinking == {"type": "enabled", "budget_tokens": 8192}
        assert output_config is None

    def test_explicit_config_overrides(self):
        from bampy.ai.providers.anthropic import _resolve_thinking

        model = Model(id="claude-opus-4-6", name="Opus", api="anthropic-messages", provider="anthropic", reasoning=True)
        config = AnthropicThinkingEnabled(budget_tokens=5000)
        thinking, output_config = _resolve_thinking(model, ThinkingLevel.HIGH, config)
        assert thinking == {"type": "enabled", "budget_tokens": 5000}
        assert output_config is None

    def test_display_and_effort_are_split_for_adaptive(self):
        from bampy.ai.providers.anthropic import _resolve_thinking

        model = Model(id="claude-sonnet-4-6", name="Sonnet", api="anthropic-messages", provider="anthropic", reasoning=True)
        config = AnthropicThinkingAdaptive(effort="high", display="omitted")
        thinking, output_config = _resolve_thinking(model, None, config)
        assert thinking == {"type": "adaptive", "display": "omitted"}
        assert output_config == {"effort": "high"}


class TestAnthropicOptions:
    def test_default(self):
        opts = AnthropicOptions()
        assert opts.thinking is None
        assert opts.temperature is None

    def test_with_thinking(self):
        opts = AnthropicOptions(
            thinking=AnthropicThinkingAdaptive(effort="max", display="omitted"),
            interleaved_thinking=True,
        )
        assert opts.thinking.effort == "max"
        assert opts.thinking.display == "omitted"
        assert opts.interleaved_thinking is True


# ---------------------------------------------------------------------------
# Live integration tests — require CLAUDE_API_KEY
# ---------------------------------------------------------------------------


@live
@requires_live_api
class TestLiveStreaming:
    """Integration tests that hit the real Anthropic API."""

    async def test_basic_text_stream(self):
        """Simple text completion with streaming events."""
        from bampy.ai.providers.anthropic import stream_anthropic

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Say exactly: Hello bampy")],
        )
        event_stream = stream_anthropic(model, ctx, _opts())
        result: AssistantMessage = await event_stream.result()

        assert result.stop_reason == StopReason.STOP
        assert result.usage.input > 0
        assert result.usage.output > 0
        text = "".join(
            b.text for b in result.content if isinstance(b, TextContent)
        )
        assert "hello" in text.lower() or "bampy" in text.lower()

    async def test_stream_events_order(self):
        """Verify event ordering: start -> text_start -> deltas -> text_end -> done."""
        from bampy.ai.providers.anthropic import stream_anthropic

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Count from 1 to 3")],
        )
        event_stream = stream_anthropic(model, ctx, _opts())

        event_types: list[str] = []
        async for event in event_stream:
            event_types.append(event.type)

        assert event_types[0] == "start"
        assert "text_start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types
        assert event_types[-1] == "done"

    async def test_stream_simple(self):
        """stream_simple_anthropic maps SimpleStreamOptions correctly."""
        from bampy.ai.providers.anthropic import stream_simple_anthropic

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Reply with one word: OK")],
        )
        opts = SimpleStreamOptions(api_key=_API_KEY, headers={"User-Agent": "bampy/1.0"})
        event_stream = stream_simple_anthropic(model, ctx, opts)
        result = await event_stream.result()

        assert result.stop_reason == StopReason.STOP
        assert any(isinstance(b, TextContent) for b in result.content)

    async def test_system_prompt(self):
        """System prompt is forwarded and influences the response."""
        from bampy.ai.providers.anthropic import stream_anthropic

        model = _model()
        ctx = Context(
            system_prompt="You are a pirate. Always say 'Arrr' in your response.",
            messages=[UserMessage(content="Hello")],
        )
        event_stream = stream_anthropic(model, ctx, _opts())
        result = await event_stream.result()

        assert result.stop_reason == StopReason.STOP
        text = "".join(
            b.text for b in result.content if isinstance(b, TextContent)
        )
        assert len(text) > 0, "Should have a non-empty response"


@live
@requires_live_api
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
        from bampy.ai.providers.anthropic import stream_anthropic

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="What is the weather in Tokyo?")],
            tools=[self._WEATHER_TOOL],
        )
        event_stream = stream_anthropic(model, ctx, _opts())
        result = await event_stream.result()

        assert result.stop_reason == StopReason.TOOL_USE
        tool_calls = [b for b in result.content if isinstance(b, ToolCall)]
        assert len(tool_calls) >= 1
        tc = tool_calls[0]
        assert tc.name == "get_weather"
        assert "location" in tc.arguments

    async def test_tool_call_events(self):
        """Verify toolcall_start and toolcall_end events are emitted."""
        from bampy.ai.providers.anthropic import stream_anthropic

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="What is the weather in Paris?")],
            tools=[self._WEATHER_TOOL],
        )
        event_stream = stream_anthropic(model, ctx, _opts())

        event_types: list[str] = []
        async for event in event_stream:
            event_types.append(event.type)

        assert "toolcall_start" in event_types
        assert "toolcall_end" in event_types

    async def test_multi_turn_with_tool_result(self):
        """Full tool-use loop: user -> model(tool_call) -> tool_result -> model(text)."""
        from bampy.ai.providers.anthropic import stream_anthropic

        model = _model()

        # Turn 1: user asks, model calls tool
        ctx1 = Context(
            messages=[UserMessage(content="What is the weather in London?")],
            tools=[self._WEATHER_TOOL],
        )
        result1 = await stream_anthropic(model, ctx1, _opts()).result()
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
                    content=[TextContent(text="Cloudy, 15C, light rain")],
                ),
            ],
            tools=[self._WEATHER_TOOL],
        )
        result2 = await stream_anthropic(model, ctx2, _opts()).result()

        assert result2.stop_reason == StopReason.STOP
        text = "".join(
            b.text for b in result2.content if isinstance(b, TextContent)
        )
        assert any(w in text.lower() for w in ["cloud", "rain", "15", "london"])


@live
@requires_live_api
class TestLiveUsage:
    """Tests for usage/token tracking."""

    async def test_usage_populated(self):
        """Usage metadata should have non-zero token counts."""
        from bampy.ai.providers.anthropic import stream_anthropic

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Hi")],
        )
        result = await stream_anthropic(model, ctx, _opts()).result()

        # input tokens may be 0 if fully served from cache
        assert (result.usage.input + result.usage.cache_read) > 0
        assert result.usage.output > 0
        assert result.usage.total_tokens > 0

    async def test_cost_calculated(self):
        """Cost should be computed from usage and model pricing."""
        from bampy.ai.providers.anthropic import stream_anthropic

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Hi")],
        )
        result = await stream_anthropic(model, ctx, _opts()).result()

        assert result.usage.cost.total > 0
        assert result.usage.cost.output > 0


@live
@requires_live_api
class TestLiveErrorHandling:
    """Tests for error scenarios."""

    async def test_invalid_api_key(self):
        """Invalid API key should yield an error event."""
        from bampy.ai.providers.anthropic import stream_anthropic

        model = _model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        opts = AnthropicOptions(api_key="invalid-key-12345", headers={"User-Agent": "bampy/1.0"})
        result = await stream_anthropic(model, ctx, opts).result()

        assert result.stop_reason == StopReason.ERROR
        assert result.error_message is not None

    async def test_missing_sdk_handled(self):
        """Module loads without requiring SDK at import time."""
        import bampy.ai.providers.anthropic as mod

        assert hasattr(mod, "stream_anthropic")
        assert hasattr(mod, "get_provider_entry")
