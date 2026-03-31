"""Tests for bampy.ai.providers.openai — unit + live integration tests.

Live tests require .env.dev at project root (or environment variables):
  GPT_API_KEY  — OpenAI API key
  GPT_BASE_URL — (optional) custom proxy base URL
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bampy.ai.models import get_model
from bampy.ai.types import (
    AssistantMessage,
    Context,
    OpenAIOptions,
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

_API_KEY = os.environ.get("GPT_API_KEY", "")
_BASE_URL = os.environ.get("GPT_BASE_URL", "")

# Default model for live tests
_TEST_MODEL = "gpt-5.4-mini"

live = pytest.mark.live
requires_live_api = pytest.mark.skipif(not _API_KEY, reason="GPT_API_KEY not set")


def _model(model_id: str = _TEST_MODEL) -> Model:
    m = get_model(model_id, provider="openai")
    assert m is not None, f"Model {model_id} not found in registry"
    if _BASE_URL:
        m = m.model_copy(update={"base_url": _BASE_URL})
    return m


def _opts(**kw) -> OpenAIOptions:
    headers = kw.pop("headers", None) or {}
    headers.setdefault("User-Agent", "bampy/1.0")
    return OpenAIOptions(api_key=_API_KEY, headers=headers, **kw)


# ---------------------------------------------------------------------------
# Unit tests — message / tool conversion (no API call)
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_user_text(self):
        from bampy.ai.providers.openai import _convert_messages

        ctx = Context(messages=[UserMessage(content="hello")])
        items = _convert_messages(ctx)
        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert items[0]["content"] == "hello"

    def test_system_prompt(self):
        from bampy.ai.providers.openai import _convert_messages

        ctx = Context(
            system_prompt="You are helpful.",
            messages=[UserMessage(content="hi")],
        )
        items = _convert_messages(ctx)
        assert items[0]["role"] == "developer"
        assert items[0]["content"] == "You are helpful."
        assert items[1]["role"] == "user"

    def test_assistant_text_and_tool_call(self):
        from bampy.ai.providers.openai import _convert_messages

        ctx = Context(messages=[
            AssistantMessage(content=[
                TextContent(text="Let me check."),
                ToolCall(id="call_1", name="search", arguments={"q": "test"}),
            ]),
        ])
        items = _convert_messages(ctx)
        # Text → assistant message, ToolCall → function_call item
        assert items[0]["role"] == "assistant"
        assert items[0]["content"] == "Let me check."
        assert items[1]["type"] == "function_call"
        assert items[1]["name"] == "search"

    def test_tool_result(self):
        from bampy.ai.providers.openai import _convert_messages

        ctx = Context(messages=[
            ToolResultMessage(
                tool_call_id="call_1", tool_name="search",
                content=[TextContent(text="result data")],
            ),
        ])
        items = _convert_messages(ctx)
        assert items[0]["type"] == "function_call_output"
        assert items[0]["output"] == "result data"

    def test_tool_result_with_image_uses_multimodal_output_when_enabled(self):
        from bampy.ai.providers.openai import _convert_messages

        ctx = Context(messages=[
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="read",
                content=[
                    TextContent(text="Read image file [image/png]"),
                    ImageContent(data="aGVsbG8=", mime_type="image/png"),
                ],
            ),
        ])

        items = _convert_messages(ctx, allow_tool_result_images=True)
        output = items[0]["output"]

        assert items[0]["type"] == "function_call_output"
        assert isinstance(output, list)
        assert output[0] == {
            "type": "input_text",
            "text": "Read image file [image/png]",
        }
        assert output[1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,aGVsbG8=",
        }

    def test_tool_result_with_image_falls_back_to_placeholder_when_disabled(self):
        from bampy.ai.providers.openai import _convert_messages

        ctx = Context(messages=[
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="read",
                content=[
                    TextContent(text="Read image file [image/png]"),
                    ImageContent(data="aGVsbG8=", mime_type="image/png"),
                ],
            ),
        ])

        items = _convert_messages(ctx, allow_tool_result_images=False)

        assert items[0]["type"] == "function_call_output"
        assert items[0]["output"] == "Read image file [image/png]\n[image]"

    def test_thinking_skipped_in_conversion(self):
        from bampy.ai.providers.openai import _convert_messages

        ctx = Context(messages=[
            AssistantMessage(content=[
                ThinkingContent(thinking="deep thought"),
                TextContent(text="The answer."),
            ]),
        ])
        items = _convert_messages(ctx)
        # Only text part, thinking skipped
        assert len(items) == 1
        assert items[0]["role"] == "assistant"
        assert items[0]["content"] == "The answer."

    def test_thinking_roundtrip_preserved_with_signature(self):
        from bampy.ai.providers.openai import _convert_messages

        ctx = Context(messages=[
            AssistantMessage(content=[
                ThinkingContent(
                    thinking="deep thought",
                    thinking_signature='{"type":"reasoning","id":"rs_123","summary":[{"type":"summary_text","text":"deep thought"}]}',
                ),
                TextContent(text="The answer."),
            ]),
        ])
        items = _convert_messages(ctx)
        assert items[0]["type"] == "reasoning"
        assert items[0]["id"] == "rs_123"
        assert items[1]["role"] == "assistant"
        assert items[1]["content"] == "The answer."

    def test_user_multimodal_content(self):
        from bampy.ai.providers.openai import _convert_messages

        ctx = Context(messages=[
            UserMessage(content=[
                TextContent(text="What is this?"),
                ImageContent(data="aGVsbG8=", mime_type="image/png"),
            ]),
        ])
        items = _convert_messages(ctx)
        content = items[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "input_text"
        assert content[1]["type"] == "input_image"


class TestToolConversion:
    def test_convert_tools(self):
        from bampy.ai.providers.openai import _convert_tools

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
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "get_weather"

    def test_convert_tools_none(self):
        from bampy.ai.providers.openai import _convert_tools

        assert _convert_tools(None) is None
        assert _convert_tools([]) is None


class TestOpenAIOptions:
    def test_default(self):
        opts = OpenAIOptions()
        assert opts.reasoning_effort is None
        assert opts.temperature is None

    def test_with_reasoning(self):
        opts = OpenAIOptions(reasoning_effort="xhigh", temperature=0.5)
        assert opts.reasoning_effort == "xhigh"
        assert opts.temperature == 0.5


class TestReasoningEffortResolution:
    def test_xhigh_supported_model(self):
        from bampy.ai.providers.openai import _resolve_reasoning_effort

        model = Model(id="gpt-5.4", name="GPT-5.4", api="openai-responses", provider="openai", reasoning=True)
        assert _resolve_reasoning_effort(model, ThinkingLevel.XHIGH) == "xhigh"

    def test_xhigh_clamped_for_unsupported_model(self):
        from bampy.ai.providers.openai import _resolve_reasoning_effort

        model = Model(id="o3", name="o3", api="openai-responses", provider="openai", reasoning=True)
        assert _resolve_reasoning_effort(model, ThinkingLevel.XHIGH) == "high"


class TestStreamEventHandling:
    def test_reasoning_item_stores_signature(self):
        from bampy.ai.providers.openai import _handle_stream_event
        from bampy.ai.stream import AssistantMessageEventStream

        output = AssistantMessage(
            api="openai-responses",
            provider="openai",
            model="gpt-5.4",
            content=[ThinkingContent(thinking="")],
        )
        stream = AssistantMessageEventStream()
        event = SimpleNamespace(
            type="response.output_item.done",
            output_index=0,
            item=SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(type="summary_text", text="deep thought")],
                model_dump=lambda exclude_none=True: {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [{"type": "summary_text", "text": "deep thought"}],
                    "encrypted_content": "opaque",
                },
            ),
        )

        _handle_stream_event(
            event,
            output,
            stream,
            output_to_content={0: 0},
            tool_json_bufs={},
        )

        thinking = output.content[0]
        assert isinstance(thinking, ThinkingContent)
        assert thinking.thinking == "deep thought"
        assert thinking.thinking_signature is not None


# ---------------------------------------------------------------------------
# Live integration tests — require GPT_API_KEY
# ---------------------------------------------------------------------------


@live
@requires_live_api
class TestLiveStreaming:
    """Integration tests that hit the real OpenAI API."""

    async def test_basic_text_stream(self):
        """Simple text completion with streaming events."""
        from bampy.ai.providers.openai import stream_openai

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Say exactly: Hello bampy")],
        )
        event_stream = stream_openai(model, ctx, _opts())
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
        from bampy.ai.providers.openai import stream_openai

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Count from 1 to 3")],
        )
        event_stream = stream_openai(model, ctx, _opts())

        event_types: list[str] = []
        async for event in event_stream:
            event_types.append(event.type)

        assert event_types[0] == "start"
        assert "text_start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types
        assert event_types[-1] == "done"

    async def test_stream_simple(self):
        """stream_simple_openai maps SimpleStreamOptions correctly."""
        from bampy.ai.providers.openai import stream_simple_openai

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Reply with one word: OK")],
        )
        opts = SimpleStreamOptions(
            api_key=_API_KEY,
            headers={"User-Agent": "bampy/1.0"},
            reasoning=ThinkingLevel.LOW,
        )
        event_stream = stream_simple_openai(model, ctx, opts)
        result = await event_stream.result()

        assert result.stop_reason == StopReason.STOP
        assert any(isinstance(b, TextContent) for b in result.content)

    async def test_system_prompt(self):
        """System prompt is forwarded correctly."""
        from bampy.ai.providers.openai import stream_openai

        model = _model()
        ctx = Context(
            system_prompt="You must always reply in uppercase only.",
            messages=[UserMessage(content="Say hello")],
        )
        event_stream = stream_openai(model, ctx, _opts())
        result = await event_stream.result()

        text = "".join(
            b.text for b in result.content if isinstance(b, TextContent)
        )
        upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text.replace(" ", "")), 1)
        assert upper_ratio > 0.5, f"Expected uppercase reply, got: {text!r}"


@live
@requires_live_api
class TestLiveReasoning:
    """Tests for reasoning/thinking support."""

    async def test_reasoning_with_effort(self):
        """Reasoning model should accept reasoning effort parameter."""
        from bampy.ai.providers.openai import stream_openai

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="What is 127 * 389? Think step by step.")],
        )
        opts = _opts(reasoning_effort="low")
        event_stream = stream_openai(model, ctx, opts)
        result = await event_stream.result()

        assert result.stop_reason == StopReason.STOP
        has_text = any(isinstance(b, TextContent) for b in result.content)
        assert has_text, "Should have text response"


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
        from bampy.ai.providers.openai import stream_openai

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="What is the weather in Tokyo?")],
            tools=[self._WEATHER_TOOL],
        )
        event_stream = stream_openai(model, ctx, _opts())
        result = await event_stream.result()

        assert result.stop_reason == StopReason.TOOL_USE
        tool_calls = [b for b in result.content if isinstance(b, ToolCall)]
        assert len(tool_calls) >= 1
        tc = tool_calls[0]
        assert tc.name == "get_weather"
        assert "location" in tc.arguments

    async def test_tool_call_events(self):
        """Verify toolcall_start and toolcall_end events are emitted."""
        from bampy.ai.providers.openai import stream_openai

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="What is the weather in Paris?")],
            tools=[self._WEATHER_TOOL],
        )
        event_stream = stream_openai(model, ctx, _opts())

        event_types: list[str] = []
        async for event in event_stream:
            event_types.append(event.type)

        assert "toolcall_start" in event_types
        assert "toolcall_end" in event_types

    async def test_multi_turn_with_tool_result(self):
        """Full tool-use loop: user -> model(tool_call) -> tool_result -> model(text)."""
        from bampy.ai.providers.openai import stream_openai

        model = _model()

        # Turn 1: user asks, model calls tool
        ctx1 = Context(
            messages=[UserMessage(content="What is the weather in London?")],
            tools=[self._WEATHER_TOOL],
        )
        result1 = await stream_openai(model, ctx1, _opts()).result()
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
        result2 = await stream_openai(model, ctx2, _opts()).result()

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
        from bampy.ai.providers.openai import stream_openai

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Hi")],
        )
        result = await stream_openai(model, ctx, _opts()).result()

        assert result.usage.input > 0
        assert result.usage.output > 0
        assert result.usage.total_tokens > 0

    async def test_cost_calculated(self):
        """Cost should be computed from usage and model pricing."""
        from bampy.ai.providers.openai import stream_openai

        model = _model()
        ctx = Context(
            messages=[UserMessage(content="Hi")],
        )
        result = await stream_openai(model, ctx, _opts()).result()

        assert result.usage.cost.total > 0
        assert result.usage.cost.input > 0
        assert result.usage.cost.output > 0


@live
@requires_live_api
class TestLiveErrorHandling:
    """Tests for error scenarios."""

    async def test_invalid_api_key(self):
        """Invalid API key should yield an error event."""
        from bampy.ai.providers.openai import stream_openai

        model = _model()
        ctx = Context(messages=[UserMessage(content="Hi")])
        opts = OpenAIOptions(api_key="invalid-key-12345", headers={"User-Agent": "bampy/1.0"})
        result = await stream_openai(model, ctx, opts).result()

        assert result.stop_reason == StopReason.ERROR
        assert result.error_message is not None

    async def test_missing_sdk_handled(self):
        """Module loads without requiring SDK at import time."""
        import bampy.ai.providers.openai as mod

        assert hasattr(mod, "stream_openai")
        assert hasattr(mod, "get_provider_entry")
