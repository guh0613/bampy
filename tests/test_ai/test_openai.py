"""Tests for bampy.ai.providers.openai — unit + live integration tests.

Live tests require .env.dev at project root (or environment variables):
  GPT_API_KEY  — OpenAI API key
  GPT_BASE_URL — (optional) custom proxy base URL
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bampy.ai.models import get_model
from bampy.ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    OpenAIChatCompat,
    OpenAIOptions,
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
_CHAT_COMPLETIONS_LIVE_MODEL = "minimax-m2.5-free"

# Default model for live tests
_TEST_MODEL = "gpt-5.4-mini"

live = pytest.mark.live
requires_live_api = pytest.mark.skipif(not _API_KEY, reason="GPT_API_KEY not set")
requires_chat_completions_live_api = pytest.mark.skipif(
    not (_API_KEY and _BASE_URL),
    reason="GPT_API_KEY or GPT_BASE_URL not set for chat-completions live test",
)


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


def _chat_model(*, reasoning: bool = False, input_types: list[str] | None = None) -> Model:
    return Model(
        id="chat-model",
        name="Chat Model",
        api="openai-completions",
        provider="openai",
        reasoning=reasoning,
        input_types=input_types or ["text", "image"],
        base_url="https://api.openai.com/v1",
        max_tokens=16384,
    )


def _kimi_chat_model() -> Model:
    return Model(
        id="kimi-k2.6",
        name="Kimi K2.6",
        api="openai-completions",
        provider="opencode-go",
        reasoning=True,
        input_types=["text", "image"],
        base_url="https://opencode.ai/zen/go/v1",
        max_tokens=65536,
        openai_chat_compat=OpenAIChatCompat(
            max_tokens_field="max_tokens",
            replay_thinking_field="reasoning_content",
            stream_reasoning_fields=["reasoning_content"],
            supports_reasoning_effort=False,
            thinking_param="kimi",
            thinking_default_enabled=True,
            thinking_tool_choice=["auto", "none"],
        ),
    )


def _responses_model(*, model_id: str = "gpt-5.4", reasoning: bool = True) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-responses",
        provider="openai",
        reasoning=reasoning,
        input_types=["text", "image"],
        base_url="https://api.openai.com/v1",
        max_tokens=16384,
    )


def _live_chat_completions_model(model_id: str = _CHAT_COMPLETIONS_LIVE_MODEL) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider="openai",
        reasoning=False,
        input_types=["text", "image"],
        base_url=_BASE_URL or "https://api.openai.com/v1",
        max_tokens=16384,
    )


# ---------------------------------------------------------------------------
# Unit tests — message / tool conversion (no API call)
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_user_text(self):
        from bampy.ai.providers.openai import _convert_messages

        model = _responses_model()
        ctx = Context(messages=[UserMessage(content="hello")])
        items = _convert_messages(model, ctx)
        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert items[0]["content"] == "hello"

    def test_system_prompt(self):
        from bampy.ai.providers.openai import _convert_messages

        model = _responses_model()
        ctx = Context(
            system_prompt="You are helpful.",
            messages=[UserMessage(content="hi")],
        )
        items = _convert_messages(model, ctx)
        assert items[0]["role"] == "developer"
        assert items[0]["content"] == "You are helpful."
        assert items[1]["role"] == "user"

    def test_assistant_text_and_tool_call(self):
        from bampy.ai.providers.openai import _convert_messages

        model = _responses_model()
        ctx = Context(messages=[
            AssistantMessage(content=[
                TextContent(text="Let me check."),
                ToolCall(id="call_1", name="search", arguments={"q": "test"}),
            ]),
        ])
        items = _convert_messages(model, ctx)
        # Text → assistant message, ToolCall → function_call item
        assert items[0]["role"] == "assistant"
        assert items[0]["content"] == "Let me check."
        assert items[1]["type"] == "function_call"
        assert items[1]["name"] == "search"

    def test_tool_result(self):
        from bampy.ai.providers.openai import _convert_messages

        model = _responses_model()
        ctx = Context(messages=[
            ToolResultMessage(
                tool_call_id="call_1", tool_name="search",
                content=[TextContent(text="result data")],
            ),
        ])
        items = _convert_messages(model, ctx)
        assert items[0]["type"] == "function_call_output"
        assert items[0]["output"] == "result data"

    def test_tool_result_with_image_uses_multimodal_output_when_enabled(self):
        from bampy.ai.providers.openai import _convert_messages

        model = _responses_model()
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

        items = _convert_messages(model, ctx, allow_tool_result_images=True)
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

        model = _responses_model()
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

        items = _convert_messages(model, ctx, allow_tool_result_images=False)

        assert items[0]["type"] == "function_call_output"
        assert items[0]["output"] == "Read image file [image/png]\n[image]"

    def test_thinking_skipped_in_conversion(self):
        from bampy.ai.providers.openai import _convert_messages

        model = _responses_model()
        ctx = Context(messages=[
            AssistantMessage(
                api="openai-responses",
                provider="openai",
                model=model.id,
                content=[
                    ThinkingContent(thinking="deep thought"),
                    TextContent(text="The answer."),
                ],
            ),
        ])
        items = _convert_messages(model, ctx)
        # Only text part, thinking skipped
        assert len(items) == 1
        assert items[0]["role"] == "assistant"
        assert items[0]["content"] == "The answer."

    def test_thinking_roundtrip_preserved_with_signature(self):
        from bampy.ai.providers.openai import _convert_messages

        model = _responses_model()
        ctx = Context(messages=[
            AssistantMessage(
                api="openai-responses",
                provider="openai",
                model=model.id,
                content=[
                    ThinkingContent(
                        thinking="deep thought",
                        thinking_signature='{"type":"reasoning","id":"rs_123","summary":[{"type":"summary_text","text":"deep thought"}]}',
                    ),
                    TextContent(text="The answer."),
                ],
            ),
        ])
        items = _convert_messages(model, ctx)
        assert items[0]["type"] == "reasoning"
        assert items[0]["id"] == "rs_123"
        assert items[1]["role"] == "assistant"
        assert items[1]["content"] == "The answer."

    def test_user_multimodal_content(self):
        from bampy.ai.providers.openai import _convert_messages

        model = _responses_model()
        ctx = Context(messages=[
            UserMessage(content=[
                TextContent(text="What is this?"),
                ImageContent(data="aGVsbG8=", mime_type="image/png"),
            ]),
        ])
        items = _convert_messages(model, ctx)
        content = items[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "input_text"
        assert content[1]["type"] == "input_image"

    def test_different_model_handoff_omits_responses_item_id(self):
        from bampy.ai.providers.openai import _convert_messages

        model = _responses_model(model_id="gpt-5.2-codex")
        ctx = Context(messages=[
            AssistantMessage(
                api="openai-responses",
                provider="openai",
                model="gpt-5-mini",
                content=[
                    TextContent(text="Let me check."),
                    ToolCall(id="call_1|fc_1", name="search", arguments={"q": "test"}),
                ],
            ),
        ])

        items = _convert_messages(model, ctx)

        assert items[0]["role"] == "assistant"
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "call_1"
        assert "id" not in items[1]


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


class TestChatCompletionMessageConversion:
    def test_system_prompt_role_depends_on_model_capability(self):
        from bampy.ai.providers.openai import _convert_chat_completion_messages

        ctx = Context(system_prompt="Be helpful.", messages=[UserMessage(content="hi")])

        reasoning_items = _convert_chat_completion_messages(_chat_model(reasoning=True), ctx)
        non_reasoning_items = _convert_chat_completion_messages(_chat_model(reasoning=False), ctx)

        assert reasoning_items[0]["role"] == "developer"
        assert non_reasoning_items[0]["role"] == "system"

    def test_assistant_tool_calls_and_tool_results_are_converted(self):
        from bampy.ai.providers.openai import _convert_chat_completion_messages

        model = Model(
            id="gpt-5.4",
            name="GPT-5.4",
            api="openai-completions",
            provider="openai",
            reasoning=True,
        )
        assistant = AssistantMessage(
            api="openai-completions",
            provider="openai",
            model="gpt-5.4",
            content=[
                TextContent(text="Let me check."),
                ToolCall(id="call_1|fc_1", name="search", arguments={"q": "test"}),
            ],
        )
        ctx = Context(messages=[
            assistant,
            ToolResultMessage(
                tool_call_id="call_1|fc_1",
                tool_name="search",
                content=[TextContent(text="result data")],
            ),
        ])

        items = _convert_chat_completion_messages(model, ctx)

        assert items[0]["role"] == "assistant"
        assert items[0]["content"] == "Let me check."
        assert items[0]["tool_calls"][0]["id"] == "call_1"
        assert items[0]["tool_calls"][0]["function"]["name"] == "search"
        assert items[1]["role"] == "tool"
        assert items[1]["tool_call_id"] == "call_1"
        assert items[1]["content"] == "result data"

    def test_tool_result_images_are_forwarded_as_user_content(self):
        from bampy.ai.providers.openai import _convert_chat_completion_messages

        ctx = Context(messages=[
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="vision_tool",
                content=[
                    TextContent(text="screenshot attached"),
                    ImageContent(data="aGVsbG8=", mime_type="image/png"),
                ],
            ),
        ])

        items = _convert_chat_completion_messages(_chat_model(), ctx)

        assert items[0]["role"] == "tool"
        assert items[0]["content"] == "screenshot attached"
        assert items[1]["role"] == "user"
        assert items[1]["content"][0]["type"] == "text"
        assert items[1]["content"][1]["type"] == "image_url"

    def test_reasoning_signature_roundtrips_for_chat_completions(self):
        from bampy.ai.providers.openai import _convert_chat_completion_messages

        assistant = AssistantMessage(
            api="openai-completions",
            provider="openai",
            model="gpt-5.4",
            content=[
                ThinkingContent(
                    thinking="deep thought",
                    thinking_signature="reasoning_content",
                ),
                TextContent(text="final answer"),
            ],
        )
        ctx = Context(messages=[assistant])
        items = _convert_chat_completion_messages(
            Model(
                id="gpt-5.4",
                name="GPT-5.4",
                api="openai-completions",
                provider="openai",
                reasoning=True,
            ),
            ctx,
        )

        assert items[0]["role"] == "assistant"
        assert items[0]["content"] == "final answer"
        assert items[0]["reasoning_content"] == "deep thought"

    def test_kimi_reasoning_replay_falls_back_to_reasoning_content(self):
        from bampy.ai.providers.openai import _convert_chat_completion_messages

        assistant = AssistantMessage(
            api="openai-completions",
            provider="opencode-go",
            model="kimi-k2.6",
            content=[
                ThinkingContent(thinking="deep thought"),
                TextContent(text="final answer"),
            ],
        )
        ctx = Context(messages=[assistant])
        items = _convert_chat_completion_messages(_kimi_chat_model(), ctx)

        assert items[0]["role"] == "assistant"
        assert items[0]["content"] == "final answer"
        assert items[0]["reasoning_content"] == "deep thought"


class TestOpenAIOptions:
    def test_default(self):
        opts = OpenAIOptions()
        assert opts.reasoning_effort is None
        assert opts.temperature is None

    def test_with_reasoning(self):
        opts = OpenAIOptions(
            reasoning_effort="xhigh",
            temperature=0.5,
            tool_choice="required",
        )
        assert opts.reasoning_effort == "xhigh"
        assert opts.temperature == 0.5
        assert opts.tool_choice == "required"

    def test_client_sets_default_user_agent(self, monkeypatch):
        from bampy.ai.providers.openai import _create_openai_client

        captured: dict[str, object] = {}

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        model = _chat_model()
        _create_openai_client(
            SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
            model,
            OpenAIOptions(api_key="test-key"),
        )

        assert captured["default_headers"]["User-Agent"] == "bampy/1.0"


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


class _FakeAsyncIterator:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class TestChatCompletionStreaming:
    def test_build_chat_completion_params_for_kimi_uses_compat_fields(self):
        from bampy.ai.providers.openai import _build_chat_completion_params

        model = _kimi_chat_model()
        params = _build_chat_completion_params(
            model,
            Context(messages=[UserMessage(content="Hello")]),
            OpenAIOptions(api_key="test-key", max_tokens=321, reasoning_effort="high"),
        )

        assert params["max_tokens"] == 321
        assert "max_completion_tokens" not in params
        assert params["extra_body"] == {"thinking": {"type": "enabled"}}
        assert "thinking" not in params
        assert "reasoning_effort" not in params

    def test_build_chat_completion_params_rejects_invalid_tool_choice_for_kimi(self):
        from bampy.ai.providers.openai import _build_chat_completion_params

        model = _kimi_chat_model()
        ctx = Context(
            messages=[UserMessage(content="Check the weather")],
            tools=[
                Tool(
                    name="get_weather",
                    description="Get weather info",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ],
        )

        with pytest.raises(ValueError, match="tool_choice"):
            _build_chat_completion_params(
                model,
                ctx,
                OpenAIOptions(api_key="test-key", tool_choice="required"),
            )

    @pytest.mark.asyncio
    async def test_stream_openai_completions_text_and_usage(self, monkeypatch):
        from bampy.ai.providers.openai import stream_openai_completions

        captured: dict[str, object] = {}
        chunks = [
            SimpleNamespace(
                id="chatcmpl_1",
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(content="Hello", refusal=None, tool_calls=None),
                    ),
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="chatcmpl_1",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        delta=SimpleNamespace(content=None, refusal=None, tool_calls=None),
                    ),
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="chatcmpl_1",
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=4,
                    total_tokens=14,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                ),
            ),
        ]

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create),
                )

            async def _create(self, **params):
                captured["params"] = params
                return _FakeAsyncIterator(chunks)

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

        result = await stream_openai_completions(
            _chat_model(),
            Context(messages=[UserMessage(content="Say hello")]),
            OpenAIOptions(api_key="test-key", tool_choice="auto"),
        ).result()

        assert result.stop_reason == StopReason.STOP
        assert result.response_id == "chatcmpl_1"
        assert result.usage.input == 8
        assert result.usage.output == 4
        assert result.usage.cache_read == 2
        assert result.usage.total_tokens == 14
        assert [block.text for block in result.content if isinstance(block, TextContent)] == ["Hello"]
        assert captured["params"]["stream_options"] == {"include_usage": True}
        assert captured["params"]["tool_choice"] == "auto"
        assert captured["params"]["max_completion_tokens"] == 16384

    @pytest.mark.asyncio
    async def test_stream_openai_completions_forwards_explicit_max_tokens(self, monkeypatch):
        from bampy.ai.providers.openai import stream_openai_completions

        captured: dict[str, object] = {}
        chunks = [
            SimpleNamespace(
                id="chatcmpl_2",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        delta=SimpleNamespace(content="ok", refusal=None, tool_calls=None),
                    ),
                ],
                usage=SimpleNamespace(
                    prompt_tokens=2,
                    completion_tokens=1,
                    total_tokens=3,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                ),
            ),
        ]

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create),
                )

            async def _create(self, **params):
                captured["params"] = params
                return _FakeAsyncIterator(chunks)

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

        await stream_openai_completions(
            _chat_model(),
            Context(messages=[UserMessage(content="hi")]),
            OpenAIOptions(api_key="test-key", max_tokens=123),
        ).result()

        assert captured["params"]["max_completion_tokens"] == 123

    @pytest.mark.asyncio
    async def test_stream_openai_completions_tool_calls(self, monkeypatch):
        from bampy.ai.providers.openai import stream_openai_completions

        chunks = [
            SimpleNamespace(
                id="chatcmpl_tool",
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content=None,
                            refusal=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    function=SimpleNamespace(name="search", arguments='{"q":"te'),
                                ),
                            ],
                        ),
                    ),
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="chatcmpl_tool",
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        delta=SimpleNamespace(
                            content=None,
                            refusal=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    function=SimpleNamespace(name=None, arguments='st"}'),
                                ),
                            ],
                        ),
                    ),
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="chatcmpl_tool",
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=7,
                    completion_tokens=3,
                    total_tokens=10,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                ),
            ),
        ]

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create),
                )

            async def _create(self, **params):
                return _FakeAsyncIterator(chunks)

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

        result = await stream_openai_completions(
            _chat_model(),
            Context(messages=[UserMessage(content="Search for test")]),
            OpenAIOptions(api_key="test-key"),
        ).result()

        tool_calls = [block for block in result.content if isinstance(block, ToolCall)]
        assert result.stop_reason == StopReason.TOOL_USE
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].name == "search"
        assert tool_calls[0].arguments == {"q": "test"}


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
@requires_chat_completions_live_api
class TestLiveChatCompletions:
    """Live tests for the OpenAI Chat Completions adapter."""

    async def test_basic_text_stream(self):
        """Basic streamed text generation through the chat-completions path."""
        from bampy.ai.providers.openai import stream_openai_completions

        model = _live_chat_completions_model()
        ctx = Context(messages=[UserMessage(content="Reply with exactly: hello from minimax")])
        result = await stream_openai_completions(model, ctx, _opts()).result()

        assert result.stop_reason == StopReason.STOP
        assert result.usage.input > 0
        assert result.usage.output > 0
        text = "".join(
            block.text for block in result.content if isinstance(block, TextContent)
        )
        assert text.strip(), "Expected non-empty text response"
        assert "hello" in text.lower() or "minimax" in text.lower()


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
        """Full reasoning tool-use loop preserves OpenAI reasoning replay across turns."""
        from bampy.ai.providers.openai import stream_openai

        model = _model()
        opts = _opts(reasoning_effort="high")

        # Turn 1: user asks, model calls tool
        ctx1 = Context(
            system_prompt="Think carefully before using tools, and always use the tool when asked about weather.",
            messages=[UserMessage(content="What is the weather in London? Use the tool and reason briefly before acting.")],
            tools=[self._WEATHER_TOOL],
        )
        result1 = await stream_openai(model, ctx1, opts).result()
        assert result1.stop_reason == StopReason.TOOL_USE

        thinking_blocks = [b for b in result1.content if isinstance(b, ThinkingContent)]
        assert thinking_blocks, "Expected reasoning content before the tool call"
        assert any(b.thinking_signature for b in thinking_blocks), "Expected replayable reasoning signature"

        tool_calls = [b for b in result1.content if isinstance(b, ToolCall)]
        assert len(tool_calls) >= 1
        tc = tool_calls[0]

        # Turn 2: provide tool result, get final answer
        ctx2 = Context(
            system_prompt=ctx1.system_prompt,
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
        result2 = await stream_openai(model, ctx2, opts).result()

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
