"""Tests for bampy.ai.providers.ollama."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from bampy.ai.types import AssistantMessage, Context, Model, StopReason, TextContent, ThinkingContent, UserMessage


class TestStreamEventHandling:
    def test_normalize_stream_delta_handles_overlapping_chunks(self):
        from bampy.ai.providers.ollama import _normalize_stream_delta

        assert _normalize_stream_delta("", "这个实验") == "这个实验"
        assert _normalize_stream_delta("这个实验", "这个") == ""
        assert _normalize_stream_delta("这个实验", "实验") == ""
        assert _normalize_stream_delta("这个实验", "要求看起来") == "要求看起来"
        assert _normalize_stream_delta("abc", "bcd") == "d"
        assert _normalize_stream_delta("hello", "hello world") == " world"

    def test_text_delta_deduplicates_overlapping_gateway_chunks(self):
        from bampy.ai.providers.ollama import _handle_stream_event
        from bampy.ai.stream import AssistantMessageEventStream

        output = AssistantMessage(
            api="ollama-responses",
            provider="ollama",
            model="kimi-k2.5",
            content=[TextContent(text="")],
        )
        stream = AssistantMessageEventStream()
        output_to_content = {0: 0}

        for delta in ["这个实验", "这个", "实验", "要求看起来比较简略"]:
            _handle_stream_event(
                SimpleNamespace(
                    type="response.output_text.delta",
                    output_index=0,
                    delta=delta,
                ),
                output,
                stream,
                output_to_content=output_to_content,
                tool_json_bufs={},
            )

        assert output.content[0].text == "这个实验要求看起来比较简略"

    def test_reasoning_delta_deduplicates_overlapping_gateway_chunks(self):
        from bampy.ai.providers.ollama import _handle_stream_event
        from bampy.ai.stream import AssistantMessageEventStream

        output = AssistantMessage(
            api="ollama-responses",
            provider="ollama",
            model="kimi-k2.5",
            content=[ThinkingContent(thinking="")],
        )
        stream = AssistantMessageEventStream()
        output_to_content = {0: 0}

        for delta in ["让我先", "让我", "先看看"]:
            _handle_stream_event(
                SimpleNamespace(
                    type="response.reasoning_summary_text.delta",
                    output_index=0,
                    delta=delta,
                ),
                output,
                stream,
                output_to_content=output_to_content,
                tool_json_bufs={},
            )

        assert output.content[0].thinking == "让我先看看"


class TestStreamingTermination:
    @pytest.mark.asyncio
    async def test_missing_response_completed_becomes_error(self, monkeypatch: pytest.MonkeyPatch):
        from bampy.ai.providers.ollama import stream_ollama

        class _FakeResponse:
            def __init__(self, events):
                self._events = iter(events)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._events)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

        class _FakeResponses:
            async def create(self, **_: object):
                return _FakeResponse([
                    SimpleNamespace(
                        type="response.content_part.added",
                        output_index=0,
                        part=SimpleNamespace(type="output_text"),
                    ),
                    SimpleNamespace(
                        type="response.output_text.delta",
                        output_index=0,
                        delta="半截内容",
                    ),
                ])

        class _FakeAsyncOpenAI:
            def __init__(self, **_: object):
                self.responses = _FakeResponses()

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI))

        model = Model(
            id="kimi-k2.5",
            name="kimi-k2.5",
            api="ollama-responses",
            provider="ollama",
        )
        ctx = Context(messages=[UserMessage(content="hello")])
        result = await stream_ollama(model, ctx).result()

        assert result.stop_reason == StopReason.ERROR
        assert result.error_message is not None
        assert "response.completed" in result.error_message
        assert any(
            isinstance(block, TextContent) and block.text == "半截内容"
            for block in result.content
        )
