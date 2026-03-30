"""Tests for bampy.ai.providers.ollama."""

from __future__ import annotations

from types import SimpleNamespace

from bampy.ai.types import AssistantMessage, TextContent, ThinkingContent


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
