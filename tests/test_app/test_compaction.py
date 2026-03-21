"""Tests for bampy.app.compaction."""

from __future__ import annotations

import importlib
import math

from bampy.ai.types import (
    AssistantMessage,
    ImageContent,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from bampy.app.compaction import (
    TURN_PREFIX_PROMPT,
    _serialize_conversation,
    CompactionPreparation,
    CompactionSettings,
    compact,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_point,
    generate_summary,
    prepare_compaction,
)
from bampy.app.session import CompactionEntry, SessionMessageEntry


def _message_entry(entry_id: str, parent_id: str | None, message) -> SessionMessageEntry:
    entry = SessionMessageEntry(id=entry_id, parent_id=parent_id, timestamp="2026-03-21T00:00:00+00:00")
    entry.message = message.model_dump(mode="json") if hasattr(message, "model_dump") else message
    return entry


class TestEstimateTokens:
    def test_counts_user_and_dict_images_conservatively(self):
        message = UserMessage(
            content=[
                TextContent(text="hi"),
                ImageContent(data="abc", mime_type="image/png"),
            ]
        )
        dict_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image", "data": "abc", "mime_type": "image/png"},
            ],
        }

        expected = math.ceil((2 + 4800) / 4)
        assert estimate_tokens(message) == expected
        assert estimate_tokens(dict_message) == expected

    def test_counts_assistant_tool_calls_and_thinking(self):
        message = AssistantMessage(
            api="test",
            provider="test",
            model="model-a",
            content=[
                TextContent(text="hello"),
                ToolCall(id="call-1", name="search", arguments={"q": "python"}),
            ],
        )

        assert estimate_tokens(message) >= math.ceil(len("hellosearch") / 4)


class TestContextEstimation:
    def test_uses_last_non_error_assistant_usage(self):
        messages = [
            AssistantMessage(
                api="test",
                provider="test",
                model="model-a",
                usage=Usage(total_tokens=120),
                stop_reason=StopReason.ERROR,
            ),
            AssistantMessage(
                api="test",
                provider="test",
                model="model-a",
                usage=Usage(total_tokens=64),
                stop_reason=StopReason.STOP,
            ),
            UserMessage(content="tail"),
        ]

        estimate = estimate_context_tokens(messages)

        assert estimate.usage_tokens == 64
        assert estimate.trailing_tokens == 1
        assert estimate.tokens == 65
        assert estimate.last_usage_index == 1


class TestCutPointsAndPreparation:
    def test_find_cut_point_detects_split_turn(self):
        entries = [
            _message_entry("u1", None, UserMessage(content="first")),
            _message_entry(
                "a1",
                "u1",
                AssistantMessage(
                    api="test",
                    provider="test",
                    model="model-a",
                    content=[TextContent(text="reply")],
                ),
            ),
            _message_entry("u2", "a1", UserMessage(content="second turn")),
            _message_entry(
                "a2",
                "u2",
                AssistantMessage(
                    api="test",
                    provider="test",
                    model="model-a",
                    content=[TextContent(text="recent reply")],
                ),
            ),
        ]

        result = find_cut_point(entries, 0, len(entries), keep_recent_tokens=1)

        assert result.first_kept_entry_index == 3
        assert result.turn_start_index == 2
        assert result.is_split_turn is True

    def test_prepare_compaction_captures_previous_summary_and_prefix(self):
        previous = CompactionEntry(
            id="c1",
            parent_id="a1",
            timestamp="2026-03-21T00:00:00+00:00",
        )
        previous.summary = "old summary"
        previous.first_kept_entry_id = "a1"
        previous.tokens_before = 80

        entries = [
            _message_entry("u1", None, UserMessage(content="first")),
            _message_entry(
                "a1",
                "u1",
                AssistantMessage(
                    api="test",
                    provider="test",
                    model="model-a",
                    content=[TextContent(text="reply")],
                ),
            ),
            previous,
            _message_entry("u2", "c1", UserMessage(content="second turn")),
            _message_entry(
                "a2",
                "u2",
                AssistantMessage(
                    api="test",
                    provider="test",
                    model="model-a",
                    content=[TextContent(text="recent reply")],
                ),
            ),
        ]

        preparation = prepare_compaction(
            entries,
            CompactionSettings(keep_recent_tokens=1, reserve_tokens=32),
        )

        assert preparation is not None
        assert preparation.first_kept_entry_id == "a2"
        assert preparation.is_split_turn is True
        assert preparation.previous_summary == "old summary"
        assert [
            msg.get("role") if isinstance(msg, dict) else msg.role
            for msg in preparation.turn_prefix_messages
        ] == ["user"]
        assert preparation.tokens_before > 0


class TestCompact:
    async def test_compact_handles_split_turn_without_prior_history(self, monkeypatch):
        history_calls: list[list[object]] = []
        prefix_calls: list[list[object]] = []

        async def fake_generate_summary(
            current_messages,
            model,
            reserve_tokens,
            api_key,
            custom_instructions=None,
            previous_summary=None,
            cancellation=None,
        ):
            history_calls.append(list(current_messages))
            return "history summary"

        async def fake_generate_turn_prefix_summary(
            current_messages,
            model,
            reserve_tokens,
            api_key,
            cancellation=None,
        ):
            prefix_calls.append(list(current_messages))
            return "prefix summary"

        monkeypatch.setattr("bampy.app.compaction.generate_summary", fake_generate_summary)
        monkeypatch.setattr(
            "bampy.app.compaction.generate_turn_prefix_summary",
            fake_generate_turn_prefix_summary,
        )

        result = await compact(
            CompactionPreparation(
                first_kept_entry_id="a2",
                messages_to_summarize=[],
                turn_prefix_messages=[UserMessage(content="recent user prompt")],
                is_split_turn=True,
                tokens_before=55,
            ),
            model=object(),
            api_key="secret",
        )

        assert result.first_kept_entry_id == "a2"
        assert result.tokens_before == 55
        assert "No prior history." in result.summary
        assert "Turn Context (split turn)" in result.summary
        assert history_calls == []
        assert len(prefix_calls) == 1
        assert len(prefix_calls[0]) == 1
        assert prefix_calls[0][0].role == "user"
        assert prefix_calls[0][0].content == "recent user prompt"

    def test_serialize_conversation_truncates_tool_results(self):
        messages = [
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="read",
                content=[TextContent(text="x" * 2105)],
            )
        ]

        serialized = _serialize_conversation(messages)

        assert "[tool_result]" in serialized
        assert "[... 105 more characters truncated]" in serialized

    async def test_generate_turn_prefix_summary_uses_prefix_prompt(self, monkeypatch):
        captured_text: list[str] = []

        async def fake_complete_simple(_model, ctx, _options):
            captured_text.append(ctx.messages[0].content[0].text)
            return AssistantMessage(
                api="test",
                provider="test",
                model="model-a",
                content=[TextContent(text="prefix summary")],
            )

        stream_module = importlib.import_module("bampy.ai.stream")
        monkeypatch.setattr(stream_module, "complete_simple", fake_complete_simple)

        from bampy.app.compaction import generate_turn_prefix_summary

        summary = await generate_turn_prefix_summary(
            [UserMessage(content="hello")],
            model=object(),
            reserve_tokens=100,
            api_key="secret",
        )

        assert summary == "prefix summary"
        assert TURN_PREFIX_PROMPT in captured_text[0]

    async def test_generate_summary_raises_on_error_response(self, monkeypatch):
        async def fake_complete_simple(_model, _ctx, _options):
            return AssistantMessage(
                api="test",
                provider="test",
                model="model-a",
                stop_reason=StopReason.ERROR,
                error_message="provider failed",
            )

        stream_module = importlib.import_module("bampy.ai.stream")
        monkeypatch.setattr(stream_module, "complete_simple", fake_complete_simple)

        try:
            await generate_summary(
                [UserMessage(content="hello")],
                model=object(),
                reserve_tokens=100,
                api_key="secret",
            )
        except RuntimeError as exc:
            assert "provider failed" in str(exc)
        else:
            raise AssertionError("Expected generate_summary() to raise on error responses")
