"""Tests for bampy.app.messages."""

from __future__ import annotations

from bampy.agent import messages as agent_messages
from bampy.agent.messages import clear_message_converters, convert_message_to_llm
from bampy.ai.types import TextContent, UserMessage
from bampy.app.messages import (
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    convert_app_messages_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
    register_app_message_converters,
)


def setup_function() -> None:
    clear_message_converters()


def teardown_function() -> None:
    clear_message_converters()


class TestMessageFactories:
    def test_create_messages_parse_iso_timestamp(self):
        timestamp = "2026-03-21T10:20:30+00:00"

        compaction = create_compaction_summary_message("summary", 42, timestamp)
        branch = create_branch_summary_message("branched", "entry-1", timestamp)
        custom = create_custom_message("note", "hello", timestamp=timestamp)

        assert isinstance(compaction, CompactionSummaryMessage)
        assert compaction.tokens_before == 42
        assert isinstance(branch, BranchSummaryMessage)
        assert branch.from_id == "entry-1"
        assert isinstance(custom, CustomMessage)
        assert compaction.timestamp == branch.timestamp == custom.timestamp

    def test_create_custom_message_keeps_details_and_display(self):
        message = create_custom_message(
            "note",
            [TextContent(text="hello")],
            display=False,
            details={"source": "ext"},
        )

        assert message.custom_type == "note"
        assert message.display is False
        assert message.details == {"source": "ext"}


class TestLlmConversion:
    def test_convert_app_messages_to_llm_handles_custom_roles(self):
        messages = [
            create_compaction_summary_message("old context", 100),
            create_branch_summary_message("branch recap", "entry-2"),
            create_custom_message("note", "extension payload"),
            UserMessage(content="keep me"),
        ]

        result = convert_app_messages_to_llm(messages)

        assert [msg.role for msg in result] == ["user", "user", "user", "user"]
        assert "old context" in result[0].content[0].text
        assert "branch recap" in result[1].content[0].text
        assert result[2].content[0].text == "extension payload"
        assert result[3].content == "keep me"

    def test_register_app_message_converters_integrates_with_agent_registry(self):
        register_app_message_converters()

        compaction = convert_message_to_llm(
            create_compaction_summary_message("checkpoint", 8)
        )
        branch = convert_message_to_llm(
            create_branch_summary_message("branch info", "entry-3")
        )
        custom = convert_message_to_llm(
            create_custom_message("note", "from extension")
        )

        assert len(compaction) == 1
        assert len(branch) == 1
        assert len(custom) == 1
        assert "checkpoint" in compaction[0].content[0].text
        assert "branch info" in branch[0].content[0].text
        assert custom[0].content[0].text == "from extension"

    def test_register_app_message_converters_overwrites_same_roles_cleanly(self):
        register_app_message_converters()
        first_registry = dict(agent_messages._registry)

        register_app_message_converters()

        assert dict(agent_messages._registry).keys() == first_registry.keys()
