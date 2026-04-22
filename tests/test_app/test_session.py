"""Tests for bampy.app.session."""

from __future__ import annotations

import json
from pathlib import Path

from bampy.ai.types import AssistantMessage, ImageContent, TextContent, ThinkingContent, UserMessage
from bampy.app.messages import convert_app_messages_to_llm
from bampy.app.session import (
    InMemoryBackend,
    SessionManager,
    _build_session_info,
)


class TestSessionContext:
    def test_build_session_context_handles_compaction_and_custom_entries(self):
        sm = SessionManager.in_memory("/repo")
        sm.append_model_change("openai", "gpt-5")
        sm.append_thinking_level_change("high")
        sm.append_message(UserMessage(content="hello"))
        kept_assistant_id = sm.append_message(
            AssistantMessage(
                api="openai-responses",
                provider="openai",
                model="gpt-5",
                content=[TextContent(text="hi there")],
            )
        )
        sm.append_custom_message_entry(
            "note",
            [TextContent(text="remember this"), ImageContent(data="abc", mime_type="image/png")],
            details={"source": "ext"},
        )
        sm.append_branch_summary(from_id=kept_assistant_id, summary="branched away")
        sm.append_compaction(
            summary="earlier history",
            first_kept_entry_id=kept_assistant_id,
            tokens_before=120,
        )
        sm.append_message(UserMessage(content="after compaction"))

        ctx = sm.build_session_context()

        assert ctx.thinking_level == "high"
        assert ctx.model == {"provider": "openai", "model_id": "gpt-5"}
        assert [
            message.get("role") if isinstance(message, dict) else message.role
            for message in ctx.messages
        ] == [
            "compaction_summary",
            "assistant",
            "custom",
            "branch_summary",
            "user",
        ]

        llm_messages = convert_app_messages_to_llm(ctx.messages)
        assert [message.role for message in llm_messages] == [
            "user",
            "assistant",
            "user",
            "user",
            "user",
        ]
        assert llm_messages[2].content[0].text == "remember this"

    def test_branching_labels_and_tree_navigation(self):
        sm = SessionManager.in_memory("/repo")
        first_user_id = sm.append_message(UserMessage(content="root"))
        first_assistant_id = sm.append_message(
            AssistantMessage(
                api="test",
                provider="test",
                model="model-a",
                content=[TextContent(text="reply")],
            )
        )
        sm.append_label_change(first_assistant_id, "checkpoint")
        sm.append_message(UserMessage(content="main branch"))
        sm.branch(first_assistant_id)
        sm.append_message(UserMessage(content="side branch"))

        branch = sm.get_branch()
        tree = sm.get_tree()

        assert sm.get_label(first_assistant_id) == "checkpoint"
        assert [entry.id for entry in branch] == [
            first_user_id,
            first_assistant_id,
            sm.leaf_id,
        ]
        assert len(tree) == 1
        assert len(tree[0].children) == 1
        assert len(tree[0].children[0].children) == 2

    def test_reset_leaf_returns_empty_context(self):
        sm = SessionManager.in_memory("/repo")
        sm.append_message(UserMessage(content="root"))

        sm.reset_leaf()

        assert sm.leaf_id is None
        assert sm.build_session_context().messages == []


class TestSessionPersistence:
    def test_persistence_flushes_only_after_first_assistant_message(self):
        backend = InMemoryBackend()
        sm = SessionManager("/repo", backend=backend, persist=True)

        sm.append_message(UserMessage(content="before assistant"))
        assert backend.read_all() == []

        sm.append_message(
            AssistantMessage(
                api="test",
                provider="test",
                model="model-a",
                content=[TextContent(text="assistant appears")],
            )
        )

        raw_entries = backend.read_all()
        assert [entry["type"] for entry in raw_entries] == [
            "session",
            "message",
            "message",
        ]

    def test_persistence_keeps_assistant_thinking_blocks(self):
        backend = InMemoryBackend()
        sm = SessionManager("/repo", backend=backend, persist=True)

        sm.append_message(UserMessage(content="think first"))
        sm.append_message(
            AssistantMessage(
                api="openai-completions",
                provider="opencode-go",
                model="kimi-k2.6",
                content=[
                    ThinkingContent(
                        thinking="deep thought",
                        thinking_signature="reasoning_content",
                    ),
                    TextContent(text="done"),
                ],
            )
        )

        raw_entries = backend.read_all()
        assistant = raw_entries[-1]["message"]
        assert assistant["role"] == "assistant"
        assert assistant["content"][0] == {
            "type": "thinking",
            "thinking": "deep thought",
            "thinking_signature": "reasoning_content",
            "redacted": False,
        }
        assert assistant["content"][1]["text"] == "done"

    def test_build_session_info_reads_persisted_file(self, tmp_path: Path):
        sm = SessionManager(
            "/repo",
            session_dir=str(tmp_path),
            persist=True,
        )
        sm.append_message(UserMessage(content="first question"))
        sm.append_message(
            AssistantMessage(
                api="test",
                provider="test",
                model="model-a",
                content=[TextContent(text="first answer")],
            )
        )
        sm.append_session_info("  Session Name  ")

        info = _build_session_info(sm.session_file)

        assert info is not None
        assert info.name == "Session Name"
        assert info.cwd == "/repo"
        assert info.message_count == 2
        assert info.first_message == "first question"
        assert "first answer" in info.all_messages_text

    async def test_list_sessions_returns_latest_first(self, tmp_path: Path):
        older = SessionManager("/repo", session_dir=str(tmp_path), persist=True)
        older.append_message(UserMessage(content="older"))
        older.append_message(
            AssistantMessage(
                api="test",
                provider="test",
                model="older-model",
                content=[TextContent(text="done")],
            )
        )

        newer = SessionManager("/repo", session_dir=str(tmp_path), persist=True)
        newer.append_message(UserMessage(content="newer"))
        newer.append_message(
            AssistantMessage(
                api="test",
                provider="test",
                model="newer-model",
                content=[TextContent(text="done")],
            )
        )

        sessions = await SessionManager.list_sessions("/repo", session_dir=str(tmp_path))

        assert len(sessions) == 2
        assert sessions[0].all_messages_text in {"newer done", "older done"}
        assert {session.id for session in sessions} == {
            older.session_id,
            newer.session_id,
        }

    def test_open_restores_cwd_from_session_header(self, tmp_path: Path):
        sm = SessionManager("/repo", session_dir=str(tmp_path), persist=True)
        sm.append_message(UserMessage(content="hi"))
        sm.append_message(
            AssistantMessage(
                api="test",
                provider="test",
                model="model-a",
                content=[TextContent(text="done")],
            )
        )

        reopened = SessionManager.open(sm.session_file)

        assert reopened.cwd == "/repo"
        assert reopened.get_header().cwd == "/repo"

    def test_open_recovers_from_invalid_existing_session_file(self, tmp_path: Path):
        session_file = tmp_path / "broken.jsonl"
        session_file.write_text(
            json.dumps(
                {
                    "type": "message",
                    "id": "msg-1",
                    "parent_id": None,
                    "timestamp": "2026-03-21T00:00:00+00:00",
                    "message": {"role": "user", "content": "oops"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        reopened = SessionManager.open(str(session_file))

        assert reopened.get_header() is not None
        assert reopened.get_entries() == []
        assert reopened.build_session_context().messages == []
