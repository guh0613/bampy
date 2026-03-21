"""Tests for bampy.app.system_prompt."""

from __future__ import annotations

from pathlib import Path

from bampy.app.system_prompt import (
    BuildSystemPromptOptions,
    ContextFile,
    build_system_prompt,
    load_context_files,
)


class TestBuildSystemPrompt:
    def test_custom_prompt_mode_includes_append_and_context_files(self):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                custom_prompt="Base prompt",
                append_system_prompt="Extra tail",
                cwd="/repo/project",
                context_files=[ContextFile(path="CLAUDE.md", content="Follow rules")],
            )
        )

        assert "Base prompt" in prompt
        assert "Extra tail" in prompt
        assert "## CLAUDE.md" in prompt
        assert "Follow rules" in prompt
        assert "Current working directory: /repo/project" in prompt
        assert "Current date:" in prompt

    def test_default_prompt_builds_visible_tools_and_guidelines(self):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                selected_tools=["read", "bash", "grep", "edit", "write", "custom"],
                tool_snippets={"custom": "Custom helper"},
                prompt_guidelines=["Keep diffs small", "Keep diffs small"],
                cwd="C:\\repo\\project",
            )
        )

        assert "- read: Read file contents" in prompt
        assert "- custom: Custom helper" in prompt
        assert "Prefer grep/find/ls tools over bash for file exploration (faster)" in prompt
        assert "Use read to examine files before editing" in prompt
        assert "Use edit for precise changes (old text must match exactly)" in prompt
        assert "Use write only for new files or complete rewrites" in prompt
        assert prompt.count("Keep diffs small") == 1
        assert "Current working directory: C:/repo/project" in prompt

    def test_default_prompt_handles_no_visible_tools(self):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(selected_tools=["unknown"])
        )

        assert "Available tools:\n(none)" in prompt


class TestLoadContextFiles:
    def test_load_context_files_reads_existing_files_only(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("claude rules", encoding="utf-8")
        bampy_dir = tmp_path / ".bampy"
        bampy_dir.mkdir()
        (bampy_dir / "context.md").write_text("project context", encoding="utf-8")

        result = load_context_files(str(tmp_path))

        assert [item.path for item in result] == ["CLAUDE.md", ".bampy/context.md"]
        assert result[0].content == "claude rules"
        assert result[1].content == "project context"
