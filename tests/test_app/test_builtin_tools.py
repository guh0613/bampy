"""Tests for built-in app tools."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bampy.app import create_all_tools, create_coding_tools, create_read_only_tools
from bampy.app.tools import (
    create_bash_tool,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_patch_tool,
    create_read_tool,
    create_write_tool,
)


class TestToolPresets:
    def test_create_tool_presets_expose_expected_names(self):
        coding = create_coding_tools("/repo")
        read_only = create_read_only_tools("/repo")
        all_tools = create_all_tools("/repo")

        assert [tool.name for tool in coding] == ["read", "bash", "edit", "patch", "write"]
        assert [tool.name for tool in read_only] == ["read", "grep", "find", "ls"]
        assert list(all_tools) == ["read", "bash", "edit", "patch", "write", "grep", "find", "ls"]


class TestReadTool:
    async def test_read_tool_supports_line_limit_and_offset(self, tmp_path: Path):
        file_path = tmp_path / "notes.txt"
        file_path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

        tool = create_read_tool(str(tmp_path))
        result = await tool.execute("call-1", {"path": "notes.txt", "limit": 2})
        next_result = await tool.execute("call-2", {"path": "notes.txt", "offset": 3, "limit": 2})

        assert result.content[0].text.startswith("one\ntwo")
        assert "Use offset=3 to continue." in result.content[0].text
        assert next_result.content[0].text.startswith("three\nfour")

    async def test_read_tool_returns_images_as_blocks(self, tmp_path: Path):
        image_path = tmp_path / "pixel.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        tool = create_read_tool(str(tmp_path))
        result = await tool.execute("call-3", {"path": "pixel.png"})

        assert result.content[0].type == "text"
        assert result.content[1].type == "image"
        assert result.content[1].mime_type == "image/png"


class TestWriteTool:
    async def test_write_tool_creates_parent_directories(self, tmp_path: Path):
        tool = create_write_tool(str(tmp_path))

        result = await tool.execute("call-4", {"path": "nested/output.txt", "content": "hello"})

        assert "Successfully wrote" in result.content[0].text
        assert (tmp_path / "nested" / "output.txt").read_text(encoding="utf-8") == "hello"


class TestEditTool:
    async def test_edit_tool_preserves_line_endings_and_returns_diff(self, tmp_path: Path):
        file_path = tmp_path / "sample.txt"
        file_path.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")

        tool = create_edit_tool(str(tmp_path))
        result = await tool.execute(
            "call-5",
            {
                "path": "sample.txt",
                "edits": [{"old_text": "alpha\nbeta", "new_text": "alpha\nBETA"}],
            },
        )

        assert "Successfully replaced 1 block(s)" in result.content[0].text
        assert b"\r\n" in file_path.read_bytes()
        assert "BETA" in file_path.read_text(encoding="utf-8")
        assert "sample.txt" in result.details["diff"]

    async def test_edit_tool_applies_multiple_disjoint_edits(self, tmp_path: Path):
        file_path = tmp_path / "sample.txt"
        file_path.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")

        tool = create_edit_tool(str(tmp_path))
        await tool.execute(
            "call-5b",
            {
                "path": "sample.txt",
                "edits": [
                    {"old_text": "alpha\n", "new_text": "ALPHA\n"},
                    {"old_text": "gamma\n", "new_text": "GAMMA\n"},
                ],
            },
        )

        assert file_path.read_text(encoding="utf-8") == "ALPHA\nbeta\nGAMMA\ndelta\n"

    async def test_edit_tool_rejects_legacy_single_replacement_shape(self, tmp_path: Path):
        tool = create_edit_tool(str(tmp_path))

        with pytest.raises(ValidationError):
            await tool.execute(
                "call-5c",
                {"path": "legacy.txt", "old_text": "before", "new_text": "after"},
            )

    async def test_edit_tool_fuzzy_matches_trailing_whitespace(self, tmp_path: Path):
        file_path = tmp_path / "fuzzy.txt"
        file_path.write_text("line one   \nline two  \nline three\n", encoding="utf-8")

        tool = create_edit_tool(str(tmp_path))
        await tool.execute(
            "call-5d",
            {
                "path": "fuzzy.txt",
                "edits": [{"old_text": "line one\nline two\n", "new_text": "replaced\n"}],
            },
        )

        assert file_path.read_text(encoding="utf-8") == "replaced\nline three\n"


class TestPatchTool:
    async def test_patch_tool_applies_unified_diff(self, tmp_path: Path):
        file_path = tmp_path / "sample.txt"
        file_path.write_text("alpha\nbeta\n", encoding="utf-8")

        tool = create_patch_tool(str(tmp_path))
        result = await tool.execute(
            "call-patch-1",
            {
                "patch": (
                    "--- a/sample.txt\n"
                    "+++ b/sample.txt\n"
                    "@@ -1,2 +1,2 @@\n"
                    " alpha\n"
                    "-beta\n"
                    "+BETA\n"
                )
            },
        )

        assert "Successfully applied patch" in result.content[0].text
        assert file_path.read_text(encoding="utf-8") == "alpha\nBETA\n"

    async def test_patch_tool_rejects_paths_that_escape_cwd(self, tmp_path: Path):
        tool = create_patch_tool(str(tmp_path))

        with pytest.raises(ValueError, match="escapes workspace"):
            await tool.execute(
                "call-patch-2",
                {
                    "patch": (
                        "--- a/../outside.txt\n"
                        "+++ b/../outside.txt\n"
                        "@@ -1 +1 @@\n"
                        "-old\n"
                        "+new\n"
                    )
                },
            )


class TestLsTool:
    async def test_ls_tool_lists_directories_with_suffix(self, tmp_path: Path):
        (tmp_path / "folder").mkdir()
        (tmp_path / "folder" / "inner.txt").write_text("x", encoding="utf-8")
        (tmp_path / ".hidden").write_text("y", encoding="utf-8")
        (tmp_path / "alpha.txt").write_text("z", encoding="utf-8")

        tool = create_ls_tool(str(tmp_path))
        result = await tool.execute("call-6", {"path": "."})

        text = result.content[0].text
        assert "folder/" in text
        assert ".hidden" in text
        assert "alpha.txt" in text


class TestFindTool:
    async def test_find_tool_returns_relative_matches_and_skips_git_dir(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('x')", encoding="utf-8")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "ignored.py").write_text("print('ignore')", encoding="utf-8")

        tool = create_find_tool(str(tmp_path))
        result = await tool.execute("call-7", {"pattern": "**/*.py"})

        assert "src/main.py" in result.content[0].text
        assert ".git/ignored.py" not in result.content[0].text


class TestGrepTool:
    async def test_grep_tool_supports_literal_search_context_and_limit(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        file_path = tmp_path / "src" / "app.py"
        file_path.write_text(
            "first line\nneedle here\nthird line\nneedle again\nfifth line\n",
            encoding="utf-8",
        )

        tool = create_grep_tool(str(tmp_path))
        result = await tool.execute(
            "call-8",
            {
                "pattern": "needle",
                "path": "src",
                "literal": True,
                "context": 1,
                "limit": 1,
            },
        )

        text = result.content[0].text
        assert "app.py-1- first line" in text
        assert "app.py:2: needle here" in text
        assert "app.py-3- third line" in text
        assert result.details["match_limit_reached"] == 1


class TestBashTool:
    async def test_bash_tool_executes_command(self, tmp_path: Path):
        tool = create_bash_tool(str(tmp_path))

        result = await tool.execute("call-9", {"command": "printf 'hello'"})

        assert result.content[0].text == "hello"

    async def test_bash_tool_raises_on_non_zero_exit(self, tmp_path: Path):
        tool = create_bash_tool(str(tmp_path))

        with pytest.raises(RuntimeError) as exc:
            await tool.execute("call-10", {"command": "printf 'oops'; exit 7"})

        assert "oops" in str(exc.value)
        assert "Command exited with code 7" in str(exc.value)
