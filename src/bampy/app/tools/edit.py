"""Built-in edit tool."""

from __future__ import annotations

import difflib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult
from bampy.ai.types import TextContent

from .path_utils import resolve_to_cwd


class EditToolInput(BaseModel):
    path: str = Field(description="Path to the file to edit (relative or absolute)")
    old_text: str = Field(min_length=1, description="Exact text to find and replace")
    new_text: str = Field(description="New text to replace the old text with")


def _coerce_params(params: Any) -> EditToolInput:
    if isinstance(params, EditToolInput):
        return params
    if isinstance(params, BaseModel):
        return EditToolInput.model_validate(params.model_dump())
    if isinstance(params, Mapping):
        return EditToolInput.model_validate(dict(params))
    return EditToolInput.model_validate({})


def _strip_bom(content: str) -> tuple[str, str]:
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


def _detect_line_ending(content: str) -> str:
    if "\r\n" in content:
        return "\r\n"
    if "\r" in content:
        return "\r"
    return "\n"


def _normalize_to_lf(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")


def _restore_line_endings(content: str, line_ending: str) -> str:
    if line_ending == "\n":
        return content
    return content.replace("\n", line_ending)


def _generate_diff(old: str, new: str, path: str) -> str:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=path,
        tofile=path,
        lineterm="",
    )
    return "\n".join(diff)


def _first_changed_line(old: str, new: str) -> int | None:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    for index, (old_line, new_line) in enumerate(zip(old_lines, new_lines), start=1):
        if old_line != new_line:
            return index
    if len(old_lines) != len(new_lines):
        return min(len(old_lines), len(new_lines)) + 1
    return None


class EditTool:
    name = "edit"
    label = "edit"
    description = (
        "Edit a file by replacing exact text. The old_text must be unique in the file. "
        "Line endings and BOM are preserved."
    )
    parameters = EditToolInput

    def __init__(self, cwd: str) -> None:
        self._cwd = cwd

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update=None,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        arguments = _coerce_params(params)
        if cancellation is not None:
            cancellation.raise_if_cancelled()

        resolved_path = resolve_to_cwd(arguments.path, self._cwd)
        path = Path(resolved_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {arguments.path}")
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {arguments.path}")

        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            raw_content = handle.read()
        if cancellation is not None:
            cancellation.raise_if_cancelled()

        bom, content = _strip_bom(raw_content)
        line_ending = _detect_line_ending(content)
        normalized_content = _normalize_to_lf(content)
        normalized_old = _normalize_to_lf(arguments.old_text)
        normalized_new = _normalize_to_lf(arguments.new_text)

        occurrences = normalized_content.count(normalized_old)
        if occurrences == 0:
            raise ValueError(
                f"Could not find the exact text in {arguments.path}. "
                "The old_text must match exactly including whitespace and newlines."
            )
        if occurrences > 1:
            raise ValueError(
                f"Found {occurrences} occurrences of the text in {arguments.path}. "
                "Please provide more context to make it unique."
            )

        new_content = normalized_content.replace(normalized_old, normalized_new, 1)
        if new_content == normalized_content:
            raise ValueError(f"No changes made to {arguments.path}")

        final_content = bom + _restore_line_endings(new_content, line_ending)
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(final_content)

        diff = _generate_diff(content, _restore_line_endings(new_content, line_ending), arguments.path)
        return AgentToolResult(
            content=[TextContent(text=f"Successfully replaced text in {arguments.path}.")],
            details={
                "diff": diff,
                "first_changed_line": _first_changed_line(content, _restore_line_endings(new_content, line_ending)),
            },
        )


def create_edit_tool(cwd: str) -> EditTool:
    return EditTool(cwd)


edit_tool = create_edit_tool(os.getcwd())
