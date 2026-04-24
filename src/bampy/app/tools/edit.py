"""Built-in edit tool."""

from __future__ import annotations

import difflib
import os
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult
from bampy.ai.types import TextContent

from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_to_cwd


_SMART_SINGLE_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
    }
)
_SMART_DOUBLE_QUOTES = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
)
_UNICODE_DASHES = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)
_SPECIAL_SPACES = str.maketrans(
    {
        "\u00a0": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
    }
)


class EditReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    old_text: str = Field(
        min_length=1,
        description="Exact text for one targeted replacement.",
    )
    new_text: str = Field(description="Replacement text.")


class EditToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path to the file to edit (relative or absolute)")
    edits: list[EditReplacement] = Field(
        min_length=1,
        description="One or more targeted replacements.",
    )


@dataclass(slots=True)
class _MatchedEdit:
    edit_index: int
    match_index: int
    match_length: int
    new_text: str


@dataclass(slots=True)
class _NormalizedTextMap:
    text: str
    starts: list[int]
    ends: list[int]


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


def _normalize_for_fuzzy_match(content: str) -> str:
    return _build_fuzzy_text_map(content).text


def _normalize_character(character: str) -> str:
    return (
        unicodedata.normalize("NFKC", character)
        .translate(_SMART_SINGLE_QUOTES)
        .translate(_SMART_DOUBLE_QUOTES)
        .translate(_UNICODE_DASHES)
        .translate(_SPECIAL_SPACES)
    )


def _append_normalized_span(
    output: list[str],
    starts: list[int],
    ends: list[int],
    span: str,
    *,
    start_offset: int,
) -> None:
    for index, character in enumerate(span):
        normalized = _normalize_character(character)
        for normalized_character in normalized:
            output.append(normalized_character)
            starts.append(start_offset + index)
            ends.append(start_offset + index + 1)


def _build_fuzzy_text_map(content: str) -> _NormalizedTextMap:
    output: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    offset = 0

    for line in content.splitlines(keepends=True):
        has_newline = line.endswith("\n")
        body = line[:-1] if has_newline else line
        stripped_body = body.rstrip()
        _append_normalized_span(output, starts, ends, stripped_body, start_offset=offset)
        if has_newline:
            newline_index = offset + len(body)
            output.append("\n")
            starts.append(newline_index)
            ends.append(newline_index + 1)
        offset += len(line)

    return _NormalizedTextMap("".join(output), starts, ends)


def _find_text(content: str, old_text: str) -> tuple[bool, int, int, bool]:
    index = content.find(old_text)
    if index != -1:
        return True, index, len(old_text), False

    fuzzy_map = _build_fuzzy_text_map(content)
    fuzzy_content = fuzzy_map.text
    fuzzy_old_text = _normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)
    if fuzzy_index == -1:
        return False, -1, 0, False
    if not fuzzy_old_text:
        return False, -1, 0, False
    start = fuzzy_map.starts[fuzzy_index]
    end = fuzzy_map.ends[fuzzy_index + len(fuzzy_old_text) - 1]
    return True, start, end - start, True


def _count_occurrences(content: str, old_text: str, *, fuzzy: bool) -> int:
    if not fuzzy:
        return content.count(old_text)
    haystack = _normalize_for_fuzzy_match(content)
    needle = _normalize_for_fuzzy_match(old_text)
    if not needle:
        return 0
    return haystack.count(needle)


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


def _not_found_error(path: str, index: int, total: int) -> ValueError:
    if total == 1:
        return ValueError(
            f"Could not find the exact text in {path}. "
            "The old_text must match exactly including whitespace and newlines."
        )
    return ValueError(
        f"Could not find edits[{index}] in {path}. "
        "The old_text must match exactly including whitespace and newlines."
    )


def _duplicate_error(path: str, index: int, total: int, occurrences: int) -> ValueError:
    if total == 1:
        return ValueError(
            f"Found {occurrences} occurrences of the text in {path}. "
            "Please provide more context to make it unique."
        )
    return ValueError(
        f"Found {occurrences} occurrences of edits[{index}] in {path}. "
        "Each old_text must be unique. Please provide more context to make it unique."
    )


def _apply_edits_to_normalized_content(
    normalized_content: str,
    edits: list[EditReplacement],
    path: str,
) -> tuple[str, str]:
    normalized_edits = [
        EditReplacement(
            old_text=_normalize_to_lf(edit.old_text),
            new_text=_normalize_to_lf(edit.new_text),
        )
        for edit in edits
    ]

    matched_edits: list[_MatchedEdit] = []
    for index, edit in enumerate(normalized_edits):
        found, match_index, match_length, used_fuzzy = _find_text(normalized_content, edit.old_text)
        if not found:
            raise _not_found_error(path, index, len(normalized_edits))

        occurrences = _count_occurrences(normalized_content, edit.old_text, fuzzy=used_fuzzy)
        if occurrences > 1:
            raise _duplicate_error(path, index, len(normalized_edits), occurrences)

        matched_edits.append(
            _MatchedEdit(
                edit_index=index,
                match_index=match_index,
                match_length=match_length,
                new_text=edit.new_text,
            )
        )

    matched_edits.sort(key=lambda item: item.match_index)
    for previous, current in zip(matched_edits, matched_edits[1:]):
        if previous.match_index + previous.match_length > current.match_index:
            raise ValueError(
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    new_content = normalized_content
    for edit in reversed(matched_edits):
        new_content = (
            new_content[: edit.match_index]
            + edit.new_text
            + new_content[edit.match_index + edit.match_length :]
        )

    if new_content == normalized_content:
        raise ValueError(f"No changes made to {path}. The replacements produced identical content.")
    return normalized_content, new_content


class EditTool:
    name = "edit"
    label = "edit"
    description = (
        "Edit a single file using exact text replacement. Every edits[].old_text "
        "must match a unique, non-overlapping region of the original file."
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
        edits = arguments.edits
        if cancellation is not None:
            cancellation.raise_if_cancelled()

        resolved_path = resolve_to_cwd(arguments.path, self._cwd)

        async def _run() -> AgentToolResult:
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
            base_content, new_content = _apply_edits_to_normalized_content(
                normalized_content,
                edits,
                arguments.path,
            )

            final_content = bom + _restore_line_endings(new_content, line_ending)
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(final_content)

            restored_base = _restore_line_endings(base_content, line_ending)
            restored_new = _restore_line_endings(new_content, line_ending)
            diff = _generate_diff(restored_base, restored_new, arguments.path)
            return AgentToolResult(
                content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {arguments.path}.")],
                details={
                    "diff": diff,
                    "first_changed_line": _first_changed_line(restored_base, restored_new),
                },
            )

        return await with_file_mutation_queue(resolved_path, _run)


def create_edit_tool(cwd: str) -> EditTool:
    return EditTool(cwd)


edit_tool = create_edit_tool(os.getcwd())
