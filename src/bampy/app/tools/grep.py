"""Built-in grep tool."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult
from bampy.ai.types import TextContent

from .path_utils import resolve_to_cwd
from .truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    format_size,
    serialize_truncation,
    truncate_head,
    truncate_line,
)

DEFAULT_LIMIT = 100
_SKIP_DIRS = {".git", "node_modules", "__pycache__"}


class GrepToolInput(BaseModel):
    pattern: str = Field(description="Search pattern (regex or literal string)")
    path: str | None = Field(default=None, description="Directory or file to search")
    glob: str | None = Field(default=None, description="Filter files by glob pattern")
    ignore_case: bool | None = Field(default=None, description="Case-insensitive search")
    literal: bool | None = Field(default=None, description="Treat pattern as a literal string")
    context: int | None = Field(default=None, description="Number of lines to show before and after each match")
    limit: int | None = Field(default=None, description="Maximum number of matches to return")


def _coerce_params(params: Any) -> GrepToolInput:
    if isinstance(params, GrepToolInput):
        return params
    if isinstance(params, BaseModel):
        return GrepToolInput.model_validate(params.model_dump())
    if isinstance(params, Mapping):
        return GrepToolInput.model_validate(dict(params))
    return GrepToolInput.model_validate({})


def _iter_files(search_root: Path, glob_pattern: str | None) -> list[Path]:
    if search_root.is_file():
        return [search_root]

    results: list[Path] = []
    for current_root, dirs, files in search_root.walk():
        dirs[:] = [name for name in dirs if name not in _SKIP_DIRS]
        for filename in files:
            path = current_root / filename
            relative = path.relative_to(search_root).as_posix()
            if glob_pattern and not path.match(glob_pattern) and not Path(relative).match(glob_pattern):
                continue
            results.append(path)
    return results


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(8_192)
    except OSError:
        return True
    return b"\x00" in sample


class GrepTool:
    name = "grep"
    label = "grep"
    description = (
        "Search file contents for a pattern. Returns matching lines with file paths and "
        f"line numbers. Output is truncated to {DEFAULT_LIMIT} matches or {DEFAULT_MAX_BYTES // 1024}KB. "
        f"Long lines are truncated to {GREP_MAX_LINE_LENGTH} characters."
    )
    parameters = GrepToolInput

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

        search_path = Path(resolve_to_cwd(arguments.path or ".", self._cwd))
        if not search_path.exists():
            raise FileNotFoundError(f"Path not found: {arguments.path or '.'}")

        flags = re.IGNORECASE if arguments.ignore_case else 0
        pattern_text = re.escape(arguments.pattern) if arguments.literal else arguments.pattern
        try:
            matcher = re.compile(pattern_text, flags)
        except re.error as exc:
            raise ValueError(f"Invalid pattern: {exc}") from exc

        context_lines = max(0, arguments.context or 0)
        limit = max(1, arguments.limit or DEFAULT_LIMIT)

        matches: list[str] = []
        lines_truncated = False
        match_limit_reached = False
        match_count = 0

        for file_path in _iter_files(search_path, arguments.glob):
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            if _looks_binary(file_path):
                continue

            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            all_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            relative = (
                file_path.relative_to(search_path).as_posix()
                if search_path.is_dir()
                else file_path.name
            )

            for index, line in enumerate(all_lines, start=1):
                if not matcher.search(line):
                    continue
                match_count += 1
                start = max(1, index - context_lines)
                end = min(len(all_lines), index + context_lines)
                for current in range(start, end + 1):
                    current_line = all_lines[current - 1].replace("\r", "")
                    truncated_text, was_truncated = truncate_line(current_line)
                    lines_truncated = lines_truncated or was_truncated
                    if current == index:
                        matches.append(f"{relative}:{current}: {truncated_text}")
                    else:
                        matches.append(f"{relative}-{current}- {truncated_text}")
                if context_lines > 0:
                    matches.append("")
                if match_count >= limit:
                    match_limit_reached = True
                    break
            if match_limit_reached:
                break

        if context_lines > 0 and matches and matches[-1] == "":
            matches.pop()

        if not matches:
            return AgentToolResult(content=[TextContent(text="No matches found")])

        truncation = truncate_head("\n".join(matches), max_lines=10**9)
        output = truncation.content
        details: dict[str, object] = {}
        notices: list[str] = []

        if match_limit_reached:
            details["match_limit_reached"] = limit
            notices.append(f"{limit} matches limit reached")
        if lines_truncated:
            details["lines_truncated"] = True
            notices.append(f"Some lines were truncated to {GREP_MAX_LINE_LENGTH} characters")
        if truncation.truncated:
            details["truncation"] = serialize_truncation(truncation)
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        if notices:
            output += f"\n\n[{' '.join(notices)}]"

        return AgentToolResult(content=[TextContent(text=output)], details=details or None)


def create_grep_tool(cwd: str) -> GrepTool:
    return GrepTool(cwd)


grep_tool = create_grep_tool(Path.cwd().as_posix())
