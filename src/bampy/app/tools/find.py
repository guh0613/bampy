"""Built-in find tool."""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult
from bampy.ai.types import TextContent

from .path_utils import resolve_to_cwd
from .truncate import DEFAULT_MAX_BYTES, format_size, serialize_truncation, truncate_head

DEFAULT_LIMIT = 1_000
_SKIP_PARTS = {".git", "node_modules", "__pycache__"}


class FindToolInput(BaseModel):
    pattern: str = Field(description="Glob pattern to match files")
    path: str | None = Field(default=None, description="Directory to search in (default: current directory)")
    limit: int | None = Field(default=None, description="Maximum number of results to return")


def _coerce_params(params: Any) -> FindToolInput:
    if isinstance(params, FindToolInput):
        return params
    if isinstance(params, BaseModel):
        return FindToolInput.model_validate(params.model_dump())
    if isinstance(params, Mapping):
        return FindToolInput.model_validate(dict(params))
    return FindToolInput.model_validate({})


def _is_skipped(relative_path: str) -> bool:
    parts = {part for part in Path(relative_path).parts if part}
    return any(part in _SKIP_PARTS for part in parts)


class FindTool:
    name = "find"
    label = "find"
    description = (
        "Search for files by glob pattern. Returns matching paths relative to the search "
        f"directory. Output is truncated to {DEFAULT_LIMIT} results or {DEFAULT_MAX_BYTES // 1024}KB."
    )
    parameters = FindToolInput

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

        search_root = Path(resolve_to_cwd(arguments.path or ".", self._cwd))
        if not search_root.exists():
            raise FileNotFoundError(f"Path not found: {arguments.path or '.'}")

        effective_limit = max(1, arguments.limit or DEFAULT_LIMIT)
        if search_root.is_file():
            relative_matches = [search_root.name] if search_root.match(arguments.pattern) else []
        else:
            raw_matches = glob.glob(
                arguments.pattern,
                root_dir=str(search_root),
                recursive=True,
                include_hidden=True,
            )
            deduped: list[str] = []
            seen: set[str] = set()
            for match in raw_matches:
                relative = match.replace(os.sep, "/")
                if not relative or relative in seen or _is_skipped(relative):
                    continue
                seen.add(relative)
                deduped.append(relative)
            relative_matches = sorted(deduped, key=str.lower)

        if not relative_matches:
            return AgentToolResult(content=[TextContent(text="No files found matching pattern")])

        results: list[str] = []
        result_limit_reached = False
        for relative in relative_matches:
            if len(results) >= effective_limit:
                result_limit_reached = True
                break
            full_path = search_root / relative
            suffix = "/" if full_path.is_dir() else ""
            results.append(relative + suffix)

        truncation = truncate_head("\n".join(results), max_lines=10**9)
        output = truncation.content
        details: dict[str, object] = {}
        notices: list[str] = []

        if result_limit_reached:
            details["result_limit_reached"] = effective_limit
            notices.append(f"{effective_limit} results limit reached")
        if truncation.truncated:
            details["truncation"] = serialize_truncation(truncation)
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        if notices:
            output += f"\n\n[{' '.join(notices)}]"

        return AgentToolResult(content=[TextContent(text=output)], details=details or None)


def create_find_tool(cwd: str) -> FindTool:
    return FindTool(cwd)


find_tool = create_find_tool(os.getcwd())
