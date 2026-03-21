"""Built-in ls tool."""

from __future__ import annotations

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

DEFAULT_LIMIT = 500


class LsToolInput(BaseModel):
    path: str | None = Field(default=None, description="Directory to list (default: current directory)")
    limit: int | None = Field(default=None, description="Maximum number of entries to return")


def _coerce_params(params: Any) -> LsToolInput:
    if isinstance(params, LsToolInput):
        return params
    if isinstance(params, BaseModel):
        return LsToolInput.model_validate(params.model_dump())
    if isinstance(params, Mapping):
        return LsToolInput.model_validate(dict(params))
    return LsToolInput.model_validate({})


class LsTool:
    name = "ls"
    label = "ls"
    description = (
        "List directory contents. Includes dotfiles, sorts entries alphabetically, "
        f"and truncates output to {DEFAULT_LIMIT} entries or {DEFAULT_MAX_BYTES // 1024}KB."
    )
    parameters = LsToolInput

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

        dir_path = Path(resolve_to_cwd(arguments.path or ".", self._cwd))
        if not dir_path.exists():
            raise FileNotFoundError(f"Path not found: {arguments.path or '.'}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {arguments.path or '.'}")

        effective_limit = max(1, arguments.limit or DEFAULT_LIMIT)
        entries = sorted(dir_path.iterdir(), key=lambda entry: entry.name.lower())
        results: list[str] = []
        entry_limit_reached = False

        for entry in entries:
            if len(results) >= effective_limit:
                entry_limit_reached = True
                break
            suffix = "/" if entry.is_dir() else ""
            results.append(entry.name + suffix)

        if not results:
            return AgentToolResult(content=[TextContent(text="(empty directory)")])

        truncation = truncate_head("\n".join(results), max_lines=10**9)
        output = truncation.content
        details: dict[str, object] = {}
        notices: list[str] = []

        if entry_limit_reached:
            details["entry_limit_reached"] = effective_limit
            notices.append(f"{effective_limit} entries limit reached. Use limit={effective_limit * 2} for more")
        if truncation.truncated:
            details["truncation"] = serialize_truncation(truncation)
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        if notices:
            output += f"\n\n[{' '.join(notices)}]"

        return AgentToolResult(content=[TextContent(text=output)], details=details or None)


def create_ls_tool(cwd: str) -> LsTool:
    return LsTool(cwd)


ls_tool = create_ls_tool(os.getcwd())
