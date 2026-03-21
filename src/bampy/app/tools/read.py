"""Built-in read tool."""

from __future__ import annotations

import base64
import mimetypes
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult
from bampy.ai.types import ImageContent, TextContent

from .path_utils import resolve_read_path
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, serialize_truncation, truncate_head

_SUPPORTED_IMAGE_MIME_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/svg+xml",
}


class ReadToolInput(BaseModel):
    path: str = Field(description="Path to the file to read (relative or absolute)")
    offset: int | None = Field(default=None, description="Line number to start reading from (1-indexed)")
    limit: int | None = Field(default=None, description="Maximum number of lines to read")


def _coerce_params(params: Any) -> ReadToolInput:
    if isinstance(params, ReadToolInput):
        return params
    if isinstance(params, BaseModel):
        return ReadToolInput.model_validate(params.model_dump())
    if isinstance(params, Mapping):
        return ReadToolInput.model_validate(dict(params))
    return ReadToolInput.model_validate({})


def _detect_image_mime_type(path: str) -> str | None:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type in _SUPPORTED_IMAGE_MIME_TYPES:
        return mime_type
    return None


class ReadTool:
    name = "read"
    label = "read"
    description = (
        "Read the contents of a file. Supports text files and common images. "
        f"Text output is truncated to {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB. "
        "Use offset/limit for large files."
    )
    parameters = ReadToolInput

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

        resolved_path = resolve_read_path(arguments.path, self._cwd)
        path = Path(resolved_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {arguments.path}")
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {arguments.path}")

        mime_type = _detect_image_mime_type(resolved_path)
        if mime_type is not None:
            data = path.read_bytes()
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            return AgentToolResult(
                content=[
                    TextContent(text=f"Read image file [{mime_type}]"),
                    ImageContent(data=base64.b64encode(data).decode("ascii"), mime_type=mime_type),
                ]
            )

        raw_text = path.read_text(encoding="utf-8", errors="replace")
        if cancellation is not None:
            cancellation.raise_if_cancelled()

        all_lines = raw_text.split("\n")
        start_index = max(0, (arguments.offset or 1) - 1)
        if start_index >= len(all_lines):
            raise ValueError(
                f"Offset {arguments.offset} is beyond end of file ({len(all_lines)} lines total)"
            )

        if arguments.limit is not None:
            end_index = min(start_index + max(arguments.limit, 0), len(all_lines))
            selected = "\n".join(all_lines[start_index:end_index])
            user_limited_lines = end_index - start_index
        else:
            selected = "\n".join(all_lines[start_index:])
            user_limited_lines = None

        truncation = truncate_head(selected)
        start_display = start_index + 1

        if truncation.first_line_exceeds_limit:
            text = (
                f"[Line {start_display} exceeds {format_size(DEFAULT_MAX_BYTES)}. "
                f"Use bash to inspect a smaller slice of {arguments.path}.]"
            )
            return AgentToolResult(
                content=[TextContent(text=text)],
                details={"truncation": serialize_truncation(truncation)},
            )

        output = truncation.content
        details: dict[str, object] = {}

        if truncation.truncated:
            end_display = start_display + truncation.output_lines - 1
            next_offset = end_display + 1
            details["truncation"] = serialize_truncation(truncation)
            if truncation.truncated_by == "lines":
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                )
        elif user_limited_lines is not None and start_index + user_limited_lines < len(all_lines):
            next_offset = start_index + user_limited_lines + 1
            remaining = len(all_lines) - (start_index + user_limited_lines)
            output += f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"

        return AgentToolResult(
            content=[TextContent(text=output)],
            details=details or None,
        )


def create_read_tool(cwd: str) -> ReadTool:
    return ReadTool(cwd)


read_tool = create_read_tool(os.getcwd())
