"""Built-in write tool."""

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


class WriteToolInput(BaseModel):
    path: str = Field(description="Path to the file to write (relative or absolute)")
    content: str = Field(description="Content to write to the file")


def _coerce_params(params: Any) -> WriteToolInput:
    if isinstance(params, WriteToolInput):
        return params
    if isinstance(params, BaseModel):
        return WriteToolInput.model_validate(params.model_dump())
    if isinstance(params, Mapping):
        return WriteToolInput.model_validate(dict(params))
    return WriteToolInput.model_validate({})


class WriteTool:
    name = "write"
    label = "write"
    description = (
        "Write content to a file. Creates parent directories if needed and "
        "overwrites the file if it already exists."
    )
    parameters = WriteToolInput

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
        path.parent.mkdir(parents=True, exist_ok=True)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        path.write_text(arguments.content, encoding="utf-8")
        return AgentToolResult(
            content=[
                TextContent(
                    text=f"Successfully wrote {len(arguments.content.encode('utf-8'))} bytes to {arguments.path}"
                )
            ]
        )


def create_write_tool(cwd: str) -> WriteTool:
    return WriteTool(cwd)


write_tool = create_write_tool(os.getcwd())
