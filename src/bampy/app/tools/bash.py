"""Built-in bash tool."""

from __future__ import annotations

import asyncio
import inspect
import os
import signal
import tempfile
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bampy.agent.cancellation import CancellationError, CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback
from bampy.ai.types import TextContent

from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, serialize_truncation, truncate_tail


class BashToolInput(BaseModel):
    command: str = Field(description="Bash command to execute")
    timeout: float | None = Field(default=None, description="Timeout in seconds")


def _coerce_params(params: Any) -> BashToolInput:
    if isinstance(params, BashToolInput):
        return params
    if isinstance(params, BaseModel):
        return BashToolInput.model_validate(params.model_dump())
    if isinstance(params, Mapping):
        return BashToolInput.model_validate(dict(params))
    return BashToolInput.model_validate({})


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.terminate()


async def _maybe_notify_update(
    callback: AgentToolUpdateCallback | None,
    result: AgentToolResult,
) -> None:
    if callback is None:
        return
    maybe = callback(result)
    if inspect.isawaitable(maybe):
        await maybe


class BashTool:
    name = "bash"
    label = "bash"
    description = (
        "Execute a shell command in the current working directory. Returns stdout and stderr. "
        f"Output is truncated to the last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB. "
        "Non-zero exit codes raise an error."
    )
    parameters = BashToolInput

    def __init__(self, cwd: str) -> None:
        self._cwd = cwd

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id
        arguments = _coerce_params(params)
        if cancellation is not None:
            cancellation.raise_if_cancelled()

        shell = os.environ.get("SHELL", "/bin/bash")
        process = await asyncio.create_subprocess_exec(
            shell,
            "-lc",
            arguments.command,
            cwd=self._cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
            start_new_session=True,
        )

        rolling_chunks: deque[bytes] = deque()
        rolling_bytes = 0
        max_rolling_bytes = DEFAULT_MAX_BYTES * 2
        initial_chunks: list[bytes] = []
        temp_handle = None
        temp_path: str | None = None
        total_bytes = 0

        def handle_chunk(data: bytes) -> None:
            nonlocal rolling_bytes, temp_handle, temp_path, total_bytes
            total_bytes += len(data)

            if temp_handle is None and total_bytes > DEFAULT_MAX_BYTES:
                temp_handle = tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix="bampy-bash-",
                    suffix=".log",
                    delete=False,
                )
                temp_path = temp_handle.name
                for chunk in initial_chunks:
                    temp_handle.write(chunk)

            if temp_handle is None:
                initial_chunks.append(data)
            else:
                temp_handle.write(data)

            rolling_chunks.append(data)
            rolling_bytes += len(data)
            while rolling_bytes > max_rolling_bytes and rolling_chunks:
                removed = rolling_chunks.popleft()
                rolling_bytes -= len(removed)

        async def read_stream(stream: asyncio.StreamReader | None) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    return
                handle_chunk(chunk)
                truncation = truncate_tail(
                    b"".join(rolling_chunks).decode("utf-8", errors="replace")
                )
                await _maybe_notify_update(
                    on_update,
                    AgentToolResult(
                        content=[TextContent(text=truncation.content or "")],
                        details={
                            "truncation": serialize_truncation(truncation) if truncation.truncated else None,
                            "full_output_path": temp_path,
                        },
                    ),
                )

        remove_cancel = None
        if cancellation is not None:
            remove_cancel = cancellation.add_callback(lambda _reason: _kill_process_group(process))

        stdout_task = asyncio.create_task(read_stream(process.stdout))
        stderr_task = asyncio.create_task(read_stream(process.stderr))

        try:
            if arguments.timeout is not None and arguments.timeout > 0:
                await asyncio.wait_for(process.wait(), timeout=arguments.timeout)
            else:
                await process.wait()
            await asyncio.gather(stdout_task, stderr_task)
        except asyncio.TimeoutError as exc:
            _kill_process_group(process)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            text = b"".join(rolling_chunks).decode("utf-8", errors="replace")
            if text:
                text += "\n\n"
            text += f"Command timed out after {arguments.timeout} seconds"
            raise RuntimeError(text) from exc
        finally:
            if remove_cancel is not None:
                remove_cancel()
            if temp_handle is not None:
                temp_handle.flush()
                temp_handle.close()

        if cancellation is not None and cancellation.cancelled:
            raise CancellationError(cancellation.reason or "Command aborted")

        full_text = b"".join(rolling_chunks).decode("utf-8", errors="replace")
        truncation = truncate_tail(full_text)
        output = truncation.content or "(no output)"
        details: dict[str, object] | None = None

        if truncation.truncated:
            details = {
                "truncation": serialize_truncation(truncation),
                "full_output_path": temp_path,
            }
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                output += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line}. "
                    f"Full output: {temp_path}]"
                )
            elif truncation.truncated_by == "lines":
                output += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {temp_path}]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {temp_path}]"
                )

        if process.returncode not in (0, None):
            if output:
                output += "\n\n"
            output += f"Command exited with code {process.returncode}"
            raise RuntimeError(output)

        return AgentToolResult(content=[TextContent(text=output)], details=details)


def create_bash_tool(cwd: str) -> BashTool:
    return BashTool(cwd)


bash_tool = create_bash_tool(Path.cwd().as_posix())
