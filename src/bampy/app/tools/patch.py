"""Built-in patch tool."""

from __future__ import annotations

import asyncio
import os
import shlex
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult
from bampy.ai.types import TextContent

from .file_mutation_queue import file_mutation_locks


class PatchToolInput(BaseModel):
    patch: str = Field(
        min_length=1,
        description="A standard unified diff to apply.",
    )


def _coerce_params(params: Any) -> PatchToolInput:
    if isinstance(params, PatchToolInput):
        return params
    if isinstance(params, BaseModel):
        return PatchToolInput.model_validate(params.model_dump())
    if isinstance(params, Mapping):
        return PatchToolInput.model_validate(dict(params))
    return PatchToolInput.model_validate({})


def _split_patch_path(payload: str) -> tuple[str, str]:
    if "\t" in payload:
        path, suffix = payload.split("\t", 1)
        return path.strip(), f"\t{suffix}"
    try:
        parts = shlex.split(payload)
    except ValueError:
        parts = payload.split()
    if not parts:
        return "", ""
    return parts[0], payload[payload.find(parts[0]) + len(parts[0]) :]


def _strip_diff_prefix(path: str) -> str:
    if path.startswith(("a/", "b/")) and len(path) > 2:
        return path[2:]
    return path


def _container_relative(path: str, container_root: str | None) -> str:
    if not container_root:
        return path
    pure = PurePosixPath(path)
    if not pure.is_absolute():
        return path
    try:
        return pure.relative_to(PurePosixPath(container_root)).as_posix()
    except ValueError:
        return path


def _display_path(path: str | None, prefix: str) -> str:
    if path is None:
        return "/dev/null"
    return f"{prefix}/{path}"


def _normalize_patch_path(raw_path: str, *, container_root: str | None = None) -> str | None:
    path = raw_path.strip()
    if not path or path == "/dev/null":
        return None
    path = _container_relative(path, container_root)
    path = _strip_diff_prefix(path)
    return path


def _validate_relative_patch_path(path: str, cwd: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute():
        raise ValueError(f"Patch path must be relative: {path}")
    if not path or path == ".":
        raise ValueError("Patch path must not be empty")
    if any(part == ".." for part in pure.parts):
        raise ValueError(f"Patch path escapes workspace: {path}")

    root = Path(cwd).resolve()
    resolved = (root / Path(pure.as_posix())).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Patch path escapes workspace: {path}") from exc
    return str(resolved)


def _extract_patch_paths(patch: str, *, cwd: str, container_root: str | None = None) -> set[str]:
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = line.split()
            for raw_path in parts[2:4]:
                path = _normalize_patch_path(raw_path, container_root=container_root)
                if path is not None:
                    paths.add(path)
            continue

        if line.startswith(("--- ", "+++ ")):
            raw_path, _suffix = _split_patch_path(line[4:])
            path = _normalize_patch_path(raw_path, container_root=container_root)
            if path is not None:
                paths.add(path)
            continue

        for prefix in ("rename from ", "rename to ", "copy from ", "copy to "):
            if line.startswith(prefix):
                path = _normalize_patch_path(line[len(prefix) :], container_root=container_root)
                if path is not None:
                    paths.add(path)

    if not paths:
        raise ValueError("Patch does not contain any file paths.")
    return {_validate_relative_patch_path(path, cwd) for path in paths}


def _rewrite_marker_line(line: str, marker: str, container_root: str | None) -> str:
    raw_path, suffix = _split_patch_path(line[len(marker) :])
    normalized = _normalize_patch_path(raw_path, container_root=container_root)
    if normalized is None:
        return line
    prefix = "a" if marker == "--- " else "b"
    rewritten = _display_path(normalized, prefix)
    return f"{marker}{rewritten}{suffix}"


def _rewrite_diff_git_line(line: str, container_root: str | None) -> str:
    try:
        parts = shlex.split(line)
    except ValueError:
        parts = line.split()
    if len(parts) != 4:
        return line

    old_path = _normalize_patch_path(parts[2], container_root=container_root)
    new_path = _normalize_patch_path(parts[3], container_root=container_root)
    return f"diff --git {_display_path(old_path, 'a')} {_display_path(new_path, 'b')}"


def _rewrite_patch_paths(patch: str, *, container_root: str | None = None) -> str:
    if not container_root:
        return patch

    rewritten: list[str] = []
    for line in patch.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        if body.startswith("diff --git "):
            body = _rewrite_diff_git_line(body, container_root)
        elif body.startswith("--- "):
            body = _rewrite_marker_line(body, "--- ", container_root)
        elif body.startswith("+++ "):
            body = _rewrite_marker_line(body, "+++ ", container_root)
        rewritten.append(f"{body}{ending}")
    return "".join(rewritten)


async def _run_git_apply(cwd: str, patch: str, *, check: bool) -> tuple[int, str, str]:
    args = ["git", "apply", "--whitespace=nowarn"]
    if check:
        args.append("--check")
    args.append("-")
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(patch.encode("utf-8"))
    return (
        process.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


class PatchTool:
    name = "patch"
    label = "patch"
    description = "Apply a standard unified diff to files in the workspace."
    parameters = PatchToolInput

    def __init__(self, cwd: str, *, container_root: str | None = None) -> None:
        self._cwd = str(Path(cwd).resolve())
        self._container_root = container_root

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

        patch = _rewrite_patch_paths(arguments.patch, container_root=self._container_root)
        locked_paths = _extract_patch_paths(patch, cwd=self._cwd)

        async with file_mutation_locks(locked_paths):
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            check_code, check_stdout, check_stderr = await _run_git_apply(self._cwd, patch, check=True)
            if check_code != 0:
                message = (check_stderr or check_stdout or "git apply --check failed").strip()
                raise ValueError(f"Patch check failed:\n{message}")

            apply_code, apply_stdout, apply_stderr = await _run_git_apply(self._cwd, patch, check=False)
            if apply_code != 0:
                message = (apply_stderr or apply_stdout or "git apply failed").strip()
                raise RuntimeError(f"Patch apply failed:\n{message}")

        rel_paths = sorted(
            Path(path).resolve().relative_to(Path(self._cwd).resolve()).as_posix()
            for path in locked_paths
        )
        plural = "s" if len(rel_paths) != 1 else ""
        return AgentToolResult(
            content=[TextContent(text=f"Successfully applied patch to {len(rel_paths)} file{plural}: {', '.join(rel_paths)}")],
            details={"paths": rel_paths, "patch": patch},
        )


def create_patch_tool(cwd: str, *, container_root: str | None = None) -> PatchTool:
    return PatchTool(cwd, container_root=container_root)


patch_tool = create_patch_tool(os.getcwd())
