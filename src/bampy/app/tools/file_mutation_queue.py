"""Serialize file mutations that target the same path."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class _LockState:
    lock: asyncio.Lock
    references: int = 0


_LOCKS: dict[str, _LockState] = {}
_LOCKS_GUARD = asyncio.Lock()


def _queue_key(path: str | Path) -> str:
    return str(Path(path).resolve(strict=False))


async def _borrow_lock(key: str) -> _LockState:
    async with _LOCKS_GUARD:
        state = _LOCKS.get(key)
        if state is None:
            state = _LockState(lock=asyncio.Lock())
            _LOCKS[key] = state
        state.references += 1
        return state


async def _return_lock(key: str, state: _LockState) -> None:
    async with _LOCKS_GUARD:
        state.references -= 1
        if state.references <= 0 and not state.lock.locked():
            _LOCKS.pop(key, None)


@asynccontextmanager
async def file_mutation_locks(paths: list[str | Path] | set[str | Path]) -> AsyncIterator[None]:
    """Hold mutation locks for one or more paths in a stable order."""

    keys = sorted({_queue_key(path) for path in paths})
    acquired: list[tuple[str, _LockState]] = []
    try:
        for key in keys:
            state = await _borrow_lock(key)
            try:
                await state.lock.acquire()
            except BaseException:
                await _return_lock(key, state)
                raise
            acquired.append((key, state))
        yield
    finally:
        for key, state in reversed(acquired):
            state.lock.release()
            await _return_lock(key, state)


async def with_file_mutation_queue(path: str | Path, fn: Callable[[], Awaitable[T]]) -> T:
    """Run one file mutation after any earlier mutations for the same file."""

    async with file_mutation_locks({path}):
        return await fn()
