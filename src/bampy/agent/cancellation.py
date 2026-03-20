"""Cancellation primitives for the agent runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class CancellationError(Exception):
    """Raised when work is cancelled through a ``CancellationToken``."""


class CancellationToken:
    """A lightweight, asyncio-friendly cancellation primitive.

    This is currently used by the layer2 runtime itself and passed down to tool
    execution hooks. Cancellation is best-effort until layer1 providers accept a
    propagated abort signal.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._callbacks: list[Callable[[str | None], None]] = []
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str | None = None) -> None:
        if self._event.is_set():
            return
        self._reason = reason
        self._event.set()
        for callback in list(self._callbacks):
            callback(reason)

    async def wait(self) -> None:
        await self._event.wait()

    def add_callback(self, callback: Callable[[str | None], None]) -> Callable[[], None]:
        self._callbacks.append(callback)
        if self.cancelled:
            callback(self._reason)

        def remove() -> None:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

        return remove

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancellationError(self._reason or "Operation cancelled")

