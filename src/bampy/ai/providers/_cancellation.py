"""Cancellation helpers shared by provider adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import AssistantMessage, ErrorEvent, StopReason


def cancellation_reason(options: Any) -> str | None:
    """Best-effort extract a human-readable cancellation reason."""
    cancellation = getattr(options, "cancellation", None)
    if cancellation is None:
        return None
    reason = getattr(cancellation, "reason", None)
    return str(reason) if reason is not None else None


def spawn_provider_task(
    *,
    event_stream: AssistantMessageEventStream,
    output: AssistantMessage,
    options: Any,
    runner: Callable[[], Awaitable[None]],
) -> asyncio.Task[None]:
    """Spawn a provider background task bound to an optional cancellation token."""
    remove_callback: Callable[[], None] | None = None

    async def _runner() -> None:
        nonlocal remove_callback
        try:
            await runner()
        except asyncio.CancelledError:
            output.stop_reason = StopReason.ABORTED
            output.error_message = cancellation_reason(options) or "Operation cancelled"
            event_stream.push(ErrorEvent(reason=StopReason.ABORTED, error=output))
            event_stream.end(output)
        finally:
            if remove_callback is not None:
                remove_callback()

    task = asyncio.get_running_loop().create_task(_runner())

    cancellation = getattr(options, "cancellation", None)
    if cancellation is not None and hasattr(cancellation, "add_callback"):
        remove_callback = cancellation.add_callback(lambda reason: task.cancel())

    return task
