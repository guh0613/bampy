"""Event stream primitives and top-level streaming entrypoints."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Generic, TypeVar

from bampy.ai.api_registry import get_api_provider
from bampy.ai.providers.registry import register_builtin_providers
from bampy.ai.types import Context, Model, SimpleStreamOptions, StreamOptions

T = TypeVar("T")
R = TypeVar("R")

_SENTINEL = object()


class EventStream(Generic[T, R]):
    """Async-iterable stream of events ``T`` that resolves to a final result ``R``.

    Usage::

        stream = EventStream(is_complete=..., extract_result=...)

        # Producer (typically in a background task)
        stream.push(event1)
        stream.push(event2)
        stream.end(result)  # or let is_complete auto-detect

        # Consumer
        async for event in stream:
            handle(event)
        final = await stream.result()
    """

    def __init__(
        self,
        is_complete: Callable[[T], bool] | None = None,
        extract_result: Callable[[T], R] | None = None,
    ) -> None:
        self._queue: asyncio.Queue[T | object] = asyncio.Queue()
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._ended = False
        self._stored_result: R | None = None
        # Lazy future — created on first access so the stream can be
        # instantiated before the event loop is running.
        self._result_future: asyncio.Future[R] | None = None

    # -- Producer API -------------------------------------------------------

    def push(self, event: T) -> None:
        """Enqueue an event.  If *is_complete* returns ``True``, the stream
        is automatically ended with the extracted result."""
        if self._ended:
            return
        self._queue.put_nowait(event)
        if self._is_complete is not None and self._is_complete(event):
            result = self._extract_result(event) if self._extract_result else None
            self.end(result)  # type: ignore[arg-type]

    def end(self, result: R | None = None) -> None:
        """Signal the end of the stream."""
        if self._ended:
            return
        self._ended = True
        self._stored_result = result
        self._queue.put_nowait(_SENTINEL)
        if self._result_future is not None and not self._result_future.done():
            if result is not None:
                self._result_future.set_result(result)

    def error(self, exc: BaseException) -> None:
        """Signal an error and terminate the stream."""
        if self._ended:
            return
        self._ended = True
        self._queue.put_nowait(_SENTINEL)
        if self._result_future is not None and not self._result_future.done():
            self._result_future.set_exception(exc)

    # -- Consumer API -------------------------------------------------------

    async def result(self) -> R:
        """Await the final result of the stream."""
        future = self._ensure_future()
        return await future

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        item = await self._queue.get()
        if item is _SENTINEL:
            raise StopAsyncIteration
        return item  # type: ignore[return-value]

    # -- Internal -----------------------------------------------------------

    def _ensure_future(self) -> asyncio.Future[R]:
        if self._result_future is None:
            loop = asyncio.get_running_loop()
            self._result_future = loop.create_future()
            # If end() was already called before anyone awaited result()
            if self._ended and self._stored_result is not None:
                self._result_future.set_result(self._stored_result)
        return self._result_future


# ---------------------------------------------------------------------------
# Top-level streaming entrypoints
# ---------------------------------------------------------------------------


def ensure_builtin_providers_registered() -> None:
    """Register built-in providers.

    Registration is intentionally lazy so importing this module does not mutate
    global registry state during test discovery or application startup.
    """
    register_builtin_providers()


def _resolve_api_provider(api: str):
    try:
        return get_api_provider(api)
    except LookupError:
        ensure_builtin_providers_registered()
    try:
        return get_api_provider(api)
    except LookupError as exc:
        raise LookupError(
            f"No provider registered for API '{api}'. "
            "Built-in providers are registered lazily on first use. "
            "If this is a custom API, register it explicitly."
        ) from exc


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
):
    """Stream an LLM response using the provider registered for ``model.api``."""
    provider = _resolve_api_provider(model.api)
    return provider.stream(model, context, options)


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
):
    """Stream and await the final assistant message."""
    return await stream(model, context, options).result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
):
    """Stream with simplified provider-agnostic options."""
    provider = _resolve_api_provider(model.api)
    return provider.stream_simple(model, context, options)


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
):
    """Stream with simplified options and await the final message."""
    return await stream_simple(model, context, options).result()


# ---------------------------------------------------------------------------
# Specialised stream for assistant messages
# ---------------------------------------------------------------------------

from bampy.ai.types import (  # noqa: E402
    AssistantMessage,
    AssistantMessageEvent,
    DoneEvent,
    ErrorEvent,
)


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """Event stream specialised for LLM assistant responses."""

    def __init__(self) -> None:
        super().__init__(
            is_complete=_is_complete,
            extract_result=_extract_result,
        )


def _is_complete(event: AssistantMessageEvent) -> bool:
    return isinstance(event, (DoneEvent, ErrorEvent))


def _extract_result(event: AssistantMessageEvent) -> AssistantMessage:
    if isinstance(event, DoneEvent):
        return event.message
    if isinstance(event, ErrorEvent):
        return event.error
    raise ValueError(f"Cannot extract result from event type: {type(event)}")
