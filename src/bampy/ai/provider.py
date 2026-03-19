"""Provider registry and top-level streaming interface.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
    ThinkingLevel,
)


# ---------------------------------------------------------------------------
# StreamFunction protocol
# ---------------------------------------------------------------------------

class StreamFunction(Protocol):
    """Protocol for provider stream functions."""

    def __call__(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...


# ---------------------------------------------------------------------------
# Provider entry
# ---------------------------------------------------------------------------

@dataclass
class ApiProviderEntry:
    api: str
    stream: StreamFunction
    stream_simple: StreamFunction


# ---------------------------------------------------------------------------
# Lazy loading support
# ---------------------------------------------------------------------------

@dataclass
class LazyApiProvider:
    """Wraps a provider that is loaded on first use."""

    api: str
    _loader: Callable[[], Awaitable[ApiProviderEntry]]
    _loaded: ApiProviderEntry | None = field(default=None, repr=False)
    _loading: asyncio.Task[ApiProviderEntry] | None = field(default=None, repr=False)

    async def load(self) -> ApiProviderEntry:
        if self._loaded is not None:
            return self._loaded
        if self._loading is None:
            self._loading = asyncio.ensure_future(self._loader())
        self._loaded = await self._loading
        return self._loaded


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[str, ApiProviderEntry | LazyApiProvider] = {}


def register_api_provider(
    api: str,
    *,
    entry: ApiProviderEntry | None = None,
    loader: Callable[[], Awaitable[ApiProviderEntry]] | None = None,
) -> None:
    """Register a provider for the given API type.

    Pass *entry* for eager registration, or *loader* for lazy loading.
    """
    if entry is not None:
        _registry[api] = entry
    elif loader is not None:
        _registry[api] = LazyApiProvider(api=api, _loader=loader)
    else:
        raise ValueError("Either entry or loader must be provided")


def unregister_api_provider(api: str) -> None:
    _registry.pop(api, None)


async def get_api_provider(api: str) -> ApiProviderEntry:
    """Resolve a provider by API type, loading lazily if needed."""
    provider = _registry.get(api)
    if provider is None:
        raise LookupError(
            f"No provider registered for API '{api}'. "
            f"Available: {list(_registry.keys())}. "
            f"Did you call register_builtin_providers()?"
        )
    if isinstance(provider, LazyApiProvider):
        entry = await provider.load()
        _registry[api] = entry  # replace lazy with loaded
        return entry
    return provider


# ---------------------------------------------------------------------------
# Top-level streaming interface
# ---------------------------------------------------------------------------

_DEFAULT_THINKING_BUDGETS: dict[ThinkingLevel, int] = {
    ThinkingLevel.MINIMAL: 1024,
    ThinkingLevel.LOW: 2048,
    ThinkingLevel.MEDIUM: 8192,
    ThinkingLevel.HIGH: 16384,
    ThinkingLevel.XHIGH: 16384,
}


def _build_base_options(
    model: Model,
    options: SimpleStreamOptions | None,
    api_key: str | None = None,
) -> StreamOptions:
    """Convert SimpleStreamOptions → StreamOptions (without reasoning)."""
    if options is None:
        return StreamOptions(api_key=api_key)
    return StreamOptions(
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        api_key=options.api_key or api_key,
        max_retry_delay_ms=options.max_retry_delay_ms,
        headers=options.headers,
    )


async def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream an LLM response using the raw provider API."""
    provider = await get_api_provider(model.api)
    return provider.stream(model, context, options)


async def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream with simplified options (auto-maps reasoning to provider config)."""
    provider = await get_api_provider(model.api)
    return provider.stream_simple(model, context, options)


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """Stream and await the final result."""
    s = await stream(model, context, options)
    return await s.result()


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """Stream (simple) and await the final result."""
    s = await stream_simple(model, context, options)
    return await s.result()
