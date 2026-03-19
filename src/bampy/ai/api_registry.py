"""API provider registry for the bampy AI layer.

it owns provider registration and returns stream functions keyed by API name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from bampy.ai.types import Context, Model, SimpleStreamOptions, StreamOptions


class StreamFunction(Protocol):
    """Protocol for provider stream functions."""

    def __call__(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | SimpleStreamOptions | None = None,
    ): ...


@dataclass(slots=True)
class ApiProviderEntry:
    """Provider implementation bound to a specific API name."""

    api: str
    stream: StreamFunction
    stream_simple: StreamFunction


@dataclass(slots=True)
class LazyApiProvider:
    """Provider wrapper loaded on first use."""

    api: str
    loader: Callable[[], ApiProviderEntry]
    _loaded: ApiProviderEntry | None = field(default=None, repr=False)

    def load(self) -> ApiProviderEntry:
        if self._loaded is None:
            self._loaded = self.loader()
        return self._loaded


@dataclass(slots=True)
class RegisteredApiProvider:
    provider: ApiProviderEntry | LazyApiProvider
    source_id: str | None = None


_registry: dict[str, RegisteredApiProvider] = {}


def _wrap_stream(api: str, stream: StreamFunction) -> StreamFunction:
    def wrapped(
        model: Model,
        context: Context,
        options: StreamOptions | SimpleStreamOptions | None = None,
    ):
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return stream(model, context, options)

    return wrapped


def register_api_provider(
    api: str,
    *,
    entry: ApiProviderEntry | None = None,
    loader: Callable[[], ApiProviderEntry] | None = None,
    source_id: str | None = None,
) -> None:
    """Register an eager or lazy provider for the given API."""
    if entry is None and loader is None:
        raise ValueError("Either entry or loader must be provided")

    provider: ApiProviderEntry | LazyApiProvider
    if entry is not None:
        provider = ApiProviderEntry(
            api=api,
            stream=_wrap_stream(api, entry.stream),
            stream_simple=_wrap_stream(api, entry.stream_simple),
        )
    else:
        provider = LazyApiProvider(api=api, loader=loader)  # type: ignore[arg-type]

    _registry[api] = RegisteredApiProvider(provider=provider, source_id=source_id)


def unregister_api_provider(api: str) -> None:
    _registry.pop(api, None)


def unregister_api_providers(source_id: str) -> None:
    for api, entry in list(_registry.items()):
        if entry.source_id == source_id:
            _registry.pop(api, None)


def clear_api_providers() -> None:
    _registry.clear()


def get_api_provider(api: str) -> ApiProviderEntry:
    """Resolve a provider by API name, loading lazily when needed."""
    registered = _registry.get(api)
    if registered is None:
        raise LookupError(f"No provider registered for API '{api}'. Available: {list(_registry)}")

    provider = registered.provider
    if isinstance(provider, LazyApiProvider):
        loaded = provider.load()
        loaded = ApiProviderEntry(
            api=api,
            stream=_wrap_stream(api, loaded.stream),
            stream_simple=_wrap_stream(api, loaded.stream_simple),
        )
        registered.provider = loaded
        return loaded

    return provider


def get_api_providers() -> list[ApiProviderEntry]:
    return [get_api_provider(api) for api in list(_registry)]
