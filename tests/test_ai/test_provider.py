"""Tests for bampy.ai.provider and bampy.ai.providers.registry."""

import pytest

from bampy.ai.provider import (
    ApiProviderEntry,
    get_api_provider,
    register_api_provider,
    unregister_api_provider,
    _registry,
)
from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    Model,
    StopReason,
    StreamOptions,
    UserMessage,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the registry before each test."""
    saved = dict(_registry)
    _registry.clear()
    yield
    _registry.clear()
    _registry.update(saved)


def _make_dummy_stream(model, context, options=None):
    """A dummy stream function that immediately emits done."""
    stream = AssistantMessageEventStream()
    output = AssistantMessage(api=model.api, provider=model.provider, model=model.id)
    stream.push(DoneEvent(reason=StopReason.STOP, message=output))
    return stream


class TestProviderRegistry:
    def test_register_eager(self):
        entry = ApiProviderEntry(
            api="test-api",
            stream=_make_dummy_stream,
            stream_simple=_make_dummy_stream,
        )
        register_api_provider("test-api", entry=entry)

    async def test_get_provider_eager(self):
        entry = ApiProviderEntry(
            api="test-api",
            stream=_make_dummy_stream,
            stream_simple=_make_dummy_stream,
        )
        register_api_provider("test-api", entry=entry)
        result = await get_api_provider("test-api")
        assert result.api == "test-api"

    async def test_get_provider_lazy(self):
        async def loader():
            return ApiProviderEntry(
                api="lazy-api",
                stream=_make_dummy_stream,
                stream_simple=_make_dummy_stream,
            )

        register_api_provider("lazy-api", loader=loader)
        result = await get_api_provider("lazy-api")
        assert result.api == "lazy-api"

    async def test_get_unknown_provider_raises(self):
        with pytest.raises(LookupError, match="No provider registered"):
            await get_api_provider("nonexistent")

    def test_unregister(self):
        entry = ApiProviderEntry(
            api="test-api",
            stream=_make_dummy_stream,
            stream_simple=_make_dummy_stream,
        )
        register_api_provider("test-api", entry=entry)
        unregister_api_provider("test-api")
        assert "test-api" not in _registry


class TestBuiltinRegistration:
    async def test_register_builtins(self):
        from bampy.ai.providers.registry import register_builtin_providers

        register_builtin_providers()
        # Should have anthropic and openai registered
        assert "anthropic-messages" in _registry
        assert "openai-responses" in _registry
