"""Tests for the API provider registry."""

import pytest

from bampy.ai.api_registry import (
    ApiProviderEntry,
    _registry,
    clear_api_providers,
    get_api_provider,
    get_api_providers,
    register_api_provider,
    unregister_api_provider,
)
from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AssistantMessage,
    DoneEvent,
    StopReason,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the registry before each test."""
    saved = dict(_registry)
    clear_api_providers()
    yield
    clear_api_providers()
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

    def test_get_provider_eager(self):
        entry = ApiProviderEntry(
            api="test-api",
            stream=_make_dummy_stream,
            stream_simple=_make_dummy_stream,
        )
        register_api_provider("test-api", entry=entry)
        result = get_api_provider("test-api")
        assert result.api == "test-api"

    def test_get_provider_lazy(self):
        def loader():
            return ApiProviderEntry(
                api="lazy-api",
                stream=_make_dummy_stream,
                stream_simple=_make_dummy_stream,
            )

        register_api_provider("lazy-api", loader=loader)
        result = get_api_provider("lazy-api")
        assert result.api == "lazy-api"

    def test_get_unknown_provider_raises(self):
        with pytest.raises(LookupError, match="No provider registered"):
            get_api_provider("nonexistent")

    def test_unregister(self):
        entry = ApiProviderEntry(
            api="test-api",
            stream=_make_dummy_stream,
            stream_simple=_make_dummy_stream,
        )
        register_api_provider("test-api", entry=entry)
        unregister_api_provider("test-api")
        assert "test-api" not in _registry

    def test_list_providers(self):
        register_api_provider(
            "test-api",
            entry=ApiProviderEntry(
                api="test-api",
                stream=_make_dummy_stream,
                stream_simple=_make_dummy_stream,
            ),
        )
        providers = get_api_providers()
        assert len(providers) == 1
        assert providers[0].api == "test-api"


class TestBuiltinRegistration:
    def test_register_builtins(self):
        from bampy.ai.providers.registry import register_builtin_providers

        register_builtin_providers()
        assert "anthropic-messages" in _registry
        assert "openai-responses" in _registry
        assert "google-genai" in _registry
