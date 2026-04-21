"""Built-in provider registration."""

from __future__ import annotations

from bampy.ai.api_registry import (
    ApiProviderEntry,
    register_api_provider,
    unregister_api_providers,
)

_BUILTIN_SOURCE_ID = "bampy.ai.providers.builtin"


def _load_anthropic() -> ApiProviderEntry:
    from bampy.ai.providers.anthropic import get_provider_entry

    return get_provider_entry()


def _load_openai() -> ApiProviderEntry:
    from bampy.ai.providers.openai import get_provider_entry

    return get_provider_entry()


def _load_openai_completions() -> ApiProviderEntry:
    from bampy.ai.providers.openai import get_completions_provider_entry

    return get_completions_provider_entry()


def _load_ollama() -> ApiProviderEntry:
    from bampy.ai.providers.ollama import get_provider_entry

    return get_provider_entry()


def _load_google() -> ApiProviderEntry:
    from bampy.ai.providers.gemini import get_provider_entry

    return get_provider_entry()


def register_builtin_providers(*, force: bool = False) -> None:
    """Register all built-in LLM providers (lazy-loaded)."""
    if force:
        unregister_api_providers(_BUILTIN_SOURCE_ID)

    register_api_provider(
        "anthropic-messages",
        loader=_load_anthropic,
        source_id=_BUILTIN_SOURCE_ID,
    )
    register_api_provider(
        "openai-responses",
        loader=_load_openai,
        source_id=_BUILTIN_SOURCE_ID,
    )
    register_api_provider(
        "openai-completions",
        loader=_load_openai_completions,
        source_id=_BUILTIN_SOURCE_ID,
    )
    register_api_provider(
        "ollama-responses",
        loader=_load_ollama,
        source_id=_BUILTIN_SOURCE_ID,
    )
    register_api_provider(
        "google-genai",
        loader=_load_google,
        source_id=_BUILTIN_SOURCE_ID,
    )
