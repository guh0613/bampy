"""Built-in provider registration with lazy loading.

"""

from __future__ import annotations

from bampy.ai.provider import ApiProviderEntry, register_api_provider


async def _load_anthropic() -> ApiProviderEntry:
    from bampy.ai.providers.anthropic import get_provider_entry
    return get_provider_entry()


async def _load_openai() -> ApiProviderEntry:
    from bampy.ai.providers.openai import get_provider_entry
    return get_provider_entry()


async def _load_google() -> ApiProviderEntry:
    from bampy.ai.providers.gemini import get_provider_entry
    return get_provider_entry()


def register_builtin_providers() -> None:
    """Register all built-in LLM providers (lazy-loaded)."""
    register_api_provider("anthropic-messages", loader=_load_anthropic)
    register_api_provider("openai-responses", loader=_load_openai)
    register_api_provider("google-genai", loader=_load_google)
