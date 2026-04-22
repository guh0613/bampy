"""Model registry and built-in model definitions."""

from __future__ import annotations

from collections.abc import Iterable

from bampy.ai.types import Model, ModelCost, OpenAIChatCompat, Usage, UsageCost


def _cost(
    *,
    input: float,
    output: float,
    cache_read: float = 0.0,
    cache_write: float = 0.0,
) -> ModelCost:
    return ModelCost(
        input=input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
    )


def _model(
    *,
    id: str,
    name: str,
    api: str,
    provider: str,
    reasoning: bool,
    input_types: list[str] | None = None,
    base_url: str = "",
    context_window: int,
    max_tokens: int,
    cost: ModelCost,
    openai_chat_compat: OpenAIChatCompat | None = None,
) -> Model:
    return Model(
        id=id,
        name=name,
        api=api,
        provider=provider,
        base_url=base_url,
        reasoning=reasoning,
        input_types=input_types or ["text", "image"],
        context_window=context_window,
        max_tokens=max_tokens,
        cost=cost,
        openai_chat_compat=openai_chat_compat,
    )


_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_OPENAI_BASE_URL = "https://api.openai.com/v1"
_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"


BUILTIN_MODELS: dict[str, tuple[Model, ...]] = {
    "anthropic": (
        _model(
            id="claude-haiku-4-5",
            name="Claude Haiku 4.5 (latest)",
            api="anthropic-messages",
            provider="anthropic",
            base_url=_ANTHROPIC_BASE_URL,
            reasoning=True,
            context_window=200_000,
            max_tokens=64_000,
            cost=_cost(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25),
        ),
        _model(
            id="claude-haiku-4-5-20251001",
            name="Claude Haiku 4.5",
            api="anthropic-messages",
            provider="anthropic",
            base_url=_ANTHROPIC_BASE_URL,
            reasoning=True,
            context_window=200_000,
            max_tokens=64_000,
            cost=_cost(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25),
        ),
        _model(
            id="claude-opus-4-1",
            name="Claude Opus 4.1 (latest)",
            api="anthropic-messages",
            provider="anthropic",
            base_url=_ANTHROPIC_BASE_URL,
            reasoning=True,
            context_window=200_000,
            max_tokens=32_000,
            cost=_cost(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
        ),
        _model(
            id="claude-opus-4-5",
            name="Claude Opus 4.5 (latest)",
            api="anthropic-messages",
            provider="anthropic",
            base_url=_ANTHROPIC_BASE_URL,
            reasoning=True,
            context_window=200_000,
            max_tokens=64_000,
            cost=_cost(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25),
        ),
        _model(
            id="claude-opus-4-6",
            name="Claude Opus 4.6",
            api="anthropic-messages",
            provider="anthropic",
            base_url=_ANTHROPIC_BASE_URL,
            reasoning=True,
            context_window=1_000_000,
            max_tokens=128_000,
            cost=_cost(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25),
        ),
        _model(
            id="claude-sonnet-4",
            name="Claude Sonnet 4 (latest)",
            api="anthropic-messages",
            provider="anthropic",
            base_url=_ANTHROPIC_BASE_URL,
            reasoning=True,
            context_window=200_000,
            max_tokens=64_000,
            cost=_cost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        ),
        _model(
            id="claude-sonnet-4-5",
            name="Claude Sonnet 4.5 (latest)",
            api="anthropic-messages",
            provider="anthropic",
            base_url=_ANTHROPIC_BASE_URL,
            reasoning=True,
            context_window=200_000,
            max_tokens=64_000,
            cost=_cost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        ),
        _model(
            id="claude-sonnet-4-5-20250929",
            name="Claude Sonnet 4.5",
            api="anthropic-messages",
            provider="anthropic",
            base_url=_ANTHROPIC_BASE_URL,
            reasoning=True,
            context_window=200_000,
            max_tokens=64_000,
            cost=_cost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        ),
        _model(
            id="claude-sonnet-4-6",
            name="Claude Sonnet 4.6",
            api="anthropic-messages",
            provider="anthropic",
            base_url=_ANTHROPIC_BASE_URL,
            reasoning=True,
            context_window=1_000_000,
            max_tokens=64_000,
            cost=_cost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        ),
    ),
    "google": (
        _model(
            id="gemini-2.5-pro",
            name="Gemini 2.5 Pro",
            api="google-genai",
            provider="google",
            base_url=_GOOGLE_BASE_URL,
            reasoning=True,
            context_window=1_048_576,
            max_tokens=65_536,
            cost=_cost(input=1.25, output=10.0, cache_read=0.31),
        ),
        _model(
            id="gemini-2.5-flash",
            name="Gemini 2.5 Flash",
            api="google-genai",
            provider="google",
            base_url=_GOOGLE_BASE_URL,
            reasoning=True,
            context_window=1_048_576,
            max_tokens=65_536,
            cost=_cost(input=0.30, output=2.50, cache_read=0.075),
        ),
        _model(
            id="gemini-2.5-flash-lite",
            name="Gemini 2.5 Flash Lite",
            api="google-genai",
            provider="google",
            base_url=_GOOGLE_BASE_URL,
            reasoning=True,
            context_window=1_048_576,
            max_tokens=65_536,
            cost=_cost(input=0.10, output=0.40, cache_read=0.025),
        ),
        _model(
            id="gemini-3-flash-preview",
            name="Gemini 3 Flash Preview",
            api="google-genai",
            provider="google",
            base_url=_GOOGLE_BASE_URL,
            reasoning=True,
            context_window=1_048_576,
            max_tokens=65_536,
            cost=_cost(input=0.50, output=3.0, cache_read=0.05),
        ),
        _model(
            id="gemini-3-pro-preview",
            name="Gemini 3 Pro Preview",
            api="google-genai",
            provider="google",
            base_url=_GOOGLE_BASE_URL,
            reasoning=True,
            context_window=1_000_000,
            max_tokens=64_000,
            cost=_cost(input=2.0, output=12.0, cache_read=0.20),
        ),
        _model(
            id="gemini-3.1-flash-lite-preview",
            name="Gemini 3.1 Flash Lite Preview",
            api="google-genai",
            provider="google",
            base_url=_GOOGLE_BASE_URL,
            reasoning=True,
            context_window=1_048_576,
            max_tokens=65_536,
            cost=_cost(input=0.25, output=1.50, cache_read=0.025, cache_write=1.0),
        ),
        _model(
            id="gemini-3.1-pro-preview",
            name="Gemini 3.1 Pro Preview",
            api="google-genai",
            provider="google",
            base_url=_GOOGLE_BASE_URL,
            reasoning=True,
            context_window=1_048_576,
            max_tokens=65_536,
            cost=_cost(input=2.0, output=12.0, cache_read=0.20),
        ),
    ),
    "ollama": (
        _model(
            id="gemini-3-flash",
            name="Gemini 3 Flash",
            api="ollama-responses",
            provider="ollama",
            reasoning=True,
            context_window=1_048_576,
            max_tokens=65_536,
            cost=_cost(input=0.50, output=3.0, cache_read=0.05),
        ),
    ),
    "opencode-go": (
        _model(
            id="kimi-k2.6",
            name="Kimi K2.6",
            api="openai-completions",
            provider="opencode-go",
            base_url=_OPENCODE_GO_BASE_URL,
            reasoning=True,
            context_window=262_144,
            max_tokens=65_536,
            cost=_cost(input=0.95, output=4.0, cache_read=0.16),
            openai_chat_compat=OpenAIChatCompat(
                max_tokens_field="max_tokens",
                replay_thinking_field="reasoning_content",
                stream_reasoning_fields=["reasoning_content"],
                supports_reasoning_effort=False,
                thinking_param="kimi",
                thinking_default_enabled=True,
                thinking_tool_choice=["auto", "none"],
            ),
        ),
    ),
    "openai": (
        _model(
            id="gpt-4.1",
            name="GPT-4.1",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=False,
            context_window=1_047_576,
            max_tokens=32_768,
            cost=_cost(input=2.0, output=8.0, cache_read=0.5),
        ),
        _model(
            id="gpt-4.1-mini",
            name="GPT-4.1 Mini",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=False,
            context_window=1_047_576,
            max_tokens=32_768,
            cost=_cost(input=0.4, output=1.6, cache_read=0.1),
        ),
        _model(
            id="gpt-4.1-nano",
            name="GPT-4.1 Nano",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=False,
            context_window=1_047_576,
            max_tokens=32_768,
            cost=_cost(input=0.1, output=0.4, cache_read=0.025),
        ),
        _model(
            id="gpt-5",
            name="GPT-5",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=1.25, output=10.0, cache_read=0.125),
        ),
        _model(
            id="gpt-5-chat-latest",
            name="GPT-5 Chat Latest",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=False,
            context_window=128_000,
            max_tokens=16_384,
            cost=_cost(input=1.25, output=10.0, cache_read=0.125),
        ),
        _model(
            id="gpt-5-codex",
            name="GPT-5-Codex",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=1.25, output=10.0, cache_read=0.125),
        ),
        _model(
            id="gpt-5-mini",
            name="GPT-5 Mini",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=0.25, output=2.0, cache_read=0.025),
        ),
        _model(
            id="gpt-5-nano",
            name="GPT-5 Nano",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=0.05, output=0.4, cache_read=0.005),
        ),
        _model(
            id="gpt-5-pro",
            name="GPT-5 Pro",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=272_000,
            cost=_cost(input=15.0, output=120.0),
        ),
        _model(
            id="gpt-5.1",
            name="GPT-5.1",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=1.25, output=10.0, cache_read=0.13),
        ),
        _model(
            id="gpt-5.1-codex",
            name="GPT-5.1 Codex",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=1.25, output=10.0, cache_read=0.125),
        ),
        _model(
            id="gpt-5.1-codex-max",
            name="GPT-5.1 Codex Max",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=1.25, output=10.0, cache_read=0.125),
        ),
        _model(
            id="gpt-5.1-codex-mini",
            name="GPT-5.1 Codex Mini",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=0.25, output=2.0, cache_read=0.025),
        ),
        _model(
            id="gpt-5.2",
            name="GPT-5.2",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=1.75, output=14.0, cache_read=0.175),
        ),
        _model(
            id="gpt-5.2-codex",
            name="GPT-5.2 Codex",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=1.75, output=14.0, cache_read=0.175),
        ),
        _model(
            id="gpt-5.2-pro",
            name="GPT-5.2 Pro",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=21.0, output=168.0),
        ),
        _model(
            id="gpt-5.3-codex",
            name="GPT-5.3 Codex",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=1.75, output=14.0, cache_read=0.175),
        ),
        _model(
            id="gpt-5.4",
            name="GPT-5.4",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=272_000,
            max_tokens=128_000,
            cost=_cost(input=2.5, output=15.0, cache_read=0.25),
        ),
        _model(
            id="gpt-5.4-mini",
            name="GPT-5.4 Mini",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=0.75, output=4.5, cache_read=0.075),
        ),
        _model(
            id="gpt-5.4-nano",
            name="GPT-5.4 Nano",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=_cost(input=0.2, output=1.25, cache_read=0.02),
        ),
        _model(
            id="gpt-5.4-pro",
            name="GPT-5.4 Pro",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=1_050_000,
            max_tokens=128_000,
            cost=_cost(input=30.0, output=180.0),
        ),
        _model(
            id="o3",
            name="o3",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=200_000,
            max_tokens=100_000,
            cost=_cost(input=2.0, output=8.0, cache_read=0.5),
        ),
        _model(
            id="o3-pro",
            name="o3 Pro",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=200_000,
            max_tokens=100_000,
            cost=_cost(input=20.0, output=80.0),
        ),
        _model(
            id="o4-mini",
            name="o4-mini",
            api="openai-responses",
            provider="openai",
            base_url=_OPENAI_BASE_URL,
            reasoning=True,
            context_window=200_000,
            max_tokens=100_000,
            cost=_cost(input=1.1, output=4.4, cache_read=0.275),
        ),
    ),
}

def _iter_builtin_models() -> Iterable[Model]:
    for models in BUILTIN_MODELS.values():
        yield from models


MODELS: list[Model] = list(_iter_builtin_models())

_model_map: dict[str, dict[str, Model]] = {
    provider: {model.id: model for model in models}
    for provider, models in BUILTIN_MODELS.items()
}


def register_model(model: Model) -> None:
    """Register or override a model definition."""
    _model_map.setdefault(model.provider, {})[model.id] = model


def get_model(model_id: str, provider: str | None = None) -> Model | None:
    """Look up a model by id, optionally scoped to a provider."""
    if provider is not None:
        return _model_map.get(provider, {}).get(model_id)

    for provider_models in _model_map.values():
        model = provider_models.get(model_id)
        if model is not None:
            return model
    return None


def get_models(provider: str | None = None) -> list[Model]:
    """List registered models, optionally filtered by provider."""
    if provider is not None:
        return list(_model_map.get(provider, {}).values())
    return [model for provider_models in _model_map.values() for model in provider_models.values()]


def get_providers() -> list[str]:
    """List all providers that currently have registered models."""
    return list(_model_map)


def calculate_cost(model: Model, usage: Usage) -> UsageCost:
    """Calculate dollar cost from token usage and model pricing."""
    input_cost = (model.cost.input / 1_000_000) * usage.input
    output_cost = (model.cost.output / 1_000_000) * usage.output
    cache_read_cost = (model.cost.cache_read / 1_000_000) * usage.cache_read
    cache_write_cost = (model.cost.cache_write / 1_000_000) * usage.cache_write
    return UsageCost(
        input=input_cost,
        output=output_cost,
        cache_read=cache_read_cost,
        cache_write=cache_write_cost,
        total=input_cost + output_cost + cache_read_cost + cache_write_cost,
    )


def supports_xhigh(model: Model | None) -> bool:
    """Return whether a model family supports ``xhigh`` reasoning."""
    if model is None:
        return False

    model_id = model.id
    return any(
        hint in model_id
        for hint in (
            "gpt-5.1-codex-max",
            "gpt-5.2",
            "gpt-5.3",
            "gpt-5.4",
            "claude-opus-4-6",
            "claude-opus-4.6",
        )
    )


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """Check whether two model objects represent the same provider/model pair."""
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider == b.provider
