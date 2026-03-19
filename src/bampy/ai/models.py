"""Model registry and built-in model definitions.
"""

from __future__ import annotations

from bampy.ai.types import Model, ModelCost, Usage, UsageCost

# ---------------------------------------------------------------------------
# Built-in model definitions
# ---------------------------------------------------------------------------

MODELS: list[Model] = [
    # -- Anthropic ────────────────────────────────────────────────────────────
    Model(
        id="claude-opus-4-6",
        name="Claude Opus 4.6",
        api="anthropic-messages",
        provider="anthropic",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200_000,
        max_tokens=32_000,
        cost=ModelCost(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
    ),
    Model(
        id="claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        api="anthropic-messages",
        provider="anthropic",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200_000,
        max_tokens=16_000,
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
    ),
    Model(
        id="claude-sonnet-4-5-20250929",
        name="Claude Sonnet 4.5",
        api="anthropic-messages",
        provider="anthropic",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200_000,
        max_tokens=16_000,
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
    ),
    Model(
        id="claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        api="anthropic-messages",
        provider="anthropic",
        reasoning=False,
        input_types=["text", "image"],
        context_window=200_000,
        max_tokens=8192,
        cost=ModelCost(input=0.8, output=4.0, cache_read=0.08, cache_write=1.0),
    ),
    # -- Google (Gemini) ─────────────────────────────────────────────────────
    Model(
        id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        api="google-genai",
        provider="google",
        reasoning=True,
        input_types=["text", "image"],
        context_window=1_048_576,
        max_tokens=65_536,
        cost=ModelCost(input=1.25, output=10.0, cache_read=0.125),
    ),
    Model(
        id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        api="google-genai",
        provider="google",
        reasoning=True,
        input_types=["text", "image"],
        context_window=1_048_576,
        max_tokens=65_536,
        cost=ModelCost(input=0.30, output=2.50, cache_read=0.03),
    ),
    Model(
        id="gemini-2.5-flash-lite",
        name="Gemini 2.5 Flash-Lite",
        api="google-genai",
        provider="google",
        reasoning=False,
        input_types=["text", "image"],
        context_window=1_048_576,
        max_tokens=65_536,
        cost=ModelCost(input=0.10, output=0.40, cache_read=0.01),
    ),
    Model(
        id="gemini-3.1-pro-preview",
        name="Gemini 3.1 Pro",
        api="google-genai",
        provider="google",
        reasoning=True,
        input_types=["text", "image"],
        context_window=1_048_576,
        max_tokens=65_536,
        cost=ModelCost(input=2.0, output=12.0, cache_read=0.20),
    ),
    Model(
        id="gemini-3-flash-preview",
        name="Gemini 3 Flash",
        api="google-genai",
        provider="google",
        reasoning=True,
        input_types=["text", "image"],
        context_window=1_048_576,
        max_tokens=65_536,
        cost=ModelCost(input=0.30, output=2.50, cache_read=0.03),
    ),
    Model(
        id="gemini-3.1-flash-lite-preview",
        name="Gemini 3.1 Flash-Lite",
        api="google-genai",
        provider="google",
        reasoning=False,
        input_types=["text", "image"],
        context_window=1_048_576,
        max_tokens=65_536,
        cost=ModelCost(input=0.25, output=1.50, cache_read=0.025),
    ),
    # -- OpenAI (GPT-5.4 flagship) ─────────────────────────────────────────
    Model(
        id="gpt-5.4",
        name="GPT-5.4",
        api="openai-responses",
        provider="openai",
        reasoning=True,
        input_types=["text", "image"],
        context_window=272_000,
        max_tokens=128_000,
        cost=ModelCost(input=2.5, output=15.0, cache_read=0.25),
    ),
    Model(
        id="gpt-5.4-mini",
        name="GPT-5.4 Mini",
        api="openai-responses",
        provider="openai",
        reasoning=True,
        input_types=["text", "image"],
        context_window=400_000,
        max_tokens=128_000,
        cost=ModelCost(input=0.75, output=4.5, cache_read=0.075),
    ),
    Model(
        id="gpt-5.4-nano",
        name="GPT-5.4 Nano",
        api="openai-responses",
        provider="openai",
        reasoning=True,
        input_types=["text", "image"],
        context_window=400_000,
        max_tokens=128_000,
        cost=ModelCost(input=0.2, output=1.25, cache_read=0.02),
    ),
    # -- OpenAI (GPT-5 base) ────────────────────────────────────────────────
    Model(
        id="gpt-5",
        name="GPT-5",
        api="openai-responses",
        provider="openai",
        reasoning=True,
        input_types=["text", "image"],
        context_window=400_000,
        max_tokens=128_000,
        cost=ModelCost(input=1.25, output=10.0, cache_read=0.125),
    ),
    Model(
        id="gpt-5-mini",
        name="GPT-5 Mini",
        api="openai-responses",
        provider="openai",
        reasoning=True,
        input_types=["text", "image"],
        context_window=400_000,
        max_tokens=128_000,
        cost=ModelCost(input=0.25, output=2.0, cache_read=0.025),
    ),
    Model(
        id="gpt-5-nano",
        name="GPT-5 Nano",
        api="openai-responses",
        provider="openai",
        reasoning=True,
        input_types=["text", "image"],
        context_window=400_000,
        max_tokens=128_000,
        cost=ModelCost(input=0.05, output=0.4, cache_read=0.005),
    ),
    # -- OpenAI (GPT-4.1 long context) ──────────────────────────────────────
    Model(
        id="gpt-4.1",
        name="GPT-4.1",
        api="openai-responses",
        provider="openai",
        reasoning=False,
        input_types=["text", "image"],
        context_window=1_047_576,
        max_tokens=32_768,
        cost=ModelCost(input=2.0, output=8.0, cache_read=0.5),
    ),
    Model(
        id="gpt-4.1-mini",
        name="GPT-4.1 Mini",
        api="openai-responses",
        provider="openai",
        reasoning=False,
        input_types=["text", "image"],
        context_window=1_047_576,
        max_tokens=32_768,
        cost=ModelCost(input=0.4, output=1.6, cache_read=0.1),
    ),
    Model(
        id="gpt-4.1-nano",
        name="GPT-4.1 Nano",
        api="openai-responses",
        provider="openai",
        reasoning=False,
        input_types=["text", "image"],
        context_window=1_047_576,
        max_tokens=32_768,
        cost=ModelCost(input=0.1, output=0.4, cache_read=0.03),
    ),
    # -- OpenAI (o-series reasoning) ────────────────────────────────────────
    Model(
        id="o3",
        name="o3",
        api="openai-responses",
        provider="openai",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200_000,
        max_tokens=100_000,
        cost=ModelCost(input=2.0, output=8.0, cache_read=0.5),
    ),
    Model(
        id="o3-pro",
        name="o3-pro",
        api="openai-responses",
        provider="openai",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200_000,
        max_tokens=100_000,
        cost=ModelCost(input=20.0, output=80.0),
    ),
    Model(
        id="o4-mini",
        name="o4-mini",
        api="openai-responses",
        provider="openai",
        reasoning=True,
        input_types=["text", "image"],
        context_window=200_000,
        max_tokens=100_000,
        cost=ModelCost(input=1.1, output=4.4, cache_read=0.28),
    ),
    # -- OpenAI (Codex) ─────────────────────────────────────────────────────
    Model(
        id="codex-mini-latest",
        name="Codex Mini",
        api="openai-responses",
        provider="openai",
        reasoning=True,
        input_types=["text"],
        context_window=200_000,
        max_tokens=100_000,
        cost=ModelCost(input=1.5, output=6.0, cache_read=0.375),
    ),
]


# ---------------------------------------------------------------------------
# Registry (provider -> model_id -> Model)
# ---------------------------------------------------------------------------

_model_map: dict[str, dict[str, Model]] = {}


def _ensure_built() -> None:
    if _model_map:
        return
    for m in MODELS:
        _model_map.setdefault(m.provider, {})[m.id] = m


def register_model(model: Model) -> None:
    """Register or override a model definition."""
    _ensure_built()
    _model_map.setdefault(model.provider, {})[model.id] = model


def get_model(model_id: str, provider: str | None = None) -> Model | None:
    """Look up a model by id, optionally scoped to a provider."""
    _ensure_built()
    if provider:
        return _model_map.get(provider, {}).get(model_id)
    # Search across all providers
    for provider_models in _model_map.values():
        if model_id in provider_models:
            return provider_models[model_id]
    return None


def get_models(provider: str | None = None) -> list[Model]:
    """List all registered models, optionally filtered by provider."""
    _ensure_built()
    if provider:
        return list(_model_map.get(provider, {}).values())
    return [m for prov in _model_map.values() for m in prov.values()]


def get_providers() -> list[str]:
    """List all providers that have registered models."""
    _ensure_built()
    return list(_model_map.keys())


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def calculate_cost(model: Model, usage: Usage) -> UsageCost:
    """Calculate dollar cost from token usage and model pricing."""
    c = model.cost
    per_m = 1_000_000
    input_cost = (usage.input * c.input) / per_m
    output_cost = (usage.output * c.output) / per_m
    cache_read_cost = (usage.cache_read * c.cache_read) / per_m
    cache_write_cost = (usage.cache_write * c.cache_write) / per_m
    total = input_cost + output_cost + cache_read_cost + cache_write_cost
    return UsageCost(
        input=input_cost,
        output=output_cost,
        cache_read=cache_read_cost,
        cache_write=cache_write_cost,
        total=total,
    )
