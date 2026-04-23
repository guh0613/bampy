"""Tests for bampy.ai.models."""

import pytest

from bampy.ai.models import (
    MODELS,
    calculate_cost,
    get_model,
    get_models,
    get_providers,
    models_are_equal,
    register_model,
    supports_xhigh,
    _model_map,
)
from bampy.ai.types import Model, ModelCost, Usage


class TestModelRegistry:
    def test_builtin_models_exist(self):
        assert len(MODELS) > 0

    def test_get_model_by_id(self):
        model = get_model("claude-sonnet-4-6")
        assert model is not None
        assert model.provider == "anthropic"

    def test_get_model_by_id_and_provider(self):
        model = get_model("gpt-4.1", provider="openai")
        assert model is not None
        assert model.api == "openai-responses"

    def test_get_builtin_ollama_model(self):
        model = get_model("gemini-3-flash", provider="ollama")
        assert model is not None
        assert model.api == "ollama-responses"
        assert model.reasoning is True
        assert model.context_window == 1_048_576
        assert model.max_tokens == 65_536

    def test_get_builtin_opencode_go_kimi_model(self):
        model = get_model("kimi-k2.6", provider="opencode-go")
        assert model is not None
        assert model.api == "openai-completions"
        assert model.reasoning is True
        assert model.base_url == "https://opencode.ai/zen/go/v1"
        assert model.openai_chat_compat is not None
        assert model.openai_chat_compat.replay_thinking_field == "reasoning"
        assert model.openai_chat_compat.stream_reasoning_fields == [
            "reasoning",
            "reasoning_details",
        ]
        assert model.openai_chat_compat.max_tokens_field == "max_tokens"

    def test_updated_model_capabilities(self):
        gpt_54 = get_model("gpt-5.4", provider="openai")
        haiku = get_model("claude-haiku-4-5-20251001", provider="anthropic")
        gpt_52 = get_model("gpt-5.2", provider="openai")

        assert gpt_54 is not None
        assert gpt_54.context_window == 272_000
        assert haiku is not None
        assert haiku.reasoning is True
        assert gpt_52 is not None
        assert gpt_52.max_tokens == 128_000

    def test_get_model_not_found(self):
        model = get_model("nonexistent-model")
        assert model is None

    def test_get_models_all(self):
        models = get_models()
        assert len(models) >= len(MODELS)

    def test_get_models_by_provider(self):
        models = get_models(provider="anthropic")
        assert all(m.provider == "anthropic" for m in models)

    def test_get_providers(self):
        providers = get_providers()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "google" in providers
        assert "ollama" in providers

    def test_register_custom_model(self):
        custom = Model(
            id="custom-model",
            name="Custom",
            api="custom-api",
            provider="custom",
        )
        register_model(custom)
        found = get_model("custom-model", provider="custom")
        assert found is not None
        assert found.name == "Custom"
        # Cleanup
        _model_map.get("custom", {}).pop("custom-model", None)


class TestCostCalculation:
    def test_calculate_cost(self):
        model = Model(
            id="test",
            name="Test",
            api="test",
            provider="test",
            cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        )
        usage = Usage(input=1000, output=500, cache_read=200, cache_write=100)
        cost = calculate_cost(model, usage)
        assert cost.input == pytest.approx(3.0 * 1000 / 1_000_000)
        assert cost.output == pytest.approx(15.0 * 500 / 1_000_000)
        assert cost.total == pytest.approx(cost.input + cost.output + cost.cache_read + cost.cache_write)
        assert cost is not usage.cost

    def test_calculate_cost_does_not_mutate_usage(self):
        model = Model(
            id="test",
            name="Test",
            api="test",
            provider="test",
            cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        )
        usage = Usage(input=1000, output=500, cache_read=200, cache_write=100)
        original_cost = usage.cost.model_copy(deep=True)

        calculate_cost(model, usage)

        assert usage.cost == original_cost

    def test_zero_usage(self):
        model = Model(id="t", name="T", api="t", provider="t")
        usage = Usage()
        cost = calculate_cost(model, usage)
        assert cost.total == 0.0


class TestModelHelpers:
    def test_supports_xhigh(self):
        model = get_model("gpt-5.4", provider="openai")
        assert supports_xhigh(model) is True

    def test_models_are_equal(self):
        a = get_model("gpt-5.4", provider="openai")
        b = get_model("gpt-5.4", provider="openai")
        c = get_model("claude-sonnet-4-6", provider="anthropic")
        assert models_are_equal(a, b) is True
        assert models_are_equal(a, c) is False
