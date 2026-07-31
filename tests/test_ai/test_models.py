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

    def test_get_builtin_opencode_go_kimi_model(self):
        model = get_model("kimi-k2.6", provider="opencode-go")
        assert model is not None
        assert model.api == "openai-completions"
        assert model.reasoning is True
        assert model.base_url == "https://opencode.ai/zen/go/v1"
        assert model.openai_chat_compat is not None
        assert model.openai_chat_compat.replay_thinking_field == "reasoning_content"
        assert model.openai_chat_compat.stream_reasoning_fields == [
            "reasoning_content",
            "reasoning",
            "reasoning_details",
        ]
        assert model.openai_chat_compat.max_tokens_field == "max_tokens"
        assert model.openai_chat_compat.system_role == "system"
        assert model.openai_chat_compat.supports_store is False

    def test_get_builtin_opencode_go_gpt_56_luna_model(self):
        model = get_model("gpt-5.6-luna", provider="opencode-go")

        assert model is not None
        assert model.name == "GPT-5.6 Luna"
        assert model.api == "openai-responses"
        assert model.base_url == "https://opencode.ai/zen/go/v1"
        assert model.reasoning is True
        assert model.reasoning_efforts == [
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]
        assert model.input_types == ["text", "image"]
        assert model.context_window == 1_050_000
        assert model.max_tokens == 128_000
        assert model.cost.input == 0.2
        assert model.cost.output == 1.2
        assert model.cost.cache_read == 0.02
        assert model.cost.cache_write == 0.25
        assert len(model.cost.tiers) == 1
        tier = model.cost.tiers[0]
        assert tier.context_over == 272_000
        assert tier.input == 0.4
        assert tier.output == 1.8
        assert tier.cache_read == 0.04
        assert tier.cache_write == 0.5

    def test_get_builtin_opencode_go_kimi_k3_model(self):
        model = get_model("kimi-k3", provider="opencode-go")

        assert model is not None
        assert model.name == "Kimi K3"
        assert model.api == "openai-completions"
        assert model.base_url == "https://opencode.ai/zen/go/v1"
        assert model.reasoning is True
        assert model.input_types == ["text", "image"]
        assert model.context_window == 1_048_576
        assert model.max_tokens == 131_072
        assert model.cost.input == 3.0
        assert model.cost.output == 15.0
        assert model.cost.cache_read == 0.3
        assert model.openai_chat_compat is not None
        assert model.openai_chat_compat.max_tokens_field == "max_tokens"
        assert model.openai_chat_compat.system_role == "system"
        assert model.openai_chat_compat.replay_thinking_field == "reasoning_content"
        assert model.openai_chat_compat.stream_reasoning_fields == ["reasoning_content"]
        assert model.openai_chat_compat.supports_reasoning_effort is True
        assert model.openai_chat_compat.supports_store is False
        assert model.openai_chat_compat.reasoning_effort_map == {
            "minimal": "low",
            "low": "low",
            "medium": "high",
            "high": "high",
            "xhigh": "max",
            "max": "max",
        }
        assert model.openai_chat_compat.thinking_param == "kimi"
        assert model.openai_chat_compat.thinking_default_enabled is True
        assert model.openai_chat_compat.thinking_tool_choice == ["auto", "none"]

    def test_get_builtin_opencode_go_glm_model_is_text_only(self):
        model = get_model("glm-5.1", provider="opencode-go")
        assert model is not None
        assert model.api == "openai-completions"
        assert model.input_types == ["text"]
        assert model.openai_chat_compat is not None
        assert model.openai_chat_compat.system_role == "system"
        assert model.openai_chat_compat.supports_store is False

    def test_get_builtin_opencode_go_latest_coding_models(self):
        kimi = get_model("kimi-k2.7-code", provider="opencode-go")
        glm = get_model("glm-5.2", provider="opencode-go")

        assert kimi is not None
        assert kimi.api == "openai-completions"
        assert kimi.reasoning is True
        assert kimi.input_types == ["text", "image"]
        assert kimi.context_window == 262_144
        assert kimi.max_tokens == 32_768
        assert kimi.cost.input == 0.95
        assert kimi.cost.output == 4.0
        assert kimi.cost.cache_read == 0.19
        assert kimi.openai_chat_compat is not None
        assert kimi.openai_chat_compat.system_role == "system"
        assert kimi.openai_chat_compat.supports_store is False
        assert kimi.openai_chat_compat.thinking_param == "kimi"
        assert kimi.openai_chat_compat.supports_reasoning_effort is False
        assert kimi.openai_chat_compat.thinking_tool_choice == ["auto", "none"]

        assert glm is not None
        assert glm.api == "openai-completions"
        assert glm.reasoning is True
        assert glm.input_types == ["text"]
        assert glm.context_window == 1_000_000
        assert glm.max_tokens == 131_072
        assert glm.cost.input == 1.4
        assert glm.cost.output == 4.4
        assert glm.cost.cache_read == 0.26
        assert glm.openai_chat_compat is not None
        assert glm.openai_chat_compat.system_role == "system"
        assert glm.openai_chat_compat.supports_store is False
        assert glm.openai_chat_compat.thinking_param == "zai"
        assert glm.openai_chat_compat.supports_reasoning_effort is True
        assert glm.openai_chat_compat.reasoning_effort_map["xhigh"] == "max"
        assert glm.openai_chat_compat.reasoning_effort_map["medium"] == "high"

    def test_get_builtin_ollama_cloud_models(self):
        glm = get_model("glm-5.2", provider="ollama")
        kimi = get_model("kimi-k2.7-code", provider="ollama")

        assert glm is not None
        assert glm.name == "GLM 5.2 (Ollama Cloud)"
        assert glm.api == "openai-completions"
        assert glm.base_url == "https://ollama.com/v1"
        assert glm.reasoning is True
        assert glm.input_types == ["text"]
        assert glm.context_window == 1_000_000
        assert glm.max_tokens == 131_072
        assert glm.cost.input == 0.0
        assert glm.cost.output == 0.0
        assert glm.openai_chat_compat is not None
        assert glm.openai_chat_compat.max_tokens_field == "max_tokens"
        assert glm.openai_chat_compat.system_role == "system"
        assert glm.openai_chat_compat.replay_thinking_field == "reasoning"
        assert glm.openai_chat_compat.stream_reasoning_fields == ["reasoning"]
        assert glm.openai_chat_compat.supports_store is False
        assert glm.openai_chat_compat.reasoning_effort_map["medium"] == "high"
        assert glm.openai_chat_compat.reasoning_effort_map["max"] == "max"

        assert kimi is not None
        assert kimi.name == "Kimi K2.7 Code (Ollama Cloud)"
        assert kimi.api == "openai-completions"
        assert kimi.base_url == "https://ollama.com/v1"
        assert kimi.reasoning is True
        assert kimi.input_types == ["text", "image"]
        assert kimi.context_window == 262_144
        assert kimi.max_tokens == 32_768
        assert kimi.openai_chat_compat is not None
        assert kimi.openai_chat_compat.replay_thinking_field == "reasoning"
        assert kimi.openai_chat_compat.reasoning_effort_map["minimal"] == "low"
        assert kimi.openai_chat_compat.reasoning_effort_map["xhigh"] == "max"

        # Keep the existing unscoped lookup stable for duplicated model IDs.
        unscoped = get_model("glm-5.2")
        assert unscoped is not None
        assert unscoped.provider == "opencode-go"

    def test_get_builtin_deepseek_v4_models(self):
        flash = get_model("deepseek-v4-flash", provider="deepseek")
        pro = get_model("deepseek-v4-pro", provider="deepseek")

        assert flash is not None
        assert flash.api == "openai-completions"
        assert flash.base_url == "https://api.deepseek.com/v1"
        assert flash.reasoning is True
        assert flash.context_window == 1_000_000
        assert flash.max_tokens == 384_000
        assert flash.input_types == ["text"]
        assert flash.openai_chat_compat is not None
        assert flash.openai_chat_compat.thinking_param == "deepseek"
        assert flash.openai_chat_compat.replay_thinking_field == "reasoning_content"
        assert flash.openai_chat_compat.supports_store is False
        assert flash.openai_chat_compat.reasoning_effort_map["xhigh"] == "max"
        assert flash.openai_chat_compat.reasoning_effort_map["max"] == "max"

        assert pro is not None
        assert pro.cost.input == 1.74
        assert pro.cost.output == 3.48

    def test_updated_model_capabilities(self):
        gpt_55 = get_model("gpt-5.5", provider="openai")
        gpt_54 = get_model("gpt-5.4", provider="openai")
        gpt_54_mini = get_model("gpt-5.4-mini", provider="openai")
        gemini_31_lite = get_model("gemini-3.1-flash-lite", provider="google")
        haiku = get_model("claude-haiku-4-5-20251001", provider="anthropic")
        opus_47 = get_model("claude-opus-4-7", provider="anthropic")

        assert gpt_55 is not None
        assert gpt_55.context_window == 1_050_000
        assert gpt_55.max_tokens == 128_000
        assert gpt_55.cost.input == 5.0
        assert gpt_55.cost.output == 30.0
        assert gpt_54 is not None
        assert gpt_54.context_window == 1_050_000
        assert gpt_54_mini is not None
        assert gpt_54_mini.context_window == 400_000
        assert gemini_31_lite is not None
        assert gemini_31_lite.context_window == 1_048_576
        assert gemini_31_lite.max_tokens == 65_536
        assert gemini_31_lite.cost.cache_read == 0.025
        assert get_model("gemini-3-pro-preview", provider="google") is None
        assert haiku is not None
        assert haiku.reasoning is True
        assert opus_47 is not None
        assert opus_47.context_window == 1_000_000
        assert opus_47.max_tokens == 128_000
        assert opus_47.cost.input == 5.0
        assert opus_47.cost.output == 25.0

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
        assert "deepseek" in providers
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

    def test_calculate_cost_uses_context_tier_for_full_request(self):
        model = get_model("gpt-5.6-luna", provider="opencode-go")
        assert model is not None

        at_threshold = calculate_cost(
            model,
            Usage(input=200_000, cache_read=72_000, output=1_000),
        )
        over_threshold = calculate_cost(
            model,
            Usage(input=200_001, cache_read=72_000, output=1_000),
        )

        assert at_threshold.input == pytest.approx(0.2 * 200_000 / 1_000_000)
        assert at_threshold.cache_read == pytest.approx(0.02 * 72_000 / 1_000_000)
        assert at_threshold.output == pytest.approx(1.2 * 1_000 / 1_000_000)
        assert over_threshold.input == pytest.approx(0.4 * 200_001 / 1_000_000)
        assert over_threshold.cache_read == pytest.approx(0.04 * 72_000 / 1_000_000)
        assert over_threshold.output == pytest.approx(1.8 * 1_000 / 1_000_000)

    def test_zero_usage(self):
        model = Model(id="t", name="T", api="t", provider="t")
        usage = Usage()
        cost = calculate_cost(model, usage)
        assert cost.total == 0.0


class TestModelHelpers:
    def test_supports_xhigh(self):
        model = get_model("gpt-5.5", provider="openai")
        assert supports_xhigh(model) is True
        model = get_model("gpt-5.6-luna", provider="opencode-go")
        assert supports_xhigh(model) is True
        model = get_model("claude-opus-4-7", provider="anthropic")
        assert supports_xhigh(model) is True

    def test_models_are_equal(self):
        a = get_model("gpt-5.4", provider="openai")
        b = get_model("gpt-5.4", provider="openai")
        c = get_model("claude-sonnet-4-6", provider="anthropic")
        assert models_are_equal(a, b) is True
        assert models_are_equal(a, c) is False
