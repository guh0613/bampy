# 提供商

bampy 内置支持三个 LLM 提供商，通过懒加载机制仅在首次使用时导入对应 SDK。

## Anthropic

```bash
uv add "bampy[anthropic]"
```

```python
from bampy.ai import get_model

model = get_model("anthropic", "claude-sonnet-4-20250514")
```

**内置模型**：

| 模型 ID | 名称 |
| ------- | ---- |
| `claude-haiku-4-5-20251001` | Claude Haiku 4.5 |
| `claude-sonnet-4-20250514` | Claude Sonnet 4 |
| `claude-sonnet-4-5-20250514` | Claude Sonnet 4.5 |
| `claude-sonnet-4-6-20250627` | Claude Sonnet 4.6 |
| `claude-opus-4-1-20250527` | Claude Opus 4.1 |
| `claude-opus-4-5-20250605` | Claude Opus 4.5 |
| `claude-opus-4-6-20250624` | Claude Opus 4.6 |

**特有选项**：

```python
from bampy.ai import AnthropicOptions, AnthropicThinkingEnabled, AnthropicThinkingAdaptive

# 显式 thinking budget
options = AnthropicOptions(
    thinking=AnthropicThinkingEnabled(budget_tokens=16384),
)

# 自适应 thinking（推荐）
options = AnthropicOptions(
    thinking=AnthropicThinkingAdaptive(effort="high"),
    interleaved_thinking=True,
    cache_retention="long",
)
```

**API Key**：通过 `ANTHROPIC_API_KEY` 环境变量或 `stream_options.api_key` 设置。

## OpenAI

```bash
uv add "bampy[openai]"
```

```python
model = get_model("openai", "gpt-4o")
```

**内置模型**：

| 模型 ID | 名称 |
| ------- | ---- |
| `gpt-4o` | GPT-4o |
| `gpt-4.1` | GPT-4.1 |
| `gpt-4.1-mini` | GPT-4.1 Mini |
| `o1` | o1 |
| `o1-mini` | o1-mini |

**特有选项**：

```python
from bampy.ai import OpenAIOptions

options = OpenAIOptions(
    reasoning_effort="high",  # minimal/low/medium/high/xhigh
)
```

**API Key**：通过 `OPENAI_API_KEY` 环境变量设置。

## Google Gemini

```bash
uv add "bampy[google]"
```

```python
model = get_model("google", "gemini-2.5-flash")
```

**特有选项**：

```python
from bampy.ai import GeminiOptions

options = GeminiOptions(
    thinking_budget=8192,
)
```

**API Key**：通过 `GOOGLE_API_KEY` 环境变量设置。

## 通用选项（SimpleStreamOptions）

所有提供商通用的选项，自动映射到提供商特有参数：

```python
from bampy.ai import SimpleStreamOptions

options = SimpleStreamOptions(
    temperature=0.7,
    max_tokens=4096,
    reasoning="medium",  # 映射到各提供商的推理配置
    api_key="sk-...",
)
```

`reasoning` 映射规则：

| ThinkingLevel | Anthropic | OpenAI | Gemini |
| ------------- | --------- | ------ | ------ |
| `minimal` | adaptive(low) | minimal | 低 budget |
| `low` | adaptive(low) | low | 低 budget |
| `medium` | adaptive(medium) | medium | 中 budget |
| `high` | adaptive(high) | high | 高 budget |
| `xhigh` | adaptive(max) | xhigh | 最大 budget |

## 自定义模型

注册使用现有提供商 API 的自定义模型：

```python
from bampy.ai import register_model, Model, ModelCost

register_model(Model(
    id="deepseek-chat",
    name="DeepSeek Chat",
    api="openai-completions",    # Chat Completions 兼容端点
    provider="deepseek",
    base_url="https://api.deepseek.com/v1",
    context_window=64_000,
    max_tokens=8192,
    cost=ModelCost(input=0.14, output=0.28),
))
```

如果你的网关实现的是 OpenAI Responses API，则将 `api` 设为 `openai-responses`；如果只兼容 `/v1/chat/completions`，则使用 `openai-completions`。

添加全新的提供商适配见 [自定义提供商](custom-provider.md)。
