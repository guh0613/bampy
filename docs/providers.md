# 提供商

bampy 内置支持三个 LLM 提供商，通过懒加载机制仅在首次使用时导入对应 SDK。

## Anthropic

```bash
uv add "bampy[anthropic]"
```

```python
from bampy.ai import get_model

model = get_model("claude-opus-4-7", provider="anthropic")
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
| `claude-opus-4-7` | Claude Opus 4.7 |

**特有选项**：

```python
from bampy.ai import AnthropicOptions, AnthropicThinkingEnabled, AnthropicThinkingAdaptive

# 显式 thinking budget（Opus 4.7 不支持；4.6 上已不推荐）
options = AnthropicOptions(
    thinking=AnthropicThinkingEnabled(budget_tokens=16384),
)

# 自适应 thinking（Opus 4.7/4.6 和 Sonnet 4.6 推荐）
options = AnthropicOptions(
    thinking=AnthropicThinkingAdaptive(effort="xhigh", display="summarized"),
    interleaved_thinking=True,
    cache_retention="long",
)
```

**API Key**：通过 `stream_options.api_key` 或上层 runtime 的 `get_api_key(provider)` 注入。

## OpenAI

```bash
uv add "bampy[openai]"
```

```python
model = get_model("gpt-5.5", provider="openai")
```

**推荐内置模型**：

| 模型 ID | 名称 |
| ------- | ---- |
| `gpt-5.5` | GPT-5.5 |
| `gpt-5.5-pro` | GPT-5.5 Pro |
| `gpt-5.4` | GPT-5.4 |
| `gpt-5.4-pro` | GPT-5.4 Pro |
| `gpt-5.4-mini` | GPT-5.4 Mini |
| `gpt-5.4-nano` | GPT-5.4 Nano |
| `gpt-5.3-codex` | GPT-5.3 Codex |
| `chat-latest` | Chat Latest |
| `gpt-4.1` | GPT-4.1 |
| `gpt-4.1-mini` | GPT-4.1 Mini |

**特有选项**：

```python
from bampy.ai import OpenAIOptions

options = OpenAIOptions(
    reasoning_effort="high",  # minimal/low/medium/high/xhigh/max
)
```

**API Key**：通过 `stream_options.api_key` 或上层 runtime 的 `get_api_key(provider)` 注入。

## DeepSeek

DeepSeek 使用 OpenAI Chat Completions 兼容端点，安装 OpenAI provider 依赖即可：

```bash
uv add "bampy[openai]"
```

```python
model = get_model("deepseek-v4-pro", provider="deepseek")
```

**内置模型**：

| 模型 ID | 名称 | Thinking |
| ------- | ---- | -------- |
| `deepseek-v4-flash` | DeepSeek V4 Flash | 默认开启 |
| `deepseek-v4-pro` | DeepSeek V4 Pro | 默认开启 |

DeepSeek v4 的 thinking mode 通过 OpenAI SDK 的 `extra_body={"thinking": {"type": "enabled"}}` 启用。`SimpleStreamOptions(reasoning="low"|"medium"|"high")` 会映射为 DeepSeek 的 `reasoning_effort="high"`，`reasoning="xhigh"` 会映射为 `reasoning_effort="max"`。如需关闭 thinking，可使用：

```python
from bampy.ai import OpenAIOptions

options = OpenAIOptions(reasoning_effort="none")
```

**API Key**：通过 `stream_options.api_key` 或上层 runtime 的 `get_api_key(provider)` 注入。

## Google Gemini

```bash
uv add "bampy[google]"
```

```python
model = get_model("gemini-3.1-flash-lite", provider="google")
```

**推荐内置模型**：

| 模型 ID | 名称 |
| ------- | ---- |
| `gemini-3.1-pro-preview` | Gemini 3.1 Pro Preview |
| `gemini-3.1-pro-preview-customtools` | Gemini 3.1 Pro Preview Custom Tools |
| `gemini-3-flash-preview` | Gemini 3 Flash Preview |
| `gemini-3.1-flash-lite` | Gemini 3.1 Flash-Lite |
| `gemini-3.1-flash-lite-preview` | Gemini 3.1 Flash-Lite Preview |
| `gemini-2.5-pro` | Gemini 2.5 Pro |
| `gemini-2.5-flash` | Gemini 2.5 Flash |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash-Lite |

**特有选项**：

```python
from bampy.ai import GeminiOptions

options = GeminiOptions(
    thinking_budget=8192,
)
```

Gemini 3 系列推荐使用 `thinking_level="minimal"|"low"|"medium"|"high"`；`SimpleStreamOptions(reasoning=...)` 会自动映射为 Gemini 3 的 `thinking_level`，Gemini 2.5 则继续映射为 `thinking_budget`。

**API Key**：通过 `stream_options.api_key` 或上层 runtime 的 `get_api_key(provider)` 注入。

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

| ThinkingLevel | Anthropic | OpenAI | DeepSeek v4 | Gemini |
| ------------- | --------- | ------ | ----------- | ------ |
| `minimal` | adaptive(low) | minimal | high | 低 budget |
| `low` | adaptive(low) | low | high | 低 budget |
| `medium` | adaptive(medium) | medium | high | 中 budget |
| `high` | adaptive(high) | high | high | 高 budget |
| `xhigh` | adaptive(xhigh) on Opus 4.7; adaptive(max) on 4.6 | xhigh | max | 最大 budget |
| `max` | adaptive(max) | max（OpenAI 模型会按支持情况归一化） | max | 最大 budget |

## 自定义模型

注册使用现有提供商 API 的自定义模型：

```python
from bampy.ai import register_model, Model, ModelCost

register_model(Model(
    id="my-chat-model",
    name="My Chat Model",
    api="openai-completions",    # Chat Completions 兼容端点
    provider="my-provider",
    base_url="https://api.example.com/v1",
    context_window=64_000,
    max_tokens=8192,
    cost=ModelCost(input=0.14, output=0.28),
))
```

如果你的网关实现的是 OpenAI Responses API，则将 `api` 设为 `openai-responses`；如果只兼容 `/v1/chat/completions`，则使用 `openai-completions`。

添加全新的提供商适配见 [自定义提供商](custom-provider.md)。
