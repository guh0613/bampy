# 提供商

内置 LLM 适配按 **API 类型**（`Model.api`）注册，按 **逻辑提供商**（`Model.provider`）组织模型。首次调用 `stream` / `stream_simple` 时懒加载对应 SDK。

## 安装

```bash
uv add "bampy[anthropic]"       # Anthropic SDK
uv add "bampy[openai]"          # OpenAI SDK（OpenAI 及兼容端点均依赖它）
uv add "bampy[google]"          # google-genai
uv add "bampy[all-providers]"   # anthropic + openai + google-genai
```

可选依赖定义见 `pyproject.toml`。

## API Key

AI 层本身**不**读取环境变量；把 key 放进选项即可：

```python
from bampy.ai import SimpleStreamOptions

SimpleStreamOptions(api_key="...")
```

未传 `api_key` 时，各官方 SDK 会使用自身默认环境变量：

| 逻辑提供商 / SDK | 常见环境变量 |
| ---------------- | ------------ |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI（及走 OpenAI SDK 的兼容端点） | `OPENAI_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` 或 `GOOGLE_API_KEY` |
| DeepSeek | 通常传 `api_key=`；也可按网关约定配置 OpenAI SDK 可读的 key |
| OpenCode Go | 通过 `api_key=` 注入网关密钥 |
| Ollama Cloud | `OLLAMA_API_KEY`；读取后通过 `api_key=` 传入 |

上层 `Agent` / `create_agent_session` 还可用 `get_api_key(provider)` 按 `Model.provider` 动态解析。

## 查找模型

签名为 **`get_model(model_id, provider=None)`**（模型 ID 在前）：

```python
from bampy.ai import get_model, get_models, get_providers

model = get_model("claude-opus-4-7", provider="anthropic")
# 或
model = get_model("gpt-5.4-mini", "openai")

get_providers()
# ['anthropic', 'google', 'opencode-go', 'ollama', 'deepseek', 'openai']

[m.id for m in get_models("deepseek")]
```

以下模型表均来自 `src/bampy/ai/models.py` 的 `BUILTIN_MODELS`。

---

## Anthropic

- **依赖**：`bampy[anthropic]`
- **API**：`anthropic-messages`
- **默认 base_url**：`https://api.anthropic.com`

```python
from bampy.ai import get_model

model = get_model("claude-opus-4-7", provider="anthropic")
```

| 模型 ID | 名称 | context / max_tokens |
| ------- | ---- | -------------------- |
| `claude-haiku-4-5` | Claude Haiku 4.5 (latest) | 200K / 64K |
| `claude-haiku-4-5-20251001` | Claude Haiku 4.5 | 200K / 64K |
| `claude-sonnet-4` | Claude Sonnet 4 (latest) | 200K / 64K |
| `claude-sonnet-4-5` | Claude Sonnet 4.5 (latest) | 200K / 64K |
| `claude-sonnet-4-5-20250929` | Claude Sonnet 4.5 | 200K / 64K |
| `claude-sonnet-4-6` | Claude Sonnet 4.6 | 1M / 64K |
| `claude-opus-4-1` | Claude Opus 4.1 | 200K / 32K |
| `claude-opus-4-5` | Claude Opus 4.5 | 200K / 64K |
| `claude-opus-4-6` | Claude Opus 4.6 | 1M / 128K |
| `claude-opus-4-7` | Claude Opus 4.7 (latest) | 1M / 128K |

全部 `reasoning=True`，`input_types=["text", "image"]`。

**特有选项**：

```python
from bampy.ai import AnthropicOptions, AnthropicThinkingEnabled, AnthropicThinkingAdaptive

# 手动 budget（Opus 4.7 会拒绝；请用 adaptive）
AnthropicOptions(thinking=AnthropicThinkingEnabled(budget_tokens=16384))

# 自适应（Opus 4.7 / 4.6、Sonnet 4.6 等）
AnthropicOptions(
    thinking=AnthropicThinkingAdaptive(effort="xhigh", display="summarized"),
    interleaved_thinking=True,
    cache_retention="long",  # short | long
    effort="high",           # 也可单独设 effort
)
```

`SimpleStreamOptions(reasoning=...)`：支持 adaptive 的型号走 adaptive + `display="summarized"`；其余走手动 `budget_tokens`。开启 reasoning 时会设 `interleaved_thinking=True`。`xhigh` 仅 Opus 4.7 保留，其它型号会归一到 `max` 或 `high`。

---

## OpenAI

- **依赖**：`bampy[openai]`
- **API**：`openai-responses`
- **默认 base_url**：`https://api.openai.com/v1`

```python
model = get_model("gpt-5.5", provider="openai")
```

| 模型 ID | 名称 | reasoning | context / max_tokens |
| ------- | ---- | --------- | -------------------- |
| `gpt-5.5` | GPT-5.5 | ✓ | 1.05M / 128K |
| `gpt-5.5-pro` | GPT-5.5 Pro | ✓ | 1.05M / 128K |
| `gpt-5.4` | GPT-5.4 | ✓ | 1.05M / 128K |
| `gpt-5.4-pro` | GPT-5.4 Pro | ✓ | 1.05M / 128K |
| `gpt-5.4-mini` | GPT-5.4 Mini | ✓ | 400K / 128K |
| `gpt-5.4-nano` | GPT-5.4 Nano | ✓ | 400K / 128K |
| `gpt-5.3-codex` | GPT-5.3 Codex | ✓ | 400K / 128K |
| `gpt-5.3-chat-latest` | GPT-5.3 Chat | ✗ | 128K / 16K |
| `gpt-5.2` | GPT-5.2 | ✓ | 400K / 128K |
| `gpt-5.2-pro` | GPT-5.2 Pro | ✓ | 400K / 128K |
| `gpt-5.2-codex` | GPT-5.2 Codex | ✓ | 400K / 128K |
| `gpt-5.2-chat-latest` | GPT-5.2 Chat | ✗ | 128K / 16K |
| `gpt-5.1` | GPT-5.1 | ✓ | 400K / 128K |
| `gpt-5.1-codex` | GPT-5.1 Codex | ✓ | 400K / 128K |
| `gpt-5.1-codex-max` | GPT-5.1 Codex Max | ✓ | 400K / 128K |
| `gpt-5.1-codex-mini` | GPT-5.1 Codex Mini | ✓ | 400K / 128K |
| `gpt-5.1-chat-latest` | GPT-5.1 Chat | ✗ | 128K / 16K |
| `gpt-5` | GPT-5 | ✓ | 400K / 128K |
| `gpt-5-pro` | GPT-5 Pro | ✓ | 400K / 272K |
| `gpt-5-codex` | GPT-5-Codex | ✓ | 400K / 128K |
| `gpt-5-mini` | GPT-5 Mini | ✓ | 400K / 128K |
| `gpt-5-nano` | GPT-5 Nano | ✓ | 400K / 128K |
| `gpt-5-chat-latest` | GPT-5 Chat Latest | ✗ | 128K / 16K |
| `chat-latest` | Chat Latest | ✗ | 128K / 16K |
| `gpt-4.1` | GPT-4.1 | ✗ | ~1.05M / 32K |
| `gpt-4.1-mini` | GPT-4.1 Mini | ✗ | ~1.05M / 32K |
| `gpt-4.1-nano` | GPT-4.1 Nano | ✗ | ~1.05M / 32K |
| `o3` | o3 | ✓ | 200K / 100K |
| `o3-pro` | o3 Pro | ✓ | 200K / 100K |
| `o4-mini` | o4-mini | ✓ | 200K / 100K |

全部 `input_types=["text", "image"]`。

**特有选项**：

```python
from bampy.ai import OpenAIOptions

OpenAIOptions(
    reasoning_effort="high",  # none | minimal | low | medium | high | xhigh | max
    tool_choice="auto",
    parallel_tool_calls=True,
    service_tier="auto",
    verbosity="medium",
    store=True,
)
```

`reasoning="max"` 在支持 `xhigh` 的型号上会归一为 `xhigh`，否则为 `high`（见 `supports_xhigh`）。

---

## Google Gemini

- **依赖**：`bampy[google]`
- **API**：`google-genai`
- **默认 base_url**：`https://generativelanguage.googleapis.com/v1beta`

```python
model = get_model("gemini-3.1-flash-lite", provider="google")
```

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

全部 `reasoning=True`，上下文约 1M，`max_tokens=65536`，支持图文。

**特有选项**：

```python
from bampy.ai import GeminiOptions

# Gemini 3：thinking_level
GeminiOptions(thinking_level="medium")  # minimal | low | medium | high

# Gemini 2.5：thinking_budget（token 预算）
GeminiOptions(thinking_budget=8192)
```

`SimpleStreamOptions(reasoning=...)`：`gemini-3*` 映射为 `thinking_level`（Pro 家族会把 `minimal`/`low` 收成 `low`，更高档收成 `high`）；`gemini-2.5*` 映射为 `thinking_budget`。

---

## DeepSeek

- **依赖**：`bampy[openai]`（走 Chat Completions）
- **API**：`openai-completions`
- **provider**：`deepseek`
- **base_url**：`https://api.deepseek.com/v1`

```python
model = get_model("deepseek-v4-pro", provider="deepseek")
```

| 模型 ID | 名称 | input_types | context / max_tokens |
| ------- | ---- | ----------- | -------------------- |
| `deepseek-v4-flash` | DeepSeek V4 Flash | text | 1M / 384K |
| `deepseek-v4-pro` | DeepSeek V4 Pro | text | 1M / 384K |

Thinking 默认开启（`openai_chat_compat.thinking_param="deepseek"`）。`SimpleStreamOptions` 的 `minimal`/`low`/`medium`/`high` 会映射为后端 `reasoning_effort="high"`，`xhigh`/`max` → `"max"`。关闭 thinking：

```python
from bampy.ai import OpenAIOptions

OpenAIOptions(reasoning_effort="none")
```

纯文本模型：历史中的图片会在发送前被降级为文本占位。

---

## OpenCode Go

- **依赖**：`bampy[openai]`
- **API**：`openai-completions`
- **provider**：`opencode-go`
- **base_url**：`https://opencode.ai/zen/go/v1`

```python
model = get_model("kimi-k3", provider="opencode-go")
```

| 模型 ID | 名称 | input_types | context / max_tokens |
| ------- | ---- | ----------- | -------------------- |
| `kimi-k3` | Kimi K3 | text+image | ~1M / 131K |
| `kimi-k2.7-code` | Kimi K2.7 Code | text+image | 262K / 32K |
| `kimi-k2.6` | Kimi K2.6 | text+image | 262K / 65K |
| `glm-5.2` | GLM 5.2 | text | 1M / 131K |
| `glm-5.1` | GLM 5.1 | text | 198K / 65K |

均通过 `OpenAIChatCompat` 处理 thinking / `reasoning_content` 等差异。例如 `kimi-k3` 默认开启 thinking，并支持后端的 `low` / `high` / `max` 三档 effort（`minimal`/`low` → `low`，`medium`/`high` → `high`，`xhigh`/`max` → `max`）；`glm-5.2` 有独立的 effort 映射表。需向网关传入有效 `api_key`。

---

## Ollama Cloud

- **依赖**：`bampy[openai]`
- **API**：`openai-completions`（复用通用 Chat Completions adapter）
- **provider**：`ollama`
- **base_url**：`https://ollama.com/v1`
- **认证**：真实 Ollama Cloud API key

```python
import os

from bampy.ai import SimpleStreamOptions, get_model

model = get_model("glm-5.2", provider="ollama")
options = SimpleStreamOptions(
    api_key=os.environ["OLLAMA_API_KEY"],
    reasoning="high",
)
```

| 直接 Cloud 模型 ID | 名称 | input_types | context / max_tokens |
| ------------------ | ---- | ----------- | -------------------- |
| `glm-5.2` | GLM 5.2 | text | 1M / 131K |
| `kimi-k2.7-code` | Kimi K2.7 Code | text+image | 262K / 32K |

这些是 `https://ollama.com/v1` 使用的直接 Cloud ID，不带 `:cloud`。`glm-5.2:cloud`、`kimi-k2.7-code:cloud` 是通过本地 Ollama 转发到 Cloud 时使用的 tag。

两个内置模型都通过 `OpenAIChatCompat` 使用 `system` role、`max_tokens` 和 Ollama 的 `reasoning` 字段进行 thinking 流式解析与历史回传，并禁用 `store`。GLM 5.2 的公开档位是 `high` / `max`，因此较低级别会映射到 `none` 或 `high`；Kimi K2.7 Code 使用 Ollama 通用的 `low` / `medium` / `high` / `max` 档位。

Ollama Cloud 按订阅额度和 GPU 使用量计费，而非公布固定的每 token 单价，因此内置 `ModelCost` 保持为零。Cloud 模型会变更或退休，当前可用 ID 应以 `https://ollama.com/v1/models` 或 `https://ollama.com/api/tags` 为准。

Ollama Chat Completions 不支持 `tool_choice`，也不应启用 prompt cache、service tier 等 OpenAI 专有选项。Ollama 也提供 stateless `/v1/responses`；若明确要求 Responses 事件协议，可将自定义模型设为 `api="openai-responses"`，但它不支持 `previous_response_id` / `conversation`，且当前对截断结束原因的表达不如 Chat Completions 完整。

本地 Ollama 模型仍可注册为 `openai-completions` 自定义模型，使用 `http://127.0.0.1:11434/v1` 和任意非空 key。API 能力以 [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) 为准；Cloud 认证见 [Ollama Cloud](https://docs.ollama.com/cloud)。

---

## SimpleStreamOptions 与 reasoning 映射

```python
from bampy.ai import SimpleStreamOptions

SimpleStreamOptions(
    temperature=0.7,
    max_tokens=4096,
    reasoning="medium",
    api_key="...",
)
```

概要映射（细节以各 adapter 源码为准）：

| ThinkingLevel | Anthropic | OpenAI Responses | DeepSeek v4 | Gemini 3 Flash | Gemini 3 Pro | Gemini 2.5 |
| ------------- | --------- | ---------------- | ----------- | -------------- | ------------ | ---------- |
| `minimal` | adaptive(low) 或 budget 1024 | minimal | high | minimal | low | budget 1024 |
| `low` | adaptive(low) 或 budget 2048 | low | high | low | low | budget 2048 |
| `medium` | adaptive(medium) 或 budget 8192 | medium | high | medium | high | budget 8192 |
| `high` | adaptive(high) 或 budget 16384 | high | high | high | high | budget 16384 |
| `xhigh` | Opus 4.7: xhigh；否则 max/high | xhigh（不支持则 high） | max | high | high | budget 16384 |
| `max` | adaptive(max) 等 | xhigh 或 high | max | high | high | budget 16384 |

---

## 自定义模型（复用已有 API）

无需写适配器时，注册 `Model` 并指向已有 `api` 即可：

```python
from bampy.ai import register_model, Model, ModelCost, OpenAIChatCompat

register_model(Model(
    id="my-chat-model",
    name="My Chat Model",
    api="openai-completions",
    provider="my-provider",
    base_url="https://api.example.com/v1",
    context_window=64_000,
    max_tokens=8192,
    cost=ModelCost(input=0.14, output=0.28),
    openai_chat_compat=OpenAIChatCompat(
        max_tokens_field="max_tokens",
        system_role="system",
    ),
))
```

- Responses 兼容（Ollama v0.13.3+ 可选）：`api="openai-responses"`
- `/v1/chat/completions`（Ollama 默认建议）：`api="openai-completions"`
- Anthropic / Gemini：使用对应 `api` 名

全新协议适配见 [custom-provider.md](custom-provider.md)。类型与流式协议见 [ai-layer.md](ai-layer.md)。
