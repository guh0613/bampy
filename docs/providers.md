# 提供商

内置 LLM 适配按 **API 类型**（`Model.api`）注册，按 **逻辑提供商**（`Model.provider`）组织模型。首次调用 `stream` / `stream_simple` 时懒加载对应 SDK。

## 安装

```bash
uv add "bampy[anthropic]"       # Anthropic SDK
uv add "bampy[openai]"          # OpenAI SDK（OpenAI 及兼容端点均依赖它）
uv add "bampy[google]"          # google-genai >= 2.30.1（Gemini / Vertex AI）
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
| Google Vertex AI | ADC；项目/区域使用 `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION`，也支持 Vertex Express API key |
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
# ['anthropic', 'google', 'opencode-go', 'ollama', 'deepseek', 'openai', 'google-vertex']

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
- **默认 base_url**：`https://generativelanguage.googleapis.com`，SDK 默认使用 `v1beta`。旧配置末尾的 `/v1beta` 也会被正确解析。

```python
model = get_model("gemini-4-argon", provider="google")
```

| 模型 ID | 名称 |
| ------- | ---- |
| `gemini-4-carbon` | Gemini 4 Carbon |
| `gemini-4-argon` | Gemini 4 Argon |
| `gemini-3.8-flash` | Gemini 3.8 Flash |
| `gemini-3.7-flash` | Gemini 3.7 Flash（Developer API 自动路由到 3.8） |
| `gemini-3.6-flash` | Gemini 3.6 Flash |
| `gemini-3.5-flash-lite` | Gemini 3.5 Flash-Lite |
| `gemini-3.1-pro-preview` | Gemini 3.1 Pro Preview |
| `gemini-3.1-pro-preview-customtools` | Gemini 3.1 Pro Preview Custom Tools |
| `gemini-3-flash-preview` | Gemini 3 Flash Preview |
| `gemini-3.1-flash-lite` | Gemini 3.1 Flash-Lite |
| `gemini-2.5-pro` | Gemini 2.5 Pro |
| `gemini-2.5-flash` | Gemini 2.5 Flash |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash-Lite |

全部 `reasoning=True`，上下文约 1M，`max_tokens=65536`，支持图文。

3.1 Flash-Lite Preview 已于 2026-05-25 下线，因此不再注册；请用 `gemini-3.1-flash-lite`。新项目可选 `gemini-3.8-flash`、`gemini-3.6-flash` 或 `gemini-3.5-flash-lite`。本提供商处理图文输入、文本/思考输出和工具调用；Live、TTS、音视频输出与图片生成需要各自的媒体接口，不在这里注册为聊天模型。

**特有选项**：

```python
from bampy.ai import GeminiOptions

# Gemini 4、3.8 / 3.7 Flash：low | medium | high；minimal 会映射为 low
GeminiOptions(thinking_level="medium")

# Gemini 3.6 Flash / 3.5 Flash-Lite / 3 Flash：minimal | low | medium | high
GeminiOptions(thinking_level="minimal")

# Gemini 2.5：thinking_budget（token 预算）
GeminiOptions(thinking_budget=8192)
```

`SimpleStreamOptions(reasoning=...)`：Gemini 3/4 使用 `thinking_level`，`xhigh`/`max` 映射为 `high`。Gemini 4、3.8 / 3.7 Flash 与 3.1 Pro 保留 `medium`，`minimal` 映射为 `low`；旧的 Gemini 3 Pro 仅支持 `low`/`high`。Gemini 2.5 使用 `thinking_budget`。不传 reasoning 时保留后端默认值，例如 3.8 Flash 为 `medium`，3.5 Flash-Lite 为 `minimal`。

Gemini 3.5+ 使用 `thinking_level`；传入旧的 `thinking_budget` 时，适配器会在发请求前返回配置错误；同时设置 budget 和 level 也会被拒绝。这些新型号不发送 `temperature`，并在本地拒绝以 assistant/model 预填充结束的请求。3.5+ 工具调用会原样保留调用 ID、thought signature，以及图片工具结果；签名独立保留在原有内容块中。

Gemini 4 的输入、thinking 和输出共享 1,048,576 tokens。内置 `max_tokens=65536` 是默认输出预算；需要长输出时显式传入 `GeminiOptions(max_tokens=131072)` 或 `SimpleStreamOptions(max_tokens=131072)`，新版 SDK 自动跟随 continuation token。每段请求默认超时 20 分钟，可用 `GeminiOptions(timeout_ms=...)` 调整。接口仍使用 GenerateContent，bampy 自行管理对话历史。

用量统计中，`input` 不含缓存命中，`cache_read` 单独计费，`output` 包含 thinking。SDK 已累加长输出各段的 usage，bampy 读取最新累计值。内置 Gemini 4 Argon（2026-10-18 快照）与 Carbon（2026-11-11 快照）价格按标准、global 费率记录：输入/缓存输入/输出分别为 $2/$0.10/$10 每百万 tokens，有效至 2026-11-30；之后官方公布的费率为 $4/$0.20/$20。Gemini 4 Carbon 在 Developer API 上没有免费层，在 Vertex 上面向通过验证的企业客户开放。模型注册表是静态快照，日期、区域或合同价格不同时应通过 `register_model` 覆盖 `cost`。

官方参考：[Gemini 4 GenerateContent 迁移](https://ai.google.dev/gemini-api/docs/generate-content/latest-model)、[thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures)。

Gemini 3.8 / 3.7 / 3.6 Flash 的标准 global 价格快照（2026-10-18）为输入 $0.75、输出 $3.75、缓存输入 $0.075 每百万 tokens，优惠至 2026-12-31，之后分别为 $1.50/$7.50/$0.15。3.5 Flash-Lite 为 $0.30/$2.50/$0.03。日期或区域费率变化时应覆盖 `Model.cost`。参考：[Gemini 定价](https://ai.google.dev/gemini-api/docs/pricing)、[3.8 Flash 迁移](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.8)、[3.6 Flash / 3.5 Flash-Lite 迁移](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.6)。

---

## Google Vertex AI

- **依赖**：同样安装 `bampy[google]`，要求 `google-genai >= 2.30.1`。
- **API / provider**：`google-vertex`。SDK 新文档将渠道称为 Gemini Enterprise Agent Platform，原 Vertex AI 配置仍适用。
- **端点**：由 SDK 按项目、区域解析，内置模型的 `base_url` 留空。

内置 Gemini 4 Carbon、Gemini 4 Argon、3.8 / 3.7 / 3.6 Flash、3.5 Flash-Lite、3.1 Pro Preview、3 Flash Preview、3.1 Flash-Lite 和 Gemini 2.5 Pro/Flash/Flash-Lite。模型可用性取决于项目权限、区域和 Google 的下线计划；其他型号可注册 `api="google-vertex"` 的自定义模型。

使用 Application Default Credentials（本地 `gcloud auth application-default login`，服务器使用工作负载身份或服务账号）：

```bash
export GOOGLE_CLOUD_PROJECT=my-project
export GOOGLE_CLOUD_LOCATION=global
# 服务账号文件方式可设置 GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

```python
from bampy.ai import Context, SimpleStreamOptions, UserMessage, get_model, stream_simple

model = get_model("gemini-4-argon", provider="google-vertex")
context = Context(messages=[UserMessage(content="Hello")])
result = await stream_simple(model, context, SimpleStreamOptions(reasoning="medium")).result()
```

直接指定项目、区域或 Google Auth 的 `Credentials` 对象：

```python
from bampy.ai import VertexAIOptions, stream

options = VertexAIOptions(
    project="my-project",
    location="global",           # Gemini 4 支持 global、us、eu
    thinking_level="medium",
    max_tokens=131072,
    timeout_ms=1200000,
    # credentials=credentials,   # 可选；默认 ADC；不会被序列化或显示在 repr 中
)
result = await stream(model, context, options).result()
```

Vertex Express 使用 `VertexAIOptions(api_key="...")`；`stream_simple` 可通过 `SimpleStreamOptions(api_key="...")` 注入 key。SDK 负责环境变量与凭据的优先级；使用 ADC 时不要向选项传入 Developer API key。显式 `credentials` 和 `api_key` 互斥。

`VertexAIOptions` 继承全部 `GeminiOptions`。通常保留 SDK 默认的 `v1beta1`：Vertex 的 Gemini 4 长输出当前不支持 `v1`，显式用 `v1` 请求超过 65,536 输出 tokens 会在本地报错。可用 `api_version` 覆盖版本，`Model.base_url` 和 headers 支持网关。Vertex 的增量函数参数协议默认关闭，bampy 在收到完整工具调用后发出结束事件。

两个渠道按 `Model.api` 明确选择，不受 `GOOGLE_GENAI_USE_VERTEXAI` / `GOOGLE_GENAI_USE_ENTERPRISE` 全局开关干扰。`stream_simple` / Agent 使用 SDK 环境变量及 ADC；需要逐次传入项目、区域或 Credentials 时，使用 `stream(..., VertexAIOptions(...))`。

官方参考：Gemini 4 Vertex AI 开发指南（[Argon](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-4-argon)、[Carbon](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-4-carbon)）、[SDK](https://googleapis.github.io/python-genai/)、[Vertex 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing)。

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
- **API**：按模型使用 `openai-responses` 或 `openai-completions`
- **provider**：`opencode-go`
- **base_url**：`https://opencode.ai/zen/go/v1`

```python
model = get_model("gpt-5.6-luna", provider="opencode-go")
```

| 模型 ID | 名称 | API | input_types | context / max_tokens |
| ------- | ---- | --- | ----------- | -------------------- |
| `gpt-5.6-luna` | GPT-5.6 Luna | Responses | text+image | 1.05M / 128K |
| `kimi-k3` | Kimi K3 | Chat Completions | text+image | ~1M / 131K |
| `kimi-k2.7-code` | Kimi K2.7 Code | Chat Completions | text+image | 262K / 32K |
| `kimi-k2.6` | Kimi K2.6 | Chat Completions | text+image | 262K / 65K |
| `glm-5.2` | GLM 5.2 | Chat Completions | text | 1M / 131K |
| `glm-5.1` | GLM 5.1 | Chat Completions | text | 198K / 65K |

`gpt-5.6-luna` 通过 `/responses` 请求，支持 `none`、`low`、`medium`、`high`、`xhigh`、`max` 六档 reasoning effort；Bampy 的 `minimal` 会映射为后端的 `low`。其 Go 用量费率在输入上下文超过 272K tokens 时会对整次请求切换到高上下文档位，Bampy 会据此计算 usage cost。

Chat Completions 模型通过 `OpenAIChatCompat` 处理 thinking / `reasoning_content` 等差异。例如 `kimi-k3` 默认开启 thinking，并支持后端的 `low` / `high` / `max` 三档 effort（`minimal`/`low` → `low`，`medium`/`high` → `high`，`xhigh`/`max` → `max`）；`glm-5.2` 有独立的 effort 映射表。所有模型均需向网关传入有效 `api_key`。

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

概要映射（细节以各 adapter 源码为准；Gemini 4 和 3.1 Pro 的 `minimal`→`low`、`medium`→`medium`，其余与 Flash 映射一致）：

| ThinkingLevel | Anthropic | OpenAI Responses | DeepSeek v4 | Gemini 3 Flash | Gemini 3.0 Pro | Gemini 2.5 |
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
