# AI 层

`bampy.ai` 是 bampy 最底层的 LLM 抽象：统一消息类型、流式事件协议、模型注册表，以及按 `Model.api` 分发的提供商适配。上层 `Agent` / `create_agent_session` 通过 `stream_simple`（或自定义 `stream_fn`）调用本层。

公开符号从 `bampy.ai` 导入即可。

## 核心类型

### 消息

```python
from bampy.ai import (
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    TextContent,
    ImageContent,
)

UserMessage(content="Hello")
UserMessage(content=[
    TextContent(text="看这张图"),
    ImageContent(data="...", mime_type="image/png"),
])

# AssistantMessage 由提供商适配器填充：
# api / provider / model / usage / stop_reason / error_message 等

ToolResultMessage(
    tool_call_id="call_123",
    tool_name="get_weather",
    content=[TextContent(text="晴天 25°C")],
    is_error=False,
)
```

`Message` 是以 `role` 区分的判别联合：`UserMessage | AssistantMessage | ToolResultMessage`。

### 内容块

```python
from bampy.ai import TextContent, ThinkingContent, ImageContent, ToolCall

TextContent(text="Hello")
ThinkingContent(thinking="Let me think...")
ImageContent(data="base64...", mime_type="image/png")
ToolCall(id="call_123", name="search", arguments={"query": "foo"})
```

- 用户 / 工具结果内容：`TextContent`、`ImageContent`
- 助手内容：`TextContent`、`ThinkingContent`、`ToolCall`

### 调用上下文

```python
from bampy.ai import Context, Tool, UserMessage

ctx = Context(
    system_prompt="You are helpful.",
    messages=[UserMessage(content="Hello")],
    tools=[
        Tool(
            name="search",
            description="搜索",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
    ],
)
```

### 停止原因与推理等级

```python
from bampy.ai import StopReason, ThinkingLevel

StopReason.STOP       # 正常结束
StopReason.LENGTH     # 达到 max_tokens
StopReason.TOOL_USE   # 需要工具调用
StopReason.ERROR      # 错误
StopReason.ABORTED    # 被取消

ThinkingLevel.MINIMAL
ThinkingLevel.LOW
ThinkingLevel.MEDIUM
ThinkingLevel.HIGH
ThinkingLevel.XHIGH
ThinkingLevel.MAX
```

### 用量与费用

```python
message = await complete_simple(model, ctx)
print(message.usage.input)
print(message.usage.output)
print(message.usage.cache_read)
print(message.usage.cache_write)
print(message.usage.total_tokens)
print(message.usage.cost.total)  # 美元，按 Model.cost（$/百万 token）计算
```

也可用 `calculate_cost(model, usage)` 单独计算。

## 直接调用 LLM

两组入口：

| 函数 | 选项类型 | 用途 |
| ---- | -------- | ---- |
| `stream` / `complete` | `StreamOptions` 或提供商特有子类 | 精确控制（如 `AnthropicOptions`） |
| `stream_simple` / `complete_simple` | `SimpleStreamOptions` | 提供商无关；`reasoning` 等会映射到各 API |

```python
import asyncio
from bampy.ai import (
    Context,
    UserMessage,
    get_model,
    stream_simple,
    complete_simple,
    SimpleStreamOptions,
)

async def main():
    model = get_model("claude-sonnet-4-5", provider="anthropic")
    ctx = Context(messages=[UserMessage(content="你好")])

    # 流式
    response = stream_simple(
        model,
        ctx,
        SimpleStreamOptions(temperature=0.7, max_tokens=4096, reasoning="medium"),
    )
    async for event in response:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
    final = await response.result()

    # 非流式（内部仍走流，只 await 最终消息）
    message = await complete_simple(
        model,
        ctx,
        SimpleStreamOptions(reasoning="medium", api_key="..."),
    )
    print(message.content)

asyncio.run(main())
```

未传 `api_key` 时，各 SDK 会读取自身默认环境变量（见 [providers.md](providers.md)）。上层 Agent 还可通过 `get_api_key(provider)` 动态注入。

### 流事件

```python
async for event in response:
    match event.type:
        case "start":
            pass
        case "text_start" | "text_delta" | "text_end":
            pass
        case "thinking_start" | "thinking_delta" | "thinking_end":
            pass
        case "toolcall_start" | "toolcall_delta" | "toolcall_end":
            pass
        case "done":
            message = event.message
        case "error":
            print(event.error.error_message)
```

每个中间事件都带有 `partial: AssistantMessage`（当前累积状态）。流以 `done` 或 `error` 结束；`await response.result()` 返回最终的 `AssistantMessage`。

## EventStream

`EventStream[T, R]` 是通用异步事件流；`AssistantMessageEventStream` 特化为 `EventStream[AssistantMessageEvent, AssistantMessage]`。

提供商适配器作为生产者：

```python
from bampy.ai import AssistantMessageEventStream

stream = AssistantMessageEventStream()
stream.push(event)       # 推入；遇到 done/error 时自动 end
stream.end(result)       # 显式结束
stream.error(exc)        # 以异常结束
```

消费者：

```python
async for event in stream:
    ...
result = await stream.result()
```

内置适配器通过 `spawn_provider_task` 在后台跑请求，并支持可选的 `cancellation` 令牌。

## 模型注册表

**重要**：`get_model` 的签名是 `get_model(model_id, provider=None)`，第一个参数是模型 ID。

```python
from bampy.ai import (
    get_model,
    get_models,
    get_providers,
    register_model,
    models_are_equal,
    supports_xhigh,
    Model,
    ModelCost,
)

# 正确
model = get_model("claude-sonnet-4-5", provider="anthropic")
model = get_model("claude-sonnet-4-5", "anthropic")
model = get_model("claude-sonnet-4-5")  # 跨提供商按注册顺序取第一个同名 ID

# 错误（旧文档常见写法）：get_model("anthropic", "claude-...")

get_providers()                 # ['anthropic', 'google', 'opencode-go', 'ollama', ...]
get_models("openai")            # 某提供商下全部 Model
models_are_equal(a, b)          # 比较 id + provider
supports_xhigh(model)           # 部分 GPT / Claude Opus 4.7 家族
```

`Model` 关键字段：

| 字段 | 含义 |
| ---- | ---- |
| `id` / `name` | 模型标识与展示名 |
| `api` | 路由键，对应 API 提供商注册（如 `anthropic-messages`） |
| `provider` | 逻辑提供商名（如 `anthropic`、`deepseek`） |
| `base_url` | 可选覆盖端点 |
| `reasoning` | 是否支持推理相关选项 |
| `input_types` | 如 `["text"]` 或 `["text", "image"]` |
| `context_window` / `max_tokens` | 上下文与输出上限 |
| `cost` | `$ / 百万 token` |
| `openai_chat_compat` | Chat Completions 兼容差异（DeepSeek / OpenCode Go 等） |

内置模型定义在 `BUILTIN_MODELS`（按 provider 分组的元组）；`MODELS` 是其扁平列表。运行时查找走 `register_model` 维护的 `_model_map`。完整列表见 [providers.md](providers.md)。

### 图片能力与消息转换

`Model.input_types` 决定请求侧能力。若模型不含 `"image"`，统一转换层会在发往提供商前把用户消息 / 工具结果中的图片替换为文本占位，且只作用于本次请求副本，不改会话历史。

### 注册自定义模型（复用已有 API）

```python
register_model(Model(
    id="my-model",
    name="My Model",
    api="openai-completions",  # 或 openai-responses / anthropic-messages / google-genai
    provider="custom",
    base_url="https://my-api.example.com/v1",
    input_types=["text"],
    context_window=128_000,
    max_tokens=8192,
    cost=ModelCost(input=1.0, output=3.0),
))
```

全新 API 适配见 [custom-provider.md](custom-provider.md)。

## 提供商特有选项

```python
from bampy.ai import (
    stream,
    AnthropicOptions,
    AnthropicThinkingAdaptive,
    OpenAIOptions,
    GeminiOptions,
)

stream(model, ctx, AnthropicOptions(
    thinking=AnthropicThinkingAdaptive(effort="high", display="summarized"),
    interleaved_thinking=True,
    cache_retention="long",
))

stream(model, ctx, OpenAIOptions(reasoning_effort="high"))

stream(model, ctx, GeminiOptions(thinking_level="medium"))
# Gemini 2.5 系列也可用 thinking_budget=8192
```

`SimpleStreamOptions` 在 `temperature` / `max_tokens` / `api_key` / `headers` / `cancellation` 之外增加 `reasoning: ThinkingLevel | None`，由各 `stream_simple_*` 映射。详见 [providers.md](providers.md)。

## API 提供商注册（懒加载）

调用 `stream` / `stream_simple` 时，会按 `model.api` 查找 `api_registry`。内置适配器在首次查找失败时通过 `ensure_builtin_providers_registered()` 懒注册，避免 import 时污染全局状态。

已注册的 API 名：

| `Model.api` | 适配模块 |
| ----------- | -------- |
| `anthropic-messages` | `providers.anthropic` |
| `openai-responses` | `providers.openai`（Responses；Ollama v0.13.3+ 可复用） |
| `openai-completions` | `providers.openai`（Chat Completions；Ollama 默认建议） |
| `google-genai` | `providers.gemini` |

也可显式调用 `register_builtin_providers()` / `ensure_builtin_providers_registered()`。

## 工具校验

```python
from bampy.ai import (
    validate_tool_arguments,
    validate_tool_call,
    schema_from_model,
    parse_partial_json,
    Tool,
    ToolCall,
)
from pydantic import BaseModel

class Params(BaseModel):
    city: str
    unit: str = "celsius"

schema = schema_from_model(Params)
validated = validate_tool_arguments({"city": "Beijing"}, schema)

validate_tool_call(
    [Tool(name="weather", description="...", parameters=schema)],
    ToolCall(id="1", name="weather", arguments={"city": "Beijing"}),
)

partial = parse_partial_json('{"city": "Bei')  # 流式参数尽力解析
```

失败时抛出 `ToolValidationError`。

## 与上层的关系

```text
create_agent_session / Agent
        │
        ▼
agent.loop（组装 Context、解析 api_key、cancellation）
        │
        ▼
stream_simple(model, Context, SimpleStreamOptions)   ← 默认
  或自定义 stream_fn / stream(... 提供商选项)
        │
        ▼
api_registry[model.api] → 具体 provider adapter
        │
        ▼
AssistantMessageEventStream → Agent 事件 / 会话消息
```

本层不负责任务循环、工具执行或会话持久化；那些由 `bampy.agent` 与 `bampy.app` 负责。直接使用本层时，自行构造 `Context` 并消费事件即可。
