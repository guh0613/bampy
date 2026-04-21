# AI 层

`bampy.ai` 是最底层的 LLM 提供商抽象，定义统一的类型系统、流式事件协议和模型注册表。

## 核心类型

### 消息

```python
from bampy.ai import UserMessage, AssistantMessage, ToolResultMessage, Message

# 用户消息
UserMessage(content="Hello")
UserMessage(content=[TextContent(text="Hello"), ImageContent(data="...", mime_type="image/png")])

# 助手消息（由 LLM 返回）
# 包含 api, provider, model, usage, stop_reason 等元数据

# 工具结果消息
ToolResultMessage(
    tool_call_id="call_123",
    tool_name="get_weather",
    content=[TextContent(text="晴天 25°C")],
    is_error=False,
)

# 联合类型
Message = UserMessage | AssistantMessage | ToolResultMessage
```

### 内容块

```python
from bampy.ai import TextContent, ThinkingContent, ImageContent, ToolCall

TextContent(text="Hello")
ThinkingContent(thinking="Let me think...")  # 推理内容
ImageContent(data="base64...", mime_type="image/png")
ToolCall(id="call_123", name="search", arguments={"query": "foo"})
```

### 调用上下文

```python
from bampy.ai import Context, Tool

ctx = Context(
    system_prompt="You are helpful.",
    messages=[UserMessage(content="Hello")],
    tools=[Tool(
        name="search",
        description="搜索",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    )],
)
```

### 停止原因

```python
from bampy.ai import StopReason

StopReason.STOP       # 正常结束
StopReason.LENGTH     # 达到 max_tokens
StopReason.TOOL_USE   # 需要工具调用
StopReason.ERROR      # 错误
StopReason.ABORTED    # 被取消
```

### 推理等级

```python
from bampy.ai import ThinkingLevel

ThinkingLevel.MINIMAL  # 最少推理
ThinkingLevel.LOW
ThinkingLevel.MEDIUM
ThinkingLevel.HIGH
ThinkingLevel.XHIGH    # 最深度推理
```

### 用量与费用

```python
# AssistantMessage.usage 自动填充
message = await response.result()
print(message.usage.input)         # 输入 token 数
print(message.usage.output)        # 输出 token 数
print(message.usage.cache_read)    # 缓存读取
print(message.usage.total_tokens)  # 总 token
print(message.usage.cost.total)    # 总费用（美元）
```

## 流式调用

两组函数：`stream_simple` 接受通用选项，`stream` 接受提供商特有选项。

```python
from bampy.ai import stream, stream_simple, SimpleStreamOptions

# 通用选项（自动映射到提供商参数）
response = stream_simple(model, ctx, SimpleStreamOptions(
    temperature=0.7,
    max_tokens=4096,
    reasoning="medium",
    api_key="sk-...",
))

# 消费事件
async for event in response:
    match event.type:
        case "start":           pass  # 流开始
        case "text_start":      pass  # 文本块开始
        case "text_delta":      print(event.delta, end="")
        case "text_end":        pass  # 文本块结束
        case "thinking_start":  pass  # 推理开始
        case "thinking_delta":  pass  # 推理增量
        case "thinking_end":    pass  # 推理结束
        case "toolcall_start":  pass  # 工具调用开始
        case "toolcall_delta":  pass  # 工具调用参数增量
        case "toolcall_end":    pass  # 工具调用结束
        case "done":            pass  # 完成
        case "error":           print(f"Error: {event.error.error_message}")

# 获取最终消息
final = await response.result()
```

## 非流式调用

```python
from bampy.ai import complete, complete_simple

message = await complete_simple(model, ctx, SimpleStreamOptions(reasoning="medium"))
print(message.content[0].text)
```

## EventStream

`EventStream[T, R]` 是通用的异步可迭代事件流：

```python
from bampy.ai import EventStream, AssistantMessageEventStream

# AssistantMessageEventStream = EventStream[AssistantMessageEvent, AssistantMessage]

# 生产者 API（提供商适配器使用）
stream = EventStream()
stream.push(event)          # 推入事件
stream.end(result)          # 标记完成
stream.error(exception)     # 标记错误

# 消费者 API
async for event in stream:  # 迭代事件
    ...
result = await stream.result()  # 获取最终结果
```

## 模型注册表

```python
from bampy.ai import get_model, get_models, get_providers, register_model, Model, ModelCost

# 获取内置模型
model = get_model("anthropic", "claude-sonnet-4-20250514")

# 列出模型/提供商
all_models = get_models()
providers = get_providers()

# 注册自定义模型
register_model(Model(
    id="my-model",
    name="My Model",
    api="openai-completions",  # 或 "openai-responses"
    provider="custom",
    base_url="https://my-api.example.com/v1",
    context_window=128_000,
    max_tokens=8192,
    cost=ModelCost(input=1.0, output=3.0),
))
```

## 提供商特有选项

```python
from bampy.ai import AnthropicOptions, AnthropicThinkingAdaptive, OpenAIOptions, GeminiOptions

# Anthropic
response = stream(model, ctx, AnthropicOptions(
    thinking=AnthropicThinkingAdaptive(effort="high"),
    interleaved_thinking=True,
    cache_retention="long",
))

# OpenAI
response = stream(model, ctx, OpenAIOptions(
    reasoning_effort="high",
))

# Gemini
response = stream(model, ctx, GeminiOptions(
    thinking_budget=8192,
))
```

详见 [提供商文档](providers.md)。

## 工具校验

```python
from bampy.ai import validate_tool_arguments, schema_from_model, parse_partial_json
from pydantic import BaseModel

class Params(BaseModel):
    city: str
    unit: str = "celsius"

schema = schema_from_model(Params)
validated = validate_tool_arguments({"city": "Beijing"}, schema)

# 流式工具调用中解析不完整 JSON
partial = parse_partial_json('{"city": "Bei')
```
