# 自定义提供商

两种扩展路径：

1. **只换端点 / 模型**：`register_model(...)`，把 `api` 设为已有值（`openai-responses`、`openai-completions`、`anthropic-messages`、`google-genai`）。见 [providers.md](providers.md)。
2. **全新协议**：实现 `stream` / `stream_simple`，用 `register_api_provider` 挂到新的 `Model.api`，再 `register_model`。

下文描述第 2 种。

## 接口约定

### ApiProviderEntry

```python
from bampy.ai.api_registry import ApiProviderEntry

ApiProviderEntry(
    api="my-provider-api",   # 必须与 Model.api 一致
    stream=stream_fn,        # (model, context, options=None) -> AssistantMessageEventStream
    stream_simple=simple_fn, # 同上，但 options 为 SimpleStreamOptions
)
```

`register_api_provider` 会在调用时校验 `model.api == api`，不匹配则抛 `ValueError`。

### 注册

```python
from bampy.ai.api_registry import (
    ApiProviderEntry,
    register_api_provider,
    unregister_api_provider,
    unregister_api_providers,
)

# 立即注册
register_api_provider(
    "my-provider-api",
    entry=ApiProviderEntry(
        api="my-provider-api",
        stream=stream_my_provider,
        stream_simple=stream_simple_my_provider,
    ),
    source_id="my-plugin",  # 可选，便于按来源批量注销
)

# 或懒加载（loader 必须返回 ApiProviderEntry，不是函数元组）
def _load_my_provider() -> ApiProviderEntry:
    from my_pkg.provider import get_provider_entry
    return get_provider_entry()

register_api_provider(
    "my-provider-api",
    loader=_load_my_provider,
    source_id="my-plugin",
)
```

内置提供商用 `register_builtin_providers()` 注册，`source_id` 为 `bampy.ai.providers.builtin`。自定义提供商**不必**改 `providers/registry.py`；在应用启动时调用上述 API 即可。

### 注册模型

```python
from bampy.ai import register_model, Model, ModelCost

register_model(Model(
    id="my-model-v1",
    name="My Model v1",
    api="my-provider-api",
    provider="my-provider",
    base_url="https://api.example.com",
    reasoning=True,
    input_types=["text", "image"],  # 纯文本写 ["text"]
    context_window=128_000,
    max_tokens=8192,
    cost=ModelCost(input=1.0, output=3.0),
))
```

查找：`get_model("my-model-v1", provider="my-provider")`（**model_id 在前**）。

`BUILTIN_MODELS` 是 `dict[str, tuple[Model, ...]]`，不是 list；运行时扩展请用 `register_model`，不要 `BUILTIN_MODELS.append(...)`。

## 实现适配器

推荐结构（与内置适配器一致）：用 `AssistantMessageEventStream` + `spawn_provider_task`，错误时推 `ErrorEvent` 而不是让任务裸抛。

```python
# my_pkg/provider.py
from __future__ import annotations

from bampy.ai.api_registry import ApiProviderEntry
from bampy.ai.models import calculate_cost
from bampy.ai.providers._cancellation import spawn_provider_task
from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    Usage,
)


def stream_my_provider(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    event_stream = AssistantMessageEventStream()
    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        content=[],
    )

    async def _run() -> None:
        try:
            # 按需导入第三方 SDK，避免未安装时 import 失败
            # import my_sdk

            event_stream.push(StartEvent(partial=output))

            api_key = options.api_key if options else None
            # client = my_sdk.AsyncClient(api_key=api_key, base_url=model.base_url or None)
            # response = await client.stream(...)

            content_index = 0
            text_block = TextContent(text="")
            output.content.append(text_block)
            event_stream.push(
                TextStartEvent(
                    content_index=content_index,
                    content=text_block,
                    partial=output,
                )
            )

            # async for chunk in response:
            #     text_block.text += chunk.text
            #     event_stream.push(TextDeltaEvent(...))

            event_stream.push(
                TextEndEvent(
                    content_index=content_index,
                    content=text_block,
                    partial=output,
                )
            )

            output.usage = Usage(input=0, output=0)
            output.usage.cost = calculate_cost(model, output.usage)
            output.stop_reason = StopReason.STOP
            event_stream.push(DoneEvent(reason=StopReason.STOP, message=output))
            event_stream.end(output)
        except Exception as exc:
            output.stop_reason = StopReason.ERROR
            output.error_message = str(exc)
            event_stream.push(ErrorEvent(reason=StopReason.ERROR, error=output))
            event_stream.end(output)

    spawn_provider_task(
        event_stream=event_stream,
        output=output,
        options=options,
        runner=_run,
    )
    return event_stream


def stream_simple_my_provider(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    mapped = StreamOptions(
        temperature=options.temperature if options else None,
        max_tokens=options.max_tokens if options else None,
        api_key=options.api_key if options else None,
        max_retry_delay_ms=options.max_retry_delay_ms if options else 60_000,
        headers=options.headers if options else None,
        cancellation=options.cancellation if options else None,
    )
    # 若支持推理，在此把 options.reasoning 映射到提供商特有字段
    return stream_my_provider(model, context, mapped)


def get_provider_entry() -> ApiProviderEntry:
    return ApiProviderEntry(
        api="my-provider-api",
        stream=stream_my_provider,
        stream_simple=stream_simple_my_provider,
    )
```

应用启动：

```python
from bampy.ai import (
    Context,
    Model,
    ModelCost,
    UserMessage,
    get_model,
    register_model,
    stream_simple,
)
from bampy.ai.api_registry import register_api_provider
from my_pkg.provider import get_provider_entry

register_api_provider("my-provider-api", entry=get_provider_entry(), source_id="my-plugin")
register_model(Model(
    id="my-model-v1",
    name="My Model v1",
    api="my-provider-api",
    provider="my-provider",
    context_window=128_000,
    max_tokens=8192,
    cost=ModelCost(input=1.0, output=3.0),
))

model = get_model("my-model-v1", "my-provider")
# response = stream_simple(model, Context(messages=[UserMessage(content="hi")]))
```

可选依赖可在你自己的发行包中声明；若贡献回 bampy，再在根目录 `pyproject.toml` 的 `[project.optional-dependencies]` 增加条目。

## 事件协议

适配器应按顺序推送：

```text
StartEvent
  └─ 每个内容块（可交错）:
     ├─ TextStartEvent → TextDeltaEvent* → TextEndEvent
     ├─ ThinkingStartEvent → ThinkingDeltaEvent* → ThinkingEndEvent
     └─ ToolCallStartEvent → ToolCallDeltaEvent* → ToolCallEndEvent
DoneEvent 或 ErrorEvent
```

要求：

- 每个事件携带最新的 `partial: AssistantMessage`
- `DoneEvent.message` 为最终完整消息；成功时调用 `event_stream.end(output)`
- 业务失败推 `ErrorEvent` 并 `end`，不要依赖未捕获异常冒泡（`spawn_provider_task` 会把 `CancelledError` 转成 `ABORTED`）
- 流对象应在任务启动后立即返回，供调用方 `async for`

## 消息与图片

需要统一 tool_call_id 清洗或非视觉模型降级图片时，可复用：

```python
from bampy.ai.providers._transform import transform_messages, sanitize_tool_call_id
```

`input_types` 必须反映真实能力；不含 `"image"` 时，转换层会把图片换成文本占位后再进入请求体。

## 注销

```python
from bampy.ai.api_registry import unregister_api_provider, unregister_api_providers, clear_api_providers

unregister_api_provider("my-provider-api")
unregister_api_providers("my-plugin")  # 按 source_id
# clear_api_providers()  # 清空全部（测试慎用）
```

## 与内置注册的关系

`stream` / `stream_simple` 在找不到 `model.api` 时会调用 `ensure_builtin_providers_registered()` 再试一次。自定义 API 必须**显式** `register_api_provider`；不会自动出现在内置列表中。

相关文档：[ai-layer.md](ai-layer.md)、[providers.md](providers.md)。
