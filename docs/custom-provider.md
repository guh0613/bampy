# 自定义提供商

如果需要接入 bampy 尚未内置的 LLM API（且非 OpenAI 兼容），可以添加自定义提供商适配器。

> 如果你的 API 是 OpenAI 兼容的，只需 `register_model` 设置 `api="openai-responses"` 和 `base_url` 即可，无需写适配器。见 [提供商文档](providers.md#自定义模型)。

## 步骤

### 1. 创建适配模块

```python
# bampy/ai/providers/my_provider.py
import asyncio

from bampy.ai.stream import AssistantMessageEventStream
from bampy.ai.types import (
    AssistantMessage, Context, DoneEvent, ErrorEvent, Model,
    SimpleStreamOptions, StartEvent, StopReason, StreamOptions,
    TextContent, TextDeltaEvent, TextStartEvent, TextEndEvent,
    Usage,
)


def stream_my_provider(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """底层流式调用，接受提供商特有选项。"""
    stream = AssistantMessageEventStream()

    async def _run():
        output = AssistantMessage(
            api=model.api,
            provider=model.provider,
            model=model.id,
        )
        try:
            stream.push(StartEvent(partial=output))

            # 调用你的 SDK
            client = MySDK(api_key=options.api_key if options else None)
            response = await client.create(
                model=model.id,
                messages=_convert_messages(context),
            )

            # 映射响应事件
            content_index = 0
            text_block = TextContent(text="")
            output.content.append(text_block)
            stream.push(TextStartEvent(
                content_index=content_index, content=text_block, partial=output,
            ))

            async for chunk in response:
                text_block.text += chunk.text
                stream.push(TextDeltaEvent(
                    content_index=content_index, delta=chunk.text, partial=output,
                ))

            stream.push(TextEndEvent(
                content_index=content_index, content=text_block, partial=output,
            ))

            # 完成
            output.usage = Usage(input=response.input_tokens, output=response.output_tokens)
            output.stop_reason = StopReason.STOP
            stream.push(DoneEvent(reason=StopReason.STOP, message=output))
        except Exception as e:
            output.stop_reason = StopReason.ERROR
            output.error_message = str(e)
            stream.push(ErrorEvent(reason=StopReason.ERROR, error=output))

    asyncio.create_task(_run())
    return stream


def stream_simple_my_provider(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """简化版，将通用选项映射到提供商特有选项。"""
    provider_options = StreamOptions(
        temperature=options.temperature if options else None,
        max_tokens=options.max_tokens if options else None,
        api_key=options.api_key if options else None,
    )
    return stream_my_provider(model, context, provider_options)
```

### 2. 注册懒加载

```python
# bampy/ai/providers/registry.py

async def _load_my_provider():
    from bampy.ai.providers.my_provider import stream_my_provider, stream_simple_my_provider
    return stream_my_provider, stream_simple_my_provider

def register_builtin_providers() -> None:
    # ... 现有注册 ...
    register_api_provider("my-provider-api", loader=_load_my_provider, source_id="builtin")
```

### 3. 添加模型

```python
# bampy/ai/models.py

BUILTIN_MODELS.append(Model(
    id="my-model-v1",
    name="My Model v1",
    api="my-provider-api",      # 对应注册的 API 名
    provider="my-provider",
    context_window=128_000,
    max_tokens=8192,
    reasoning=True,             # 是否支持推理
    input_types=["text", "image"],
    cost=ModelCost(input=1.0, output=3.0),
))
```

### 4. 添加可选依赖

```toml
# pyproject.toml
[project.optional-dependencies]
my-provider = ["my-sdk>=1.0"]
```

## 事件协议

适配器必须按顺序推送以下事件：

```text
StartEvent
  └─ 每个内容块:
     ├─ TextStartEvent → TextDeltaEvent* → TextEndEvent
     ├─ ThinkingStartEvent → ThinkingDeltaEvent* → ThinkingEndEvent
     └─ ToolCallStartEvent → ToolCallDeltaEvent* → ToolCallEndEvent
DoneEvent 或 ErrorEvent
```

关键要求：
- 每个事件都携带 `partial: AssistantMessage`，是当前累积状态
- `DoneEvent.message` 是最终完整消息
- 如果出错，推送 `ErrorEvent` 而非抛异常
- `asyncio.create_task(_run())` 确保流立即返回
