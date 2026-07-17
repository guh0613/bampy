# 快速开始

## 安装

用 `uv` 管理环境与依赖：

```bash
# 基础安装（不含提供商 SDK）
uv add bampy

# 按需安装提供商 SDK
uv add "bampy[anthropic]"       # Anthropic Claude
uv add "bampy[openai]"          # OpenAI
uv add "bampy[google]"          # Google Gemini
uv add "bampy[all-providers]"   # 上述全部

# 开发依赖（pytest / ruff）
uv add "bampy[dev]"
```

要求：Python >= 3.12。

## 环境变量

未显式传入 `api_key` 时，各提供商 SDK 会读取常见环境变量：

| 提供商 | 环境变量 |
| ------ | -------- |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google | `GEMINI_API_KEY` 或 `GOOGLE_API_KEY` |

也可在调用时注入：

```python
from bampy.ai import SimpleStreamOptions

# 传给 create_agent_session / Agent / stream_simple
stream_options = SimpleStreamOptions(api_key="...")
```

或通过 `create_agent_session(get_api_key=...)` / `Agent(get_api_key=...)` 按 provider 解析。

## 模型查找

`get_model` 的签名是 `(model_id, provider=None)`，**第一个参数是模型 ID**：

```python
from bampy.ai import get_model, get_models, get_providers

model = get_model("claude-sonnet-4-5", "anthropic")
# 或
model = get_model("claude-sonnet-4-5", provider="anthropic")

print(get_providers())                 # ['anthropic', 'google', ...]
print([m.id for m in get_models("anthropic")])
```

## 最小 Agent 示例

`create_agent_session` 会装配会话、扩展、Skills 与系统提示，并自动 `start()`：

```python
import asyncio
from bampy.app import create_agent_session
from bampy.ai import get_model

async def main():
    result = await create_agent_session(
        model=get_model("claude-sonnet-4-5", "anthropic"),
        tools=[],  # None 时默认加载编码工具集
        discover_extensions=False,
        include_default_skills=False,
    )
    session = result.session

    def on_event(event):
        if (
            event.type == "message_update"
            and event.assistant_message_event.type == "text_delta"
        ):
            print(event.assistant_message_event.delta, end="", flush=True)
        elif event.type == "agent_end":
            print()

    session.subscribe(on_event)
    await session.prompt("用一句话介绍你自己")
    await session.close()

asyncio.run(main())
```

## 带自定义工具

```python
import asyncio
from bampy.app import create_agent_session, tool
from bampy.ai import get_model

@tool(name="get_weather", description="获取指定城市的天气")
async def get_weather(city: str) -> str:
    return f"{city}今天晴，25°C"

async def main():
    result = await create_agent_session(
        model=get_model("claude-sonnet-4-5", "anthropic"),
        tools=[get_weather],
        custom_system_prompt="你是一个天气助手。",
        discover_extensions=False,
        include_default_skills=False,
    )
    session = result.session

    def on_event(event):
        if (
            event.type == "message_update"
            and event.assistant_message_event.type == "text_delta"
        ):
            print(event.assistant_message_event.delta, end="", flush=True)

    session.subscribe(on_event)
    await session.prompt("北京今天天气怎么样？")
    await session.close()

asyncio.run(main())
```

## 纯 LLM 调用（不经过 Agent）

只需调用模型时，直接用 `bampy.ai`：

```python
import asyncio
from bampy.ai import (
    stream_simple,
    get_model,
    UserMessage,
    Context,
    SimpleStreamOptions,
)

async def main():
    model = get_model("claude-sonnet-4-5", "anthropic")
    ctx = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="What is 2+2?")],
    )

    response = stream_simple(model, ctx, SimpleStreamOptions())
    async for event in response:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
    print()

    result = await response.result()
    print(f"Tokens: {result.usage.total_tokens}, Cost: ${result.usage.cost.total:.4f}")

asyncio.run(main())
```

阻塞式一次性完成可用 `complete_simple(model, ctx)`。

## 下一步

- [架构说明](architecture.md) — 三层职责与数据流
- [AgentSession](agent-session.md) — 应用层完整 API
- [工具](tools.md) — 内置工具与自定义工具
- [扩展](extensions.md) — 扩展与生命周期事件
- [提供商](providers.md) — 模型与 API Key 配置
