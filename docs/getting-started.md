# 快速开始

## 安装

```bash
# 基础安装
uv add bampy

# 按需安装提供商 SDK
uv add "bampy[anthropic]"       # Anthropic Claude
uv add "bampy[openai]"          # OpenAI GPT
uv add "bampy[google]"          # Google Gemini
uv add "bampy[all-providers]"   # 全部

# 开发依赖
uv add "bampy[dev]"
```

> **要求**: Python >= 3.12

## 第一个 Agent

使用 `create_agent_session` 一行创建完整的 Agent 应用：

```python
import asyncio
from bampy.app import create_agent_session
from bampy.ai import get_model

async def main():
    result = await create_agent_session(
        model=get_model("anthropic", "claude-sonnet-4-20250514"),
    )
    session = result.session

    # 订阅流式输出
    def on_event(event):
        if event.type == "message_update":
            ame = event.assistant_message_event
            if hasattr(ame, "delta"):
                print(ame.delta, end="", flush=True)
        elif event.type == "agent_end":
            print()

    session.subscribe(on_event)

    await session.prompt("用 Python 写一个快速排序")
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
        model=get_model("anthropic", "claude-sonnet-4-20250514"),
        tools=[get_weather],
        custom_system_prompt="你是一个天气助手。",
    )
    session = result.session

    session.subscribe(lambda e: (
        print(e.assistant_message_event.delta, end="", flush=True)
        if e.type == "message_update" and hasattr(e.assistant_message_event, "delta")
        else None
    ))

    await session.prompt("北京和上海今天天气怎么样？")
    await session.close()

asyncio.run(main())
```

## 纯 LLM 调用（不需要 Agent）

如果只需要调用 LLM，可以直接使用 `bampy.ai`：

```python
import asyncio
from bampy.ai import stream_simple, get_model, UserMessage, Context, SimpleStreamOptions

async def main():
    model = get_model("anthropic", "claude-sonnet-4-20250514")
    ctx = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="What is 2+2?")],
    )

    response = stream_simple(model, ctx, SimpleStreamOptions(reasoning="medium"))
    async for event in response:
        if event.type == "text_delta":
            print(event.delta, end="")
    print()

    result = await response.result()
    print(f"Tokens: {result.usage.total_tokens}, Cost: ${result.usage.cost.total:.4f}")

asyncio.run(main())
```

## 下一步

- [架构说明](architecture.md) — 了解三层架构
- [AgentSession](agent-session.md) — 完整的应用层 API
- [工具](tools.md) — 创建自定义工具
- [扩展](extensions.md) — 开发扩展插件
