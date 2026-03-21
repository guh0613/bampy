# bampy

模块化、异步优先的 Python LLM Agent 框架。

bampy 提供从底层 LLM API 调用到上层 Agent 应用的完整抽象，采用三层分离架构，支持多提供商、流式事件、工具调用、会话持久化和扩展系统。

## 快速开始

```bash
uv add "bampy[anthropic]"   # 或 openai / google / all-providers
```

```python
import asyncio
from bampy.app import create_agent_session
from bampy.ai import get_model

async def main():
    result = await create_agent_session(
        model=get_model("anthropic", "claude-sonnet-4-20250514"),
    )
    session = result.session
    session.subscribe(lambda e: (
        print(e.assistant_message_event.delta, end="", flush=True)
        if e.type == "message_update"
        and hasattr(e.assistant_message_event, "delta")
        else None
    ))
    await session.prompt("Hello!")
    await session.close()

asyncio.run(main())
```

更多示例 → [快速开始](docs/getting-started.md)

## 架构

```text
bampy.app    — AgentSession 编排 · 会话持久化 · 扩展 · 内置工具 · Skills
bampy.agent  — Agent 运行时 · 状态管理 · 工具执行 · 事件系统
bampy.ai     — 统一 LLM 类型 · 流式事件 · 模型注册 · 提供商适配
```

每层均可独立使用，详见 [架构说明](docs/architecture.md)。

## 文档

| 文档                                     | 说明                           |
| ---------------------------------------- | ------------------------------ |
| [快速开始](docs/getting-started.md)      | 安装、配置与示例               |
| [架构说明](docs/architecture.md)         | 三层架构设计与数据流            |
| [AgentSession](docs/agent-session.md)    | 应用层编排器（推荐入口）       |
| [Agent 运行时](docs/agent.md)            | Layer 2 Agent 类与 agent loop  |
| [AI 层](docs/ai-layer.md)               | LLM 调用、类型系统与流式事件   |
| [工具](docs/tools.md)                    | 内置工具与自定义工具开发       |
| [扩展](docs/extensions.md)               | 扩展系统、事件与生命周期       |
| [会话](docs/session.md)                  | NDJSON 会话持久化与分支        |
| [Skills](docs/skills.md)                 | Skill 发现与 SKILL.md 格式     |
| [上下文压缩](docs/compaction.md)         | 长对话自动压缩                 |
| [提供商](docs/providers.md)              | 内置 LLM 提供商配置            |
| [自定义提供商](docs/custom-provider.md)  | 添加新的 LLM 提供商            |

## 要求

- Python >= 3.12
- 核心依赖：`pydantic >= 2.0`
- 提供商 SDK 按需安装
