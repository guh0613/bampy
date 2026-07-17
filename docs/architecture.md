# 架构说明

bampy 采用三层分离架构（实现上对齐 pi-mono 的分层思想）：

```text
┌─────────────────────────────────────────────────────────┐
│  Layer 3: bampy.app — 应用层                             │
│  AgentSession · 会话持久化 · 扩展 · 内置工具 · Skills     │
├─────────────────────────────────────────────────────────┤
│  Layer 2: bampy.agent — Agent 运行时                      │
│  Agent Loop · 状态管理 · 工具执行 · 事件系统               │
├─────────────────────────────────────────────────────────┤
│  Layer 1: bampy.ai — LLM 提供商抽象                       │
│  统一类型 · 流式事件 · 模型注册 · 提供商适配                │
└─────────────────────────────────────────────────────────┘
```

**依赖方向**：`app → agent → ai`（上层依赖下层，下层不依赖上层）。

顶层入口包 `bampy` 仅导出 `__version__`；日常使用从各层子包导入，例如 `from bampy.app import create_agent_session`。

## 各层职责

### Layer 1: `bampy.ai`

统一多提供商 LLM API。定义消息 / 内容块、流式事件、模型注册表，以及懒加载的提供商适配器。

核心概念：

| 概念 | 说明 |
| ---- | ---- |
| `Message` | `UserMessage \| AssistantMessage \| ToolResultMessage` |
| `AssistantMessageEvent` | 12 种细粒度流式事件（`text_delta`、`toolcall_delta` 等） |
| `EventStream` | 异步可迭代事件流，`result()` 取最终 `AssistantMessage` |
| `Model` | 模型元数据（api / provider / 费用 / 上下文窗口） |
| `stream` / `stream_simple` | 底层与简化选项的流式调用入口 |
| `get_model` | `(model_id, provider=None) → Model \| None` |

公开 API 见 `bampy.ai`；提供商细节见 [AI 层](ai-layer.md) 与 [提供商](providers.md)。

### Layer 2: `bampy.agent`

Agent 运行时：驱动对话、工具调用与取消。

核心概念：

| 概念 | 说明 |
| ---- | ---- |
| `Agent` | 有状态运行时（模型、消息、工具、订阅） |
| `agent_loop` / `run_agent_loop` | 底层循环；`Agent` 是其有状态封装 |
| `AgentTool` | 工具 Protocol：`name` / `description` / `parameters` / `execute` |
| `AgentEvent` | 10 种运行时事件（`message_update`、`tool_execution_*` 等） |
| `CancellationToken` | 异步取消 |

详见 [Agent 运行时](agent.md)。

### Layer 3: `bampy.app`

面向应用的装配层：会话、扩展、工具装饰器、压缩与 Skills。

核心概念：

| 概念 | 说明 |
| ---- | ---- |
| `create_agent_session` | 工厂：加载扩展 / Skills / 上下文文件，创建并 `start` session |
| `AgentSession` | 顶层编排器；`prompt` / `subscribe` / 自动压缩 |
| `SessionManager` | NDJSON 追加写入的会话持久化 |
| `ExtensionRunner` | 扩展生命周期与事件分发 |
| `@tool` | 将 async 函数包装为 `AgentTool` |
| `CompactionSettings` | 长对话上下文压缩 |
| `load_skills` | 发现并加载 `SKILL.md` |

详见 [AgentSession](agent-session.md)、[会话](session.md)、[扩展](extensions.md)、[工具](tools.md)、[Skills](skills.md)、[上下文压缩](compaction.md)。

## 数据流

一次典型的 `AgentSession.prompt()`：

```text
用户输入
  │
  ▼
AgentSession.prompt()
  ├─ InputEvent          → 扩展可拦截 / 变换
  ├─ BeforeAgentStartEvent → 扩展可改 system prompt
  │
  ▼
Agent.prompt()
  │
  ▼
Agent Loop（内层）
  ├─ transform_context() → ContextEvent（扩展）
  ├─ convert_to_llm()    → 转为 LLM 消息
  ├─ stream_simple()     → Provider Adapter → SDK
  ├─ MessageUpdate       → 扩展 + session 订阅者
  ├─ 若有 tool_use
  │   ├─ ToolCallEvent → 扩展可阻止
  │   ├─ AgentTool.execute()
  │   └─ ToolResultEvent → 扩展可改结果
  └─ 无工具调用则结束内层
  │
  ▼
Agent Loop（外层 follow-up）
  └─ 有 follow-up 则回到内层
  │
  ▼
AgentSession
  ├─ 持久化到 SessionManager
  ├─ 按阈值触发 auto-compaction
  └─ AgentEndEvent
```

`AgentSessionEvent` = `AgentEvent` ∪ `auto_compaction_start` / `auto_compaction_end`。

## 何时独立使用各层

```python
# Layer 1：只要 LLM 调用
from bampy.ai import stream_simple, complete_simple, get_model, Context, UserMessage

# Layer 2：自建循环 / 嵌入自定义运行时
from bampy.agent import Agent, agent_loop

# Layer 3：完整应用（推荐默认入口）
from bampy.app import create_agent_session, tool
```

选择建议：

- 脚本式问答、评测、代理转发 → `bampy.ai`
- 需要工具循环但不想要会话 / 扩展 → `bampy.agent.Agent`
- CLI / 产品应用、要持久化与插件 → `create_agent_session`

## 设计原则

- **懒加载**：提供商模块在首次 `stream` / `stream_simple` 时加载
- **协议优先**：`AgentTool`、`SessionBackend`、`StreamFunction` 等用 Protocol
- **事件驱动**：`AsyncIterator` 流式事件 + `subscribe` 回调
- **Pydantic v2**：消息与选项模型统一校验
- **追加写入**：会话日志 append-only，崩溃安全
- **组合优于继承**：工厂（`create_agent_session`）与 hook 组装行为

## 相关文档

| 文档 | 内容 |
| ---- | ---- |
| [快速开始](getting-started.md) | 安装与可运行示例 |
| [AgentSession](agent-session.md) | Layer 3 API |
| [Agent 运行时](agent.md) | Layer 2 |
| [AI 层](ai-layer.md) | Layer 1 |
| [自定义提供商](custom-provider.md) | 扩展 Layer 1 |
