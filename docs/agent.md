# Agent 运行时

`bampy.agent` 提供 Layer 2 的状态机与 agent loop，驱动 LLM 对话与工具调用。

> 大多数场景建议使用 [AgentSession](agent-session.md)。直接使用 `Agent` 适合：无持久化会话、自定义编排、或只需要纯内存 agent loop 的场景。

导入：`from bampy.agent import Agent, agent_loop, AgentContext, AgentLoopConfig, ...`。

## 何时用 Agent 而非 AgentSession

| 需求 | 选择 |
| ---- | ---- |
| 会话落盘、扩展、Skills、自动压缩 | [`AgentSession`](agent-session.md) |
| 纯内存对话 / 嵌入到自有编排 | `Agent` |
| 无状态、可流式消费事件的底层循环 | `agent_loop` / `run_agent_loop` |

## Agent 类

```python
from bampy.agent import Agent
from bampy.ai import get_model, SimpleStreamOptions

agent = Agent(
    initial_state={
        "system_prompt": "你是一个助手。",
        "model": get_model("claude-sonnet-4-5", provider="anthropic"),
        "tools": [my_tool],
        "thinking_level": "off",
    },
    stream_options=SimpleStreamOptions(),
    tool_execution="parallel",
    steering_mode="one-at-a-time",
    follow_up_mode="one-at-a-time",
    max_turns=30,
)
```

未传入 `model` 时，默认尝试 `get_model("gemini-3.1-flash-lite", provider="google")`。

### 构造参数

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `initial_state` | `AgentState \| Mapping \| None` | 空状态 + 默认模型 | 初始状态 |
| `convert_to_llm` | `ConvertToLlmFn \| None` | `default_convert_to_llm` | 消息转换 |
| `transform_context` | `TransformContextFn \| None` | None | LLM 调用前变换 |
| `steering_mode` | `str` | `"one-at-a-time"` | steering 出队模式 |
| `follow_up_mode` | `str` | `"one-at-a-time"` | follow-up 出队模式 |
| `stream_fn` | `StreamFn \| None` | None | 自定义流式函数 |
| `stream_options` | `SimpleStreamOptions \| None` | 默认 | 流式选项 |
| `get_api_key` | `ApiKeyResolverFn \| None` | None | 动态 API Key |
| `tool_execution` | `ToolExecutionMode \| str` | `"parallel"` | `parallel` / `sequential` |
| `before_tool_call` | `BeforeToolCallHook \| None` | None | 工具调用前 |
| `after_tool_call` | `AfterToolCallHook \| None` | None | 工具调用后 |
| `max_turns` | `int` | `50` | 最大轮数（≥ 1） |

### 执行

```python
from bampy.ai import UserMessage, ImageContent

await agent.prompt("Hello")
await agent.prompt(UserMessage(content="Hello"))
await agent.prompt([msg1, msg2])
await agent.prompt("描述这张图", images=[ImageContent(data="...", mime_type="image/png")])

await agent.continue_()
agent.abort("User cancelled")
await agent.wait_for_idle()
agent.reset()
```

### 状态 `AgentState`

通过 `agent.state` 只读访问当前状态；用 setter / 消息 API 修改。

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `system_prompt` | `str` | 系统提示 |
| `model` | `Model` | 当前模型 |
| `thinking_level` | `AgentThinkingLevel` | `off` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max` |
| `tools` | `list[AgentTool]` | 工具列表 |
| `messages` | `list[AgentMessage]` | 对话消息 |
| `is_streaming` | `bool` | 是否在 loop 中 |
| `stream_message` | `AgentMessage \| None` | 当前流式消息 |
| `pending_tool_calls` | `set[str]` | 进行中的 tool call id |
| `error` | `str \| None` | 最近错误信息 |

```python
agent.set_system_prompt("...")
agent.set_model(model)
agent.set_thinking_level("high")
agent.set_tools([tool1, tool2])
agent.set_stream_options(SimpleStreamOptions())
agent.set_tool_execution("sequential")
agent.set_before_tool_call(hook)
agent.set_after_tool_call(hook)
agent.set_max_turns(20)
agent.append_message(msg)
agent.replace_messages([...])
agent.clear_messages()
```

`AgentThinkingLevel.OFF` 在发往 `bampy.ai` 前会转为 `None`（不开启 reasoning）。

### 事件

```python
def on_event(event):
    match event.type:
        case "agent_start":
            ...
        case "agent_end":
            _ = event.messages
        case "turn_start":
            ...
        case "turn_end":
            _ = event.message, event.tool_results
        case "message_start" | "message_end":
            _ = event.message
        case "message_update":
            _ = event.message, event.assistant_message_event
        case "tool_execution_start":
            _ = event.tool_call_id, event.tool_name, event.args
        case "tool_execution_update":
            _ = event.partial_result
        case "tool_execution_end":
            _ = event.result, event.is_error

unsub = agent.subscribe(on_event)
unsub()
```

| 事件类 | `type` | 字段 |
| ------ | ------ | ---- |
| `AgentStartEvent` | `agent_start` | — |
| `AgentEndEvent` | `agent_end` | `messages` |
| `TurnStartEvent` | `turn_start` | — |
| `TurnEndEvent` | `turn_end` | `message`, `tool_results` |
| `MessageStartEvent` | `message_start` | `message` |
| `MessageUpdateEvent` | `message_update` | `message`, `assistant_message_event` |
| `MessageEndEvent` | `message_end` | `message` |
| `ToolExecutionStartEvent` | `tool_execution_start` | `tool_call_id`, `tool_name`, `args` |
| `ToolExecutionUpdateEvent` | `tool_execution_update` | 同上 + `partial_result` |
| `ToolExecutionEndEvent` | `tool_execution_end` | 同上 + `result`, `is_error` |

`AgentEvent` 是上述类型的联合。

## Agent Loop

底层循环可独立于 `Agent` 使用：

```python
from bampy.agent import (
    agent_loop,
    AgentContext,
    AgentLoopConfig,
    default_convert_to_llm,
)
from bampy.ai import UserMessage, get_model

stream = agent_loop(
    prompts=[UserMessage(content="Hello")],
    context=AgentContext(system_prompt="...", messages=[], tools=[my_tool]),
    config=AgentLoopConfig(
        model=get_model("claude-sonnet-4-5", provider="anthropic"),
        convert_to_llm=default_convert_to_llm,
    ),
)

async for event in stream:
    print(event.type)

new_messages = await stream.result()
```

相关 API：

| 函数 | 说明 |
| ---- | ---- |
| `agent_loop(...)` | 返回 `AgentEventStream`（异步可迭代，`.result()` 取最终消息） |
| `agent_loop_continue(...)` | 无新 prompt，从现有 context 继续 |
| `run_agent_loop(...)` | 回调式 `emit`，直接返回 `list[AgentMessage]` |
| `run_agent_loop_continue(...)` | 对应的 continue 版本 |

### 循环流程

```text
agent_start
  │
  ▼
┌── 外层（follow-up）─────────────────────────┐
│  ┌── 内层（工具 + steering）────────────────┐│
│  │  1. 注入 steering 消息                   ││
│  │  2. transform_context → convert_to_llm   ││
│  │  3. 调用 LLM                             ││
│  │  4. 提取 tool_calls                      ││
│  │  5. 校验 → before_tool_call              ││
│  │  6. 执行工具（parallel / sequential）    ││
│  │  7. after_tool_call                      ││
│  │  8. turn_end                             ││
│  └── 无 tool_calls → 退出内层 ───────────────┘│
│  有 follow-up → 继续外层                      │
└── 无 follow-up → 退出 ───────────────────────┘
  │
  ▼
agent_end
```

## Steering 与 Follow-up

```python
from bampy.ai import UserMessage

agent.steer(UserMessage(content="换一个思路"))
agent.follow_up(UserMessage(content="接着做一件事"))

agent.set_steering_mode("one-at-a-time")  # 或 "all"
agent.set_follow_up_mode("all")
agent.clear_steering_queue()
agent.clear_follow_up_queue()
agent.clear_all_queues()
agent.has_queued_messages()
```

## 取消

`Agent.abort(reason)` 会取消当前 loop 使用的 `CancellationToken`。工具执行可接收同一 token：

```python
from bampy.agent import CancellationToken, CancellationError

token = CancellationToken()
remove = token.add_callback(lambda reason: print(f"Cancelled: {reason}"))

token.raise_if_cancelled()  # 已取消则抛 CancellationError
token.cancel("User cancelled")
token.cancelled   # bool
token.reason      # str | None
await token.wait()
```

## 自定义消息

```python
from bampy.agent import register_message_converter
from bampy.ai import UserMessage

register_message_converter(
    "my_custom_role",
    lambda msg: UserMessage(content=f"[System] {getattr(msg, 'text', msg)}"),
    source_id="my_extension",
)
```

也可用 `unregister_message_converter` / `unregister_message_converters` / `clear_message_converters`。

## AgentTool

工具需满足 `AgentTool` 协议（`name` / `label` / `description` / `parameters` / `async execute(...)`），返回 `AgentToolResult(content=..., details=...)`。详见 [tools.md](tools.md)。

辅助函数：`tool_schema(parameters)`、`to_ai_tool(tool)`。
