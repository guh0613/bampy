# Agent 运行时

`bampy.agent` 提供 Agent 状态机和 agent loop，驱动 LLM 对话和工具调用。

> 大多数场景建议使用 [AgentSession](agent-session.md) 而非直接使用 Agent。AgentSession 自动集成会话持久化、扩展和压缩。

## Agent 类

```python
from bampy.agent import Agent
from bampy.ai import get_model, SimpleStreamOptions

agent = Agent(
    initial_state={
        "system_prompt": "你是一个助手。",
        "model": get_model("anthropic", "claude-sonnet-4-20250514"),
        "tools": [my_tool],
    },
    stream_options=SimpleStreamOptions(reasoning="medium"),
    tool_execution="parallel",
    max_turns=30,
)
```

### 构造参数

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `initial_state` | `AgentState \| Mapping \| None` | None | 初始状态 |
| `convert_to_llm` | `ConvertToLlmFn \| None` | 默认 | 消息转换函数 |
| `transform_context` | `TransformContextFn \| None` | None | LLM 调用前消息变换 |
| `stream_fn` | `StreamFn \| None` | None | 自定义流式函数 |
| `stream_options` | `SimpleStreamOptions \| None` | 默认 | 温度等流式选项 |
| `get_api_key` | `ApiKeyResolverFn \| None` | None | 动态 API Key 解析 |
| `tool_execution` | `"parallel" \| "sequential"` | `"parallel"` | 工具执行模式 |
| `before_tool_call` | hook | None | 工具调用前钩子 |
| `after_tool_call` | hook | None | 工具调用后钩子 |
| `max_turns` | `int` | 50 | 最大对话轮数 |

### 执行

```python
# 发送提示
await agent.prompt("Hello")
await agent.prompt(UserMessage(content="Hello"))
await agent.prompt([msg1, msg2])
await agent.prompt("描述这张图", images=[ImageContent(...)])

# 继续对话
await agent.continue_()

# 中断
agent.abort("User cancelled")

# 等待完成
await agent.wait_for_idle()

# 重置
agent.reset()
```

### 状态

```python
state = agent.state
state.system_prompt      # str
state.model              # Model
state.thinking_level     # AgentThinkingLevel
state.tools              # list[AgentTool]
state.messages           # list[AgentMessage]
state.is_streaming       # bool
state.stream_message     # 当前流式消息
state.pending_tool_calls # set[str]
state.error              # str | None

# 修改
agent.set_system_prompt("...")
agent.set_model(model)
agent.set_thinking_level("high")  # off/minimal/low/medium/high/xhigh
agent.set_tools([tool1, tool2])
agent.set_max_turns(20)
agent.append_message(msg)
agent.replace_messages([...])
agent.clear_messages()
```

### 事件

```python
def on_event(event):
    match event.type:
        case "agent_start":           ...  # Agent 循环开始
        case "agent_end":             ...  # 结束，event.messages 包含所有新消息
        case "turn_start":            ...  # 单轮开始
        case "turn_end":              ...  # 单轮结束
        case "message_start":         ...  # 消息开始
        case "message_update":        ...  # 流式更新，event.assistant_message_event
        case "message_end":           ...  # 消息完成
        case "tool_execution_start":  ...  # 工具开始，event.tool_name
        case "tool_execution_update": ...  # 工具进度
        case "tool_execution_end":    ...  # 工具完成

unsub = agent.subscribe(on_event)
unsub()  # 取消
```

## Agent Loop

Agent Loop 是驱动对话的核心机制，可独立于 Agent 类使用：

```python
from bampy.agent import agent_loop, AgentContext, AgentLoopConfig

stream = agent_loop(
    prompts=[UserMessage(content="Hello")],
    context=AgentContext(system_prompt="...", messages=[], tools=[my_tool]),
    config=AgentLoopConfig(
        model=model,
        convert_to_llm=default_convert_to_llm,
    ),
)

async for event in stream:
    print(event.type)

new_messages = await stream.result()
```

### 循环流程

```text
agent_start
  │
  ▼
┌── 外层循环（follow-up 驱动）──────────────┐
│  ┌── 内层循环（工具调用 + steering 驱动）──┐│
│  │  1. 注入 steering 消息                  ││
│  │  2. transform_context → convert_to_llm  ││
│  │  3. 调用 LLM (stream_simple)            ││
│  │  4. 提取 tool_calls                     ││
│  │  5. 校验 → before_tool_call             ││
│  │  6. 执行工具（顺序/并行）               ││
│  │  7. after_tool_call                     ││
│  │  8. turn_end                            ││
│  └── 无 tool_calls → 退出 ─────────────────┘│
│  检查 follow-up → 有则继续                    │
└── 无 follow-up → 退出 ───────────────────────┘
  │
  ▼
agent_end
```

## Steering 与 Follow-up

```python
# Steering: 在下一次 LLM 调用前注入消息
agent.steer(UserMessage(content="换一个思路"))

# Follow-up: 当前 agent loop 完成后触发新一轮
agent.follow_up(UserMessage(content="接着做一件事"))

# 队列模式
agent.set_steering_mode("one-at-a-time")  # 默认
agent.set_steering_mode("all")
```

## CancellationToken

```python
from bampy.agent import CancellationToken, CancellationError

token = CancellationToken()

# 注册回调
remove = token.add_callback(lambda reason: print(f"Cancelled: {reason}"))

# 在工具中检查
token.raise_if_cancelled()  # 抛出 CancellationError

# 取消
token.cancel("User cancelled")

# 属性
token.cancelled  # bool
token.reason     # str | None

# 异步等待
await token.wait()
```

## 自定义消息类型

```python
from bampy.agent import register_message_converter

# 注册转换器
register_message_converter(
    "my_custom_role",
    lambda msg: UserMessage(content=f"[System] {msg.text}"),
    source_id="my_extension",
)
```

## AgentTool 协议

见 [工具文档](tools.md#agenttool-协议)。
