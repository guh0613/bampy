# AgentSession

`AgentSession` 是 bampy 的顶层编排器，自动装配会话持久化、扩展系统、内置工具、Skills 和上下文压缩。**这是大多数场景下的推荐入口**。

## 快速创建

```python
from bampy.app import create_agent_session
from bampy.ai import get_model

result = await create_agent_session(
    model=get_model("anthropic", "claude-sonnet-4-20250514"),
)
session = result.session
# result.extensions — 扩展加载结果
# result.skills — Skills 加载结果
```

`create_agent_session` 自动完成：
1. 加载上下文文件（CONTEXT.md 等）
2. 发现并加载扩展
3. 发现并加载 Skills
4. 创建 SessionManager
5. 构建 system prompt
6. 组装 Agent + ExtensionRunner
7. 调用 `session.start()`

## 手动创建

需要更细粒度控制时：

```python
from bampy.app import AgentSession, SessionManager, load_extensions, load_skills

session = AgentSession(
    cwd="/my/project",
    model=get_model("anthropic", "claude-sonnet-4-20250514"),
    thinking_level="medium",
    tools=[my_tool_1, my_tool_2],  # 或 None 使用默认 coding tools
    custom_system_prompt="你是一个代码助手。",
    compaction_settings=CompactionSettings(
        enabled=True,
        reserve_tokens=16384,
        keep_recent_tokens=20000,
    ),
    auto_compaction=True,
    max_turns=30,
)
await session.start()
```

## 构造参数

### 基础配置

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `cwd` | `str \| None` | 当前目录 | 工作目录 |
| `model` | `Model \| None` | 自动选择 | LLM 模型 |
| `thinking_level` | `str \| None` | `"off"` | 推理等级：off/minimal/low/medium/high/xhigh/max |
| `steering_mode` | `str` | `"one-at-a-time"` | steering 队列模式：`one-at-a-time` 或 `all` |
| `follow_up_mode` | `str` | `"one-at-a-time"` | follow-up 队列模式：`one-at-a-time` 或 `all` |
| `tools` | `list \| dict \| None` | coding_tools | 工具集，None 使用默认编码工具 |
| `active_tool_names` | `list[str] \| None` | 全部 | 激活的工具子集 |
| `custom_system_prompt` | `str \| None` | None | 自定义 system prompt |
| `append_system_prompt` | `str \| None` | None | 追加到 system prompt 末尾 |
| `max_turns` | `int` | 50 | 单次 prompt 最大对话轮数 |

### 会话与扩展

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `session_manager` | `SessionManager \| None` | 自动创建 | 会话管理器 |
| `extension_runner` | `ExtensionRunner \| None` | 自动创建 | 扩展运行器 |
| `extensions` | `list[Extension] \| None` | None | 已加载的扩展 |
| `context_files` | `list[ContextFile] \| None` | None | 上下文文件 |
| `skills` | `list[Skill] \| None` | None | Skills 列表 |

### 流式与 API

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `stream_options` | `SimpleStreamOptions \| None` | 默认 | 温度、max_tokens 等 |
| `get_api_key` | `ApiKeyResolverFn \| None` | None | 动态 API Key 解析 |
| `stream_fn` | `StreamFn \| None` | None | 自定义流式函数 |
| `convert_to_llm` | `ConvertToLlmFn \| None` | 默认 | 自定义消息转换 |
| `transform_context` | `TransformContextFn \| None` | None | LLM 调用前消息变换 |
| `tool_execution` | `str` | `"parallel"` | 工具执行模式 |
| `before_tool_call` | `BeforeToolCallHook \| None` | None | 工具调用前钩子 |
| `after_tool_call` | `AfterToolCallHook \| None` | None | 工具调用后钩子 |

### 压缩

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `compaction_settings` | `CompactionSettings \| None` | 默认 | 压缩配置 |
| `auto_compaction` | `bool` | `True` | 自动触发压缩 |
| `summarization_model` | `Model \| None` | 同主模型 | 压缩摘要用模型 |
| `summarization_api_key` | `str \| None` | None | 摘要 API Key |
| `summarization_custom_instructions` | `str \| None` | None | 摘要自定义指令 |

## 使用

### 发送消息

```python
# 字符串
await session.prompt("帮我写一个排序算法")

# 带图片
from bampy.ai import ImageContent
await session.prompt("描述这张图", images=[ImageContent(data="...", mime_type="image/png")])

# 直接传消息对象
from bampy.ai import UserMessage
await session.prompt(UserMessage(content="Hello"))

# 继续对话
await session.continue_()
```

### Steering 与 Follow-up

`AgentSession` 现在直接暴露底层 `Agent` 的队列控制能力，不必再手动访问 `session.agent`：

```python
from bampy.ai import UserMessage

# Steering: 在下一次 LLM 调用前注入消息
session.steer(UserMessage(content="换一个实现方向"))

# Follow-up: 当前 agent loop 结束后发起新一轮
session.follow_up(UserMessage(content="接着补测试"))

# 执行已排队消息
await session.continue_()

# 队列模式
session.set_steering_mode("one-at-a-time")
session.set_follow_up_mode("all")

# 队列状态
session.has_queued_messages()
session.clear_all_queues()
```

### 事件订阅

```python
def on_event(event):
    match event.type:
        case "message_update":
            ame = event.assistant_message_event
            if ame.type == "text_delta":
                print(ame.delta, end="")
        case "tool_execution_start":
            print(f"\n[Tool] {event.tool_name}")
        case "tool_execution_end":
            print(f"[Done] {event.tool_name}")
        case "auto_compaction_start":
            print("[Compacting...]")
        case "auto_compaction_end":
            if event.result:
                print(f"[Compacted] saved {event.result.tokens_before - event.result.tokens_after} tokens")

unsub = session.subscribe(on_event)
# unsub() 取消订阅
```

`AgentSessionEvent` 包含所有 `AgentEvent`（见 [Agent 文档](agent.md#事件)）加上：
- `AutoCompactionStartEvent` — 自动压缩开始
- `AutoCompactionEndEvent` — 自动压缩结束（含结果或错误）

### 状态管理

```python
# 读取
session.model              # 当前模型
session.thinking_level     # 当前推理等级
session.steering_mode      # 当前 steering 队列模式
session.follow_up_mode     # 当前 follow-up 队列模式
session.messages           # 消息列表
session.system_prompt      # 当前 system prompt
session.active_tool_names  # 激活的工具名
session.cwd                # 工作目录

# 修改
session.set_model(get_model("openai", "gpt-4o"))
session.set_thinking_level("high")
session.set_steering_mode("all")
session.set_follow_up_mode("one-at-a-time")
session.set_active_tools(["bash", "read", "write"])
```

### 手动压缩

```python
result = await session.compact()
if result:
    print(f"压缩完成: {result.tokens_before} → {result.tokens_after} tokens")
```

### 生命周期

```python
await session.start()           # 触发 SessionStartEvent
await session.prompt("...")     # 使用
await session.wait_for_idle()   # 等待当前操作完成
await session.close()           # 触发 SessionShutdownEvent
```

## create_agent_session 参数

除 `AgentSession` 的全部参数外，工厂函数额外支持：

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `extension_paths` | `list[str] \| None` | None | 扩展文件路径 |
| `extension_modules` | `list[str] \| None` | None | 扩展模块名 |
| `discover_extensions` | `bool` | `True` | 自动发现扩展 |
| `skill_paths` | `list[str] \| None` | None | Skill 搜索路径 |
| `include_default_skills` | `bool` | `True` | 包含默认 Skills |
| `context_filenames` | `list[str] \| None` | None | 上下文文件名 |

## 内部组件访问

```python
session.agent              # Layer 2 Agent 实例
session.session_manager    # SessionManager
session.extension_runner   # ExtensionRunner
```

需要更底层的控制时可直接操作这些组件，详见各自文档。
