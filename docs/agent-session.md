# AgentSession

`AgentSession` 是 bampy 的应用层编排器，把会话持久化、扩展、内置工具、Skills 与上下文压缩接到 Layer 2 [`Agent`](agent.md) 上。**大多数场景应使用 `AgentSession` / `create_agent_session`，而不是直接使用 `Agent`。**

导入路径：`from bampy.app import create_agent_session, AgentSession`。

## 快速创建

```python
from bampy.app import create_agent_session
from bampy.ai import get_model

result = await create_agent_session(
    model=get_model("claude-sonnet-4-5", provider="anthropic"),
)
session = result.session
# result.extensions — LoadExtensionsResult
# result.skills — LoadSkillsResult
```

`create_agent_session` 会依次：

1. 加载上下文文件（如 `CONTEXT.md`）
2. 发现并加载扩展
3. 发现并加载 Skills
4. 构造 `AgentSession`（内部创建 [`SessionManager`](session.md)、`Agent`、`ExtensionRunner`）
5. 调用 `await session.start()`

`tools=None` 时默认通过 `create_coding_tools(cwd)` 加载编码工具集。

## 手动创建

需要更细粒度控制时，可直接构造 `AgentSession`，再自行 `start()`：

```python
from bampy.app import AgentSession, CompactionSettings
from bampy.ai import get_model

session = AgentSession(
    cwd="/my/project",
    model=get_model("claude-sonnet-4-5", provider="anthropic"),
    thinking_level="medium",
    tools=None,  # 默认 coding tools；也可传入 list/dict
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

### 基础

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `cwd` | `str \| None` | 当前目录 / `session_manager.cwd` | 工作目录 |
| `model` | `Model \| None` | 会话恢复或内置回退 | 主模型 |
| `thinking_level` | `AgentThinkingLevel \| str \| None` | 会话恢复或 `"off"` | `off` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max` |
| `steering_mode` | `str` | `"one-at-a-time"` | steering 队列：`one-at-a-time` 或 `all` |
| `follow_up_mode` | `str` | `"one-at-a-time"` | follow-up 队列：`one-at-a-time` 或 `all` |
| `tools` | `list[AgentTool] \| dict[str, AgentTool] \| None` | coding tools | `None` 使用默认编码工具 |
| `active_tool_names` | `list[str] \| None` | 全部已注册工具 | 激活子集 |
| `custom_system_prompt` | `str \| None` | None | 自定义 system prompt |
| `append_system_prompt` | `str \| None` | None | 追加到 system prompt 末尾 |
| `augment_custom_system_prompt` | `bool` | `True` | 自定义 prompt 时是否仍附加上下文/Skills 等增强 |
| `max_turns` | `int` | `50` | 单次 agent loop 最大轮数 |

### 会话与扩展

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `session_manager` | `SessionManager \| None` | `SessionManager.create(cwd)` | 会话管理器 |
| `extension_runner` | `ExtensionRunner \| None` | 新建 | 扩展运行器 |
| `extensions` | `list[Extension] \| None` | None | 已加载扩展 |
| `context_files` | `list[ContextFile] \| None` | None | 上下文文件 |
| `skills` | `list[Skill] \| None` | None | Skills |

### 流式与钩子

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `stream_options` | `SimpleStreamOptions \| None` | 默认实例 | 流式选项 |
| `get_api_key` | `ApiKeyResolverFn \| None` | None | 按 provider 解析 API Key |
| `stream_fn` | `StreamFn \| None` | None | 自定义流式函数 |
| `convert_to_llm` | `ConvertToLlmFn \| None` | `default_convert_to_llm` | 消息 → LLM 消息 |
| `transform_context` | `TransformContextFn \| None` | None | LLM 调用前变换上下文 |
| `tool_execution` | `ToolExecutionMode \| str` | `"parallel"` | `parallel` / `sequential` |
| `before_tool_call` | `BeforeToolCallHook \| None` | None | 工具调用前钩子 |
| `after_tool_call` | `AfterToolCallHook \| None` | None | 工具调用后钩子 |

### 压缩

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `compaction_settings` | `CompactionSettings \| None` | 默认设置 | 压缩阈值等，见 [compaction.md](compaction.md) |
| `auto_compaction` | `bool` | `True` | 是否在 prompt 前自动压缩 |
| `summarization_model` | `Model \| None` | 同主模型 | 摘要模型 |
| `summarization_api_key` | `str \| None` | None | 摘要用 API Key |
| `summarization_custom_instructions` | `str \| None` | None | 摘要附加指令 |

## 使用

### 发送消息

```python
await session.prompt("帮我写一个排序算法")

from bampy.ai import ImageContent, UserMessage

await session.prompt(
    "描述这张图",
    images=[ImageContent(data="...", mime_type="image/png")],
)
await session.prompt(UserMessage(content="Hello"))
await session.continue_()  # 别名: session.resume
```

`prompt` 的 `source` 关键字参数默认 `"interactive"`，会进入扩展的 `InputEvent` 变换链路。

### Steering 与 Follow-up

```python
from bampy.ai import UserMessage

session.steer(UserMessage(content="换一个实现方向"))
session.follow_up(UserMessage(content="接着补测试"))

session.set_steering_mode("one-at-a-time")
session.set_follow_up_mode("all")
session.has_queued_messages()
session.clear_all_queues()
```

### 事件订阅

```python
def on_event(event):
    match event.type:
        case "message_update":
            ame = event.assistant_message_event
            if getattr(ame, "type", None) == "text_delta":
                print(ame.delta, end="")
        case "tool_execution_start":
            print(f"\n[Tool] {event.tool_name}")
        case "tool_execution_end":
            print(f"[Done] {event.tool_name}")
        case "auto_compaction_start":
            print("[Compacting...]")
        case "auto_compaction_end":
            if event.result:
                print(f"[Compacted] tokens_before={event.result.tokens_before}")
            elif event.error_message:
                print(f"[Compact error] {event.error_message}")

unsub = session.subscribe(on_event)
# unsub() 取消订阅
```

`AgentSessionEvent` = 全部 [`AgentEvent`](agent.md#事件) ∪：

| 类型 | `type` 字段 | 主要字段 |
| ---- | ----------- | -------- |
| `AutoCompactionStartEvent` | `auto_compaction_start` | `reason`（当前为 `"threshold"`） |
| `AutoCompactionEndEvent` | `auto_compaction_end` | `result: CompactionResult \| None`、`aborted`、`error_message` |

`CompactionResult` 字段：`summary`、`first_kept_entry_id`、`tokens_before`、`details`（**没有** `tokens_after`）。

### 状态与配置

```python
session.cwd
session.model
session.thinking_level
session.steering_mode
session.follow_up_mode
session.messages
session.system_prompt
session.active_tool_names
session.is_processing
session.extension_errors

session.set_model(get_model("gpt-5.4-mini", provider="openai"))
session.set_thinking_level("high")
session.set_steering_mode("all")
session.set_follow_up_mode("one-at-a-time")
session.set_active_tools(["bash", "read", "write"])
session.get_all_tools()
session.reload_session_context()  # 从 SessionManager 重建 messages / model / thinking
```

### 压缩与中断

```python
result = await session.compact()  # 手动；不需要时返回 None
session.abort("User cancelled")
await session.wait_for_idle()
```

### 生命周期

```python
await session.start()          # SessionStartEvent；create_agent_session 已调用
await session.prompt("...")
await session.wait_for_idle()
await session.close()          # SessionShutdownEvent，并取消 Agent 订阅
```

## `create_agent_session` 额外参数

除 `AgentSession` 构造参数外（无 `extension_runner` / `extensions` / `context_files` / `skills`，由工厂内部加载），还支持：

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `extension_paths` | `list[str] \| None` | None | 扩展文件路径 |
| `extension_modules` | `list[str] \| None` | None | 扩展模块名 |
| `discover_extensions` | `bool` | `True` | 自动发现扩展 |
| `skill_paths` | `list[str] \| None` | None | Skill 搜索路径 |
| `include_default_skills` | `bool` | `True` | 包含默认 Skills |
| `context_filenames` | `list[str] \| None` | None | 上下文文件名列表 |

返回 `CreateAgentSessionResult(session, extensions, skills)`。

## 内部组件

```python
session.agent              # Layer 2 Agent
session.session_manager    # SessionManager，见 session.md
session.extension_runner   # ExtensionRunner，见 extensions.md
```

工具与 Skills 集成细节见 [tools.md](tools.md)、[skills.md](skills.md)、[extensions.md](extensions.md)。
