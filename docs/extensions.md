# 扩展

扩展是导出 `setup(api: ExtensionAPI)` 的 Python 模块，用于注册事件处理、工具和斜杠命令。生命周期由 `ExtensionRunner` 管理，通常经 [AgentSession](agent-session.md) 自动加载。

## 创建扩展

```python
# my_extension.py
from bampy.app import ExtensionAPI, ToolDefinition, ToolCallEventResult
from bampy.agent import AgentToolResult
from bampy.ai import TextContent


def setup(api: ExtensionAPI) -> None:
    api.on("agent_end", on_agent_end)
    api.on("tool_call", on_tool_call)

    api.register_tool(ToolDefinition(
        name="my_tool",
        label="My Tool",
        description="Does something",
        parameters={
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
        execute=execute_my_tool,
        prompt_snippet="Use my_tool when ...",
        prompt_guidelines=["Always validate input"],
    ))

    api.register_command(
        "clear",
        description="清空会话",
        handler=handle_clear,  # Callable[..., Awaitable[None]]
    )


async def execute_my_tool(tool_call_id, params, cancellation, on_update, ctx):
    # ctx: ExtensionContext
    return AgentToolResult(content=[TextContent(text="Done")])


async def on_agent_end(event, ctx):
    print(f"Agent completed with {len(event.messages)} messages")


def on_tool_call(event, ctx):
    if event.tool_name == "bash" and "rm -rf" in str(event.input):
        return ToolCallEventResult(block=True, reason="Dangerous command")


async def handle_clear(*args, **kwargs):
    ...
```

`setup` 可以是同步或异步函数。扩展工具的 `execute` 签名为：

`(tool_call_id, params, cancellation, on_update, ctx) -> AgentToolResult`

## 发现与加载

### 发现顺序

`create_agent_session` / `load_extensions(discover=True)` 按以下来源收集：

1. **显式路径** — `extension_paths=["./my_ext.py"]`
2. **项目本地** — `<cwd>/.bampy/extensions/*.py`
3. **用户全局** — `~/.bampy/extensions/*.py`
4. **显式模块** — `extension_modules=["my_package.ext"]`
5. **entry_points** — 组名 `bampy.extensions`

```toml
# pyproject.toml
[project.entry-points."bampy.extensions"]
my_extension = "my_package.extension"
```

路径类扩展先加载，模块类后加载；同名工具/命令按**先注册者优先**。

### API

```python
from bampy.app import load_extensions, discover_and_load_extensions, create_agent_session

# 底层加载
result = await load_extensions(
    paths=["./ext.py"],
    modules=["my_package.ext"],
    cwd=".",
    discover=True,  # 同时扫描本地/全局目录与 entry_points
)
# result.extensions / result.errors

# 便捷封装（等价于 paths=extra_paths + discover）
result = await discover_and_load_extensions(cwd=".", extra_paths=["./ext.py"])

# 会话入口
await create_agent_session(
    extension_paths=["./ext.py"],
    extension_modules=["my_package.ext"],
    discover_extensions=True,
)
```

模块必须提供可调用的 `setup`；缺失时记入 `LoadError`，不中断其他扩展。

## 事件类型

处理器签名：`(event, ctx: ExtensionContext) -> ...`，可为 sync/async。

### 会话生命周期

| 事件 | 触发时机 |
| ---- | -------- |
| `session_start` | `AgentSession.start()` |
| `session_shutdown` | `AgentSession.close()` |
| `session_compact` | 上下文压缩完成后 |

### Agent 生命周期

| 事件 | 触发时机 | 可返回 |
| ---- | -------- | ------ |
| `before_agent_start` | Agent 启动前 | `BeforeAgentStartEventResult(system_prompt=...)` |
| `agent_start` | Agent 循环开始 | — |
| `agent_end` | Agent 循环结束（含 `messages`） | — |

### 对话轮与消息

| 事件 | 说明 |
| ---- | ---- |
| `turn_start` / `turn_end` | 对话轮起止（`turn_index` 等） |
| `message_start` / `message_update` / `message_end` | 消息流生命周期 |

### 工具

| 事件 | 触发时机 | 可返回 |
| ---- | -------- | ------ |
| `tool_call` | 工具执行前 | `ToolCallEventResult(block=True, reason=...)` |
| `tool_result` | 工具执行后 | `ToolResultEventResult`（可改 `content` / `details` / `is_error`） |
| `tool_execution_start` / `update` / `end` | 执行进度通知 | — |

### 上下文与输入

| 事件 | 触发时机 | 可返回 |
| ---- | -------- | ------ |
| `context` | 每次 LLM 调用前 | `ContextEventResult(messages=[...])` |
| `input` | 用户输入进入 Agent 前 | `InputEventResult(action=...)` |

`InputEventResult.action`：

- `"continue"` — 原样继续（默认）
- `"transform"` — 用 `text` / `images` 替换输入
- `"handled"` — 消费输入，不交给 Agent

## 事件处理示例

### 修改上下文

```python
from bampy.app import ContextEventResult
from bampy.ai import UserMessage

def on_context(event, ctx):
    return ContextEventResult(
        messages=[*event.messages, UserMessage(content="[System: be concise]")],
    )

api.on("context", on_context)
```

### 拦截工具调用

```python
from bampy.app import ToolCallEventResult

def on_tool_call(event, ctx):
    if event.tool_name == "write" and "/etc/" in str(event.input):
        return ToolCallEventResult(block=True, reason="Cannot write to /etc/")

api.on("tool_call", on_tool_call)
```

### 修改工具结果

```python
from bampy.app import ToolResultEventResult
from bampy.ai import TextContent

def on_tool_result(event, ctx):
    if event.is_error:
        return ToolResultEventResult(
            content=[TextContent(text=f"[Error handled] {event.content[0].text}")],
            is_error=False,
        )

api.on("tool_result", on_tool_result)
```

### 拦截用户输入

```python
from bampy.app import InputEventResult

def on_input(event, ctx):
    if event.text.startswith("/help"):
        return InputEventResult(action="handled")
    if "secret" in event.text:
        return InputEventResult(
            action="transform",
            text=event.text.replace("secret", "***"),
        )

api.on("input", on_input)
```

## ExtensionAPI

| 方法 | 作用 |
| ---- | ---- |
| `on(event, handler)` | 订阅事件 |
| `register_tool(ToolDefinition)` | 注册 LLM 可调用工具 |
| `register_command(name, *, description="", handler=...)` | 注册斜杠命令 |
| `send_message(custom_type, content, *, display=True, details=None, trigger_turn=False)` | 写入自定义消息 |
| `send_user_message(content)` | 发送用户消息并触发新一轮 |
| `append_entry(custom_type, data=None)` | 追加自定义会话条目（不发给 LLM） |

`ToolDefinition` 字段：`name`、`label`、`description`、`parameters`、`execute`，以及可选的 `prompt_snippet`、`prompt_guidelines`（会并入 system prompt 的工具描述与 guidelines）。斜杠命令经 `register_command` 登记后可由 `ExtensionRunner.get_all_commands()` 取出；会话层如何派发由调用方决定。

## ExtensionContext

事件处理器第二个参数：

```python
def on_event(event, ctx):
    ctx.cwd                 # 工作目录
    ctx.session_manager     # SessionManager | None
    ctx.model               # 当前 Model | None
    ctx.is_idle()           # Agent 是否空闲
    ctx.abort()             # 中断 Agent
    ctx.has_pending_messages()
    ctx.get_system_prompt()
```

## ExtensionRunner

```python
from bampy.app import ExtensionRunner, load_extensions
from bampy.app import SessionStartEvent, ToolCallEvent

loaded = await load_extensions(paths=["./ext.py"], cwd=".", discover=False)
runner = ExtensionRunner()
runner.set_extensions(loaded.extensions)

await runner.emit(SessionStartEvent())
block = await runner.emit_tool_call(ToolCallEvent(tool_name="bash", input={...}))
messages = await runner.emit_context(messages)

tools = runner.get_all_registered_tools()
commands = runner.get_all_commands()
```

`AgentSession` 会把扩展的 `tool_call` / `tool_result` 结果桥接到 Agent 的 `before_tool_call` / `after_tool_call` 钩子；也可与用户传入的钩子叠加。详见 [tools.md](tools.md)、[compaction.md](compaction.md)。
