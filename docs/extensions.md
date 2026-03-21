# 扩展

扩展是导出 `setup(api: ExtensionAPI)` 函数的 Python 模块，用于向 Agent 添加工具、命令和事件处理。

## 创建扩展

```python
# my_extension.py
from bampy.app import ExtensionAPI, ToolDefinition

def setup(api: ExtensionAPI) -> None:
    # 订阅事件
    api.on("agent_end", on_agent_end)
    api.on("tool_call", on_tool_call)

    # 注册工具
    api.register_tool(ToolDefinition(
        name="my_tool",
        label="My Tool",
        description="Does something",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}},
        execute=execute_my_tool,
        prompt_snippet="Use my_tool when ...",        # 添加到 system prompt
        prompt_guidelines=["Always validate input"],  # 添加到工具指南
    ))

    # 注册命令
    api.register_command("clear", description="清空会话", handler=handle_clear)


async def execute_my_tool(tool_call_id, params, cancellation, on_update, ctx):
    from bampy.agent import AgentToolResult
    from bampy.ai import TextContent
    return AgentToolResult(content=[TextContent(text="Done")])


async def on_agent_end(event, ctx):
    print(f"Agent completed with {len(event.messages)} messages")


def on_tool_call(event, ctx):
    from bampy.app import ToolCallEventResult
    if event.tool_name == "bash" and "rm -rf" in str(event.input):
        return ToolCallEventResult(block=True, reason="Dangerous command")
```

## 放置扩展

按优先级顺序发现：

1. **显式路径** — `create_agent_session(extension_paths=["./my_ext.py"])`
2. **显式模块** — `create_agent_session(extension_modules=["my_package.ext"])`
3. **项目本地** — `<cwd>/.bampy/extensions/*.py`
4. **用户全局** — `~/.bampy/extensions/*.py`
5. **entry_points** — pyproject.toml 中注册

```toml
# pyproject.toml
[project.entry-points."bampy.extensions"]
my_extension = "my_package.extension"
```

## 事件类型

### 会话生命周期

| 事件 | 触发时机 |
| ---- | -------- |
| `session_start` | `AgentSession.start()` |
| `session_shutdown` | `AgentSession.close()` |
| `session_compact` | 上下文压缩完成后 |

### Agent 生命周期

| 事件 | 触发时机 | 可操作 |
| ---- | -------- | ------ |
| `before_agent_start` | Agent 启动前 | 返回 `BeforeAgentStartEventResult` 修改 system prompt |
| `agent_start` | Agent 循环开始 | - |
| `agent_end` | Agent 循环结束 | - |

### 对话轮与消息

| 事件 | 触发时机 |
| ---- | -------- |
| `turn_start` | 对话轮开始 |
| `turn_end` | 对话轮结束 |
| `message_start` | 消息开始 |
| `message_update` | 消息流式更新 |
| `message_end` | 消息完成 |

### 工具

| 事件 | 触发时机 | 可操作 |
| ---- | -------- | ------ |
| `tool_call` | 工具调用前 | 返回 `ToolCallEventResult(block=True)` 阻止 |
| `tool_result` | 工具调用后 | 返回 `ToolResultEventResult` 修改结果 |
| `tool_execution_start` | 工具开始执行 | - |
| `tool_execution_update` | 工具执行进度 | - |
| `tool_execution_end` | 工具执行完成 | - |

### 上下文与输入

| 事件 | 触发时机 | 可操作 |
| ---- | -------- | ------ |
| `context` | LLM 调用前 | 返回 `ContextEventResult(messages=[...])` 修改消息 |
| `input` | 用户输入后 | 返回 `InputEventResult(action="transform"\|"handled")` |

## 事件处理示例

### 修改上下文

```python
from bampy.app import ContextEventResult
from bampy.ai import UserMessage

def on_context(event, ctx):
    # 注入额外消息
    return ContextEventResult(
        messages=[*event.messages, UserMessage(content="[System: be concise]")]
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
        return InputEventResult(action="handled")  # 完全消费，不传给 Agent
    if "secret" in event.text:
        return InputEventResult(action="transform", text=event.text.replace("secret", "***"))

api.on("input", on_input)
```

## ExtensionAPI 方法

### 事件订阅

```python
api.on("event_name", handler)
```

### 工具注册

```python
api.register_tool(ToolDefinition(
    name="...",
    label="...",
    description="...",
    parameters=ParamsModel,
    execute=execute_fn,
    prompt_snippet="...",           # 可选
    prompt_guidelines=["..."],      # 可选
))
```

### 命令注册

```python
api.register_command("name", description="...", handler=async_handler)
```

### 运行时动作

```python
# 发送自定义消息到会话
api.send_message("custom_type", "content", display=True, trigger_turn=False)

# 发送用户消息（触发新 Agent 轮）
api.send_user_message("Follow up question")

# 追加自定义条目到会话（不发送给 LLM）
api.append_entry("custom_data", {"key": "value"})
```

## ExtensionContext

事件处理器的第二个参数：

```python
def on_event(event, ctx):
    ctx.cwd                  # 工作目录
    ctx.session_manager      # SessionManager
    ctx.model                # 当前 Model
    ctx.is_idle()            # Agent 是否空闲
    ctx.abort()              # 中断 Agent
    ctx.has_pending_messages()
    ctx.get_system_prompt()
```

## ExtensionRunner

管理扩展生命周期和事件分发：

```python
from bampy.app import ExtensionRunner, load_extensions

result = await load_extensions(paths=["./ext.py"], cwd=".")
runner = ExtensionRunner()
runner.set_extensions(result.extensions)

# 分发事件
await runner.emit(SessionStartEvent())
block = await runner.emit_tool_call(ToolCallEvent(...))
messages = await runner.emit_context(messages)

# 获取所有扩展工具
tools = runner.get_all_registered_tools()
commands = runner.get_all_commands()
```
