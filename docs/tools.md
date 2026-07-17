# 工具

bampy 提供 `@tool` 装饰器、`AgentTool` 协议，以及 8 个内置编码工具。工具最终交给 [Agent](agent.md) / [AgentSession](agent-session.md) 执行。

## `@tool` 装饰器

最简单的定义方式：

```python
from bampy.app import tool

@tool(name="greet", description="Say hello")
async def greet(name: str) -> str:
    return f"Hello, {name}!"
```

装饰器返回 `ToolFromFunction`，实现 `AgentTool` 协议。

### 返回值

```python
from bampy.agent import AgentToolResult
from bampy.ai import TextContent

# str → TextContent
@tool(name="hello")
async def hello(name: str) -> str:
    return f"Hello, {name}!"

# AgentToolResult → 完全控制（details 不发给 LLM）
@tool(name="search")
async def search(query: str) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(text=f"Found: {query}")],
        details={"raw": []},
    )

# list[str | content block] → 多块内容
@tool(name="multi")
async def multi() -> list:
    return [TextContent(text="part1"), TextContent(text="part2")]
```

### 参数 Schema

函数签名自动生成 JSON Schema（跳过注入参数）：

```python
@tool(name="calculate", description="计算数学表达式")
async def calculate(expression: str, precision: int = 2) -> str:
    ...
```

也可用 Pydantic 模型：

```python
from pydantic import BaseModel, Field
from bampy.app import tool

class SearchParams(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=10, ge=1, le=100)

@tool(name="search", description="搜索", parameters=SearchParams)
async def search(query: str, max_results: int = 10) -> str:
    ...
```

### 可选注入参数

函数签名中可声明以下名称，运行时自动注入：

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| `tool_call_id` | `str` | 当前工具调用 ID |
| `cancellation` | `CancellationToken \| None` | 取消令牌 |
| `on_update` | `AgentToolUpdateCallback \| None` | 进度回调 |

```python
from bampy.agent import AgentToolResult, CancellationToken, AgentToolUpdateCallback
from bampy.ai import TextContent
from bampy.app import tool

@tool(name="long_task", description="耗时任务")
async def long_task(
    input: str,
    tool_call_id: str,
    cancellation: CancellationToken | None = None,
    on_update: AgentToolUpdateCallback | None = None,
) -> str:
    if cancellation is not None:
        cancellation.raise_if_cancelled()
    if on_update is not None:
        on_update(AgentToolResult(content=[TextContent(text="working...")]))
    return "Done"
```

## AgentTool 协议

复杂工具可直接实现协议（duck typing，不必继承）：

```python
from pydantic import BaseModel, Field
from bampy.agent import AgentToolResult, CancellationToken, AgentToolUpdateCallback
from bampy.ai import TextContent

class QueryInput(BaseModel):
    sql: str
    database: str = "main"

class DatabaseTool:
    name = "database_query"
    label = "Database Query"
    description = "Execute SQL queries"
    parameters = QueryInput  # Pydantic 模型或 JSON Schema dict

    def __init__(self, conn_str: str):
        self._conn = conn_str

    async def execute(
        self,
        tool_call_id: str,
        params: QueryInput,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")])
```

必需属性：`name`、`label`、`description`、`parameters`，以及异步方法 `execute(...)`。

## 内置工具

共 8 个工具：

| 类 | 名称 | 主要参数 | 说明 |
| -- | ---- | -------- | ---- |
| `ReadTool` | `read` | `path`, `offset?`, `limit?` | 读文件（含常见图片） |
| `BashTool` | `bash` | `command`, `timeout?` | 执行 shell 命令 |
| `EditTool` | `edit` | `path`, `edits[{old_text, new_text}]` | 精确文本替换（可多处） |
| `PatchTool` | `patch` | `patch` | 应用 unified diff |
| `WriteTool` | `write` | `path`, `content` | 创建/覆盖文件 |
| `GrepTool` | `grep` | `pattern`, `path?`, `glob?`, `ignore_case?`, `literal?`, `context?`, `limit?` | 正则/字面量内容搜索 |
| `FindTool` | `find` | `pattern`, `path?`, `limit?` | Glob 文件搜索 |
| `LsTool` | `ls` | `path?`, `limit?` | 列出目录 |

### 工具集与工厂

```python
from bampy.app import (
    # 预实例化（cwd="."）
    bash_tool, read_tool, write_tool, edit_tool, patch_tool,
    find_tool, grep_tool, ls_tool,

    # 工具集
    coding_tools,      # [read, bash, edit, patch, write]
    read_only_tools,   # [read, grep, find, ls]

    # 工厂（绑定自定义 cwd）
    create_bash_tool,
    create_coding_tools,
    create_read_only_tools,
    create_all_tools,  # dict[name, tool]
)

tools = create_coding_tools(cwd="/my/project")
all_named = create_all_tools(cwd="/my/project")
```

> `AgentSession` / `create_agent_session` 在 `tools=None` 时默认使用 `create_coding_tools(cwd)`（不含 `grep` / `find` / `ls`）。若需要完整 8 工具，传入 `list(create_all_tools(cwd).values())` 或自行组合。

### 参数细节

```python
# edit：一次可提交多处替换，每处 old_text 须在原文件中唯一且不重叠
{
  "path": "src/app.py",
  "edits": [
    {"old_text": "foo = 1", "new_text": "foo = 2"},
    {"old_text": "bar = 3", "new_text": "bar = 4"},
  ],
}

# patch：标准 unified diff 文本
{"patch": "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n"}

# grep
{
  "pattern": "TODO",
  "path": "src",
  "glob": "*.py",
  "ignore_case": True,
  "literal": False,
  "context": 2,
  "limit": 50,
}
```

## 与 AgentSession 的关系

- 默认工具集：`create_coding_tools(cwd)`
- 扩展通过 `api.register_tool(ToolDefinition(...))` 注册的工具会合并进会话工具表（见 [extensions.md](extensions.md)）
- `active_tool_names` 可限制实际启用的子集
- Agent 层还可用 `before_tool_call` / `after_tool_call` 钩子拦截执行（见下文）；扩展侧对应 `tool_call` / `tool_result` 事件

## 工具钩子（Agent 层）

这些钩子挂在 `Agent` / `AgentSession` 上，与扩展事件互补。

### before_tool_call

```python
from bampy.agent import BeforeToolCallContext, BeforeToolCallResult

async def before_hook(ctx: BeforeToolCallContext, cancellation=None):
    if ctx.tool_call.name == "bash" and "rm" in str(ctx.args):
        return BeforeToolCallResult(block=True, reason="Dangerous command")
    return None
```

### after_tool_call

```python
from bampy.agent import AfterToolCallContext, AfterToolCallResult
from bampy.ai import TextContent

async def after_hook(ctx: AfterToolCallContext, cancellation=None):
    if ctx.is_error:
        return AfterToolCallResult(
            content=[TextContent(text=f"Tool failed: {ctx.result.content[0].text}")],
        )
    return None
```

相关文档：[agent-session.md](agent-session.md)、[extensions.md](extensions.md)、[agent.md](agent.md)。
