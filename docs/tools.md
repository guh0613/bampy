# 工具

## @tool 装饰器

最简单的工具定义方式：

```python
from bampy.app import tool

@tool(name="greet", description="Say hello")
async def greet(name: str) -> str:
    return f"Hello, {name}!"
```

### 返回值

装饰器自动处理多种返回类型：

```python
# 返回字符串 → 自动包装为 TextContent
@tool(name="hello")
async def hello(name: str) -> str:
    return f"Hello, {name}!"

# 返回 AgentToolResult → 完全控制
from bampy.agent import AgentToolResult
from bampy.ai import TextContent

@tool(name="search")
async def search(query: str) -> AgentToolResult:
    results = find(query)
    return AgentToolResult(
        content=[TextContent(text=f"Found {len(results)} results")],
        details={"files": results},  # details 不发送给 LLM
    )

# 返回列表 → 自动转换为内容块
@tool(name="multi")
async def multi() -> list:
    return [TextContent(text="part1"), TextContent(text="part2")]
```

### 参数

函数签名自动生成 JSON Schema：

```python
@tool(name="calculate", description="计算数学表达式")
async def calculate(expression: str, precision: int = 2) -> str:
    ...
# → {"type": "object", "properties": {"expression": {"type": "string"}, "precision": {"type": "integer", "default": 2}}, "required": ["expression"]}
```

也可用 Pydantic 模型获得更丰富的 Schema：

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=10, ge=1, le=100, description="最大结果数")

@tool(name="search", description="搜索", parameters=SearchParams)
async def search(query: str, max_results: int = 10) -> str:
    ...
```

### 可选注入参数

函数签名中可声明以下名称，运行时自动注入：

```python
from bampy.agent import CancellationToken, AgentToolUpdateCallback

@tool(name="long_task", description="耗时任务")
async def long_task(
    input: str,
    tool_call_id: str,                        # 当前工具调用 ID
    cancellation: CancellationToken,           # 取消令牌
    on_update: AgentToolUpdateCallback,        # 进度回调
) -> str:
    for i, chunk in enumerate(process(input)):
        cancellation.raise_if_cancelled()
        on_update(AgentToolResult(content=[TextContent(text=f"Progress: {i}")]))
    return "Done"
```

## AgentTool 协议

对于复杂工具，实现 `AgentTool` 协议：

```python
from pydantic import BaseModel
from bampy.agent import AgentTool, AgentToolResult, CancellationToken, AgentToolUpdateCallback
from bampy.ai import TextContent

class QueryInput(BaseModel):
    sql: str
    database: str = "main"

class DatabaseTool:
    name = "database_query"
    label = "Database Query"
    description = "Execute SQL queries"
    parameters = QueryInput  # Pydantic model 或 dict (JSON Schema)

    def __init__(self, conn_str: str):
        self._conn = conn_str

    async def execute(
        self,
        tool_call_id: str,
        params: QueryInput,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        rows = await run_query(params.sql, params.database)
        return AgentToolResult(
            content=[TextContent(text=format_table(rows))],
            details={"row_count": len(rows)},
        )
```

`parameters` 可以是：
- **Pydantic BaseModel 子类**（推荐）— 自动生成 JSON Schema
- **dict** — 直接提供 JSON Schema

## 内置工具

bampy 提供 7 个开箱即用的编码工具：

| 工具 | 名称 | 说明 |
| ---- | ---- | ---- |
| `BashTool` | `bash` | 执行 shell 命令，支持超时和工作目录 |
| `ReadTool` | `read` | 读取文件，支持行偏移/限制和图片 |
| `WriteTool` | `write` | 创建/覆盖文件 |
| `EditTool` | `edit` | 精确文本替换（find → replace） |
| `FindTool` | `find` | Glob 模式文件搜索 |
| `GrepTool` | `grep` | 正则表达式内容搜索 |
| `LsTool` | `ls` | 列出目录内容 |

### 使用方式

```python
from bampy.app import (
    # 预实例化工具
    bash_tool, read_tool, write_tool, edit_tool, find_tool, grep_tool, ls_tool,

    # 工具集
    read_only_tools,   # [read, find, grep, ls]
    coding_tools,      # [bash, read, write, edit, find, grep, ls]

    # 工厂函数（自定义配置）
    create_bash_tool,
    create_coding_tools,
    create_read_only_tools,
    create_all_tools,
)

# 自定义工作目录
tools = create_coding_tools(cwd="/my/project")
```

> `AgentSession` 在 `tools=None` 时默认使用 `create_coding_tools(cwd)`。

## 工具钩子

### before_tool_call

在工具执行前拦截，可阻止执行：

```python
from bampy.agent import BeforeToolCallContext, BeforeToolCallResult

async def before_hook(ctx: BeforeToolCallContext, cancellation=None):
    if ctx.tool_call.name == "bash" and "rm" in str(ctx.args):
        return BeforeToolCallResult(block=True, reason="Dangerous command")
    return None  # 不阻止
```

### after_tool_call

在工具执行后修改结果：

```python
from bampy.agent import AfterToolCallContext, AfterToolCallResult

async def after_hook(ctx: AfterToolCallContext, cancellation=None):
    if ctx.is_error:
        # 修改错误消息
        return AfterToolCallResult(
            content=[TextContent(text=f"Tool failed: {ctx.result.content[0].text}")],
        )
    return None  # 不修改
```
