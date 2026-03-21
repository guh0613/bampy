# 会话

bampy 使用 NDJSON（每行一个 JSON 对象）的追加写入模式持久化会话，支持树形分支结构。

## SessionManager

```python
from bampy.app import SessionManager, NDJSONBackend, InMemoryBackend

# 默认创建（自动生成路径）
session = SessionManager.create(cwd="/my/project")

# 自定义后端
session = SessionManager(backend=NDJSONBackend("./session.jsonl"), cwd=".")

# 内存后端（测试用）
session = SessionManager(backend=InMemoryBackend(), cwd=".")
```

### 写入

```python
from bampy.ai import UserMessage, AssistantMessage

# 追加消息
session.append_message(UserMessage(content="Hello"))

# 追加模型变更
session.append_model_change("anthropic", "claude-sonnet-4-20250514")

# 追加推理等级变更
session.append_thinking_level_change("high")

# 追加压缩记录
session.append_compaction(summary="...", first_kept_entry_id="m5", tokens_before=50000)

# 追加扩展消息（发送给 LLM）
session.append_custom_message_entry("my_type", "content", display=True)

# 追加扩展数据（不发送给 LLM）
session.append_custom_entry("my_data", {"key": "value"})
```

### 读取

```python
# 构建当前分支的上下文
context = session.build_session_context()
context.messages          # list[AgentMessage]
context.thinking_level    # str
context.model             # dict | None

# 获取当前分支条目
entries = session.get_branch()
```

### 分支

```python
# 从某个条目创建新分支
session.branch(branch_from_id="entry_id")
```

## 条目类型

NDJSON 文件中每行是一个条目，通过 `parent_id` 构成树形结构：

| 类型 | 说明 |
| ---- | ---- |
| `SessionHeader` | 会话头，包含版本号 |
| `SessionMessageEntry` | 标准 LLM 消息（user/assistant/tool_result） |
| `ModelChangeEntry` | 模型切换记录 |
| `ThinkingLevelChangeEntry` | 推理等级变更记录 |
| `CompactionEntry` | 压缩摘要 + 保留起点 |
| `BranchSummaryEntry` | 分支偏离摘要 |
| `CustomEntry` | 扩展数据（不发送给 LLM） |
| `CustomMessageEntry` | 扩展消息（发送给 LLM） |
| `LabelEntry` | 条目标签 |
| `SessionInfoEntry` | 会话元数据 |

## NDJSON 格式示例

```jsonl
{"type":"session","id":"abc","version":1,"timestamp":"2024-01-01T00:00:00Z"}
{"type":"message","id":"m1","parent_id":null,"message":{"role":"user","content":"Hello"}}
{"type":"message","id":"m2","parent_id":"m1","message":{"role":"assistant","content":[...]}}
{"type":"model_change","id":"mc1","parent_id":"m2","provider":"openai","model_id":"gpt-4o"}
{"type":"compaction","id":"c1","parent_id":"mc1","summary":"...","first_kept_entry_id":"m5"}
```

## 自定义后端

实现 `SessionBackend` 协议即可替换存储：

```python
from bampy.app import SessionBackend

class SQLiteBackend:
    async def append(self, entry: dict) -> None:
        ...

    async def read_all(self) -> list[dict]:
        ...

    async def rewrite(self, entries: list[dict]) -> None:
        ...  # 用于压缩后重写
```

## 应用消息类型

app 层定义了几种特殊消息类型，通过消息转换器注册后可参与 LLM 对话：

```python
from bampy.app import (
    CompactionSummaryMessage,  # 压缩摘要
    BranchSummaryMessage,      # 分支摘要
    CustomMessage,             # 扩展自定义消息
    register_app_message_converters,
)

# AgentSession 初始化时自动调用
register_app_message_converters()
```
