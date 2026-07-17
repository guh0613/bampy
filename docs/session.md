# 会话持久化

bampy 用**追加写入的 NDJSON（JSONL）**保存会话：每个 entry 有 `id` 与 `parent_id`，形成树；**leaf** 指针表示当前分支末端。分支只移动 leaf，不改写历史。

LLM 上下文通过 `SessionManager.build_session_context()`（或模块级 `build_session_context`）沿 root→leaf 解析，并处理 compaction / branch summary。

导入：`from bampy.app import SessionManager, NDJSONBackend, InMemoryBackend, ...`。

## SessionManager

### 创建与打开

```python
from bampy.app import SessionManager

# 新建会话（默认目录 ~/.bampy/sessions/--<cwd-safe>--/）
sm = SessionManager.create(cwd="/my/project")

# 打开已有 .jsonl
sm = SessionManager.open("/path/to/session.jsonl")

# 内存会话（测试 / 临时）
sm = SessionManager.in_memory(cwd=".")

# 底层构造（一般用上面的工厂方法）
sm = SessionManager(
    cwd="/my/project",
    backend=None,          # persist=True 时懒创建 NDJSONBackend
    session_dir=None,
    session_file=None,
    persist=True,
)
```

属性：`cwd`、`session_id`、`session_file`、`session_dir`、`leaf_id`。

### 写入

```python
from bampy.ai import UserMessage

sm.append_message(UserMessage(content="Hello"))
sm.append_model_change("anthropic", "claude-sonnet-4-5")
sm.append_thinking_level_change("high")
sm.append_compaction(
    summary="...",
    first_kept_entry_id="m5",
    tokens_before=50000,
)
sm.append_custom_message_entry("my_type", "content", display=True)
sm.append_custom_entry("my_data", {"key": "value"})
sm.append_session_info("我的会话")
sm.append_label_change("entry_id", "checkpoint")
sm.append_branch_summary(from_id="m3", summary="偏离说明")
```

持久化策略：在出现**第一条 assistant 消息之前**不会真正写盘；之后先整文件 `rewrite`，再对新 entry `append`。

### 读取与树

```python
context = sm.build_session_context()
context.messages         # 解析后的消息列表
context.thinking_level   # str，默认 "off"
context.model            # {"provider", "model_id"} | None

sm.get_branch()          # 当前 leaf 到 root 的路径（root→leaf）
sm.get_tree()            # list[SessionTreeNode]
sm.get_entries()
sm.get_entry(entry_id)
sm.get_leaf_entry()
sm.get_header()
sm.get_label(entry_id)
sm.get_session_name()
```

### 分支

```python
sm.branch(branch_from_id="entry_id")          # 将 leaf 移到该 entry
sm.reset_leaf()                               # leaf = None
sm.branch_with_summary("entry_id", "摘要...")  # branch + append_branch_summary
```

### 列出会话

```python
sessions = await SessionManager.list_sessions(cwd="/my/project")
# list[SessionInfo]: path, id, cwd, name, created, modified, message_count, ...
```

## 条目类型

| 类 | `type` | 说明 |
| -- | ------ | ---- |
| `SessionHeader` | `session` | 文件头：`version`、`id`、`timestamp`、`cwd`、`parent_session` |
| `SessionMessageEntry` | `message` | LLM 消息（`message` dict） |
| `ModelChangeEntry` | `model_change` | `provider`、`model_id` |
| `ThinkingLevelChangeEntry` | `thinking_level_change` | `thinking_level` |
| `CompactionEntry` | `compaction` | `summary`、`first_kept_entry_id`、`tokens_before`、`details`、`from_hook` |
| `BranchSummaryEntry` | `branch_summary` | `from_id`、`summary`、`details`、`from_hook` |
| `CustomEntry` | `custom` | 扩展数据，**不进** LLM |
| `CustomMessageEntry` | `custom_message` | 扩展消息，**可进** LLM |
| `LabelEntry` | `label` | `target_id`、`label` |
| `SessionInfoEntry` | `session_info` | `name` |

公共基类字段（header 除外）：`id`、`parent_id`、`timestamp`。

联合类型别名：`SessionEntry`。上下文结果：`SessionContext`；树节点：`SessionTreeNode`；列表项：`SessionInfo`。

## NDJSON 示例

```jsonl
{"type":"session","version":1,"id":"abc","timestamp":"2024-01-01T00:00:00+00:00","cwd":"/my/project","parent_session":null}
{"type":"message","id":"m1","parent_id":null,"timestamp":"...","message":{"role":"user","content":"Hello"}}
{"type":"message","id":"m2","parent_id":"m1","timestamp":"...","message":{"role":"assistant","content":[...]}}
{"type":"model_change","id":"mc1","parent_id":"m2","timestamp":"...","provider":"openai","model_id":"gpt-5.4-mini"}
{"type":"compaction","id":"c1","parent_id":"mc1","timestamp":"...","summary":"...","first_kept_entry_id":"m5","tokens_before":50000,"details":null,"from_hook":false}
```

当前版本常量：`CURRENT_SESSION_VERSION = 1`。

## 自定义后端

实现同步协议 `SessionBackend`：

```python
from bampy.app import SessionBackend  # Protocol

class MyBackend:
    def append(self, entry: dict) -> None: ...
    def read_all(self) -> list[dict]: ...
    def rewrite(self, entries: list[dict]) -> None: ...
```

内置：

- `NDJSONBackend(path)` — 文件追加
- `InMemoryBackend()` — 内存列表

传入方式：`SessionManager(cwd, backend=MyBackend(), persist=True)`，或 `in_memory()`。

## 应用层消息类型

压缩 / 分支摘要等会以自定义消息角色进入 agent 消息列表，需注册转换器后才能送给 LLM：

```python
from bampy.app import (
    CompactionSummaryMessage,
    BranchSummaryMessage,
    CustomMessage,
    register_app_message_converters,
    create_compaction_summary_message,
    create_branch_summary_message,
    create_custom_message,
    convert_app_messages_to_llm,
)

# AgentSession 初始化时会自动调用
register_app_message_converters()
```

与 [AgentSession](agent-session.md) 配合时，消息会在 agent 事件流中自动 `append_message` 到 `SessionManager`。更多压缩行为见 [compaction.md](compaction.md)。
