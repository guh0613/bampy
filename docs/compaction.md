# 上下文压缩

长对话会逼近模型上下文窗口。bampy 将较早消息摘要化，保留近期内容，并写入会话中的 `CompactionEntry`，随后重载分支消息。

## 自动压缩

`AgentSession` 默认开启 `auto_compaction=True`。在 Agent 循环正常结束（非 abort/error）后估算上下文用量，超过阈值则触发压缩。

```python
from bampy.app import create_agent_session, CompactionSettings

result = await create_agent_session(
    model=model,
    auto_compaction=True,
    compaction_settings=CompactionSettings(
        enabled=True,
        reserve_tokens=16_384,     # 为后续对话预留的安全余量
        keep_recent_tokens=20_000,  # 尽量保留的近期消息 token 量
    ),
    summarization_model=summary_model,              # 可选：专用摘要模型
    summarization_api_key=None,                     # 可选；否则走 get_api_key / stream_options
    summarization_custom_instructions="保留文件路径",  # 可选
)
session = result.session
```

触发条件（`should_compact`）：

```text
settings.enabled
且 context_tokens > context_window - reserve_tokens
```

`context_tokens` 优先取最近一条有效 assistant `usage`，再加上其后消息的启发式估算（约 `chars / 4`）。

### 会话事件

自动压缩会向 `session.subscribe` 发出：

```python
from bampy.app import AutoCompactionStartEvent, AutoCompactionEndEvent

def on_event(event):
    if event.type == "auto_compaction_start":
        # reason == "threshold"
        print("Compacting...")
    elif event.type == "auto_compaction_end":
        if event.result is not None:
            print(f"tokens_before={event.result.tokens_before}")
            print(event.result.summary[:100])
        elif event.aborted or event.error_message:
            print(f"Compaction failed: {event.error_message}")

session.subscribe(on_event)
```

压缩成功后还会向扩展发出 `session_compact`（见 [extensions.md](extensions.md)）。

## 手动压缩

```python
result = await session.compact()
if result is not None:
    print(result.tokens_before)
    print(result.first_kept_entry_id)
    print(result.summary[:200])
```

`CompactionResult` 字段：`summary`、`first_kept_entry_id`、`tokens_before`、`details`（**没有** `tokens_after`）。

手动压缩会等待会话空闲后执行；若分支末尾已是 `CompactionEntry`，或无法确定切点，则返回 `None`。

## 底层 API

可脱离 `AgentSession` 使用：

```python
from bampy.app import (
    CompactionSettings,
    estimate_tokens,
    estimate_context_tokens,
    should_compact,
    prepare_compaction,
    compact,
    generate_summary,
)

# 单条消息启发式 token
tokens = estimate_tokens(message)

# 上下文估算（usage + trailing）
estimate = estimate_context_tokens(messages)
if should_compact(estimate.tokens, model.context_window, CompactionSettings()):
    preparation = prepare_compaction(session_entries, CompactionSettings())
    if preparation is not None:
        result = await compact(
            preparation,
            model,
            api_key="sk-...",
            custom_instructions=None,
        )
```

常用符号：

| 符号 | 作用 |
| ---- | ---- |
| `CompactionSettings` | `enabled` / `reserve_tokens` / `keep_recent_tokens` |
| `estimate_tokens` | 单消息 chars/4 估算 |
| `estimate_context_tokens` | 返回 `ContextUsageEstimate` |
| `should_compact` | 阈值判断 |
| `prepare_compaction` | 从当前分支 `SessionEntry` 计算切点与待摘要消息 |
| `generate_summary` | 仅生成摘要文本 |
| `compact` | 基于 `CompactionPreparation` 生成 `CompactionResult`（含 split-turn 前缀摘要） |

## 压缩流程

1. **估算** — `estimate_context_tokens`（或会话内等价逻辑）
2. **判断** — `context_tokens > context_window - reserve_tokens`
3. **准备** — `prepare_compaction`：自上一次 `CompactionEntry` 之后找切点，保留约 `keep_recent_tokens` 的近期条目；若切在一轮中间则拆分 turn prefix
4. **摘要** — LLM 生成结构化 checkpoint（可叠加 `previous_summary` 增量更新）；split-turn 时并行摘要历史与 turn 前缀
5. **持久化** — `SessionManager.append_compaction(...)` 写入 `CompactionEntry`
6. **重载** — 会话重建消息列表（摘要 + 保留消息），并 `emit(SessionCompactEvent)`

默认摘要格式包含 Goal / Constraints / Progress / Key Decisions / Next Steps / Critical Context 等固定章节，便于后续模型续写。

相关文档：[agent-session.md](agent-session.md)、[session.md](session.md)、[extensions.md](extensions.md)。
