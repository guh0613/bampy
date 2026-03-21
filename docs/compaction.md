# 上下文压缩

长对话会逐渐逼近模型的上下文窗口限制。bampy 提供自动压缩机制，将早期消息摘要化以释放空间。

## 自动压缩

`AgentSession` 默认开启自动压缩。每次 Agent 循环结束后检查 token 用量，超过阈值时自动触发。

```python
from bampy.app import create_agent_session, CompactionSettings

result = await create_agent_session(
    model=model,
    auto_compaction=True,  # 默认开启
    compaction_settings=CompactionSettings(
        enabled=True,
        reserve_tokens=16384,    # 留给新对话的安全余量
        keep_recent_tokens=20000,  # 保留最近消息的 token 数
    ),
    summarization_model=summary_model,       # 可选：摘要用不同模型
    summarization_custom_instructions="...",  # 可选：自定义摘要指令
)
```

### 事件监听

```python
def on_event(event):
    if event.type == "auto_compaction_start":
        print("Compacting...")
    elif event.type == "auto_compaction_end":
        if event.result:
            saved = event.result.tokens_before - event.result.tokens_after
            print(f"Saved {saved} tokens")
        elif event.error_message:
            print(f"Compaction failed: {event.error_message}")

session.subscribe(on_event)
```

## 手动压缩

```python
result = await session.compact()
if result:
    print(f"{result.tokens_before} → {result.tokens_after} tokens")
    print(f"Summary: {result.summary[:100]}...")
```

## 底层 API

可独立于 AgentSession 使用压缩功能：

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

# Token 估算（chars / 4 启发式）
tokens = estimate_tokens(message)

# 检查是否需要压缩
context_tokens = estimate_context_tokens(messages).tokens
if should_compact(context_tokens, model.context_window, settings):
    preparation = prepare_compaction(session_entries, settings)
    if preparation:
        result = await compact(preparation, model, api_key="sk-...")
```

## 压缩流程

1. **估算** — 计算当前上下文 token 数
2. **判断** — `context_tokens > context_window - reserve_tokens`
3. **准备** — 找到分割点，保留最近 `keep_recent_tokens` 的消息
4. **摘要** — 调用 LLM 生成早期消息的摘要
5. **持久化** — 写入 `CompactionEntry` 到会话
6. **重载** — 从会话重建消息列表（摘要 + 保留的消息）
