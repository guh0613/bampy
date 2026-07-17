# Skills

Skills 是带 YAML frontmatter 的 Markdown 能力描述。加载后，名称与描述会写入 system prompt；完整正文由模型在需要时用 `read` 工具按需读取（渐进披露）。

## SKILL.md 格式

```markdown
---
name: code-review
description: 审查代码并提供改进建议
---

# Code Review

审查提供的代码，关注以下方面：
- 代码质量和可读性
- 性能优化建议
- 安全漏洞检查
```

Frontmatter 字段：

| 字段 | 必须 | 说明 |
| ---- | ---- | ---- |
| `description` | 是 | 简短描述；缺失或为空则该 Skill 不会被加载 |
| `name` | 否 | 标识名；省略时用目录名（`SKILL.md`）或文件名 stem（根级 `.md`） |
| `disable-model-invocation` | 否 | 设为 `true` 时不出现在 prompt 的可用列表中 |

名称约定（不满足时产生 warning，仍可能加载）：

- 仅小写 `a-z`、`0-9`、连字符 `-`
- 最长 64 字符；描述最长 1024 字符
- 不以 `-` 开头/结尾，不含连续 `--`
- 建议与目录名 / 文件名一致

## 发现路径

`load_skills` / `create_agent_session` 的加载顺序（**同名先到者优先**）：

1. **用户全局** — `~/.bampy/skills/`（`include_defaults=True`）
2. **项目本地** — `<cwd>/.bampy/skills/`（`include_defaults=True`）
3. **显式路径** — `skill_paths=[...]`（目录或单个 `.md` 文件）

目录扫描规则（对齐渐进披露）：

- 若目录含 `SKILL.md`，将该目录视为单个 Skill 根，**不再递归子目录**
- 否则递归子目录；跳过隐藏目录与 `node_modules`
- 仅在扫描根目录时，额外接受直接放置的 `*.md` 文件

```text
.bampy/skills/
  code-review/SKILL.md          # skill 根
  research/                     # 继续向下找
    web-search/SKILL.md
  notes.md                      # 仅根级 .md 会作为 skill
```

## 加载与启用

```python
from bampy.app import load_skills, format_skills_for_prompt, create_agent_session

result = load_skills(
    cwd="/my/project",
    skill_paths=["/extra/skills", "./one-off.md"],
    include_defaults=True,
)

for skill in result.skills:
    print(skill.name, skill.description, skill.file_path, skill.source)

for diag in result.diagnostics:
    # type: "warning" | "collision"
    print(diag.type, diag.message, diag.path)

# 格式化为 system prompt 片段（跳过 disable_model_invocation）
text = format_skills_for_prompt(result.skills)
```

通过会话自动启用：

```python
result = await create_agent_session(
    skill_paths=["/custom/skills"],
    include_default_skills=True,  # 对应 load_skills(include_defaults=...)
)
# result.skills → LoadSkillsResult
```

`AgentSession` 构建 system prompt 时：

- 当前激活工具包含 `read`（或自定义 prompt 且未关闭增强）时，追加 Skills 列表
- Prompt 提示模型：任务匹配时用 `read` 读取 `location` 指向的 Skill 文件；相对路径相对 Skill 目录解析

也可以只加载某一目录：

```python
from bampy.app import load_skills_from_dir

result = load_skills_from_dir("/path/to/skills", source="path")
```

## 冲突处理

同名 Skill 再次出现时记录 `SkillDiagnostic(type="collision")`，保留先加载的，忽略后者。诊断中的 `SkillCollision` 含 `winner_path` / `loser_path`。

同一真实路径重复发现会被去重，不产生 collision。

相关文档：[agent-session.md](agent-session.md)、[tools.md](tools.md)、[extensions.md](extensions.md)。
