# Skills

Skills 是通过 Markdown 文件声明的能力描述，加载后会附加到 system prompt 中，告知 LLM 可用的技能。

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

YAML frontmatter 字段：

| 字段 | 必须 | 说明 |
| ---- | ---- | ---- |
| `name` | 是 | Skill 名称（唯一标识） |
| `description` | 是 | 简短描述 |
| `disable-model-invocation` | 否 | 设为 true 禁止模型主动调用 |

## 放置 Skills

Skills 按以下位置发现：

1. **项目本地** — `<cwd>/.bampy/skills/`
2. **用户全局** — `~/.bampy/skills/`
3. **显式路径** — `create_agent_session(skill_paths=["/custom/skills"])`

目录内递归扫描 `SKILL.md` 文件。如果某目录根下有 `SKILL.md`，该目录视为一个 Skill 的根，不再递归子目录。

## 加载

```python
from bampy.app import load_skills, format_skills_for_prompt

result = load_skills(
    cwd="/my/project",
    skill_paths=["/extra/skills"],
    include_defaults=True,
)

for skill in result.skills:
    print(f"{skill.name}: {skill.description}")

# 诊断信息
for diag in result.diagnostics:
    print(f"Warning: {diag}")

# 格式化为 system prompt 片段
text = format_skills_for_prompt(result.skills)
```

> `AgentSession` 自动加载 Skills 并注入 system prompt，通常无需手动操作。

## 冲突处理

如果多个位置存在同名 Skill，`load_skills` 会记录 `SkillCollision` 诊断，先发现的优先。
