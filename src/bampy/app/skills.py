"""Skill discovery, validation, and prompt formatting.

Skills are discovered from markdown files with YAML-like frontmatter. The
frontmatter provides a short name/description pair for prompt-time discovery,
while the full ``SKILL.md`` contents are loaded on demand by the agent.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DEFAULT_SKILL_FILE = "SKILL.md"
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
_NAME_RE = re.compile(r"^[a-z0-9-]+$")


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    source: str
    disable_model_invocation: bool = False


@dataclass(slots=True)
class SkillCollision:
    resource_type: str
    name: str
    winner_path: str
    loser_path: str


@dataclass(slots=True)
class SkillDiagnostic:
    type: Literal["warning", "collision"]
    message: str
    path: str
    collision: SkillCollision | None = None


@dataclass(slots=True)
class LoadSkillsResult:
    skills: list[Skill]
    diagnostics: list[SkillDiagnostic]


class _FrontmatterParseError(ValueError):
    def __init__(self, line: int, message: str) -> None:
        super().__init__(f"Invalid frontmatter at line {line}: {message}")
        self.line = line


def load_skills_from_dir(
    dir_path: str,
    *,
    source: str,
) -> LoadSkillsResult:
    """Load skills from a directory tree.

    Discovery rules mirror pi-mono's progressive disclosure model:

    - if a directory contains ``SKILL.md``, treat it as a single skill root and
      do not recurse further
    - otherwise, recurse into child directories
    - direct ``.md`` files are only considered at the root level
    """

    return _load_skills_from_dir_internal(
        Path(dir_path).expanduser(),
        source=source,
        include_root_files=True,
    )


def load_skills(
    *,
    cwd: str | None = None,
    skill_paths: list[str] | None = None,
    include_defaults: bool = True,
) -> LoadSkillsResult:
    """Load skills from default locations and explicit file/directory paths."""

    resolved_cwd = Path(cwd or os.getcwd()).expanduser().resolve()
    requested_paths = list(skill_paths or [])
    skill_map: dict[str, Skill] = {}
    seen_real_paths: set[str] = set()
    diagnostics: list[SkillDiagnostic] = []

    def _add(result: LoadSkillsResult) -> None:
        diagnostics.extend(result.diagnostics)
        for skill in result.skills:
            try:
                real_path = str(Path(skill.file_path).resolve())
            except OSError:
                real_path = skill.file_path

            if real_path in seen_real_paths:
                continue

            existing = skill_map.get(skill.name)
            if existing is not None:
                diagnostics.append(
                    SkillDiagnostic(
                        type="collision",
                        message=f'name "{skill.name}" collision',
                        path=skill.file_path,
                        collision=SkillCollision(
                            resource_type="skill",
                            name=skill.name,
                            winner_path=existing.file_path,
                            loser_path=skill.file_path,
                        ),
                    )
                )
                continue

            skill_map[skill.name] = skill
            seen_real_paths.add(real_path)

    if include_defaults:
        _add(
            load_skills_from_dir(
                str(Path.home() / ".bampy" / "skills"),
                source="user",
            )
        )
        _add(
            load_skills_from_dir(
                str(resolved_cwd / ".bampy" / "skills"),
                source="project",
            )
        )

    for raw_path in requested_paths:
        resolved_path = _resolve_skill_path(raw_path, resolved_cwd)
        if not resolved_path.exists():
            diagnostics.append(
                SkillDiagnostic(
                    type="warning",
                    message="skill path does not exist",
                    path=str(resolved_path),
                )
            )
            continue

        if resolved_path.is_dir():
            _add(load_skills_from_dir(str(resolved_path), source="path"))
            continue

        if resolved_path.is_file() and resolved_path.suffix.lower() == ".md":
            skill, file_diagnostics = _load_skill_from_file(resolved_path, source="path")
            diagnostics.extend(file_diagnostics)
            if skill is not None:
                _add(LoadSkillsResult(skills=[skill], diagnostics=[]))
            continue

        diagnostics.append(
            SkillDiagnostic(
                type="warning",
                message="skill path is not a markdown file",
                path=str(resolved_path),
            )
        )

    return LoadSkillsResult(skills=list(skill_map.values()), diagnostics=diagnostics)


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """Format skills as an XML section appended to the system prompt."""

    visible_skills = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible_skills:
        return ""

    lines = [
        "",
        "",
        "The following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory and use the absolute path in tool commands.",
        "",
        "<available_skills>",
    ]

    for skill in visible_skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{_escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")

    lines.append("</available_skills>")
    return "\n".join(lines)


def _load_skills_from_dir_internal(
    dir_path: Path,
    *,
    source: str,
    include_root_files: bool,
) -> LoadSkillsResult:
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []

    if not dir_path.exists():
        return LoadSkillsResult(skills=skills, diagnostics=diagnostics)

    try:
        skill_file = dir_path / DEFAULT_SKILL_FILE
        if skill_file.is_file():
            skill, file_diagnostics = _load_skill_from_file(skill_file, source=source)
            diagnostics.extend(file_diagnostics)
            if skill is not None:
                skills.append(skill)
            return LoadSkillsResult(skills=skills, diagnostics=diagnostics)

        entries = sorted(dir_path.iterdir(), key=lambda entry: entry.name.lower())
    except OSError:
        return LoadSkillsResult(skills=skills, diagnostics=diagnostics)

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            if entry.name == "node_modules":
                continue
            result = _load_skills_from_dir_internal(
                entry,
                source=source,
                include_root_files=False,
            )
            skills.extend(result.skills)
            diagnostics.extend(result.diagnostics)
            continue

        if include_root_files and entry.is_file() and entry.suffix.lower() == ".md":
            skill, file_diagnostics = _load_skill_from_file(entry, source=source)
            diagnostics.extend(file_diagnostics)
            if skill is not None:
                skills.append(skill)

    return LoadSkillsResult(skills=skills, diagnostics=diagnostics)


def _load_skill_from_file(
    file_path: Path,
    *,
    source: str,
) -> tuple[Skill | None, list[SkillDiagnostic]]:
    diagnostics: list[SkillDiagnostic] = []

    try:
        raw_text = file_path.read_text(encoding="utf-8")
        frontmatter = _parse_frontmatter(raw_text)
    except (OSError, _FrontmatterParseError) as exc:
        diagnostics.append(
            SkillDiagnostic(
                type="warning",
                message=str(exc),
                path=str(file_path),
            )
        )
        return None, diagnostics

    expected_name = file_path.parent.name if file_path.name == DEFAULT_SKILL_FILE else file_path.stem
    raw_name = frontmatter.get("name")
    name = str(raw_name).strip() if raw_name else expected_name

    raw_description = frontmatter.get("description")
    description = str(raw_description).strip() if raw_description is not None else ""

    diagnostics.extend(_validate_description(description, file_path))
    diagnostics.extend(_validate_name(name, expected_name, file_path))

    if not description:
        return None, diagnostics

    return (
        Skill(
            name=name,
            description=description,
            file_path=str(file_path.resolve()),
            base_dir=str(file_path.parent.resolve()),
            source=source,
            disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
        ),
        diagnostics,
    )


def _parse_frontmatter(raw_text: str) -> dict[str, Any]:
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    end_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break

    if end_index is None:
        raise _FrontmatterParseError(1, "missing closing --- delimiter")

    return _parse_frontmatter_lines(lines[1:end_index], start_line=2)


def _parse_frontmatter_lines(lines: list[str], *, start_line: int) -> dict[str, Any]:
    data: dict[str, Any] = {}
    index = 0

    while index < len(lines):
        raw_line = lines[index]
        line_number = start_line + index

        if not raw_line.strip():
            index += 1
            continue

        if raw_line.startswith((" ", "\t")):
            raise _FrontmatterParseError(line_number, "unexpected indentation")

        if ":" not in raw_line:
            raise _FrontmatterParseError(line_number, "expected 'key: value'")

        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        if not key:
            raise _FrontmatterParseError(line_number, "missing key name")

        value = raw_value.lstrip()
        if value in {"|", ">"}:
            block_lines: list[str] = []
            index += 1
            while index < len(lines):
                block_line = lines[index]
                if block_line.startswith((" ", "\t")) or not block_line.strip():
                    block_lines.append(block_line[2:] if block_line.startswith("  ") else block_line.lstrip())
                    index += 1
                    continue
                break

            if value == ">":
                data[key] = " ".join(part for part in block_lines if part).strip()
            else:
                data[key] = "\n".join(block_lines).strip()
            continue

        data[key] = _parse_scalar(value, line_number)
        index += 1

    return data


def _parse_scalar(value: str, line_number: int) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if value.startswith('"') or value.startswith("'"):
        quote = value[0]
        if len(value) < 2 or not value.endswith(quote):
            raise _FrontmatterParseError(line_number, "unterminated quoted string")
        return value[1:-1]

    if value.startswith("[") and not value.endswith("]"):
        raise _FrontmatterParseError(line_number, "unterminated list value")
    if value.startswith("{") and not value.endswith("}"):
        raise _FrontmatterParseError(line_number, "unterminated mapping value")

    return value


def _validate_name(name: str, expected_name: str, file_path: Path) -> list[SkillDiagnostic]:
    diagnostics: list[SkillDiagnostic] = []

    if name != expected_name:
        diagnostics.append(
            SkillDiagnostic(
                type="warning",
                message=f'name "{name}" does not match expected name "{expected_name}"',
                path=str(file_path),
            )
        )

    if len(name) > MAX_NAME_LENGTH:
        diagnostics.append(
            SkillDiagnostic(
                type="warning",
                message=f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})",
                path=str(file_path),
            )
        )

    if not _NAME_RE.fullmatch(name):
        diagnostics.append(
            SkillDiagnostic(
                type="warning",
                message="name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)",
                path=str(file_path),
            )
        )

    if name.startswith("-") or name.endswith("-"):
        diagnostics.append(
            SkillDiagnostic(
                type="warning",
                message="name must not start or end with a hyphen",
                path=str(file_path),
            )
        )

    if "--" in name:
        diagnostics.append(
            SkillDiagnostic(
                type="warning",
                message="name must not contain consecutive hyphens",
                path=str(file_path),
            )
        )

    return diagnostics


def _validate_description(description: str, file_path: Path) -> list[SkillDiagnostic]:
    if not description:
        return [
            SkillDiagnostic(
                type="warning",
                message="description is required",
                path=str(file_path),
            )
        ]

    if len(description) > MAX_DESCRIPTION_LENGTH:
        return [
            SkillDiagnostic(
                type="warning",
                message=f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})",
                path=str(file_path),
            )
        ]

    return []


def _resolve_skill_path(path: str, cwd: Path) -> Path:
    expanded = Path(os.path.expanduser(path))
    if expanded.is_absolute():
        return expanded.resolve()
    return (cwd / expanded).resolve()


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
