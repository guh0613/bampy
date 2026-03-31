"""System prompt construction.

Builds the LLM system prompt with tool descriptions, guidelines, project
context files, and date/cwd information.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .skills import Skill, format_skills_for_prompt


@dataclass(slots=True)
class ContextFile:
    """A project context file loaded into the system prompt."""

    path: str
    content: str


@dataclass
class BuildSystemPromptOptions:
    """Options for :func:`build_system_prompt`."""

    custom_prompt: str | None = None
    selected_tools: list[str] | None = None
    tool_snippets: dict[str, str] | None = None
    prompt_guidelines: list[str] | None = None
    append_system_prompt: str | None = None
    augment_custom_prompt: bool = True
    cwd: str | None = None
    context_files: list[ContextFile] | None = None
    skills: list[Skill] | None = None


# Default tool descriptions
TOOL_DESCRIPTIONS: dict[str, str] = {
    "read": "Read file contents",
    "bash": "Execute bash commands",
    "edit": "Make surgical edits to files (find exact text and replace)",
    "write": "Create or overwrite files",
    "grep": "Search file contents for patterns",
    "find": "Find files by glob pattern",
    "ls": "List directory contents",
}


def build_system_prompt(options: BuildSystemPromptOptions | None = None) -> str:
    """Build the system prompt with tools, guidelines, and context."""
    if options is None:
        options = BuildSystemPromptOptions()

    cwd = options.cwd or os.getcwd()
    prompt_cwd = cwd.replace("\\", "/")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    append_section = f"\n\n{options.append_system_prompt}" if options.append_system_prompt else ""
    context_files = options.context_files or []
    skills = options.skills or []

    # Custom prompt mode
    if options.custom_prompt:
        prompt = options.custom_prompt
        if append_section:
            prompt += append_section
        if not options.augment_custom_prompt:
            return prompt
        if context_files:
            prompt += "\n\n# Project Context\n\nProject-specific instructions and guidelines:\n\n"
            for cf in context_files:
                prompt += f"## {cf.path}\n\n{cf.content}\n\n"
        if (not options.selected_tools or "read" in options.selected_tools) and skills:
            prompt += format_skills_for_prompt(skills)
        prompt += f"\nCurrent date: {date}"
        prompt += f"\nCurrent working directory: {prompt_cwd}"
        return prompt

    # Default prompt construction
    tools = options.selected_tools or ["read", "bash", "edit", "write"]
    snippets = options.tool_snippets or {}

    visible = [t for t in tools if t in TOOL_DESCRIPTIONS or t in snippets]
    if visible:
        tools_list = "\n".join(
            f"- {t}: {snippets.get(t, TOOL_DESCRIPTIONS.get(t, t))}" for t in visible
        )
    else:
        tools_list = "(none)"

    # Build guidelines
    guidelines_set: set[str] = set()
    guidelines_list: list[str] = []

    def _add(g: str) -> None:
        if g not in guidelines_set:
            guidelines_set.add(g)
            guidelines_list.append(g)

    has_bash = "bash" in tools
    has_edit = "edit" in tools
    has_write = "write" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools
    has_read = "read" in tools

    if has_bash and not (has_grep or has_find or has_ls):
        _add("Use bash for file operations like ls, rg, find")
    elif has_bash and (has_grep or has_find or has_ls):
        _add("Prefer grep/find/ls tools over bash for file exploration (faster)")

    if has_read and has_edit:
        _add("Use read to examine files before editing")
    if has_edit:
        _add("Use edit for precise changes (old text must match exactly)")
    if has_write:
        _add("Use write only for new files or complete rewrites")
    if has_edit or has_write:
        _add("When summarizing your actions, output plain text directly")

    for g in options.prompt_guidelines or []:
        g = g.strip()
        if g:
            _add(g)

    _add("Be concise in your responses")
    _add("Show file paths clearly when working with files")

    guidelines = "\n".join(f"- {g}" for g in guidelines_list)

    prompt = f"""You are an expert coding assistant. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{tools_list}

Guidelines:
{guidelines}"""

    if append_section:
        prompt += append_section

    if context_files:
        prompt += "\n\n# Project Context\n\nProject-specific instructions and guidelines:\n\n"
        for cf in context_files:
            prompt += f"## {cf.path}\n\n{cf.content}\n\n"

    if has_read and skills:
        prompt += format_skills_for_prompt(skills)

    prompt += f"\nCurrent date: {date}"
    prompt += f"\nCurrent working directory: {prompt_cwd}"

    return prompt


def load_context_files(
    cwd: str,
    filenames: list[str] | None = None,
) -> list[ContextFile]:
    """Load project context files (e.g. CLAUDE.md, .bampy/context.md)."""
    if filenames is None:
        filenames = ["CLAUDE.md", ".bampy/context.md"]

    result: list[ContextFile] = []
    for name in filenames:
        p = Path(cwd) / name
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8")
                result.append(ContextFile(path=name, content=content))
            except OSError:
                continue
    return result
