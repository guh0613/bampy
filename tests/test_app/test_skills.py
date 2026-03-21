"""Tests for bampy.app.skills."""

from __future__ import annotations

from pathlib import Path

from bampy.app.skills import (
    Skill,
    format_skills_for_prompt,
    load_skills,
    load_skills_from_dir,
)


def _write_skill(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestLoadSkillsFromDir:
    def test_loads_valid_skill_root(self, tmp_path: Path):
        skill_file = tmp_path / "valid-skill" / "SKILL.md"
        _write_skill(
            skill_file,
            "---\n"
            "name: valid-skill\n"
            "description: A valid skill.\n"
            "---\n\n"
            "# Valid Skill\n",
        )

        result = load_skills_from_dir(str(tmp_path / "valid-skill"), source="test")

        assert [skill.name for skill in result.skills] == ["valid-skill"]
        assert result.skills[0].description == "A valid skill."
        assert result.diagnostics == []

    def test_skips_missing_description_and_reports_warning(self, tmp_path: Path):
        skill_file = tmp_path / "missing-description" / "SKILL.md"
        _write_skill(
            skill_file,
            "---\n"
            "name: missing-description\n"
            "---\n\n"
            "# Missing Description\n",
        )

        result = load_skills_from_dir(str(tmp_path / "missing-description"), source="test")

        assert result.skills == []
        assert any(diagnostic.message == "description is required" for diagnostic in result.diagnostics)

    def test_parses_multiline_description_and_disable_model_invocation(self, tmp_path: Path):
        skill_file = tmp_path / "manual-only" / "SKILL.md"
        _write_skill(
            skill_file,
            "---\n"
            "name: manual-only\n"
            "description: |\n"
            "  First line.\n"
            "  Second line.\n"
            "disable-model-invocation: true\n"
            "---\n\n"
            "# Manual Only\n",
        )

        result = load_skills_from_dir(str(tmp_path / "manual-only"), source="test")

        assert len(result.skills) == 1
        assert result.skills[0].description == "First line.\nSecond line."
        assert result.skills[0].disable_model_invocation is True

    def test_prefers_root_skill_over_nested_skills(self, tmp_path: Path):
        root_skill = tmp_path / "root-skill" / "SKILL.md"
        nested_skill = tmp_path / "root-skill" / "nested-child" / "SKILL.md"
        _write_skill(
            root_skill,
            "---\n"
            "name: root-skill\n"
            "description: Root wins.\n"
            "---\n\n"
            "# Root Skill\n",
        )
        _write_skill(
            nested_skill,
            "---\n"
            "name: nested-child\n"
            "description: Nested should not load.\n"
            "---\n\n"
            "# Nested Skill\n",
        )

        result = load_skills_from_dir(str(tmp_path / "root-skill"), source="test")

        assert [skill.name for skill in result.skills] == ["root-skill"]

    def test_reports_invalid_frontmatter(self, tmp_path: Path):
        skill_file = tmp_path / "broken-skill" / "SKILL.md"
        _write_skill(
            skill_file,
            "---\n"
            "name: broken-skill\n"
            "description: [unclosed\n"
            "---\n\n"
            "# Broken Skill\n",
        )

        result = load_skills_from_dir(str(tmp_path / "broken-skill"), source="test")

        assert result.skills == []
        assert any("Invalid frontmatter at line 3" in diagnostic.message for diagnostic in result.diagnostics)


class TestLoadSkills:
    def test_loads_default_locations_and_reports_collisions(self, tmp_path: Path, monkeypatch):
        user_skill = tmp_path / "home" / ".bampy" / "skills" / "shared-skill" / "SKILL.md"
        project_skill = tmp_path / "project" / ".bampy" / "skills" / "shared-skill" / "SKILL.md"
        _write_skill(
            user_skill,
            "---\n"
            "name: shared-skill\n"
            "description: User version.\n"
            "---\n\n"
            "# User Skill\n",
        )
        _write_skill(
            project_skill,
            "---\n"
            "name: shared-skill\n"
            "description: Project version.\n"
            "---\n\n"
            "# Project Skill\n",
        )

        monkeypatch.setattr("bampy.app.skills.Path.home", staticmethod(lambda: tmp_path / "home"))

        result = load_skills(cwd=str(tmp_path / "project"))

        assert [skill.description for skill in result.skills] == ["User version."]
        assert any(diagnostic.type == "collision" for diagnostic in result.diagnostics)

    def test_supports_explicit_skill_paths(self, tmp_path: Path):
        explicit_skill = tmp_path / "extras" / "api-docs" / "SKILL.md"
        _write_skill(
            explicit_skill,
            "---\n"
            "name: api-docs\n"
            "description: Look up API docs.\n"
            "---\n\n"
            "# API Docs\n",
        )

        result = load_skills(
            cwd=str(tmp_path),
            include_defaults=False,
            skill_paths=["extras/api-docs"],
        )

        assert [skill.name for skill in result.skills] == ["api-docs"]
        assert result.diagnostics == []


class TestFormatSkillsForPrompt:
    def test_formats_skills_as_xml_and_filters_hidden_skills(self):
        prompt = format_skills_for_prompt(
            [
                Skill(
                    name="docs-search",
                    description='Docs search with <angle> & "quotes".',
                    file_path="/skills/docs-search/SKILL.md",
                    base_dir="/skills/docs-search",
                    source="project",
                ),
                Skill(
                    name="manual-only",
                    description="Only for explicit invocation.",
                    file_path="/skills/manual-only/SKILL.md",
                    base_dir="/skills/manual-only",
                    source="project",
                    disable_model_invocation=True,
                ),
            ]
        )

        assert "The following skills provide specialized instructions" in prompt
        assert "<available_skills>" in prompt
        assert "&lt;angle&gt;" in prompt
        assert "&amp;" in prompt
        assert "&quot;quotes&quot;" in prompt
        assert "manual-only" not in prompt
