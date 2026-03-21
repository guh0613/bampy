"""Tests for bampy.app.loader."""

from __future__ import annotations

import sys
from pathlib import Path

from bampy.app import loader
from bampy.app.loader import (
    _get_setup_fn,
    _load_module_from_path,
    discover_and_load_extensions,
    discover_entry_point_modules,
    discover_extension_paths,
    load_extensions,
)


def _write_extension(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


class TestLoadingHelpers:
    def test_load_module_from_path_and_get_setup(self, tmp_path: Path):
        extension_file = tmp_path / "demo_ext.py"
        _write_extension(
            extension_file,
            "def setup(api):\n"
            "    api.register_command('hello', description='world')\n",
        )

        module = _load_module_from_path(str(extension_file))
        setup = _get_setup_fn(module)

        assert module is sys.modules[module.__name__]
        assert callable(setup)

    def test_discover_extension_paths_collects_extra_local_and_global(self, tmp_path: Path, monkeypatch):
        home_dir = tmp_path / "home"
        global_dir = home_dir / ".bampy" / "extensions"
        local_dir = tmp_path / "project" / ".bampy" / "extensions"
        global_dir.mkdir(parents=True)
        local_dir.mkdir(parents=True)

        extra = tmp_path / "extra.py"
        local = local_dir / "local.py"
        global_file = global_dir / "global.py"
        for path in (extra, local, global_file):
            path.write_text("# ext\n", encoding="utf-8")

        monkeypatch.setattr(loader.Path, "home", staticmethod(lambda: home_dir))

        paths = discover_extension_paths(
            cwd=str(tmp_path / "project"),
            extra_paths=[str(extra)],
        )

        assert paths == [str(extra.resolve()), str(local), str(global_file)]

    def test_discover_entry_point_modules_supports_selectable_groups(self, monkeypatch):
        class EntryPoint:
            def __init__(self, value: str) -> None:
                self.value = value

        class EntryPoints:
            def select(self, *, group: str):
                assert group == "bampy.extensions"
                return [EntryPoint("pkg.alpha"), EntryPoint("pkg.beta")]

        monkeypatch.setattr(loader.importlib.metadata, "entry_points", lambda: EntryPoints())

        assert discover_entry_point_modules() == ["pkg.alpha", "pkg.beta"]


class TestLoadExtensions:
    async def test_load_extensions_from_paths_and_modules(self, tmp_path: Path, monkeypatch):
        path_ext = tmp_path / "path_ext.py"
        _write_extension(
            path_ext,
            "def setup(api):\n"
            "    api.register_command('from_path', description='path')\n",
        )

        module_ext = tmp_path / "module_ext.py"
        _write_extension(
            module_ext,
            "async def setup(api):\n"
            "    api.register_command('from_module', description='module')\n",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        result = await load_extensions(
            paths=[str(path_ext), str(path_ext)],
            modules=["module_ext", "module_ext"],
            discover=False,
        )

        assert len(result.errors) == 0
        assert len(result.extensions) == 2
        assert result.extensions[0].commands["from_path"].description == "path"
        assert result.extensions[1].commands["from_module"].description == "module"

    async def test_load_extensions_collects_errors(self, tmp_path: Path):
        no_setup = tmp_path / "no_setup.py"
        _write_extension(no_setup, "VALUE = 1\n")

        result = await load_extensions(
            paths=[str(no_setup)],
            modules=["missing_extension_module"],
            discover=False,
        )

        assert len(result.extensions) == 0
        assert len(result.errors) == 2
        assert result.errors[0].path == str(no_setup)
        assert "No setup() function found" in result.errors[0].error
        assert result.errors[1].path == "missing_extension_module"

    async def test_discover_and_load_extensions_uses_project_discovery(self, tmp_path: Path):
        project_dir = tmp_path / "project"
        ext_dir = project_dir / ".bampy" / "extensions"
        ext_dir.mkdir(parents=True)
        ext_path = ext_dir / "project_ext.py"
        _write_extension(
            ext_path,
            "def setup(api):\n"
            "    api.register_command('project_cmd', description='project')\n",
        )

        result = await discover_and_load_extensions(cwd=str(project_dir))

        assert len(result.errors) == 0
        assert len(result.extensions) == 1
        assert "project_cmd" in result.extensions[0].commands
