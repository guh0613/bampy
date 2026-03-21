"""Extension discovery and loading.

Extensions are Python modules exporting a ``setup(api: ExtensionAPI)`` factory.
They are discovered from:

1. Explicit module paths
2. Project-local ``<cwd>/.bampy/extensions/``
3. Global ``~/.bampy/extensions/``
4. ``entry_points`` group ``bampy.extensions``
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .extension import Extension, ExtensionAPI, ExtensionFactory

logger = logging.getLogger(__name__)

ENTRY_POINTS_GROUP = "bampy.extensions"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_module_from_path(path: str) -> Any:
    """Import a Python module from an absolute file path."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Extension not found: {resolved}")

    module_name = f"bampy_ext_{resolved.stem}_{id(resolved)}"
    spec = importlib.util.spec_from_file_location(module_name, str(resolved))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {resolved}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_module_by_name(module_name: str) -> Any:
    """Import a module by its dotted name."""
    return importlib.import_module(module_name)


def _get_setup_fn(module: Any) -> ExtensionFactory | None:
    """Extract the ``setup`` function from an extension module."""
    setup = getattr(module, "setup", None)
    if callable(setup):
        return setup
    return None


async def _run_setup(
    setup: ExtensionFactory,
    api: ExtensionAPI,
) -> None:
    """Run the setup function (sync or async)."""
    result = setup(api)
    if asyncio.iscoroutine(result):
        await result


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_extension_paths(
    cwd: str | None = None,
    extra_paths: list[str] | None = None,
) -> list[str]:
    """Return a list of extension file paths to load.

    Searches:
    1. ``extra_paths`` (explicit)
    2. ``<cwd>/.bampy/extensions/*.py``
    3. ``~/.bampy/extensions/*.py``
    """
    paths: list[str] = []

    if extra_paths:
        for p in extra_paths:
            resolved = Path(p).expanduser().resolve()
            if resolved.exists():
                paths.append(str(resolved))

    # Project-local extensions
    if cwd:
        local_dir = Path(cwd) / ".bampy" / "extensions"
        if local_dir.is_dir():
            for f in sorted(local_dir.glob("*.py")):
                paths.append(str(f))

    # Global extensions
    global_dir = Path.home() / ".bampy" / "extensions"
    if global_dir.is_dir():
        for f in sorted(global_dir.glob("*.py")):
            if str(f) not in paths:
                paths.append(str(f))

    return paths


def discover_entry_point_modules() -> list[str]:
    """Discover extension modules registered via ``entry_points``."""
    modules: list[str] = []
    try:
        eps = importlib.metadata.entry_points()
        # Python 3.12+ returns SelectableGroups
        group = eps.select(group=ENTRY_POINTS_GROUP) if hasattr(eps, "select") else eps.get(ENTRY_POINTS_GROUP, [])
        for ep in group:
            modules.append(ep.value)
    except Exception:
        pass
    return modules


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class LoadExtensionsResult:
    extensions: list[Extension]
    errors: list[LoadError]


@dataclass
class LoadError:
    path: str
    error: str


async def load_extensions(
    *,
    paths: list[str] | None = None,
    modules: list[str] | None = None,
    cwd: str | None = None,
    discover: bool = True,
) -> LoadExtensionsResult:
    """Load extensions from file paths, module names, and entry_points.

    Parameters
    ----------
    paths:
        Explicit extension file paths.
    modules:
        Explicit dotted module names.
    cwd:
        Working directory for project-local discovery.
    discover:
        If ``True``, also discover from filesystem and entry_points.
    """
    extensions: list[Extension] = []
    errors: list[LoadError] = []

    # Collect all sources
    all_paths: list[str] = list(paths or [])
    all_modules: list[str] = list(modules or [])

    if discover:
        all_paths.extend(discover_extension_paths(cwd=cwd))
        all_modules.extend(discover_entry_point_modules())

    # De-duplicate
    seen_paths: set[str] = set()
    unique_paths: list[str] = []
    for p in all_paths:
        if p not in seen_paths:
            seen_paths.add(p)
            unique_paths.append(p)

    seen_modules: set[str] = set()
    unique_modules: list[str] = []
    for m in all_modules:
        if m not in seen_modules:
            seen_modules.add(m)
            unique_modules.append(m)

    # Load from file paths
    for path in unique_paths:
        try:
            module = _load_module_from_path(path)
            setup = _get_setup_fn(module)
            if setup is None:
                errors.append(LoadError(path=path, error="No setup() function found"))
                continue
            api = ExtensionAPI(extension_path=path)
            await _run_setup(setup, api)
            extensions.append(api._build_extension())
            logger.debug("Loaded extension from path: %s", path)
        except Exception as exc:
            errors.append(LoadError(path=path, error=str(exc)))
            logger.warning("Failed to load extension %s: %s", path, exc)

    # Load from module names
    for mod_name in unique_modules:
        try:
            module = _load_module_by_name(mod_name)
            setup = _get_setup_fn(module)
            if setup is None:
                errors.append(LoadError(path=mod_name, error="No setup() function found"))
                continue
            api = ExtensionAPI(extension_path=mod_name)
            await _run_setup(setup, api)
            extensions.append(api._build_extension())
            logger.debug("Loaded extension from module: %s", mod_name)
        except Exception as exc:
            errors.append(LoadError(path=mod_name, error=str(exc)))
            logger.warning("Failed to load extension module %s: %s", mod_name, exc)

    return LoadExtensionsResult(extensions=extensions, errors=errors)


async def discover_and_load_extensions(
    cwd: str | None = None,
    extra_paths: list[str] | None = None,
) -> LoadExtensionsResult:
    """Convenience: discover and load all extensions."""
    return await load_extensions(
        paths=extra_paths,
        cwd=cwd,
        discover=True,
    )
