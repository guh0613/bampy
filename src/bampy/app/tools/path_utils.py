"""Path helpers shared by built-in tools."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

_UNICODE_SPACES = {
    "\u00A0",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200A",
    "\u202F",
    "\u205F",
    "\u3000",
}
_NARROW_NO_BREAK_SPACE = "\u202F"


def _normalize_unicode_spaces(value: str) -> str:
    return "".join(" " if ch in _UNICODE_SPACES else ch for ch in value)


def _normalize_at_prefix(value: str) -> str:
    return value[1:] if value.startswith("@") else value


def _try_macos_screenshot_variant(path: str) -> str:
    return path.replace(" AM.", f"{_NARROW_NO_BREAK_SPACE}AM.").replace(
        " PM.",
        f"{_NARROW_NO_BREAK_SPACE}PM.",
    )


def _try_nfd_variant(path: str) -> str:
    return unicodedata.normalize("NFD", path)


def _try_curly_quote_variant(path: str) -> str:
    return path.replace("'", "\u2019")


def expand_path(path: str) -> str:
    """Expand ``~`` and normalize prefixes commonly typed by users."""
    normalized = _normalize_at_prefix(_normalize_unicode_spaces(path))
    return os.path.expanduser(normalized)


def resolve_to_cwd(path: str, cwd: str) -> str:
    expanded = expand_path(path)
    return expanded if os.path.isabs(expanded) else str(Path(cwd, expanded).resolve())


def resolve_read_path(path: str, cwd: str) -> str:
    """Resolve a read path and try a few macOS filename variants if needed."""
    resolved = resolve_to_cwd(path, cwd)
    if os.path.exists(resolved):
        return resolved

    variants = [
        _try_macos_screenshot_variant(resolved),
        _try_nfd_variant(resolved),
        _try_curly_quote_variant(resolved),
        _try_curly_quote_variant(_try_nfd_variant(resolved)),
    ]
    for candidate in variants:
        if candidate != resolved and os.path.exists(candidate):
            return candidate
    return resolved
