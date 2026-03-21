"""Session management with append-only NDJSON tree persistence.

Each session is a tree of entries stored in a JSONL file. Every entry has an
``id`` and ``parent_id`` forming a tree.  The *leaf* pointer tracks the current
position; appending creates a child of the current leaf.  Branching moves the
leaf to an earlier entry without modifying history.

Use :meth:`SessionManager.build_session_context` (or the module-level
:func:`build_session_context`) to resolve the message list for the LLM, which
walks root → leaf handling compaction summaries along the path.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from bampy.ai.types import ImageContent, TextContent

from .messages import (
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)

CURRENT_SESSION_VERSION = 1


# ---------------------------------------------------------------------------
# Entry types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SessionHeader:
    type: str = field(default="session", init=False)
    version: int = CURRENT_SESSION_VERSION
    id: str = ""
    timestamp: str = ""
    cwd: str = ""
    parent_session: str | None = None


@dataclass(slots=True)
class SessionEntryBase:
    type: str = ""
    id: str = ""
    parent_id: str | None = None
    timestamp: str = ""


@dataclass(slots=True)
class SessionMessageEntry(SessionEntryBase):
    type: str = field(default="message", init=False)
    message: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelChangeEntry(SessionEntryBase):
    type: str = field(default="model_change", init=False)
    provider: str = ""
    model_id: str = ""


@dataclass(slots=True)
class ThinkingLevelChangeEntry(SessionEntryBase):
    type: str = field(default="thinking_level_change", init=False)
    thinking_level: str = ""


@dataclass(slots=True)
class CompactionEntry(SessionEntryBase):
    type: str = field(default="compaction", init=False)
    summary: str = ""
    first_kept_entry_id: str = ""
    tokens_before: int = 0
    details: Any = None
    from_hook: bool = False


@dataclass(slots=True)
class BranchSummaryEntry(SessionEntryBase):
    type: str = field(default="branch_summary", init=False)
    from_id: str = ""
    summary: str = ""
    details: Any = None
    from_hook: bool = False


@dataclass(slots=True)
class CustomEntry(SessionEntryBase):
    """Extension-stored data (NOT sent to LLM)."""

    type: str = field(default="custom", init=False)
    custom_type: str = ""
    data: Any = None


@dataclass(slots=True)
class CustomMessageEntry(SessionEntryBase):
    """Extension message that participates in LLM context."""

    type: str = field(default="custom_message", init=False)
    custom_type: str = ""
    content: str | list[dict[str, Any]] = ""
    details: Any = None
    display: bool = True


@dataclass(slots=True)
class LabelEntry(SessionEntryBase):
    type: str = field(default="label", init=False)
    target_id: str = ""
    label: str | None = None


@dataclass(slots=True)
class SessionInfoEntry(SessionEntryBase):
    type: str = field(default="session_info", init=False)
    name: str | None = None


SessionEntry = (
    SessionMessageEntry
    | ModelChangeEntry
    | ThinkingLevelChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
    | CustomMessageEntry
    | LabelEntry
    | SessionInfoEntry
)

FileEntry = SessionHeader | SessionEntry


@dataclass(slots=True)
class SessionContext:
    """Resolved context for the LLM from a session tree traversal."""

    messages: list[Any] = field(default_factory=list)
    thinking_level: str = "off"
    model: dict[str, str] | None = None


@dataclass(slots=True)
class SessionTreeNode:
    entry: SessionEntry
    children: list[SessionTreeNode] = field(default_factory=list)
    label: str | None = None


@dataclass(slots=True)
class SessionInfo:
    path: str = ""
    id: str = ""
    cwd: str = ""
    name: str | None = None
    parent_session_path: str | None = None
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0
    first_message: str = "(no messages)"
    all_messages_text: str = ""


# ---------------------------------------------------------------------------
# Session backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SessionBackend(Protocol):
    """Pluggable persistence backend."""

    def append(self, entry: dict[str, Any]) -> None: ...
    def read_all(self) -> list[dict[str, Any]]: ...
    def rewrite(self, entries: list[dict[str, Any]]) -> None: ...


class NDJSONBackend:
    """Append-only NDJSON file backend."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def rewrite(self, entries: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


class InMemoryBackend:
    """In-memory backend for testing and ephemeral sessions."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def append(self, entry: dict[str, Any]) -> None:
        self._entries.append(entry)

    def read_all(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def rewrite(self, entries: list[dict[str, Any]]) -> None:
        self._entries = list(entries)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _generate_id(existing: set[str]) -> str:
    for _ in range(100):
        candidate = uuid.uuid4().hex[:8]
        if candidate not in existing:
            return candidate
    return uuid.uuid4().hex


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry_to_dict(entry: FileEntry) -> dict[str, Any]:
    """Convert a dataclass entry to a JSON-serialisable dict."""
    from dataclasses import asdict

    d = asdict(entry)  # type: ignore[arg-type]
    return d


def _dict_to_entry(d: dict[str, Any]) -> FileEntry:
    """Reconstruct a typed entry from a raw dict."""
    t = d.get("type")
    if t == "session":
        return SessionHeader(
            version=d.get("version", 1),
            id=d.get("id", ""),
            timestamp=d.get("timestamp", ""),
            cwd=d.get("cwd", ""),
            parent_session=d.get("parent_session"),
        )
    base_kwargs = dict(
        id=d.get("id", ""),
        parent_id=d.get("parent_id"),
        timestamp=d.get("timestamp", ""),
    )
    if t == "message":
        e = SessionMessageEntry(**base_kwargs)
        e.message = d.get("message", {})
        return e
    if t == "model_change":
        e2 = ModelChangeEntry(**base_kwargs)
        e2.provider = d.get("provider", "")
        e2.model_id = d.get("model_id", "")
        return e2
    if t == "thinking_level_change":
        e3 = ThinkingLevelChangeEntry(**base_kwargs)
        e3.thinking_level = d.get("thinking_level", "")
        return e3
    if t == "compaction":
        e4 = CompactionEntry(**base_kwargs)
        e4.summary = d.get("summary", "")
        e4.first_kept_entry_id = d.get("first_kept_entry_id", "")
        e4.tokens_before = d.get("tokens_before", 0)
        e4.details = d.get("details")
        e4.from_hook = d.get("from_hook", False)
        return e4
    if t == "branch_summary":
        e5 = BranchSummaryEntry(**base_kwargs)
        e5.from_id = d.get("from_id", "")
        e5.summary = d.get("summary", "")
        e5.details = d.get("details")
        e5.from_hook = d.get("from_hook", False)
        return e5
    if t == "custom":
        e6 = CustomEntry(**base_kwargs)
        e6.custom_type = d.get("custom_type", "")
        e6.data = d.get("data")
        return e6
    if t == "custom_message":
        e7 = CustomMessageEntry(**base_kwargs)
        e7.custom_type = d.get("custom_type", "")
        e7.content = d.get("content", "")
        e7.details = d.get("details")
        e7.display = d.get("display", True)
        return e7
    if t == "label":
        e8 = LabelEntry(**base_kwargs)
        e8.target_id = d.get("target_id", "")
        e8.label = d.get("label")
        return e8
    if t == "session_info":
        e9 = SessionInfoEntry(**base_kwargs)
        e9.name = d.get("name")
        return e9
    # Fallback: return as message entry with the raw dict
    return SessionMessageEntry(message=d, **base_kwargs)


def _message_to_serializable(message: Any) -> dict[str, Any]:
    """Convert an agent message to a JSON-serialisable dict."""
    from pydantic import BaseModel

    if isinstance(message, BaseModel):
        return message.model_dump(mode="json")
    if isinstance(message, dict):
        return message
    from dataclasses import asdict as dc_asdict, fields as dc_fields

    try:
        dc_fields(message)
        return dc_asdict(message)
    except TypeError:
        return {"role": getattr(message, "role", "unknown")}


_LEAF_UNSET = object()


# ---------------------------------------------------------------------------
# build_session_context (module-level, pure function)
# ---------------------------------------------------------------------------

def build_session_context(
    entries: list[SessionEntry],
    leaf_id: str | None | object = _LEAF_UNSET,
    by_id: dict[str, SessionEntry] | None = None,
) -> SessionContext:
    """Walk root→leaf through entries and build LLM context.

    Handles compaction summaries, branch summaries, and custom messages.
    """
    if by_id is None:
        by_id = {e.id: e for e in entries}

    # Find leaf
    if leaf_id is None:
        return SessionContext()
    if leaf_id is _LEAF_UNSET and entries:
        leaf = entries[-1]
    elif leaf_id:
        leaf = by_id.get(leaf_id)
        if leaf is None and entries:
            leaf = entries[-1]
    else:
        return SessionContext()

    if leaf is None:
        return SessionContext()

    # Walk leaf → root
    path: list[SessionEntry] = []
    current: SessionEntry | None = leaf
    while current is not None:
        path.append(current)
        current = by_id.get(current.parent_id) if current.parent_id else None
    path.reverse()

    # Extract settings and find compaction
    thinking_level = "off"
    model: dict[str, str] | None = None
    compaction: CompactionEntry | None = None

    for entry in path:
        if isinstance(entry, ThinkingLevelChangeEntry):
            thinking_level = entry.thinking_level
        elif isinstance(entry, ModelChangeEntry):
            model = {"provider": entry.provider, "model_id": entry.model_id}
        elif isinstance(entry, SessionMessageEntry):
            msg = entry.message
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                model = {
                    "provider": msg.get("provider", ""),
                    "model_id": msg.get("model", ""),
                }
        elif isinstance(entry, CompactionEntry):
            compaction = entry

    # Build messages
    messages: list[Any] = []

    def _append_message(entry: SessionEntry) -> None:
        if isinstance(entry, SessionMessageEntry):
            messages.append(entry.message)
        elif isinstance(entry, CustomMessageEntry):
            messages.append(
                create_custom_message(
                    entry.custom_type,
                    entry.content if isinstance(entry.content, str) else entry.content,
                    entry.display,
                    entry.details,
                    entry.timestamp,
                )
            )
        elif isinstance(entry, BranchSummaryEntry) and entry.summary:
            messages.append(
                create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)
            )

    if compaction:
        # 1. Emit compaction summary
        messages.append(
            create_compaction_summary_message(compaction.summary, compaction.tokens_before, compaction.timestamp)
        )
        # 2. Find compaction index in path
        compaction_idx = next((i for i, e in enumerate(path) if isinstance(e, CompactionEntry) and e.id == compaction.id), -1)
        # 3. Emit kept messages (before compaction, starting from first_kept_entry_id)
        found_first_kept = False
        for i in range(compaction_idx):
            entry = path[i]
            if entry.id == compaction.first_kept_entry_id:
                found_first_kept = True
            if found_first_kept:
                _append_message(entry)
        # 4. Emit messages after compaction
        for i in range(compaction_idx + 1, len(path)):
            _append_message(path[i])
    else:
        for entry in path:
            _append_message(entry)

    return SessionContext(messages=messages, thinking_level=thinking_level, model=model)


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages conversation sessions as append-only trees.

    Each entry has an ``id`` and ``parent_id`` forming a tree. The *leaf*
    pointer tracks the current position.
    """

    def __init__(
        self,
        cwd: str,
        *,
        backend: SessionBackend | None = None,
        session_dir: str | None = None,
        session_file: str | None = None,
        persist: bool = True,
    ) -> None:
        self._cwd = cwd
        self._persist = persist
        self._session_dir = session_dir or ""
        self._session_id = ""
        self._session_file: str | None = None
        self._file_entries: list[FileEntry] = []
        self._by_id: dict[str, SessionEntry] = {}
        self._labels_by_id: dict[str, str] = {}
        self._leaf_id: str | None = None
        self._flushed = False
        self._backend = backend

        if session_file:
            self._load_session_file(session_file)
        else:
            self.new_session()

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @classmethod
    def create(cls, cwd: str, session_dir: str | None = None) -> SessionManager:
        d = session_dir or cls._default_session_dir(cwd)
        return cls(cwd, session_dir=d, persist=True)

    @classmethod
    def open(cls, path: str, session_dir: str | None = None) -> SessionManager:
        d = session_dir or str(Path(path).parent)
        header = _read_session_header(path)
        cwd = header.cwd if header and header.cwd else os.getcwd()
        return cls(cwd, session_dir=d, session_file=path, persist=True)

    @classmethod
    def in_memory(cls, cwd: str = ".") -> SessionManager:
        return cls(cwd, backend=InMemoryBackend(), persist=False)

    @staticmethod
    def _default_session_dir(cwd: str) -> str:
        safe = cwd.lstrip("/\\").replace("/", "-").replace("\\", "-").replace(":", "-")
        base = Path.home() / ".bampy" / "sessions" / f"--{safe}--"
        base.mkdir(parents=True, exist_ok=True)
        return str(base)

    # -----------------------------------------------------------------------
    # Session lifecycle
    # -----------------------------------------------------------------------

    def new_session(self, *, session_id: str | None = None, parent_session: str | None = None) -> str | None:
        self._session_id = session_id or uuid.uuid4().hex
        ts = _now_iso()
        header = SessionHeader(
            version=CURRENT_SESSION_VERSION,
            id=self._session_id,
            timestamp=ts,
            cwd=self._cwd,
            parent_session=parent_session,
        )
        self._file_entries = [header]
        self._by_id.clear()
        self._labels_by_id.clear()
        self._leaf_id = None
        self._flushed = False

        if self._persist and self._session_dir:
            safe_ts = ts.replace(":", "-").replace(".", "-")
            self._session_file = str(Path(self._session_dir) / f"{safe_ts}_{self._session_id}.jsonl")
            if self._backend is None:
                self._backend = NDJSONBackend(self._session_file)
        return self._session_file

    def _load_session_file(self, path: str) -> None:
        resolved_path = str(Path(path).resolve())
        self._session_file = resolved_path
        if self._backend is None:
            self._backend = NDJSONBackend(self._session_file)
        raw = self._backend.read_all()
        header_is_valid = bool(raw) and raw[0].get("type") == "session"
        if not header_is_valid:
            self.new_session()
            if self._persist:
                self._session_file = resolved_path
                self._backend = NDJSONBackend(self._session_file)
                if Path(resolved_path).exists():
                    self._backend.rewrite([_entry_to_dict(e) for e in self._file_entries])
                self._flushed = False
            return

        self._file_entries = [_dict_to_entry(d) for d in raw]
        header = next((e for e in self._file_entries if isinstance(e, SessionHeader)), None)
        if header and header.cwd:
            self._cwd = header.cwd
        self._session_id = header.id if header else uuid.uuid4().hex
        self._build_index()
        self._flushed = True

    def _build_index(self) -> None:
        self._by_id.clear()
        self._labels_by_id.clear()
        self._leaf_id = None
        for entry in self._file_entries:
            if isinstance(entry, SessionHeader):
                continue
            self._by_id[entry.id] = entry
            self._leaf_id = entry.id
            if isinstance(entry, LabelEntry):
                if entry.label:
                    self._labels_by_id[entry.target_id] = entry.label
                else:
                    self._labels_by_id.pop(entry.target_id, None)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _persist_entry(self, entry: SessionEntry) -> None:
        if not self._persist or self._backend is None:
            return
        # Lazy flush: wait until first assistant message before writing
        has_assistant = any(
            isinstance(e, SessionMessageEntry) and isinstance(e.message, dict) and e.message.get("role") == "assistant"
            for e in self._file_entries
            if isinstance(e, SessionMessageEntry)
        )
        if not has_assistant:
            self._flushed = False
            return

        if not self._flushed:
            self._backend.rewrite([_entry_to_dict(e) for e in self._file_entries])
            self._flushed = True
        else:
            self._backend.append(_entry_to_dict(entry))

    def _append_entry(self, entry: SessionEntry) -> None:
        self._file_entries.append(entry)
        self._by_id[entry.id] = entry
        self._leaf_id = entry.id
        self._persist_entry(entry)

    # -----------------------------------------------------------------------
    # Append methods
    # -----------------------------------------------------------------------

    def append_message(self, message: Any) -> str:
        """Append a message as child of current leaf. Returns entry id."""
        entry = SessionMessageEntry(
            id=_generate_id(set(self._by_id)),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
        )
        entry.message = _message_to_serializable(message)
        self._append_entry(entry)
        return entry.id

    def append_model_change(self, provider: str, model_id: str) -> str:
        entry = ModelChangeEntry(
            id=_generate_id(set(self._by_id)),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
        )
        entry.provider = provider
        entry.model_id = model_id
        self._append_entry(entry)
        return entry.id

    def append_thinking_level_change(self, thinking_level: str) -> str:
        entry = ThinkingLevelChangeEntry(
            id=_generate_id(set(self._by_id)),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
        )
        entry.thinking_level = thinking_level
        self._append_entry(entry)
        return entry.id

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        entry = CompactionEntry(
            id=_generate_id(set(self._by_id)),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
        )
        entry.summary = summary
        entry.first_kept_entry_id = first_kept_entry_id
        entry.tokens_before = tokens_before
        entry.details = details
        entry.from_hook = from_hook
        self._append_entry(entry)
        return entry.id

    def append_branch_summary(
        self,
        from_id: str,
        summary: str,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        entry = BranchSummaryEntry(
            id=_generate_id(set(self._by_id)),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
        )
        entry.from_id = from_id
        entry.summary = summary
        entry.details = details
        entry.from_hook = from_hook
        self._append_entry(entry)
        return entry.id

    def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        entry = CustomEntry(
            id=_generate_id(set(self._by_id)),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
        )
        entry.custom_type = custom_type
        entry.data = data
        self._append_entry(entry)
        return entry.id

    def append_custom_message_entry(
        self,
        custom_type: str,
        content: str | list[TextContent | ImageContent],
        display: bool = True,
        details: Any = None,
    ) -> str:
        entry = CustomMessageEntry(
            id=_generate_id(set(self._by_id)),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
        )
        entry.custom_type = custom_type
        if isinstance(content, str):
            entry.content = content
        else:
            from pydantic import BaseModel

            entry.content = [c.model_dump(mode="json") if isinstance(c, BaseModel) else c for c in content]
        entry.display = display
        entry.details = details
        self._append_entry(entry)
        return entry.id

    def append_session_info(self, name: str) -> str:
        entry = SessionInfoEntry(
            id=_generate_id(set(self._by_id)),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
        )
        entry.name = name.strip()
        self._append_entry(entry)
        return entry.id

    def append_label_change(self, target_id: str, label: str | None) -> str:
        if target_id not in self._by_id:
            raise KeyError(f"Entry {target_id} not found")
        entry = LabelEntry(
            id=_generate_id(set(self._by_id)),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
        )
        entry.target_id = target_id
        entry.label = label
        self._append_entry(entry)
        if label:
            self._labels_by_id[target_id] = label
        else:
            self._labels_by_id.pop(target_id, None)
        return entry.id

    # -----------------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------------

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_file(self) -> str | None:
        return self._session_file

    @property
    def session_dir(self) -> str:
        return self._session_dir

    @property
    def leaf_id(self) -> str | None:
        return self._leaf_id

    def get_leaf_entry(self) -> SessionEntry | None:
        return self._by_id.get(self._leaf_id) if self._leaf_id else None

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        return self._by_id.get(entry_id)

    def get_label(self, entry_id: str) -> str | None:
        return self._labels_by_id.get(entry_id)

    def get_header(self) -> SessionHeader | None:
        return next((e for e in self._file_entries if isinstance(e, SessionHeader)), None)

    def get_entries(self) -> list[SessionEntry]:
        return [e for e in self._file_entries if not isinstance(e, SessionHeader)]

    def get_session_name(self) -> str | None:
        for entry in reversed(self._file_entries):
            if isinstance(entry, SessionInfoEntry):
                return entry.name.strip() if entry.name else None
        return None

    # -----------------------------------------------------------------------
    # Tree navigation
    # -----------------------------------------------------------------------

    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]:
        """Walk from entry to root, returning path in root→leaf order."""
        path: list[SessionEntry] = []
        start_id = from_id or self._leaf_id
        current = self._by_id.get(start_id) if start_id else None
        while current is not None:
            path.append(current)
            current = self._by_id.get(current.parent_id) if current.parent_id else None
        path.reverse()
        return path

    def get_tree(self) -> list[SessionTreeNode]:
        entries = self.get_entries()
        node_map: dict[str, SessionTreeNode] = {}
        roots: list[SessionTreeNode] = []

        for entry in entries:
            label = self._labels_by_id.get(entry.id)
            node_map[entry.id] = SessionTreeNode(entry=entry, label=label)

        for entry in entries:
            node = node_map[entry.id]
            if entry.parent_id is None or entry.parent_id == entry.id:
                roots.append(node)
            else:
                parent = node_map.get(entry.parent_id)
                if parent:
                    parent.children.append(node)
                else:
                    roots.append(node)
        return roots

    def build_session_context(self) -> SessionContext:
        return build_session_context(self.get_entries(), self._leaf_id, self._by_id)

    # -----------------------------------------------------------------------
    # Branching
    # -----------------------------------------------------------------------

    def branch(self, branch_from_id: str) -> None:
        """Move the leaf pointer to an earlier entry (start a new branch)."""
        if branch_from_id not in self._by_id:
            raise KeyError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        """Reset leaf to None (before any entries)."""
        self._leaf_id = None

    def branch_with_summary(
        self,
        branch_from_id: str | None,
        summary: str,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        """Branch and append a branch_summary entry."""
        if branch_from_id is not None and branch_from_id not in self._by_id:
            raise KeyError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id
        return self.append_branch_summary(
            from_id=branch_from_id or "root",
            summary=summary,
            details=details,
            from_hook=from_hook,
        )

    # -----------------------------------------------------------------------
    # Session listing (static / async)
    # -----------------------------------------------------------------------

    @staticmethod
    async def list_sessions(
        cwd: str,
        session_dir: str | None = None,
    ) -> list[SessionInfo]:
        """List all sessions in a directory, sorted by modified time descending."""
        d = session_dir or SessionManager._default_session_dir(cwd)
        p = Path(d)
        if not p.exists():
            return []

        sessions: list[SessionInfo] = []
        for f in sorted(p.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
            info = _build_session_info(str(f))
            if info:
                sessions.append(info)
        return sessions


def _build_session_info(file_path: str) -> SessionInfo | None:
    try:
        entries: list[dict[str, Any]] = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not entries:
            return None
        header = entries[0]
        if header.get("type") != "session":
            return None

        stat = os.stat(file_path)
        message_count = 0
        first_message = ""
        all_messages: list[str] = []
        name: str | None = None

        for entry in entries:
            if entry.get("type") == "session_info":
                n = entry.get("name", "")
                name = n.strip() if n else None

            if entry.get("type") != "message":
                continue
            message_count += 1
            msg = entry.get("message", {})
            if msg.get("role") not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = ""
            if text:
                all_messages.append(text)
                if not first_message and msg.get("role") == "user":
                    first_message = text

        ts_str = header.get("timestamp", "")
        try:
            created = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)

        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        return SessionInfo(
            path=file_path,
            id=header.get("id", ""),
            cwd=header.get("cwd", ""),
            name=name,
            parent_session_path=header.get("parent_session"),
            created=created,
            modified=modified,
            message_count=message_count,
            first_message=first_message or "(no messages)",
            all_messages_text=" ".join(all_messages),
        )
    except OSError:
        return None


def _read_session_header(path: str) -> SessionHeader | None:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    return None
                header = _dict_to_entry(entry)
                return header if isinstance(header, SessionHeader) else None
    except OSError:
        return None
    return None
