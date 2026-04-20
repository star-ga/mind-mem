"""BlockStore abstraction — decouples block access from storage format.

Provides a Protocol-based interface for block CRUD operations, with
MarkdownBlockStore as the default implementation wrapping the existing
file-based Markdown parsing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re as _re
import shutil
import tempfile
import warnings
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from .block_parser import get_active, get_by_id, parse_file
from .corpus_registry import SNAPSHOT_DIRS, SNAPSHOT_EXCLUDE_DIRS
from .mind_filelock import FileLock
from .observability import get_logger

_log = get_logger("block_store")

# ─── Snapshot constants (v3.2.0 §1.4 PR-3) ───────────────────────────────────

# Config / root files always included in every snapshot for rollback integrity.
SNAPSHOT_FILES: list[str] = ["AGENTS.md", "MEMORY.md", "IDENTITY.md", "mind-mem.json"]


# ─── Snapshot helper functions ────────────────────────────────────────────────


def _is_in_excluded_dir(ws: str, path: str) -> bool:
    """True when ``path`` falls under one of SNAPSHOT_EXCLUDE_DIRS.

    Normalises separators so a Windows-native path under
    ``maintenance\\append-only\\...`` still matches the canonical
    forward-slash-declared exclude list.
    """
    rel = os.path.relpath(path, ws).replace(os.sep, "/")
    for excluded in SNAPSHOT_EXCLUDE_DIRS:
        if rel == excluded or rel.startswith(excluded + "/"):
            return True
    return False


def _safe_copy(src: str, dst: str) -> None:
    """Copy a file for snapshot purposes. Always uses copy2 (not hardlinks).

    Hardlinks are unsuitable for mutable-file snapshots because Python's
    open("w") truncates the inode in-place, corrupting both the workspace
    file and its hardlinked snapshot copy.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _build_cleanup_inventory(ws: str, roots: set[str]) -> dict[str, list[str]]:
    """Capture the pre-snapshot file inventory for touched top-level roots."""
    inventory: dict[str, list[str]] = {}
    normalized_roots = {root.replace("\\", "/").strip("/") for root in roots if root}
    for root in sorted(normalized_roots):
        entries: list[str] = []
        root_path = os.path.join(ws, root)
        if not os.path.isdir(root_path):
            inventory[root] = entries
            continue
        for walk_root, dirs, files in os.walk(root_path):
            rel_walk_root = os.path.relpath(walk_root, ws)
            if root == "intelligence" and "applied" in rel_walk_root.split(os.sep):
                dirs.clear()
                continue
            for fname in files:
                rel = os.path.relpath(os.path.join(walk_root, fname), ws)
                entries.append(rel.replace(os.sep, "/"))
        inventory[root] = sorted(entries)
    return inventory


def _build_manifest(snap_dir: str, files: list[str], cleanup_inventory: dict[str, list[str]] | None = None) -> None:
    """Write snapshot manifest for efficient delta-based restore."""
    normalized = [f.replace(os.sep, "/") for f in files]
    manifest_path = os.path.join(snap_dir, "MANIFEST.json")
    payload: dict[str, Any] = {"files": normalized, "version": 2}
    if cleanup_inventory:
        payload["cleanup_inventory"] = cleanup_inventory
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _read_manifest(snap_dir: str) -> dict[str, Any] | None:
    """Read snapshot manifest, or None for legacy snapshots."""
    manifest_path = os.path.join(snap_dir, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return {"files": data, "cleanup_inventory": {}, "version": 1}
    return {
        "files": data.get("files", []),
        "cleanup_inventory": data.get("cleanup_inventory", {}),
        "version": data.get("version", 1),
    }


def _cleanup_orphans_from_manifest(ws: str, manifest: list[str], cleanup_inventory: dict[str, list[str]] | None = None) -> None:
    """Remove files in snapshotted directories that aren't in the manifest.

    This handles the case where ops created new files after the snapshot —
    those files must be removed on rollback for true atomic restore.

    Paths under ``SNAPSHOT_EXCLUDE_DIRS`` are skipped because they were
    deliberately left out of the snapshot capture.
    """
    manifest_set = {m.replace(os.sep, "/") for m in manifest}
    inventory_sets: dict[str, set[str]] = {
        root.replace("\\", "/").strip("/"): {entry.replace(os.sep, "/") for entry in entries}
        for root, entries in (cleanup_inventory or {}).items()
    }

    snapshotted_dirs: set[str] = set()
    for rel in manifest:
        top_dir = rel.split("/")[0]
        snapshotted_dirs.add(top_dir)
    snapshotted_dirs.update(inventory_sets.keys())

    for d in snapshotted_dirs:
        allowed = inventory_sets.get(d, manifest_set)
        if d in ("intelligence",):
            intel_dir = os.path.join(ws, "intelligence")
            if os.path.isdir(intel_dir):
                for root, dirs, files in os.walk(intel_dir):
                    rel_root = os.path.relpath(root, ws)
                    if "applied" in rel_root.split(os.sep):
                        dirs.clear()
                        continue
                    for fname in files:
                        rel = os.path.relpath(os.path.join(root, fname), ws)
                        if rel.replace(os.sep, "/") not in allowed:
                            os.remove(os.path.join(root, fname))
        else:
            dirpath = os.path.join(ws, d)
            if os.path.isdir(dirpath):
                for root, dirs, files in os.walk(dirpath):
                    if _is_in_excluded_dir(ws, root):
                        dirs.clear()
                        continue
                    dirs[:] = [sub for sub in dirs if not _is_in_excluded_dir(ws, os.path.join(root, sub))]
                    for fname in files:
                        rel = os.path.relpath(os.path.join(root, fname), ws)
                        if rel.replace(os.sep, "/") not in allowed:
                            os.remove(os.path.join(root, fname))


# Block-id prefix → (corpus subdir, filename) routing. Shared with
# ``mcp.tools.memory_ops._BLOCK_PREFIX_MAP``; duplicated here so the
# write surface doesn't pull an MCP-layer import. Keep in lockstep.
_BLOCK_PREFIX_MAP: dict[str, tuple[str, str]] = {
    "D": ("decisions", "DECISIONS.md"),
    "T": ("tasks", "TASKS.md"),
    "C": ("intelligence", "CONTRADICTIONS.md"),
    "INC": ("entities", "incidents.md"),
    "PRJ": ("entities", "projects.md"),
    "PER": ("entities", "people.md"),
    "TOOL": ("entities", "tools.md"),
}

_BLOCK_ID_RE = _re.compile(r"^([A-Z]+)-[a-zA-Z0-9_.-]+$")


def _resolve_block_file(workspace: str, block_id: str) -> Optional[str]:
    """Return the absolute path of the canonical file for ``block_id``.

    Returns ``None`` for unrecognised prefixes. Callers must fall back
    to full-corpus scan when the mapping is absent (e.g., signals,
    one-off entity types not in the prefix map).
    """
    m = _BLOCK_ID_RE.match(block_id)
    if not m:
        return None
    prefix = m.group(1)
    mapped = _BLOCK_PREFIX_MAP.get(prefix)
    if mapped is None:
        return None
    subdir, filename = mapped
    return os.path.join(workspace, subdir, filename)


# Fields emitted in a fixed order so block round-trips are
# deterministic. Unknown fields are appended alphabetically after
# the canonical head. ``_id`` is the synthetic parse-time field
# (emitted as the ``[id]`` header) and ``_source_file`` is a
# tool-side hint that must never be written back.
_CANONICAL_FIELD_ORDER: tuple[str, ...] = (
    "Statement",
    "Date",
    "Status",
    "Priority",
    "Risk",
    "Type",
    "Subject",
    "Object",
    "Tags",
    "Rationale",
    "Evidence",
    "Source",
    "Confidence",
    "ContentHash",
    "Excerpt",
    "Action",
)
_FORBIDDEN_WRITE_FIELDS: frozenset[str] = frozenset({"_id", "_source_file", "_line_number", "_raw"})


def _render_block(block: dict[str, Any]) -> str:
    """Serialize a parsed block dict back to its Markdown form.

    Output layout::

        [ID]
        Field1: value
        Field2: value
        ...

        ---

    Lists are rendered as ``"- item"`` bullets on lines following the
    field. Multi-line field values are emitted verbatim except that
    newline-plus-left-bracket ("\\n[") is neutralised to avoid
    accidentally starting a new block header mid-value.
    """
    block_id = block.get("_id")
    if not block_id:
        raise ValueError("block is missing '_id'; cannot render without an ID")
    if not _BLOCK_ID_RE.match(str(block_id)):
        raise ValueError(f"invalid block id: {block_id!r}")

    out: list[str] = [f"[{block_id}]"]

    seen: set[str] = set()

    def _emit(key: str, value: Any) -> None:
        if key in seen or key.startswith("_") or key in _FORBIDDEN_WRITE_FIELDS:
            return
        seen.add(key)
        if isinstance(value, list):
            out.append(f"{key}:")
            for item in value:
                safe = str(item).replace("\n[", "\n ")
                out.append(f"- {safe}")
        else:
            safe = str(value).replace("\n[", "\n ")
            out.append(f"{key}: {safe}")

    for key in _CANONICAL_FIELD_ORDER:
        if key in block:
            _emit(key, block[key])

    for key in sorted(block.keys()):
        _emit(key, block[key])

    out.append("")  # trailing blank line before the separator
    out.append("---")
    return "\n".join(out) + "\n"


def _atomic_write(path: str, text: str) -> None:
    """Write ``text`` to ``path`` via a temp-file + ``os.replace`` swap.

    Produces no partial-write window even when another process is
    tailing the destination. The temp file is created in the same
    directory so ``os.replace`` is guaranteed to be atomic on POSIX
    and NTFS.
    """
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _record_deletion(workspace: str, block_id: str, content: str) -> None:
    """Append a deletion receipt to ``memory/deleted_blocks.jsonl``.

    Matches the receipt format already written by
    ``mcp.tools.memory_ops.delete_memory_item`` so both write paths
    converge on the same recovery journal.
    """
    log_path = os.path.join(workspace, "memory", "deleted_blocks.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "block_id": block_id,
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "content": content,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _locate_block_in_text(text: str, block_id: str) -> Optional[tuple[int, int, str]]:
    """Return ``(start_line, end_line_exclusive, deleted_content)`` or ``None``.

    Mirrors the scan logic in ``mcp.tools.memory_ops.delete_memory_item``
    so both implementations see the same boundary rules:
    * A block starts at a line exactly equal to ``[<id>]``.
    * A block ends at the next ``[<ID>]`` header line, an isolated
      ``---`` separator (preceded by a blank line), or EOF.
    """
    lines = text.split("\n")
    header = f"[{block_id}]"
    block_start: Optional[int] = None
    block_end: Optional[int] = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            block_start = i
        elif block_start is not None and block_end is None:
            stripped = line.strip()
            if line.startswith("[") and stripped.endswith("]") and _re.match(r"^\[[A-Z]+-", stripped):
                block_end = i
            elif stripped == "---":
                preceding_blank = (i == 0) or (lines[i - 1].strip() == "")
                if preceding_blank:
                    block_end = i + 1
    if block_start is None:
        return None
    if block_end is None:
        block_end = len(lines)
    content = "\n".join(lines[block_start:block_end])
    return block_start, block_end, content


class BlockStoreError(Exception):
    """Raised when a storage operation fails in a BlockStore implementation.

    Re-exported here so callers can catch ``mind_mem.block_store.BlockStoreError``
    regardless of which backend is active.  ``PostgresBlockStore`` imports and
    raises this same class.
    """


@runtime_checkable
class BlockStore(Protocol):
    """Protocol for block storage backends.

    v3.2.0 §1.4 renames ``list_files`` → ``list_blocks`` as part of
    the apply-engine routing work. ``list_files`` remains as a thin
    alias with a ``DeprecationWarning``; it will be removed in v4.0.
    """

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        """Return all blocks, optionally filtered to active only."""
        ...

    def get_by_id(self, block_id: str) -> Optional[dict[str, Any]]:
        """Return a single block by ID, or None if not found."""
        ...

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Search blocks by text query."""
        ...

    def list_blocks(self) -> list[str]:
        """Return the list of block-containing artifacts managed by this store.

        For filesystem-backed stores this is the set of .md files in
        the corpus. For database-backed stores this is the logical
        equivalent (e.g., table partitions or rowids). Callers should
        treat it as an opaque identifier list.
        """
        ...

    # ─── write surface (v3.2.0 §1.4 PR-2) ────────────────────────────
    #
    # These methods are declared on the Protocol but implementations
    # may choose to raise ``NotImplementedError`` for read-only stores.
    # ``MarkdownBlockStore`` implements them as atomic file ops.
    def write_block(self, block: dict[str, Any]) -> str:
        """Persist or replace a block. Returns the block's ``_id``.

        If a block with the same ``_id`` already exists in the
        store's canonical file, it is replaced in place; otherwise
        the block is appended. All writes are atomic (temp-file +
        rename) and hold an exclusive file lock for the duration of
        the read-modify-write.
        """
        ...

    def delete_block(self, block_id: str) -> bool:
        """Remove a block. Returns True if a block was removed.

        Implementations should log the deletion so operators can
        recover removed content if needed.
        """
        ...

    # ─── snapshot surface (v3.2.0 §1.4 PR-3) ─────────────────────────
    def snapshot(
        self,
        snap_dir: str,
        *,
        files_touched: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a point-in-time snapshot. Returns the manifest dict."""
        ...

    def restore(self, snap_dir: str) -> None:
        """Restore the workspace from a snapshot directory."""
        ...

    def diff(self, snap_dir: str) -> list[str]:
        """Per-file diff (added / modified / deleted) vs. a snapshot.

        Returns relative POSIX paths of files that differ.
        """
        ...

    # ─── lock surface (v3.2.0 §1.4 PR-4) ─────────────────────────────

    def lock(
        self,
        *,
        blocking: bool = True,
        timeout: float = 30.0,
    ) -> "Any":
        """Acquire an exclusive workspace-wide lock.

        Returns a context manager; the lock is released on ``__exit__``.
        ``blocking=False`` raises :class:`~mind_mem.mind_filelock.LockTimeout`
        immediately if the lock is held elsewhere. ``timeout`` is the
        max wait when ``blocking=True``.

        The Markdown backend maps this to a single workspace-level
        ``.workspace.lock`` file; the Postgres backend uses a row in
        the ``workspace_lock`` table. Semantics are identical from
        the caller's perspective — exclusive while held, released on
        context exit.
        """
        ...


class MarkdownBlockStore:
    """BlockStore backed by Markdown files on disk.

    Wraps the existing block_parser functions to provide a uniform interface.
    """

    def __init__(self, workspace: str, corpus_dirs: tuple[str, ...] | None = None):
        self._workspace = workspace
        if corpus_dirs is None:
            from .corpus_registry import CORPUS_DIRS

            corpus_dirs = CORPUS_DIRS
        self._corpus_dirs = corpus_dirs
        self._files: list[str] | None = None
        # v3.2.0 §1.4 PR-4: workspace-wide lock target. ``FileLock``
        # appends ``.lock`` to form the sidecar, so the resulting lock
        # file lives at ``<workspace>/.workspace.lock``. Co-located
        # with the workspace so it follows a rename / mount-point
        # change; per-file FileLocks on individual block files remain
        # in use for fine-grained write coordination inside
        # ``write_block`` / ``delete_block``.
        self._lock_target = os.path.join(workspace, ".workspace")

    def _discover_files(self) -> list[str]:
        """Discover all .md files in corpus directories."""
        if self._files is not None:
            return self._files
        files: list[str] = []
        for d in self._corpus_dirs:
            dir_path = os.path.join(self._workspace, d)
            if os.path.isdir(dir_path):
                for fname in sorted(os.listdir(dir_path)):
                    if fname.endswith(".md"):
                        files.append(os.path.join(dir_path, fname))
        self._files = files
        return files

    def get_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        """Return all blocks from all corpus files.

        Args:
            active_only: If True, only return blocks with Status=active.
        """
        blocks: list[dict[str, Any]] = []
        for fpath in self._discover_files():
            parsed = parse_file(fpath)
            if active_only:
                parsed = get_active(parsed)
            blocks.extend(parsed)
        return blocks

    def get_by_id(self, block_id: str) -> Optional[dict[str, Any]]:
        """Return a single block by ID, or None if not found."""
        for fpath in self._discover_files():
            parsed = parse_file(fpath)
            result = get_by_id(parsed, block_id)
            if result:
                return result
        return None

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Simple substring search across all blocks.

        Args:
            query: Case-insensitive substring to match against block values.
            limit: Maximum number of results to return.
        """
        query_lower = query.lower()
        matches: list[dict[str, Any]] = []
        for block in self.get_all():
            text = " ".join(str(v) for v in block.values()).lower()
            if query_lower in text:
                matches.append(block)
                if len(matches) >= limit:
                    break
        return matches

    def list_blocks(self) -> list[str]:
        """Return list of corpus .md file paths managed by this store.

        v3.2.0 §1.4: renamed from ``list_files``. The old name is
        preserved as a deprecation shim on both this class and every
        wrapping store (``EncryptedBlockStore``) — callers migrating
        from v3.1.x should switch to ``list_blocks`` at their
        convenience; the shim stays through v3.2.x.
        """
        return list(self._discover_files())

    def list_files(self) -> list[str]:
        """Deprecated alias for :meth:`list_blocks` — removed in v4.0."""
        warnings.warn(
            "BlockStore.list_files() is deprecated; use list_blocks() instead. The alias will be removed in v4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_blocks()

    def invalidate_cache(self) -> None:
        """Clear file discovery cache (call after corpus changes)."""
        self._files = None

    # ─── lock surface (v3.2.0 §1.4 PR-4) ─────────────────────────────

    def lock(self, *, blocking: bool = True, timeout: float = 30.0) -> FileLock:
        """Return an exclusive workspace-wide lock as a context manager.

        Implementation maps to a single ``.workspace.lock`` file at the
        workspace root. The underlying :class:`FileLock` honors:

        * ``blocking=False`` → ``timeout=0`` so the first failed
          acquire raises :class:`LockTimeout` immediately.
        * ``blocking=True`` → ``timeout`` seconds (default 30) before
          the caller gets ``LockTimeout``.

        Cross-process semantics: two mind-mem processes targeting the
        same workspace serialize on this lock. Within a process the
        :class:`FileLock` also holds a :class:`threading.Lock` keyed
        on the lock path so multithreaded callers don't trample one
        another.

        Note: the per-file locks used inside :meth:`write_block` /
        :meth:`delete_block` are orthogonal — they protect individual
        block-file read-modify-writes even when the workspace-wide
        lock is not held. Callers that want strict end-to-end
        serialization should acquire the workspace lock first, then
        the per-file lock inside.

        Example::

            with store.lock(timeout=10):
                block = store.get_by_id("D-20260420-001")
                block["Status"] = "superseded"
                store.write_block(block)
        """
        # Ensure the workspace directory exists so FileLock can create
        # its .lock sidecar — first-run on a fresh workspace would
        # otherwise fail when the parent directory is absent.
        os.makedirs(self._workspace, exist_ok=True)
        effective_timeout = 0.0 if not blocking else timeout
        return FileLock(self._lock_target, timeout=effective_timeout)

    # ─── write surface (v3.2.0 §1.4 PR-2) ────────────────────────────

    def write_block(self, block: dict[str, Any]) -> str:
        """Persist or replace a block.

        The target file is resolved from the block's ``_id`` prefix
        via :data:`_BLOCK_PREFIX_MAP`. If a block with the same ID
        already exists in that file, it's replaced in place; if not,
        the rendered block is appended.

        Returns the block's ``_id`` so callers can chain
        ``store.write_block(b).get_by_id(...)``-style flows.

        Raises:
            ValueError: block is missing ``_id`` or has an
                unrecognised prefix (no canonical file mapping).
        """
        block_id = block.get("_id")
        if not block_id:
            raise ValueError("block is missing '_id'; cannot write")
        if not _BLOCK_ID_RE.match(str(block_id)):
            raise ValueError(f"invalid block id: {block_id!r}")
        target = _resolve_block_file(self._workspace, block_id)
        if target is None:
            raise ValueError(f"no canonical file mapping for block id {block_id!r}; add an entry to _BLOCK_PREFIX_MAP to enable writes")

        rendered = _render_block(block)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        with FileLock(target):
            if os.path.isfile(target):
                with open(target, "r", encoding="utf-8") as fh:
                    existing_text = fh.read()
            else:
                existing_text = ""

            loc = _locate_block_in_text(existing_text, block_id)
            if loc is not None:
                start, end, _prior = loc
                lines = existing_text.split("\n")
                # Remove prior block (lines[start:end]) and splice in
                # the new rendered form. ``rendered`` already ends
                # with a trailing newline; splitlines strips it, so
                # reassemble with a single joiner.
                new_lines = lines[:start] + rendered.rstrip("\n").split("\n") + lines[end:]
                new_text = "\n".join(new_lines)
                if not new_text.endswith("\n"):
                    new_text += "\n"
            else:
                # Append — ensure a separator between existing content
                # and the new block when the file is non-empty and
                # doesn't already end with ``---``.
                if existing_text and not existing_text.endswith("\n"):
                    existing_text += "\n"
                new_text = existing_text + "\n" + rendered

            _atomic_write(target, new_text)

        # Any write may have added a previously-missing .md file
        # (first block into a new-entity-type file), so invalidate
        # the discovery cache.
        self.invalidate_cache()
        _log.info("block_store_write", block_id=block_id, file=os.path.relpath(target, self._workspace))
        return str(block_id)

    def delete_block(self, block_id: str) -> bool:
        """Remove a block by ID. Returns True if a block was removed.

        Logs the removed content to ``memory/deleted_blocks.jsonl``
        so the deletion is recoverable. The journal format matches
        what :func:`mcp.tools.memory_ops.delete_memory_item` writes —
        both write paths converge on the same recovery record.
        """
        target = _resolve_block_file(self._workspace, block_id)
        if target is None or not os.path.isfile(target):
            return False

        with FileLock(target):
            with open(target, "r", encoding="utf-8") as fh:
                text = fh.read()
            loc = _locate_block_in_text(text, block_id)
            if loc is None:
                return False
            start, end, removed = loc
            lines = text.split("\n")
            new_lines = lines[:start] + lines[end:]
            new_text = "\n".join(new_lines)
            if new_text and not new_text.endswith("\n"):
                new_text += "\n"
            _record_deletion(self._workspace, block_id, removed)
            _atomic_write(target, new_text)

        self.invalidate_cache()
        _log.info("block_store_delete", block_id=block_id, file=os.path.relpath(target, self._workspace))
        return True

    # ─── snapshot surface (v3.2.0 §1.4 PR-3) ────────────────────────────

    def snapshot(
        self,
        snap_dir: str,
        *,
        files_touched: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a point-in-time snapshot in ``snap_dir``. Returns manifest dict.

        When ``files_touched`` is provided only those files are captured —
        O(touched) instead of O(workspace). Falls back to a full snapshot when
        ``files_touched`` is empty or None.

        ``intelligence/applied/`` is always excluded to prevent recursive
        nesting (snapshots containing snapshots).
        """
        ws = self._workspace
        os.makedirs(snap_dir, exist_ok=True)

        manifest_files: list[str] = []
        cleanup_inventory: dict[str, list[str]] | None = None

        if files_touched:
            ws_real = os.path.realpath(ws)
            cleanup_inventory = _build_cleanup_inventory(
                ws,
                {p.replace("\\", "/").split("/", 1)[0] for p in files_touched if p},
            )
            for rel_path in files_touched:
                resolved = os.path.realpath(os.path.join(ws_real, rel_path))
                if not resolved.startswith(ws_real + os.sep) and resolved != ws_real:
                    continue
                if os.path.isfile(resolved):
                    _safe_copy(resolved, os.path.join(snap_dir, rel_path))
                    manifest_files.append(rel_path)
            for f in SNAPSHOT_FILES:
                src = os.path.join(ws, f)
                if os.path.isfile(src):
                    _safe_copy(src, os.path.join(snap_dir, f))
                    manifest_files.append(f)
        else:
            for d in SNAPSHOT_DIRS:
                src_dir = os.path.join(ws, d)
                if os.path.isdir(src_dir):
                    for root, _dirs, files in os.walk(src_dir):
                        for fname in files:
                            src_file = os.path.join(root, fname)
                            rel = os.path.relpath(src_file, ws)
                            dst_file = os.path.join(snap_dir, rel)
                            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                            _safe_copy(src_file, dst_file)
                            manifest_files.append(rel)

            intel_src = os.path.join(ws, "intelligence")
            if os.path.isdir(intel_src):
                for root, dirs, files in os.walk(intel_src):
                    rel_root = os.path.relpath(root, ws)
                    if "applied" in rel_root.split(os.sep):
                        dirs.clear()
                        continue
                    for fname in files:
                        src_file = os.path.join(root, fname)
                        rel = os.path.relpath(src_file, ws)
                        dst_file = os.path.join(snap_dir, rel)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        _safe_copy(src_file, dst_file)
                        manifest_files.append(rel)

            for f in SNAPSHOT_FILES:
                src = os.path.join(ws, f)
                if os.path.isfile(src):
                    _safe_copy(src, os.path.join(snap_dir, f))
                    manifest_files.append(f)

        _build_manifest(snap_dir, manifest_files, cleanup_inventory=cleanup_inventory)
        manifest_data = _read_manifest(snap_dir)
        assert manifest_data is not None
        _log.info("block_store_snapshot", snap_dir=snap_dir, file_count=len(manifest_files))
        return manifest_data

    def restore(self, snap_dir: str) -> None:
        """Restore workspace from snapshot.

        Uses the MANIFEST.json fast-path (O(manifest)) when available;
        falls back to the legacy copytree approach for pre-manifest snapshots.

        ``intelligence/applied/`` is always skipped to prevent deleting
        active snapshots during a restore.
        """
        ws = self._workspace
        manifest_data = _read_manifest(snap_dir)
        if manifest_data is not None:
            manifest = manifest_data.get("files", [])
            cleanup_inventory = manifest_data.get("cleanup_inventory", {})
            for rel_posix in manifest:
                rel_path = rel_posix.replace("/", os.sep)
                src = os.path.join(snap_dir, rel_path)
                dst = os.path.join(ws, rel_path)
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
            _cleanup_orphans_from_manifest(ws, manifest, cleanup_inventory)
            _log.info("block_store_restore", snap_dir=snap_dir, file_count=len(manifest))
            return

        # Legacy fallback: copytree-based restore for pre-manifest snapshots.
        for d in SNAPSHOT_DIRS:
            src = os.path.join(snap_dir, d)
            dst = os.path.join(ws, d)
            if os.path.isdir(src):
                tmp_dst = dst + ".rollback_tmp"
                if os.path.islink(tmp_dst):
                    os.unlink(tmp_dst)
                elif os.path.isdir(tmp_dst):
                    shutil.rmtree(tmp_dst)
                shutil.copytree(src, tmp_dst)
                if os.path.islink(dst):
                    os.unlink(dst)
                elif os.path.isdir(dst):
                    shutil.rmtree(dst)
                os.rename(tmp_dst, dst)

        intel_snap = os.path.join(snap_dir, "intelligence")
        intel_ws = os.path.join(ws, "intelligence")
        if os.path.isdir(intel_snap):
            for item in os.listdir(intel_snap):
                if item == "applied":
                    continue
                src = os.path.join(intel_snap, item)
                dst = os.path.join(intel_ws, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    tmp_dst = dst + ".rollback_tmp"
                    if os.path.islink(tmp_dst):
                        os.unlink(tmp_dst)
                    elif os.path.isdir(tmp_dst):
                        shutil.rmtree(tmp_dst)
                    shutil.copytree(src, tmp_dst)
                    if os.path.islink(dst):
                        os.unlink(dst)
                    elif os.path.isdir(dst):
                        shutil.rmtree(dst)
                    os.rename(tmp_dst, dst)

        for f in SNAPSHOT_FILES:
            src = os.path.join(snap_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(ws, f))

        _log.info("block_store_restore_legacy", snap_dir=snap_dir)

    def diff(self, snap_dir: str) -> list[str]:
        """Return sorted list of relative POSIX paths that differ vs. snapshot.

        Uses the manifest fast-path when available; walks the snapshot tree
        for legacy (pre-manifest) snapshots. Compares files by SHA-256 hash.
        """
        ws = self._workspace
        diffs: list[str] = []
        manifest_data = _read_manifest(snap_dir)

        if manifest_data is not None:
            files_to_check = manifest_data.get("files", [])
        else:
            files_to_check = []
            for root, _dirs, files in os.walk(snap_dir):
                for fname in files:
                    snap_file = os.path.join(root, fname)
                    rel = os.path.relpath(snap_file, snap_dir).replace(os.sep, "/")
                    if rel in ("MANIFEST.json", "APPLY_RECEIPT.md"):
                        continue
                    files_to_check.append(rel)

        for rel_posix in files_to_check:
            rel_native = rel_posix.replace("/", os.sep)
            ws_file = os.path.join(ws, rel_native)
            snap_file = os.path.join(snap_dir, rel_native)

            ws_exists = os.path.isfile(ws_file)
            snap_exists = os.path.isfile(snap_file)

            if snap_exists and not ws_exists:
                diffs.append(rel_posix)
            elif not snap_exists and ws_exists:
                diffs.append(rel_posix)
            elif snap_exists and ws_exists:
                with open(snap_file, "rb") as fh:
                    snap_hash = hashlib.sha256(fh.read()).hexdigest()
                with open(ws_file, "rb") as fh:
                    ws_hash = hashlib.sha256(fh.read()).hexdigest()
                if snap_hash != ws_hash:
                    diffs.append(rel_posix)

        return sorted(diffs)
