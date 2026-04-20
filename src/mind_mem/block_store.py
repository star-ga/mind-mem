"""BlockStore abstraction — decouples block access from storage format.

Provides a Protocol-based interface for block CRUD operations, with
MarkdownBlockStore as the default implementation wrapping the existing
file-based Markdown parsing.
"""

from __future__ import annotations

import json
import os
import re as _re
import tempfile
import warnings
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from .block_parser import get_active, get_by_id, parse_file
from .mind_filelock import FileLock
from .observability import get_logger

_log = get_logger("block_store")

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
_FORBIDDEN_WRITE_FIELDS: frozenset[str] = frozenset(
    {"_id", "_source_file", "_line_number", "_raw"}
)


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
            if (
                line.startswith("[")
                and stripped.endswith("]")
                and _re.match(r"^\[[A-Z]+-", stripped)
            ):
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
            "BlockStore.list_files() is deprecated; use list_blocks() instead. "
            "The alias will be removed in v4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_blocks()

    def invalidate_cache(self) -> None:
        """Clear file discovery cache (call after corpus changes)."""
        self._files = None

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
            raise ValueError(
                f"no canonical file mapping for block id {block_id!r}; "
                f"add an entry to _BLOCK_PREFIX_MAP to enable writes"
            )

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
