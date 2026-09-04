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

from .admission import require_admission, require_delete_admission, require_restore_admission
from .block_parser import get_active, get_by_id, parse_file
from .corpus_registry import (
    BLOCK_PREFIX_MAP,
    CORPUS_RELPATHS,
    SNAPSHOT_DIRS,
    SNAPSHOT_EXCLUDE_DIRS,
    assert_ledger_free,
    is_ledger_path,
)
from .mind_filelock import FileLock
from .observability import get_logger


def _safe_child_path(root: str, relative: str) -> str:
    """Resolve ``relative`` inside ``root`` and reject any traversal escape.

    Hardens the snapshot restore / diff paths against a crafted
    MANIFEST.json that tries to write outside the workspace via
    ``../../etc/passwd`` or via a symlink pointing outside. Every
    file copy in :meth:`BlockStore.restore` and the manifest-diff
    walkers passes each entry through this helper before touching
    the filesystem.

    Returns the resolved absolute path.
    Raises ``ValueError`` when the resolved path escapes ``root``.
    """
    root_real = os.path.realpath(root)
    joined = os.path.join(root, relative)
    resolved = os.path.realpath(joined)
    if resolved != root_real and not resolved.startswith(root_real + os.sep):
        raise ValueError(f"path escapes root {root!r}: {relative!r}")
    return resolved


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
    """Capture the pre-snapshot file inventory for touched top-level roots.

    Stores POSIX-form paths relative to ``ws``. The inventory is the
    authoritative ``allowed`` set for orphan cleanup on rollback —
    every entry must compare byte-for-byte against
    ``relpath(walked_file, ws)`` on restore.

    Cross-platform invariant (v3.7.0 H3 fix): the walk root MUST be the
    un-resolved ``os.path.join(ws, root)``, not its realpath. On macOS
    (``/var`` → ``/private/var``) and Windows short-name runners
    (``RUNNER~1`` → ``runneradmin``), realpath flips the prefix; a
    later ``relpath(child, ws)`` against the un-resolved ``ws`` then
    produces an upward-traversing key (``../../private/...``) that
    never matches ``relpath`` on restore — so every file in the touched
    root gets deleted as an "orphan" even when it IS in the inventory.
    Validate via ``_safe_child_path`` for traversal protection, then
    discard the resolved value and walk the un-resolved join.
    """
    inventory: dict[str, list[str]] = {}
    normalized_roots = {root.replace("\\", "/").strip("/") for root in roots if root}
    for root in sorted(normalized_roots):
        entries: list[str] = []
        try:
            _safe_child_path(ws, root)
        except ValueError:
            _log.warning("cleanup_inventory_root_escape", root=root)
            inventory[root] = entries
            continue
        walk_dir = os.path.join(ws, root)
        if not os.path.isdir(walk_dir):
            inventory[root] = entries
            continue
        for walk_root, dirs, files in os.walk(walk_dir):
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
    """Write snapshot manifest for efficient delta-based restore.

    Refuses a file list naming a ledger of record. Every snapshot walk
    filters for itself, but this is the one call all of them have to make,
    so a walk that forgets cannot produce an artifact a later restore
    would rewind the audit chain with. It raises at capture time, when
    nothing has been lost yet, rather than at the rollback.
    """
    assert_ledger_free(files, what=f"snapshot manifest for {snap_dir}")
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


def _is_removable_orphan(rel_posix: str, allowed: set[str]) -> bool:
    """True when a file under a snapshotted directory must go on restore.

    A ledger of record never qualifies, whatever the manifest says. The
    snapshot walk refuses to capture the ledgers, so they are absent from
    every manifest built after 5.0.2 — and *that absence is what would make
    this sweep delete them*, replacing a rewound chain with no chain at
    all. The exclusion has to be stated on both sides or the capture-side
    half is worse than the defect.
    """
    return rel_posix not in allowed and not is_ledger_path(rel_posix)


def _carry_ledgers_into(ws: str, live_dir: str, staged_dir: str) -> None:
    """Copy the ledgers under *live_dir* into *staged_dir* before a swap.

    The legacy (pre-manifest) restore replaces a whole directory: copytree
    into a temp, rmtree the live one, rename the temp over it. ``memory/``
    holds corpus files *and* the append-only ledgers, so that swap destroys
    the live chain outright — a harder failure than the rewind this change
    exists to remove. The snapshot holds no ledger to put back, so the live
    ones are carried across the swap instead of being restored from it.
    """
    if not os.path.isdir(live_dir):
        return
    for root, _dirs, files in os.walk(live_dir):
        for fname in files:
            live_path = os.path.join(root, fname)
            if not is_ledger_path(os.path.relpath(live_path, ws)):
                continue
            staged_path = os.path.join(staged_dir, os.path.relpath(live_path, live_dir))
            os.makedirs(os.path.dirname(staged_path), exist_ok=True)
            shutil.copy2(live_path, staged_path)


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

    # v3.6.9 path-injection hardening (Gemini CLI finding 1):
    # Every entry in ``snapshotted_dirs`` is an attacker-controlled
    # manifest key — a crafted MANIFEST.json with a root key of ``..``
    # would make ``os.path.join(ws, "..")`` walk the parent of the
    # workspace and ``os.remove`` any file under it not listed in the
    # inventory. Validate each ``d`` through ``_safe_child_path`` first
    # and skip any that escape. Also reject empty / dotted names
    # defensively before they ever reach the filesystem.
    for d in list(snapshotted_dirs):
        if not d or d in (".", "..") or "/" in d or "\\" in d:
            snapshotted_dirs.discard(d)

    ws_real = os.path.realpath(ws)
    for d in snapshotted_dirs:
        try:
            safe_d = _safe_child_path(ws, d)
        except ValueError:
            # Manifest entry attempted to escape the workspace. Skip
            # silently — this is the malicious-manifest path.
            continue
        # Refuse to walk the workspace root itself even if ``safe_d``
        # resolves there (a zero-length ``d`` would).
        if safe_d == ws_real:
            continue
        allowed = inventory_sets.get(d, manifest_set)
        # v3.7.0 H3 fix: walk via the un-resolved ``os.path.join(ws, d)``,
        # NOT ``safe_d`` (which is realpath-resolved). On macOS
        # (``/var`` → ``/private/var``) and Windows short-name runners
        # (``RUNNER~1`` → ``runneradmin``), realpath flips the prefix;
        # walking the resolved path makes ``os.walk`` yield child paths
        # under the resolved prefix, so a later
        # ``relpath(child, ws)`` against the un-resolved ``ws`` produces
        # an upward-traversing key (``../../private/...``) that never
        # matches the manifest — and every legitimate file gets deleted
        # as an orphan. ``safe_d`` is still computed for path-traversal
        # validation; it is just not used as the walk root.
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
                        if _is_removable_orphan(rel.replace(os.sep, "/"), allowed):
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
                        if _is_removable_orphan(rel.replace(os.sep, "/"), allowed):
                            os.remove(os.path.join(root, fname))


#: Block-id prefix → (corpus subdir, filename) routing.
#:
#: Derived, not declared: :data:`corpus_registry.CORPUS_TABLE` is the one
#: corpus definition, and this is its write-routing projection. It used to be
#: a literal here, a second literal in ``mcp.tools.memory_ops`` and a third
#: view of the same corpus in ``_recall_constants.CORPUS_FILES`` — three
#: tables that disagreed about which files hold blocks, which is how a
#: released ``INBOX-`` block became readable by recall and unreachable by the
#: store (see :meth:`MarkdownBlockStore._discover_files`). Adding a corpus is
#: now one row in the table; there is no second place to forget it.
#:
#: Kept under this name because ``admissibility._releasable_id_pattern``,
#: ``mcp_server`` and several tests import it from here.
_BLOCK_PREFIX_MAP: dict[str, tuple[str, str]] = BLOCK_PREFIX_MAP


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
    # Provenance fields (roadmap Group E + T-001) — optional,
    # schema-additive; see block_provenance.PROVENANCE_FIELDS. Existing
    # blocks without them render byte-identically.
    "ActorId",
    "ActorRole",
    "SessionId",
    "ToolId",
    "Purpose",
    "ContentSource",
    "ContentHash",
    "Excerpt",
    "Action",
)
_FORBIDDEN_WRITE_FIELDS: frozenset[str] = frozenset({"_id", "_source_file", "_line_number", "_raw"})


#: A line INSIDE a field value that would be read as a block boundary.
#: ``block_parser`` ends a block at any line starting with ``---``
#: (:mod:`mind_mem.block_parser`, "Section separator").
_VALUE_SEPARATOR_RE = _re.compile(r"\n(-{3,})")


def _neutralise_value(value: str) -> str:
    """Stop a field value from breaking out of the block that contains it.

    Two escapes, and the second one is a SECURITY fix, not cosmetics.

    ``\n[`` would start a new block header mid-value — forging a block.
    That escape has always been here.

    ``\n---`` would *terminate* the block mid-value, and everything the
    renderer emits after that point is then read as loose text outside any
    block. ``Status`` is emitted after ``Statement`` in
    :data:`_CANONICAL_FIELD_ORDER`, so a payload containing a ``---`` line
    dropped its own ``Status: quarantined`` — and an **unstated status is
    servable** (``admissibility.is_admissible_status``). Measured on the
    live inbox door before this fix: a dropped file whose text contained a
    ``---`` line was returned by ``recall``, with no proposal and no
    release. Every door that writes attacker-supplied text through this
    renderer had the same hole (inbox, agent messages, importers, the
    5.0.1 webhook), so the fix belongs here rather than in any one door.

    The escape is a single leading space, matching the ``\n[`` precedent:
    the parser's separator rule is ``line.startswith("---")``, so an
    indented one is not a separator. :func:`_locate_block_in_text` was
    using a *different* rule (``line.strip() == "---"``) and is realigned
    with the parser, so the write/delete boundary and the read boundary
    agree — otherwise a rewrite would splice at the escaped line and leave
    the block's real tail orphaned in the file.
    """
    return _VALUE_SEPARATOR_RE.sub(r"\n \1", value.replace("\n[", "\n "))


def _render_block(block: dict[str, Any]) -> str:
    """Serialize a parsed block dict back to its Markdown form.

    Output layout::

        [ID]
        Field1: value
        Field2: value
        ...

        ---

    Lists are rendered as ``"- item"`` bullets on lines following the
    field. Multi-line field values are emitted verbatim except that a
    value can never break out of its own block: see
    :func:`_neutralise_value`.
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
                out.append(f"- {_neutralise_value(str(item))}")
        else:
            out.append(f"{key}: {_neutralise_value(str(value))}")

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
    * A block ends at the next ``[<ID>]`` header line, a ``---``
      separator (preceded by a blank line), or EOF.

    The separator rule is ``line.startswith("---")``, which is
    ``block_parser``'s rule verbatim. It used to be ``line.strip() ==
    "---"``, which is a *different* rule: it treats an INDENTED ``---`` as
    a boundary where the parser does not. That divergence became load-
    bearing once :func:`_neutralise_value` started escaping a value's
    ``---`` line by indenting it — under the old rule a rewrite of such a
    block would have ended at the escaped line and orphaned the block's
    real tail (Status included) in the file.
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
            elif line.startswith("---"):
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


class CorpusEncodingError(BlockStoreError, ValueError):
    """A corpus file on the read-modify-write path is not valid UTF-8.

    Carries the offending ``path`` so a caller can say *which* file needs
    fixing instead of reporting an anonymous internal failure.

    **Also a ``ValueError``, deliberately.** This replaces the bare
    ``UnicodeDecodeError`` the strict decode used to raise, and that is a
    ``ValueError`` — so every existing handler catching one still catches
    this. ``apply_engine.execute_op`` guards a mid-transaction write with
    ``except (OSError, ValueError, KeyError, IndexError)`` and rolls back;
    narrowing the base class to ``BlockStoreError`` alone would have let an
    undecodable corpus file escape that handler *past the rollback*, which
    is the failure mode the handler exists to prevent. Naming an error more
    precisely must not quietly widen what it escapes.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"corpus file is not valid UTF-8: {path} ({reason})")
        self.path = path
        self.reason = reason


def _read_corpus_for_edit(path: str) -> str:
    """Read a corpus file that is about to be REWRITTEN, strictly as UTF-8.

    Deliberately different from :func:`block_parser.parse_file`, which reads
    with ``errors="replace"`` and says why: a stray byte should shadow one
    character rather than hide a whole file from recall. That is right for a
    read-only path and wrong for this one. ``write_block`` and
    ``delete_block`` read the text, edit it, and write the result back over
    the original with :func:`_atomic_write`. Decoding with ``errors="replace"``
    here would substitute U+FFFD for the byte it could not read and then
    PERSIST that substitution — a silent, irreversible edit to stored memory,
    performed by an operation the caller asked to do something else entirely.
    So the read-modify-write path fails instead, and fails by name.

    The failure is reported rather than swallowed for the same reason
    ``_delete_candidates`` scans the whole corpus: a file this store cannot
    read may be the file that holds the id, and answering "not found" for a
    block that exists is the one answer a memory product must never give.

    Raises:
        CorpusEncodingError: *path* holds bytes that are not UTF-8 — most
            often a legacy file written in a locale codepage (an em dash
            arrives as the single byte 0x97 under cp1252).
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except UnicodeDecodeError as exc:
        raise CorpusEncodingError(
            path,
            f"byte 0x{exc.object[exc.start]:02x} at position {exc.start} is not valid UTF-8; "
            "re-save the file as UTF-8 (mind-mem writes UTF-8 on every platform)",
        ) from exc


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

        Implementations must call
        :func:`~mind_mem.admission.require_delete_admission` as their
        **first** statement — before resolving the target — and report
        the removed text back through
        :meth:`~mind_mem.admission.AdmissionReceipt.record_removal` when
        one is actually removed. That ordering is what makes a missing
        id and an ungated caller fail differently by *authorisation*
        rather than by *existence*, and it is what puts the removed
        content's hash into the evidence chain.

        Implementations should also log the deletion so operators can
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
        """Restore the workspace from a snapshot directory.

        Implementations must call
        :func:`~mind_mem.admission.require_restore_admission` as their
        **first** statement — before the snapshot is read — so a restore
        with no RESTORE scope open fails by *authorisation* rather than
        by whatever the snapshot happens to contain, exactly as
        :meth:`delete_block` fails before resolving its target.

        A restore withdraws every block written since the snapshot. The
        record naming what it reinstated and what it withdrew is written
        by the scope (``apply_engine.restore_snapshot``); the check here
        is what makes that scope non-optional.
        """
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
        # ``corpus_dirs`` is a *narrowing*: the default (None) is the whole
        # corpus as ``corpus_registry`` defines it — the CORPUS_DIRS markdown
        # walk plus every file the corpus table names, wherever it lives. A
        # caller that passes an explicit tuple is asking for a subset, and
        # gets the table's files only for the directories it named. Nothing
        # in ``src/`` narrows; the parameter exists for tests and for a
        # caller opening one part of a workspace.
        self._narrowed_to: tuple[str, ...] | None = corpus_dirs
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
        """Every file this store can read: the ONE corpus definition (I-14).

        Two sources, unioned:

        * every ``.md`` directly inside :data:`corpus_registry.CORPUS_DIRS`
          — the historical walk, which also picks up files no prefix routes
          to (``entities/signals.md``, an archive split out of
          ``DECISIONS.md``); and
        * every file :data:`corpus_registry.CORPUS_TABLE` names that exists
          on disk — which is what adds the four untrusted-ingest corpora
          under ``memory/`` (``INBOX.md``, ``MESSAGES.md``, ``IMPORTED.md``,
          ``INGEST.md``).

        The second half is the fix. Those four files are written through the
        prefix map and served by recall (they are in ``CORPUS_FILES``), while
        the walk never looked under ``memory/`` — so a released ``INBOX-``
        block was returned by ``recall()`` and by ``iter_active_blocks``,
        answered ``None`` from :meth:`get_by_id`, was absent from
        :meth:`get_all`, and made ``DELETE /memories/{id}`` reply ``404 block
        not found`` while the block sat on disk and the store's own
        :meth:`delete_block` removed it fine under a scope. Measured, and the
        ``POST /clear`` docstring asserted the opposite. The three
        definitions of "the corpus" disagreeing was the defect; the 404 was a
        symptom.

        ``memory/`` is **not** walked wholesale — daily logs, the ledgers and
        the deletion journal live there. Only the files the table names are
        added, so a new drop corpus is one table row and nothing else is
        exposed.

        Order is stable and additive: the directory walk first, exactly as
        before, then any table file the walk did not already produce, in
        table order. An existing corpus keeps its enumeration order.
        """
        if self._files is not None:
            return self._files
        files: list[str] = []
        seen: set[str] = set()
        for d in self._corpus_dirs:
            dir_path = os.path.join(self._workspace, d)
            if os.path.isdir(dir_path):
                for fname in sorted(os.listdir(dir_path)):
                    if fname.endswith(".md"):
                        path = os.path.join(dir_path, fname)
                        if path not in seen:
                            seen.add(path)
                            files.append(path)
        for rel in CORPUS_RELPATHS:
            subdir = rel.split("/", 1)[0]
            if self._narrowed_to is not None and subdir not in self._narrowed_to:
                continue
            path = os.path.join(self._workspace, *rel.split("/"))
            if path not in seen and os.path.isfile(path):
                seen.add(path)
                files.append(path)
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
            UngatedWriteError: no governance admission is open for this
                block. See :mod:`mind_mem.admission`.
            ValueError: block is missing ``_id`` or has an
                unrecognised prefix (no canonical file mapping).
        """
        block_id = block.get("_id")
        if not block_id:
            raise ValueError("block is missing '_id'; cannot write")
        # Governance choke-point. Refuses the write outright when no
        # admission scope is open — checked before the id is even parsed,
        # so an ungated caller cannot learn anything by probing id shapes.
        require_admission(str(block_id), status=block.get("Status"))
        if not _BLOCK_ID_RE.match(str(block_id)):
            raise ValueError(f"invalid block id: {block_id!r}")
        target = _resolve_block_file(self._workspace, block_id)
        if target is None:
            raise ValueError(
                f"no canonical file mapping for block id {block_id!r}; add a row to corpus_registry.CORPUS_TABLE to enable writes"
            )

        rendered = _render_block(block)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        with FileLock(target):
            if os.path.isfile(target):
                existing_text = _read_corpus_for_edit(target)
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

        Requires an open ``DELETE`` admission covering *block_id*
        (:meth:`~mind_mem.governance_gate.GovernanceGate.admit_delete`).
        The check runs before the target is resolved, so a delete of an
        id that is not here returns ``False`` while a delete with no
        scope open raises — existence never leaks through the refusal.

        Logs the removed content to ``memory/deleted_blocks.jsonl``
        so the deletion is recoverable. The journal format matches
        what :func:`mcp.tools.memory_ops.delete_memory_item` writes —
        both write paths converge on the same recovery record. The
        journal is local recovery; the chain record the scope writes
        from :meth:`~mind_mem.admission.AdmissionReceipt.record_removal`
        is the audit fact.

        Resolution is by :meth:`_delete_candidates`, not by the prefix
        map alone: a block this store can *read* must be a block it can
        *remove*, or a governed delete becomes a governed refusal to
        delete.
        """
        receipt = require_delete_admission(str(block_id))
        for target in self._delete_candidates(str(block_id)):
            with FileLock(target):
                text = _read_corpus_for_edit(target)
                loc = _locate_block_in_text(text, block_id)
                if loc is None:
                    continue
                start, end, removed = loc
                lines = text.split("\n")
                new_lines = lines[:start] + lines[end:]
                new_text = "\n".join(new_lines)
                if new_text and not new_text.endswith("\n"):
                    new_text += "\n"
                _record_deletion(self._workspace, block_id, removed)
                _atomic_write(target, new_text)

            receipt.record_removal(str(block_id), removed)
            self.invalidate_cache()
            _log.info("block_store_delete", block_id=block_id, file=os.path.relpath(target, self._workspace))
            return True
        return False

    def _delete_candidates(self, block_id: str) -> list[str]:
        """Files that could hold *block_id*, canonical one first.

        :func:`_resolve_block_file` answers only for the prefixes in
        :data:`_BLOCK_PREFIX_MAP`, and its own docstring says so: it
        "returns ``None`` for unrecognised prefixes. Callers must fall
        back to full-corpus scan when the mapping is absent (e.g.,
        signals, one-off entity types not in the prefix map)."
        ``delete_block`` did not fall back, so a block the store could
        read was a block it refused to delete — measured: a ``SIG-…``
        block sitting in ``entities/signals.md`` is returned by
        ``get_by_id`` and served by recall, while ``DELETE
        /memories/{id}`` answered ``404 block not found`` and ``POST
        /clear`` reported ``ok`` and left it behind. A *partial* purge
        reported as a whole one, and an id an operator cannot destroy
        through any door — which for a memory product is the shape of an
        undeletable record.

        The canonical file is tried first, so a mapped prefix costs
        exactly what it cost before: one open. Everything else is walked
        only when that misses — a path that previously returned the wrong
        answer, so the scan is spent on the cases it fixes rather than on
        the common one.
        """
        candidates: list[str] = []
        mapped = _resolve_block_file(self._workspace, block_id)
        if mapped is not None and os.path.isfile(mapped):
            candidates.append(mapped)
        for path in self._discover_files():
            if path not in candidates and os.path.isfile(path):
                candidates.append(path)
        return candidates

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
                if is_ledger_path(rel_path):
                    continue  # a ledger of record is never snapshot content
                resolved = os.path.realpath(os.path.join(ws_real, rel_path))  # nosec — realpath resolves symlinks; traversal filtered by startswith check below
                if not resolved.startswith(ws_real + os.sep) and resolved != ws_real:
                    continue  # nosec — path escapes workspace; skip it
                if os.path.isfile(resolved):  # nosec — resolved is within ws_real (validated above)
                    _safe_copy(resolved, os.path.join(snap_dir, rel_path))  # nosec — resolved validated; snap_dir is operator-controlled workspace subdirectory
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
                            if is_ledger_path(rel):
                                # ``memory/`` is corpus AND ledger. Taking the
                                # ledger is what makes the restore a rewind.
                                continue
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
        if manifest_data is None:
            raise RuntimeError(f"invariant violated: snapshot manifest was not written to {snap_dir}")
        _log.info("block_store_snapshot", snap_dir=snap_dir, file_count=len(manifest_files))
        return manifest_data

    def restore(self, snap_dir: str) -> None:
        """Restore workspace from snapshot.

        Uses the MANIFEST.json fast-path (O(manifest)) when available;
        falls back to the legacy copytree approach for pre-manifest snapshots.

        ``intelligence/applied/`` is always skipped to prevent deleting
        active snapshots during a restore.

        A restore is best-effort against a damaged snapshot: a manifest
        entry whose snapshot copy is missing cannot be put back, and the
        workspace keeps whatever the ops left there. Those entries are
        logged individually as ``restore_missing_snapshot_source`` and
        counted in the ``missing`` field of the ``block_store_restore``
        record, which also carries ``complete`` — do not read a restore as
        a completed rollback without checking it.

        Raises:
            UngatedRestoreError: No RESTORE admission is open. Checked
                before the manifest is read, so an ungated caller and a
                caller naming a damaged snapshot fail differently.
        """
        receipt = require_restore_admission(snap_dir)
        ws = self._workspace
        manifest_data = _read_manifest(snap_dir)
        if manifest_data is not None:
            manifest = manifest_data.get("files", [])
            cleanup_inventory = manifest_data.get("cleanup_inventory", {})
            safe_manifest: list[str] = []
            #: Manifest entries whose snapshot copy is gone — restore cannot
            #: put those files back, so they are reported, never counted as
            #: restored.
            missing: list[str] = []
            #: Ledger entries a pre-5.0.2 manifest named. Reported apart from
            #: ``missing`` on purpose: a refused ledger is the invariant
            #: holding, not a damaged snapshot, so it must not make
            #: ``complete`` read False and drain that signal of meaning.
            refused_ledgers: list[str] = []
            restored = 0
            for rel_posix in manifest:
                rel_path = rel_posix.replace("/", os.sep)
                try:
                    # v3.6.6 path-injection hardening: reject any manifest
                    # entry whose resolved path (after symlinks) escapes
                    # either the snapshot dir or the workspace root.
                    src = _safe_child_path(snap_dir, rel_path)
                    dst = _safe_child_path(ws, rel_path)
                except ValueError as exc:
                    _log.warning("restore_unsafe_manifest_entry", entry=rel_posix, reason=str(exc))
                    continue
                if is_ledger_path(rel_posix):
                    # A snapshot taken before 5.0.2 names the ledgers. Putting
                    # one back IS the rewind, so the write is refused — but the
                    # entry stays in ``safe_manifest`` so the orphan sweep does
                    # not delete the live ledger this branch just spared.
                    safe_manifest.append(rel_posix)
                    refused_ledgers.append(rel_posix)
                    continue
                safe_manifest.append(rel_posix)
                if os.path.exists(src):  # nosec — src is the resolved absolute path returned by _safe_child_path; path traversal already rejected above
                    os.makedirs(os.path.dirname(dst), exist_ok=True)  # nosec — dst validated by _safe_child_path
                    shutil.copy2(src, dst)  # nosec — both src and dst validated by _safe_child_path
                    restored += 1
                else:
                    # The manifest names a file the snapshot no longer holds
                    # (a partially removed / truncated snapshot dir). There is
                    # nothing to copy back, so whatever the ops left at that
                    # path stays. Name it and count it: reporting these as
                    # restored is how a rollback that did not roll back gets
                    # logged as complete.
                    #
                    # The entry deliberately stays in ``safe_manifest``.
                    # Dropping it would make _cleanup_orphans_from_manifest
                    # treat the live workspace file as an orphan and delete
                    # it — destroying content the snapshot recorded as present
                    # on the strength of a damaged snapshot.
                    missing.append(rel_posix)
                    _log.warning("restore_missing_snapshot_source", snap_dir=snap_dir, entry=rel_posix)
            _cleanup_orphans_from_manifest(ws, safe_manifest, cleanup_inventory)
            _log.info(
                "block_store_restore",
                snap_dir=snap_dir,
                file_count=restored,
                skipped=len(manifest) - len(safe_manifest),
                missing=len(missing),
                refused_ledgers=len(refused_ledgers),
                complete=not missing,
                admission=receipt.entry_id,
            )
            return

        # Legacy fallback: copytree-based restore for pre-manifest snapshots.
        # SNAPSHOT_DIRS is a module constant (no user input) but we still
        # route via _safe_child_path so audits see one uniform guard.
        for d in SNAPSHOT_DIRS:
            try:
                src = _safe_child_path(snap_dir, d)  # nosec — _safe_child_path rejects traversal; d from SNAPSHOT_DIRS constant
                dst = _safe_child_path(ws, d)  # nosec — same guard
            except ValueError:
                continue
            if os.path.isdir(src):
                tmp_dst = dst + ".rollback_tmp"
                if os.path.islink(tmp_dst):
                    os.unlink(tmp_dst)
                elif os.path.isdir(tmp_dst):
                    shutil.rmtree(tmp_dst)
                shutil.copytree(src, tmp_dst)
                # ``memory/`` is swapped wholesale here; carry the live
                # ledgers across so the rmtree below cannot destroy them.
                _carry_ledgers_into(ws, dst, tmp_dst)
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
                try:
                    # ``item`` comes from os.listdir on disk so it's ours,
                    # but we still validate so a rogue symlink at the
                    # destination path can't redirect the copy outside ws.
                    src = _safe_child_path(intel_snap, item)
                    dst = _safe_child_path(intel_ws, item)
                except ValueError as exc:
                    _log.warning("restore_unsafe_intel_entry", entry=item, reason=str(exc))
                    continue
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
            try:
                src = _safe_child_path(snap_dir, f)
                dst = _safe_child_path(ws, f)
            except ValueError:
                continue
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        _log.info("block_store_restore_legacy", snap_dir=snap_dir, admission=receipt.entry_id)

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
