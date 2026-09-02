#!/usr/bin/env python3
"""Mind Mem SQLite FTS5 Index — incremental lexical indexing. Zero external deps.

Replaces O(corpus) per-query scanning with O(log N) indexed lookup via SQLite
FTS5. Supports incremental updates (only re-indexes changed files), field-weighted
ranking, and deterministic post-processing (recency/status boosts, query type
detection, reranker) matching the existing recall.py pipeline.

Schema:
    blocks(id PK, type, file, line, status, date, speaker, tags, json_blob)
    blocks_fts(Statement, Title, Tags, Description, Context) — FTS5 virtual table
    xref_edges(src, dst) — cross-reference graph
    file_state(path, mtime, size, hash) — incremental rebuild tracking

Usage:
    python3 -m mind_mem.sqlite_index build --workspace .
    python3 -m mind_mem.sqlite_index build --workspace . --incremental
    python3 -m mind_mem.sqlite_index query --workspace . --query "PostgreSQL"
    python3 -m mind_mem.sqlite_index status --workspace .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import threading
from datetime import date, datetime

from .admissibility import is_admissible_status, workspace_release_ids
from .block_parser import parse_file
from .block_provenance import PROVENANCE_FIELD_NAMES
from .connection_manager import ConnectionManager
from .enums import TaskStatus
from .extractor import extract_facts
from .observability import get_logger, metrics
from .recall import (
    _BLOCK_ID_RE,
    _QUERY_TYPE_PARAMS,
    CORPUS_FILES,
    SEARCH_FIELDS,
    _parse_speaker_from_tags,
    date_score,
    detect_query_type,
    expand_months,
    expand_query,
    get_block_type,
    get_excerpt,
    rerank_hits,
    tokenize,
)
from .scoring_instant import as_utc_datetime, resolve_scoring_instant

_log = get_logger("sqlite_index")

# DB location relative to workspace
DB_REL_PATH = ".mind-mem-index/recall.db"

# FTS5 columns and their weights (order matters for bm25() function)
# bm25() returns negative values; lower = better match
FTS5_COLUMNS = [
    ("statement", 3.0),
    ("title", 2.5),
    ("name", 2.0),
    ("description", 1.2),
    ("tags", 0.8),
    ("context", 0.5),
    ("all_text", 1.0),  # catch-all for other fields
]

# Leading weight for the block_id column. The blocks_fts table is
# fts5(block_id, <FTS5_COLUMNS...>), so block_id is the first INDEXED column
# and bm25() positional weights must include it. A neutral 1.0 keeps block_id
# searchable without distorting the tuned field weights.
_BLOCK_ID_WEIGHT = 1.0


def _bm25_weights() -> str:
    """Return the comma-separated bm25() weights aligned to blocks_fts.

    One weight per indexed FTS5 column: block_id first, then FTS5_COLUMNS.
    Omitting the leading block_id weight shifts every field's weight by one
    column (block_id steals statement's weight, etc.).
    """
    return ", ".join(str(w) for w in (_BLOCK_ID_WEIGHT, *(w for _, w in FTS5_COLUMNS)))


# Only allow alphanumeric tokens (plus _ - .) through to FTS5 queries
# to prevent wildcard injection (e.g. "*" matching entire corpus)
_FTS5_SAFE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


# ---------------------------------------------------------------------------
# Admission — which blocks the index may carry statistics for
# ---------------------------------------------------------------------------
#
# The searchable table (``blocks_fts``) is the STATISTICS surface: SQLite's
# ``bm25()`` computes IDF and the average document length over every row in
# it, corpus-wide, and no ``WHERE`` clause narrows that. So a withheld block
# sitting in ``blocks_fts`` moves the score of every *admitted* result — for
# terms it shares AND for terms it does not, through the document count and
# the length average. Measured on a 12-block corpus: adding one quarantined
# block moved a shared term's bm25 from -1.9688 to -1.6026 and an unrelated
# term's from -2.8718 to -3.1645. That is the withheld content shaping what a
# caller sees, which is exactly what the gate exists to prevent.
#
# So ``blocks_fts`` carries the ADMITTED set only. ``blocks``, ``index_meta``
# and ``xref_edges`` still carry every block, which is what keeps the two
# properties the old design was protecting: the attested index anchor
# (:func:`merkle_leaves` reads ``index_meta``) does not churn, and a release
# is a membership flip on one table rather than a re-parse of the corpus.


def _release_set(workspace: str | None) -> frozenset[str]:
    """Governance release set for *workspace*; empty and never raising.

    An unreadable decisions file admits nothing — the fail-closed
    direction, matching :func:`admissibility.workspace_release_ids`.
    """
    if not workspace:
        return frozenset()
    try:
        return workspace_release_ids(workspace)
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("release_lookup_failed", error=str(exc))
        return frozenset()


def _admit_ids(
    pairs: list[tuple[str, object]],
    *,
    workspace: str | None = None,
    releases: frozenset[str] | None = None,
) -> set[str]:
    """The ids in ``(block_id, status)`` *pairs* the index may serve.

    The predicate is :func:`admissibility.is_admissible_status` — the one
    allow-list — plus the governance release set. Nothing here restates
    the status vocabulary, so a tier that mints a new withheld status
    withholds its blocks from the index with no edit in this module.

    The release set is resolved **lazily**: an all-admissible batch
    returns without touching the filesystem, so the common case pays no
    probe. Pass *releases* when the caller already resolved it once for a
    whole index pass.
    """
    kept: set[str] = set()
    withheld = 0
    for bid, status in pairs:
        if is_admissible_status(status):
            kept.add(bid)
        else:
            withheld += 1
    if not withheld:
        return kept
    if releases is None:
        releases = _release_set(workspace)
    if releases:
        kept |= {bid for bid, _status in pairs if bid in releases}
    return kept


def _release_set_hash(releases: frozenset[str]) -> str:
    """Stable digest of the release set, for build-to-build comparison."""
    return hashlib.sha256("\x00".join(sorted(releases)).encode("utf-8")).hexdigest()


def _db_path(workspace: str) -> str:
    """Return absolute path to the index database."""
    return os.path.join(os.path.abspath(workspace), DB_REL_PATH)


def _connect(workspace: str, readonly: bool = False) -> sqlite3.Connection:
    """Open (or create) the index database.

    For new code, prefer _get_conn_manager() which provides connection
    pooling with read/write separation (#466).
    """
    path = _db_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if readonly:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(path)
    if not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Connection Manager — pooled read/write separation (#466)
# ---------------------------------------------------------------------------

# BOUNDED, and closed on eviction. Each ConnectionManager holds an open
# SQLite read connection, which costs three descriptors on disk: the db, its
# -wal and its -shm. This cache used to be unbounded, and its only eviction
# was REACTIVE -- a stale entry was dropped when _get_conn_manager was called
# again FOR THAT SAME PATH and the file had vanished. Nothing re-accesses a
# workspace once its work is done, so entries were never reached again and
# their connections were held for the life of the process.
#
# Measured: a full test run reached 2,101 open fds -- 55 handles on one
# recall.db, 38 on its -wal, 13 on its -shm, many marked (deleted) -- then
# died with "ValueError: I/O operation on closed file" / "lost sys.stderr",
# the process having crossed the fd limit with pytest's own stream as the
# casualty. In a long-running MCP server the same shape leaks three
# descriptors per workspace, permanently.
#
# Plain dict, which has been insertion-ordered since 3.7, so LRU needs no
# extra type: ``d[k] = d.pop(k)`` moves an entry to the end and the oldest
# key is ``next(iter(d))``. The evicted manager is CLOSED, not merely
# dropped -- dropping the reference would leave the descriptors held until
# the collector happened to run, which is precisely the non-determinism this
# is meant to end.
_CONN_MANAGER_CACHE_MAX = 32

_conn_managers: dict[str, ConnectionManager] = {}
_conn_managers_lock = threading.Lock()


def _evict_conn_managers_locked() -> None:
    """Close and drop least-recently-used managers past the bound.

    Caller must hold ``_conn_managers_lock``. Closing is best-effort for the
    same reason the stale-file path is: a manager whose backing file is gone
    can raise on close, and that must not block the eviction it is part of.
    """
    while len(_conn_managers) > _CONN_MANAGER_CACHE_MAX:
        _path = next(iter(_conn_managers))
        victim = _conn_managers.pop(_path)
        try:
            victim.close()
        except Exception:  # nosec B110 - eviction must not be blocked by a dying manager
            _log.warning("conn_manager_evict_close_failed", db_path=_path)


# Per-thread re-entrancy guard for query_index.  When the index file is
# missing and query_index falls back to recall(), and recall() is configured
# to use the sqlite backend, it would call query_index() again — causing
# mutual infinite recursion.  Tracking entry per workspace string per thread
# breaks the cycle: a re-entrant call returns [] instead of looping.
_query_index_active: threading.local = threading.local()


def _get_conn_manager(workspace: str) -> ConnectionManager:
    """Return a shared ConnectionManager for *workspace*.

    Ensures the index directory exists and caches one manager per db_path.

    If the cached manager's DB file has been deleted from disk (e.g. because
    the workspace was wiped by a test harness or ``shutil.rmtree``), the stale
    manager is evicted and a fresh one is returned.  Without this check the
    stale ``sqlite3.Connection`` inside the old manager keeps writing to the
    deleted inode, the file path remains absent on disk, and
    ``query_index()``'s "index_missing_fallback" branch then calls
    ``recall()`` → ``query_index()`` → ``recall()`` ... in an unbounded
    mutual recursion.
    """
    path = _db_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _conn_managers_lock:
        mgr = _conn_managers.get(path)
        if mgr is not None and not os.path.isfile(path):
            # DB file was removed (workspace deleted between runs). Drop the
            # cache entry so a fresh manager is created below, producing a
            # new on-disk file. ``close()`` here is best-effort: it closes
            # the calling thread's pooled connections; any other thread's
            # thread-local connection to the now-unlinked inode is harmless
            # (the manager is removed from the cache, so no new caller can
            # reach it, and the OS reclaims the inode once those fds close).
            try:
                mgr.close()
            except Exception:  # nosec B110 — see rationale below.
                # Best-effort close of a stale ``ConnectionManager`` whose
                # backing DB file has already been unlinked (workspace was
                # deleted between runs). Any failure here is non-fatal:
                # the manager is dropped from the cache on the next line,
                # so no caller can reach the dead manager again, and the
                # OS reclaims fds + inode once existing thread-local
                # connections close. Re-raising would block the legitimate
                # fresh-manager creation in the common "workspace rebuilt"
                # path. Audited 2026-05-19 for alert #191.
                pass
            mgr = None
        if mgr is None:
            mgr = ConnectionManager(path)
            _conn_managers[path] = mgr
        else:
            # Touch: most-recently-used goes to the end, so eviction takes
            # the genuinely cold entries rather than whichever was created
            # first.
            _conn_managers[path] = _conn_managers.pop(path)
        _evict_conn_managers_locked()
    return mgr


# Core index tables whose presence indicates the FTS schema has been built.
# ``index_status``/``merkle_leaves`` probe these via ``sqlite_master`` instead
# of calling ``_init_schema`` (a write) on a read-only connection — recall.db
# is frequently created by side-tables (calibration, retrieval_log) before
# ``build_index`` ever runs, leaving it without these tables. On the Postgres
# backend the FTS schema is often never built at all, making the read-only
# CREATE TABLE crash reliable. See audit bugs 13 & 14.
_INDEX_SCHEMA_TABLES: frozenset[str] = frozenset({"blocks", "meta"})


def _index_schema_present(conn: sqlite3.Connection) -> bool:
    """Return True when the core FTS index tables exist in *conn*.

    Read-only safe: issues a single ``sqlite_master`` SELECT and never
    executes DDL. Used by the read-only status/merkle paths so they do
    not attempt a ``CREATE TABLE`` on a ``mode=ro`` connection (which
    raises ``OperationalError: attempt to write a readonly database``).
    """
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name IN ('blocks','meta')").fetchall()
    except sqlite3.OperationalError:
        return False
    present = {r[0] for r in rows}
    return _INDEX_SCHEMA_TABLES.issubset(present)


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blocks (
            id          TEXT PRIMARY KEY,
            type        TEXT NOT NULL,
            file        TEXT NOT NULL,
            line        INTEGER NOT NULL DEFAULT 0,
            status      TEXT NOT NULL DEFAULT '',
            date        TEXT NOT NULL DEFAULT '',
            speaker     TEXT NOT NULL DEFAULT '',
            tags        TEXT NOT NULL DEFAULT '',
            dia_id      TEXT NOT NULL DEFAULT '',
            parent_id   TEXT NOT NULL DEFAULT '',
            json_blob   TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS xref_edges (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            PRIMARY KEY (src, dst)
        );

        CREATE TABLE IF NOT EXISTS file_state (
            path     TEXT PRIMARY KEY,
            mtime    REAL NOT NULL,
            mtime_ns INTEGER,
            size     INTEGER NOT NULL,
            hash     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS block_vectors (
            id        TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            model     TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS block_meta (
            id TEXT PRIMARY KEY,
            importance REAL DEFAULT 1.0,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            keywords TEXT DEFAULT '',
            connections TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS index_meta (
            file_path    TEXT NOT NULL,
            block_id     TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            indexed_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            admitted     INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (file_path, block_id)
        );
    """)

    # Create standalone FTS5 virtual tables (we manage sync ourselves).
    #
    # TWO of them, and the split is the admission boundary. ``bm25()``
    # averages over every row of the table it is given and no WHERE clause
    # narrows that, so which table a block sits in IS the decision about
    # whether it shapes other blocks' scores. ``blocks_fts`` holds the
    # admitted set and is what queries score against; ``blocks_fts_withheld``
    # holds the rest, so a governance release can still surface a block with
    # no reindex (see ``query_index``) without that block having contributed
    # a single document to the statistics of the corpus it was withheld from.
    # A block is in exactly one of the two, never both.
    cols = ", ".join(col for col, _ in FTS5_COLUMNS)
    for _fts_table in ("blocks_fts", "blocks_fts_withheld"):
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {_fts_table}
            USING fts5(block_id, {cols}, tokenize='porter unicode61')
        """)

    # Migration: add parent_id column if missing (existing databases)
    try:
        conn.execute("ALTER TABLE blocks ADD COLUMN parent_id TEXT NOT NULL DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # 5.0.2 migration: record the ADMISSION verdict beside the content hash.
    # Content-hash equality is not enough to decide whether a block still
    # belongs in ``blocks_fts``: a governance release (or a quarantine) flips
    # admissibility with the block's bytes untouched, so without this column
    # an incremental pass would classify it "unchanged" and the membership
    # flip would never land. Legacy rows default to 1 (admitted), which is
    # what they were: the recorded verdict then disagrees with the computed
    # one for exactly the withheld blocks, and the first pass corrects them.
    try:
        conn.execute("ALTER TABLE index_meta ADD COLUMN admitted INTEGER NOT NULL DEFAULT 1")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # v3.7.0 M8 migration: add nanosecond mtime column. Legacy rows
    # left at NULL force a one-time rehash (treated as "changed" by
    # ``_get_changed_files``) which is cheap because the full SHA-256
    # is now what gets compared anyway.
    try:
        conn.execute("ALTER TABLE file_state ADD COLUMN mtime_ns INTEGER")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Secondary indexes for common query patterns
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_parent_id ON blocks(parent_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_file ON blocks(file)")

    conn.commit()


# ---------------------------------------------------------------------------
# File State Tracking
# ---------------------------------------------------------------------------


_FILE_HASH_CHUNK = 1 << 20  # 1 MiB


def _file_hash(path: str) -> str:
    """Compute SHA-256 of the entire file (v3.7.0 M8).

    Pre-3.7.0 only hashed the first 64 KiB plus the file size, so a
    file whose size + mtime stayed identical but whose content past
    64 KiB changed (large markdown corpus rewritten in place; an
    editor that truncates-and-rewrites within the same second on
    coarse-mtime filesystems) was treated as unchanged and skipped on
    the next reindex. Reading the whole file is unconditionally
    correct; the cheap pre-filter in ``_get_changed_files`` keeps
    the common steady-state path I/O-free.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_FILE_HASH_CHUNK), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _get_changed_files(conn: sqlite3.Connection, workspace: str) -> list[tuple[str, str]]:
    """Return list of (label, rel_path) for corpus files that changed since last index.

    A file is considered changed if:
    - It doesn't exist in file_state table
    - Its size differs from the recorded value
    - Its ``mtime_ns`` differs from the recorded value (or the
      recorded value is NULL — happens once per pre-3.7.0 row after
      the M8 migration, which forces a single rehash)
    - Its full SHA-256 differs (catches the same-size + same-mtime_ns
      case where an in-place edit kept the metadata stable)
    """
    changed = []
    ws = os.path.abspath(workspace)

    for label, rel_path in CORPUS_FILES.items():
        full_path = os.path.join(ws, rel_path)
        if not os.path.isfile(full_path):
            # File doesn't exist — check if it was previously indexed
            row = conn.execute("SELECT path FROM file_state WHERE path = ?", (rel_path,)).fetchone()
            if row:
                changed.append((label, rel_path))  # file was deleted
            continue

        stat = os.stat(full_path)
        row = conn.execute(
            "SELECT mtime, mtime_ns, size, hash FROM file_state WHERE path = ?",
            (rel_path,),
        ).fetchone()

        if row is None:
            changed.append((label, rel_path))
            continue

        if stat.st_size != row["size"]:
            changed.append((label, rel_path))
            continue

        # Prefer the nanosecond mtime when present; fall back to the
        # legacy float mtime for pre-3.7.0 rows that haven't been
        # rehashed yet. Either way we still verify the full hash next.
        recorded_mtime_ns = row["mtime_ns"]
        if recorded_mtime_ns is None:
            # Legacy row — force a rehash so the migration completes
            # for this file on the next index pass.
            changed.append((label, rel_path))
            continue
        if stat.st_mtime_ns != recorded_mtime_ns:
            changed.append((label, rel_path))
            continue

        # Same size + same nanosecond mtime — verify the full hash to
        # catch in-place edits that kept the metadata stable.
        current_hash = _file_hash(full_path)
        if current_hash != row["hash"]:
            changed.append((label, rel_path))

    return changed


def _update_file_state(conn: sqlite3.Connection, workspace: str, rel_path: str) -> None:
    """Update file_state for a corpus file."""
    ws = os.path.abspath(workspace)
    full_path = os.path.join(ws, rel_path)

    if not os.path.isfile(full_path):
        conn.execute("DELETE FROM file_state WHERE path = ?", (rel_path,))
        return

    stat = os.stat(full_path)
    h = _file_hash(full_path)
    conn.execute(
        "INSERT OR REPLACE INTO file_state (path, mtime, mtime_ns, size, hash) VALUES (?, ?, ?, ?, ?)",
        (rel_path, stat.st_mtime, stat.st_mtime_ns, stat.st_size, h),
    )


# ---------------------------------------------------------------------------
# Block-level hashing
# ---------------------------------------------------------------------------


def _compute_block_hash(block: dict) -> str:
    """Compute content hash of a parsed block for change detection.

    Hashes a stable JSON representation of all fields except _line
    (which changes when blocks above shift). Uses SHA-256, stdlib only.
    """
    # Copy without volatile fields
    stable = {k: v for k, v in block.items() if k != "_line"}
    raw = json.dumps(stable, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def _extract_fts_fields(block: dict) -> dict:
    """Extract FTS5 column values from a block."""
    return {
        "statement": block.get("Statement", ""),
        "title": block.get("Title", "") or block.get("Name", ""),
        "name": block.get("Name", ""),
        "description": block.get("Description", "") or block.get("Summary", ""),
        "tags": block.get("Tags", "") or block.get("Keywords", ""),
        "context": block.get("Context", "") or block.get("Rationale", ""),
        "all_text": " ".join(str(block.get(f, "")) for f in SEARCH_FIELDS if block.get(f)),
    }


def _extract_xrefs(block: dict, all_block_ids: set) -> list[str]:
    """Extract cross-reference IDs from a block."""
    texts = []
    xref_fields = SEARCH_FIELDS + [
        "Supersedes",
        "SupersededBy",
        "AlignsWith",
        "Dependencies",
        "Next",
        "Sources",
        "Evidence",
        "Rollback",
        "History",
    ]
    for field in xref_fields:
        val = block.get(field, "")
        if isinstance(val, str):
            texts.append(val)
        elif isinstance(val, list):
            texts.extend(str(v) for v in val)

    full_text = " ".join(texts)
    bid = block.get("_id", "")
    refs = []
    for m in _BLOCK_ID_RE.finditer(full_text):
        ref_id = m.group(1)
        if ref_id != bid and ref_id in all_block_ids:
            refs.append(ref_id)
    return refs


def _index_file(
    conn: sqlite3.Connection,
    workspace: str,
    label: str,
    rel_path: str,
    all_block_ids: set,
    force: bool = False,
    releases: frozenset[str] | None = None,
) -> dict:
    """Index a single corpus file with block-level incremental updates.

    Returns dict with counts: new, modified, deleted, unchanged, total.
    When force=True, skips hash comparison and re-indexes all blocks.

    *releases* is the workspace's governance release set, resolved once
    per build pass by :func:`build_index`. A block is admitted to the
    searchable table when :func:`_admit_ids` says so; a block whose
    admission verdict changed since the last pass counts as **modified**
    even when its bytes did not, because that is the only way a release
    or a quarantine reaches ``blocks_fts``.
    """
    ws = os.path.abspath(workspace)
    full_path = os.path.join(ws, rel_path)

    counts = {"new": 0, "modified": 0, "deleted": 0, "unchanged": 0, "total": 0}

    # Load existing block hashes + admission verdicts for this file
    existing_hashes = {}
    existing_admitted: dict[str, bool] = {}
    for row in conn.execute(
        "SELECT block_id, content_hash, admitted FROM index_meta WHERE file_path = ?",
        (rel_path,),
    ).fetchall():
        existing_hashes[row["block_id"]] = row["content_hash"]
        existing_admitted[row["block_id"]] = bool(row["admitted"])

    # Handle deleted file
    if not os.path.isfile(full_path):
        if existing_hashes:
            old_ids = list(existing_hashes.keys())
            _delete_blocks(conn, old_ids, rel_path)
            counts["deleted"] = len(old_ids)
        _update_file_state(conn, workspace, rel_path)
        return counts

    try:
        blocks = parse_file(full_path)
    except (OSError, UnicodeDecodeError, ValueError):
        _update_file_state(conn, workspace, rel_path)
        return counts

    # Build current block map: {block_id: (block_dict, content_hash)}
    current_blocks = {}
    for block in blocks:
        bid = block.get("_id", "")
        if not bid:
            continue
        current_blocks[bid] = (block, _compute_block_hash(block))

    # ADMISSION. Decided here, once, off the block headers this pass just
    # parsed — the same allow-list every other egress path uses.
    admitted_ids = _admit_ids(
        [(bid, blk.get("Status")) for bid, (blk, _hash) in current_blocks.items()],
        releases=releases if releases is not None else frozenset(),
    )

    current_ids = set(current_blocks.keys())
    existing_ids = set(existing_hashes.keys())

    # Classify blocks
    new_ids = current_ids - existing_ids
    deleted_ids = existing_ids - current_ids
    common_ids = current_ids & existing_ids

    modified_ids = set()
    unchanged_ids = set()
    for bid in common_ids:
        admission_flipped = (bid in admitted_ids) != existing_admitted.get(bid, True)
        if force or current_blocks[bid][1] != existing_hashes[bid] or admission_flipped:
            modified_ids.add(bid)
        else:
            unchanged_ids.add(bid)

    # Delete removed blocks
    if deleted_ids:
        _delete_blocks(conn, list(deleted_ids), rel_path)

    # Delete modified blocks (will be re-inserted)
    if modified_ids:
        _delete_blocks(conn, list(modified_ids), rel_path)

    # Insert new + modified blocks
    for bid in new_ids | modified_ids:
        block, content_hash = current_blocks[bid]
        _insert_block(conn, block, bid, rel_path, all_block_ids, admitted=bid in admitted_ids)
        conn.execute(
            """INSERT OR REPLACE INTO index_meta
               (file_path, block_id, content_hash, admitted)
               VALUES (?, ?, ?, ?)""",
            (rel_path, bid, content_hash, 1 if bid in admitted_ids else 0),
        )

    # Update index_meta for unchanged blocks (keep existing entries)
    # No-op — they're already correct in index_meta

    # Clean up index_meta for deleted blocks
    if deleted_ids:
        placeholders = ",".join("?" for _ in deleted_ids)
        conn.execute(
            f"DELETE FROM index_meta WHERE file_path = ? AND block_id IN ({placeholders})",  # nosec B608 — placeholders is `? * N`, all values passed as bind params; no user input in query string
            [rel_path] + list(deleted_ids),
        )

    counts["new"] = len(new_ids)
    counts["modified"] = len(modified_ids)
    counts["deleted"] = len(deleted_ids)
    counts["unchanged"] = len(unchanged_ids)
    counts["total"] = len(new_ids) + len(modified_ids)

    _update_file_state(conn, workspace, rel_path)
    return counts


def _insert_block(
    conn: sqlite3.Connection,
    block: dict,
    bid: str,
    rel_path: str,
    all_block_ids: set,
    admitted: bool = True,
) -> None:
    """Insert a single block into blocks, blocks_fts, and xref_edges.

    Also extracts atomic fact cards from the block's Statement field and indexes
    them as sub-blocks (parent_id = bid) for small-to-big retrieval.

    *admitted* is the egress verdict from :func:`_admit_ids`, and it
    routes the searchable text to ``blocks_fts`` or to the withheld
    shadow ``blocks_fts_withheld``. ``blocks``, ``index_meta`` and
    ``xref_edges`` carry every block either way — the index knows the
    whole corpus, so the attested anchor covers it and a release is a
    membership flip rather than a re-parse. What the withheld block does
    NOT do is contribute a document to the ``bm25()`` statistics that
    shape every admitted result's score. Its fact sub-blocks inherit the
    verdict, for the same reason.
    """
    tags_str = block.get("Tags", "")
    speaker = _parse_speaker_from_tags(tags_str)

    conn.execute(
        """INSERT OR REPLACE INTO blocks
           (id, type, file, line, status, date, speaker, tags, dia_id, parent_id, json_blob)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            bid,
            get_block_type(bid),
            rel_path,
            block.get("_line", 0),
            block.get("Status", ""),
            block.get("Date", ""),
            speaker,
            tags_str,
            block.get("DiaID", ""),
            "",  # parent_id — empty for top-level blocks
            json.dumps(block, default=str),
        ),
    )

    fts_table = "blocks_fts" if admitted else "blocks_fts_withheld"
    fts = _extract_fts_fields(block)
    conn.execute(
        f"""INSERT INTO {fts_table} (block_id, statement, title, name,
           description, tags, context, all_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",  # nosec B608 — fts_table is one of two literals chosen by the admission verdict, never user input
        (
            bid,
            fts["statement"],
            fts["title"],
            fts["name"],
            fts["description"],
            fts["tags"],
            fts["context"],
            fts["all_text"],
        ),
    )

    refs = _extract_xrefs(block, all_block_ids)
    for ref in refs:
        conn.execute(
            "INSERT OR IGNORE INTO xref_edges (src, dst) VALUES (?, ?)",
            (bid, ref),
        )
        conn.execute(
            "INSERT OR IGNORE INTO xref_edges (src, dst) VALUES (?, ?)",
            (ref, bid),
        )

    # --- Feature 2: Extract and index atomic fact cards as sub-blocks ---
    statement = block.get("Statement", "")
    if statement and len(statement) > 15:
        block_date = block.get("Date", "")
        try:
            facts = extract_facts(statement, speaker=speaker, date=block_date, source_id=bid)
        except (ValueError, TypeError):
            facts = []
        for i, card in enumerate(facts):
            fact_id = f"{bid}::F{i + 1}"
            fact_tags = card.get("type", "FACT")
            if card.get("speaker"):
                fact_tags += f", {card['speaker']}"
            fact_block = {
                "Statement": card["content"],
                "Tags": fact_tags,
                "Date": card.get("date", block_date),
                "Status": block.get("Status", "active"),
                "_id": fact_id,
                "_parent_id": bid,
            }
            conn.execute(
                """INSERT OR REPLACE INTO blocks
                   (id, type, file, line, status, date, speaker, tags, dia_id, parent_id, json_blob)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact_id,
                    card.get("type", "FACT"),
                    rel_path,
                    block.get("_line", 0),
                    block.get("Status", "active"),
                    card.get("date", block_date),
                    card.get("speaker", ""),
                    fact_tags,
                    block.get("DiaID", ""),
                    bid,
                    json.dumps(fact_block, default=str),
                ),
            )
            fact_fts = _extract_fts_fields(fact_block)
            conn.execute(
                f"""INSERT INTO {fts_table} (block_id, statement, title, name,
                   description, tags, context, all_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",  # nosec B608 — see above: two literals, chosen by the admission verdict
                (
                    fact_id,
                    fact_fts["statement"],
                    fact_fts["title"],
                    fact_fts["name"],
                    fact_fts["description"],
                    fact_fts["tags"],
                    fact_fts["context"],
                    fact_fts["all_text"],
                ),
            )


def _delete_blocks(
    conn: sqlite3.Connection,
    block_ids: list,
    rel_path: str,
) -> None:
    """Delete blocks from blocks, blocks_fts, and xref_edges.

    Also deletes child fact sub-blocks (parent_id matching any deleted block).
    """
    if not block_ids:
        return
    placeholders = ",".join("?" for _ in block_ids)

    # Find child fact sub-blocks before deleting parents
    child_rows = conn.execute(
        f"SELECT id FROM blocks WHERE parent_id IN ({placeholders})",  # nosec B608 — placeholders is `? * N`, values passed as bind params
        block_ids,
    ).fetchall()
    child_ids = [r["id"] for r in child_rows]

    all_ids = block_ids + child_ids
    all_ph = ",".join("?" for _ in all_ids)

    conn.execute(f"DELETE FROM blocks WHERE id IN ({all_ph})", all_ids)  # nosec B608 — placeholders is `? * N`, values passed as bind params
    # Both FTS tables: a block moves between them when its admission verdict
    # flips, and a delete that only cleared one would leave the stale twin.
    conn.execute(f"DELETE FROM blocks_fts WHERE block_id IN ({all_ph})", all_ids)  # nosec B608 — same as above
    conn.execute(f"DELETE FROM blocks_fts_withheld WHERE block_id IN ({all_ph})", all_ids)  # nosec B608 — same as above
    conn.execute(
        f"DELETE FROM xref_edges WHERE src IN ({all_ph}) OR dst IN ({all_ph})",  # nosec B608 — same as above
        all_ids + all_ids,
    )


def _is_markdown_backend(workspace: str) -> bool:
    """Return True when *workspace*'s blocks of record live on the Markdown corpus.

    Defaults to the markdown corpus (the zero-config SQLite default) when
    no config / no ``block_store`` section is present. Never raises — a
    malformed config degrades to the markdown path so the default
    SQLite / Markdown experience is byte-for-byte unchanged. The
    enumeration helper :func:`mind_mem.storage.iter_active_blocks` owns
    the single source of truth for the same backend classification.
    """
    # Lazy import: avoids an import cycle (storage -> block_store -> ...)
    # at module-import time and matches the lazy ``from .storage import``
    # convention used elsewhere in the package.
    from .storage import _MARKDOWN_BACKENDS, _backend_name

    return _backend_name(workspace) in _MARKDOWN_BACKENDS


def _build_index_from_store(workspace: str, conn: sqlite3.Connection, start: datetime) -> dict:
    """Rebuild the FTS5 index from the configured (non-markdown) block store.

    For backends whose blocks of record live in the store (e.g. Postgres)
    the local Markdown corpus files are empty init templates, so the
    markdown-driven ``build_index`` path would index nothing (or worse,
    index stray template blocks — the ``sqlite_only_count`` drift the
    audit warns about). This path enumerates the store's active blocks
    via :func:`mind_mem.storage.iter_active_blocks` and reindexes them
    through the same :func:`_insert_block` machinery (FTS fields, xrefs,
    fact-cards) the markdown path uses, so recall sees the store-resident
    blocks out of the box.

    The store rebuild is always a full rebuild (the store is the source
    of truth; there is no per-file mtime to diff against), so the
    ``blocks``/``blocks_fts``/``xref_edges``/``index_meta`` tables are
    cleared first to drop any rows orphaned by deletes in the store.
    """
    from .storage import iter_active_blocks

    blocks = iter_active_blocks(workspace)

    # Full rebuild: clear all index state so deletes in the store
    # propagate and no markdown-template rows linger (drift fix).
    conn.execute("DELETE FROM blocks")
    conn.execute("DELETE FROM blocks_fts")
    conn.execute("DELETE FROM blocks_fts_withheld")
    conn.execute("DELETE FROM xref_edges")
    conn.execute("DELETE FROM index_meta")
    conn.execute("DELETE FROM file_state")

    all_block_ids = {b.get("_id", "") for b in blocks if b.get("_id")}

    # ADMISSION, on the store path too. ``iter_active_blocks`` filters on the
    # store's own notion of "active", which is not the egress allow-list, so
    # the gate runs here rather than being assumed upstream.
    releases = _release_set(workspace)
    admitted_ids = _admit_ids(
        [(b.get("_id", ""), b.get("Status")) for b in blocks if b.get("_id")],
        releases=releases,
    )

    indexed = 0
    for block in blocks:
        bid = block.get("_id", "")
        if not bid:
            continue
        rel_path = block.get("_source_file", "") or ""
        _insert_block(conn, block, bid, rel_path, all_block_ids, admitted=bid in admitted_ids)
        conn.execute(
            "INSERT OR REPLACE INTO index_meta (file_path, block_id, content_hash, admitted) VALUES (?, ?, ?, ?)",
            (rel_path, bid, _compute_block_hash(block), 1 if bid in admitted_ids else 0),
        )
        indexed += 1

    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("last_build", datetime.now().isoformat()),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("build_mode", "store-full"),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("release_set_hash", _release_set_hash(releases)),
    )
    conn.commit()

    elapsed = (datetime.now() - start).total_seconds() * 1000
    admitted_count, withheld_count = _indexed_block_counts(conn, workspace)
    summary = {
        "files_checked": 0,
        "files_indexed": 0,
        "blocks_indexed": indexed,
        "blocks_new": indexed,
        "blocks_modified": 0,
        "blocks_deleted": 0,
        "blocks_unchanged": 0,
        "elapsed_ms": round(elapsed, 1),
        "total_blocks": admitted_count + withheld_count,
        "blocks_admitted": admitted_count,
        "blocks_withheld": withheld_count,
        "source": "block_store",
    }
    _log.info("build_complete", **summary)
    metrics.inc("index_builds")
    metrics.inc("index_blocks_indexed", indexed)
    return summary


def build_index(workspace: str, incremental: bool = True) -> dict:
    """Build or incrementally update the FTS5 index.

    Uses ConnectionManager (#466) for write connection pooling with
    chunked commits (one commit per file) to reduce lock hold time.

    Backend-aware (audit bugs 4 & 9): for the default Markdown / SQLite
    backend, the corpus files are the blocks of record and are indexed
    incrementally via :data:`CORPUS_FILES` + :func:`parse_file`. For a
    non-markdown backend (e.g. Postgres) the blocks of record live in
    the configured store, so the index is rebuilt from
    :func:`mind_mem.storage.iter_active_blocks` instead — otherwise the
    PG user's blocks would be invisible to recall and stray markdown
    template rows would introduce ``sqlite_only_count`` drift.

    Args:
        workspace: Workspace root path.
        incremental: If True, only re-index changed files. If False, rebuild all.
            Ignored for non-markdown backends (a store rebuild is always full).

    Returns:
        Summary dict with files_checked, files_indexed, blocks_indexed, elapsed_ms.
    """
    ws = os.path.abspath(workspace)
    start = datetime.now()

    markdown_backend = _is_markdown_backend(workspace)

    mgr = _get_conn_manager(workspace)
    with mgr.write_lock:
        conn = mgr.get_write_connection()
        conn.row_factory = sqlite3.Row
        try:
            _init_schema(conn)

            if not markdown_backend:
                return _build_index_from_store(workspace, conn, start)

            # Collect all block IDs for xref resolution
            all_block_ids = set()
            for label, rel_path in CORPUS_FILES.items():
                path = os.path.join(ws, rel_path)
                if os.path.isfile(path):
                    try:
                        for b in parse_file(path):
                            bid = b.get("_id", "")
                            if bid:
                                all_block_ids.add(bid)
                    except (OSError, UnicodeDecodeError, ValueError) as e:
                        _log.debug("xref_scan_parse_failed", file=rel_path, error=str(e))

            # The governance release set decides admissibility alongside the
            # block's own status, so it is resolved ONCE per pass and threaded
            # into every file.
            releases = _release_set(workspace)
            release_hash = _release_set_hash(releases)

            force = not incremental
            if incremental:
                changed = _get_changed_files(conn, workspace)
                # A release names ids in decisions/DECISIONS.md but ADMITS
                # blocks that live in another file — memory/INBOX.md, say —
                # whose bytes never moved. Change detection is per file, so
                # without this the released block's file is never revisited
                # and the admission flip never reaches ``blocks_fts``. Rare
                # (a release is a governed act) and cheap: every file is
                # re-parsed, but only the blocks whose verdict actually
                # flipped are rewritten.
                prior = conn.execute("SELECT value FROM meta WHERE key = 'release_set_hash'").fetchone()
                if (prior["value"] if prior else None) != release_hash:
                    _log.info("release_set_changed", workspace=ws, files=len(CORPUS_FILES))
                    changed = list(CORPUS_FILES.items())
            else:
                changed = list(CORPUS_FILES.items())
                # Clear the index for a full rebuild — the SAME set of tables
                # ``_build_index_from_store`` clears, which is where this list
                # comes from. Clearing only file_state/index_meta made every
                # block "new" while its old rows survived, and ``blocks_fts``
                # has no primary key to absorb the second copy: three full
                # rebuilds of a one-block corpus left blocks=1 and
                # blocks_fts=3 (measured). Duplicated documents inflate the
                # bm25 document count and the length average — the same
                # corpus-statistics distortion this module's admission split
                # exists to prevent, arriving from the other direction — and
                # they double up in the result list. Stale rows for blocks
                # since deleted from the corpus were never dropped either.
                conn.execute("DELETE FROM file_state")
                conn.execute("DELETE FROM index_meta")
                conn.execute("DELETE FROM blocks")
                conn.execute("DELETE FROM blocks_fts")
                conn.execute("DELETE FROM blocks_fts_withheld")
                conn.execute("DELETE FROM xref_edges")

            total_blocks = 0
            total_new = 0
            total_modified = 0
            total_deleted = 0
            total_unchanged = 0
            for label, rel_path in changed:
                counts = _index_file(conn, workspace, label, rel_path, all_block_ids, force=force, releases=releases)
                total_blocks += counts["total"]
                total_new += counts["new"]
                total_modified += counts["modified"]
                total_deleted += counts["deleted"]
                total_unchanged += counts["unchanged"]
                # Chunked commit: commit after each file to reduce lock hold time
                conn.commit()
                _log.info(
                    "indexed_file",
                    file=rel_path,
                    new=counts["new"],
                    modified=counts["modified"],
                    deleted=counts["deleted"],
                    unchanged=counts["unchanged"],
                )

            # Update metadata
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("last_build", datetime.now().isoformat()),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("build_mode", "incremental" if incremental else "full"),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("release_set_hash", release_hash),
            )

            conn.commit()

            elapsed = (datetime.now() - start).total_seconds() * 1000
            summary = {
                "files_checked": len(CORPUS_FILES),
                "files_indexed": len(changed),
                "blocks_indexed": total_blocks,
                "blocks_new": total_new,
                "blocks_modified": total_modified,
                "blocks_deleted": total_deleted,
                "blocks_unchanged": total_unchanged,
                "elapsed_ms": round(elapsed, 1),
            }

            # Count the blocks in the index, and SAY WHICH SET each number
            # counts. The bare ``SELECT COUNT(*) FROM blocks`` that used to
            # stand here is the same statistic ``index_status`` serves, and
            # it answered differently: admission-blind, so it moved by one
            # every time a quarantined block was indexed. Two counters in
            # one module disagreeing about "how many blocks?" is how the
            # egress count drifted in the first place, so both now come
            # from :func:`_indexed_block_counts` — the one counting
            # authority, which puts every status to the shared allow-list.
            #
            # ``total_blocks`` stays the WHOLE index on purpose: this is the
            # indexer reporting the work it did, an operator-side number,
            # and hiding the withheld rows from it would make the builder
            # lie about what it indexed. What changes is that the number is
            # no longer ambiguous — ``blocks_admitted`` is the egress-safe
            # count a caller should propagate to any served surface, and
            # ``blocks_withheld`` is the difference, stated rather than
            # buried. Both keys are additive; a reader that knows only
            # ``total_blocks`` still finds it, with the same meaning.
            admitted_count, withheld_count = _indexed_block_counts(conn, workspace)
            summary["total_blocks"] = admitted_count + withheld_count
            summary["blocks_admitted"] = admitted_count
            summary["blocks_withheld"] = withheld_count
        finally:
            # Don't close the manager — it's shared and cached
            pass

    _log.info("build_complete", **summary)
    metrics.inc("index_builds")
    metrics.inc("index_blocks_indexed", total_blocks)
    return summary


# ---------------------------------------------------------------------------
# Fact aggregation — small-to-big retrieval
# ---------------------------------------------------------------------------


def _aggregate_facts_to_parents(
    conn: sqlite3.Connection,
    results: list[dict],
    workspace: str | None = None,
) -> list[dict]:
    """Merge fact sub-block scores into their parent blocks.

    Fact sub-blocks have IDs like "D-20230507-001::F1". This function:
    1. Groups fact sub-blocks by parent ID (everything before "::F")
    2. Boosts parent score by the best fact sub-block score
    3. Removes fact sub-blocks from results (folded into parent)
    4. If a parent is not already in results, fetches it from the DB

    Step 4 reads ``blocks``, which carries the WHOLE corpus including
    what admission withholds, so the injected parents run through the
    same allow-list on the way out. A fact card whose parent is withheld
    cannot normally reach here (it inherits the parent's verdict and so
    is absent from ``blocks_fts``), but an index whose cached status has
    gone stale is exactly the case a backstop is for.

    Returns results with fact sub-blocks replaced by boosted parents.
    """
    if not results:
        return results

    # Separate fact sub-blocks and regular blocks
    fact_scores: dict[str, float] = {}  # parent_id -> max fact score
    regular = []
    result_ids = set()

    for r in results:
        rid = r.get("_id", "")
        if "::F" in rid:
            parent_id = rid.split("::F")[0]
            score = r.get("score", 0)
            if score > fact_scores.get(parent_id, 0):
                fact_scores[parent_id] = score
        else:
            regular.append(r)
            result_ids.add(rid)

    if not fact_scores:
        return results

    # Boost parents already in results
    boosted = set()
    for r in regular:
        rid = r.get("_id", "")
        if rid in fact_scores:
            fact_sc = fact_scores[rid]
            parent_sc = r.get("score", 0)
            r["score"] = round(max(parent_sc, fact_sc * 0.8 + parent_sc * 0.2), 4)
            r["_fact_boost"] = True
            boosted.add(rid)

    # Inject parents not in results (fact card matched but parent didn't)
    missing = set(fact_scores.keys()) - result_ids
    if missing:
        placeholders = ",".join("?" for _ in missing)
        rows = conn.execute(
            f"SELECT * FROM blocks WHERE id IN ({placeholders}) AND parent_id = ''",  # nosec B608 — placeholders is `? * N`, values passed as bind params
            list(missing),
        ).fetchall()
        admitted = _admit_ids([(r["id"], r["status"]) for r in rows], workspace=workspace)
        for row in rows:
            if row["id"] not in admitted:
                continue
            block_data = json.loads(row["json_blob"]) if row["json_blob"] else {}
            regular.append(
                {
                    "_id": row["id"],
                    "type": row["type"],
                    "score": round(fact_scores[row["id"]] * 0.8, 4),
                    "excerpt": get_excerpt(block_data),
                    "speaker": row["speaker"],
                    "tags": row["tags"],
                    "file": row["file"],
                    "line": row["line"],
                    "status": row["status"],
                    "_fact_boost": True,
                }
            )

    _log.debug(
        "fact_aggregation",
        facts_found=sum(1 for r in results if "::F" in r.get("_id", "")),
        parents_boosted=len(boosted),
        parents_injected=len(missing),
    )
    return regular


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query_index(
    workspace: str,
    query: str,
    limit: int = 10,
    active_only: bool = False,
    graph_boost: bool = False,
    retrieve_wide_k: int = 200,
    rerank: bool = True,
    rerank_debug: bool = False,
    scoring_instant: date | None = None,
) -> list[dict]:
    """Query the FTS5 index. Returns ranked results matching recall() format.

    Falls back to filesystem scan (recall.recall()) if index doesn't exist.

    ``scoring_instant`` is the UTC date the recency ramp and the calibration
    window score against — the same seam ``recall()`` carries, threaded here so
    the FTS5 leg is reproducible on exactly the same terms as the scan leg
    rather than reading its own clock. ``None`` resolves to today in UTC.

    .. warning::
       Admitted **as of the last index pass**, not as of now. ``blocks_fts``
       carries the admitted set only (see the admission section at the top of
       this module), so a block withheld when the index was built contributes
       neither a candidate nor a byte of ``bm25()`` corpus statistics. What
       this function cannot see is a status that changed *since* that pass:
       the ``status`` column is a cache, and it goes stale in the fail-OPEN
       direction. Callers must still route these rows through
       ``_recall_core._withhold_inadmissible`` (which refreshes from
       ``admissibility.live_statuses`` first), exactly as the legs before
       fusion in ``hybrid_recall`` and the recall funnels do.
    """
    # Re-entrancy guard: if this thread is already inside query_index for the
    # same workspace (can happen when the index-missing fallback calls recall()
    # which — with backend="sqlite" — would immediately call query_index again),
    # return an empty list instead of recursing infinitely.
    _active: set | None = getattr(_query_index_active, "workspaces", None)
    if _active is None:
        _active = set()
        _query_index_active.workspaces = _active
    if workspace in _active:
        _log.error(
            "query_index_reentrant_call_blocked",
            workspace=workspace,
            hint="sqlite index missing and recall() re-entered query_index; returning []",
        )
        return []

    db_path = _db_path(workspace)
    if not os.path.isfile(db_path):
        _log.info("index_missing_fallback", db=db_path)
        from .recall import recall

        _active.add(workspace)
        try:
            return recall(
                workspace,
                query,
                limit=limit,
                active_only=active_only,
                graph_boost=graph_boost,
                retrieve_wide_k=retrieve_wide_k,
                rerank=rerank,
                rerank_debug=rerank_debug,
                # A fallback that drops the instant would silently score
                # against "today" and break replay on exactly the paths a
                # caller cannot see — the leak this seam exists to close.
                scoring_instant=scoring_instant,
            )
        finally:
            _active.discard(workspace)

    # Initialize calibration manager (optional — graceful degradation)
    _cal_mgr = None
    try:
        from .calibration import CalibrationManager

        _cal_mgr = CalibrationManager(workspace)
    except (ImportError, Exception) as _cal_err:
        _log.debug("calibration_unavailable_in_fts", error=str(_cal_err))

    # Staleness check: warn but don't auto-rebuild (#34)
    if is_stale(workspace):
        _log.info("index_stale", hint="Run 'reindex' tool to update the FTS5 index")
        metrics.inc("index_stale_queries")

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    query_type = detect_query_type(query)
    qparams: dict = _QUERY_TYPE_PARAMS.get(query_type, _QUERY_TYPE_PARAMS["single-hop"])  # type: ignore[assignment]

    # Month normalization
    query_tokens = expand_months(query, query_tokens)

    expand_mode = qparams.get("expand_query", True)
    if expand_mode:
        mode = expand_mode if isinstance(expand_mode, str) else "full"
        query_tokens = expand_query(query_tokens, mode=mode)

    if qparams.get("graph_boost_override", False):
        graph_boost = True

    mgr = _get_conn_manager(workspace)
    conn = mgr.get_read_connection()
    conn.row_factory = sqlite3.Row

    # Build FTS5 MATCH query from tokens
    # Quote each token to prevent FTS5 operator injection (NOT, AND, NEAR, etc.)
    # Also reject tokens that aren't alphanumeric to prevent wildcard injection (e.g. "*")
    fts_query = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in query_tokens if _FTS5_SAFE.match(t))
    if not fts_query:
        return []

    try:
        # FTS5 bm25() returns negative scores (lower = better)
        # We negate to get positive scores (higher = better)
        #
        # bm25() weights must align to the blocks_fts columns
        # (block_id first, then FTS5_COLUMNS); see _bm25_weights().
        # BM25 statistics come from the PREBUILT FTS5 index, and no WHERE
        # clause narrows what bm25() averages over — so index membership IS
        # the statistics decision. ``blocks_fts`` therefore carries the
        # admitted set only, which puts this leg's IDF and length-average on
        # the same corpus the in-memory scan leg computes over.
        #
        # Residual, stated honestly: membership is decided at index time, so
        # a block quarantined AFTER the last pass still contributes until the
        # next one. That window is the ordinary stale-index window — it is
        # detected by ``is_stale`` (logged as ``index_stale``, counted as
        # ``index_stale_queries``, both a few lines above), it closes on any
        # incremental build, and the CONTENT of such a block is withheld
        # throughout by the live-status refresh in
        # ``_recall_core._withhold_inadmissible``. What remains inside it is
        # a second-order effect on the ranking of admitted documents.
        weights = _bm25_weights()
        rows = conn.execute(
            f"""SELECT b.*, f.rank as fts_rank,
                       -bm25(blocks_fts, {weights}) as bm25_score
                FROM blocks_fts f
                JOIN blocks b ON b.id = f.block_id
                WHERE blocks_fts MATCH ?
                ORDER BY bm25_score DESC
                LIMIT ?""",  # nosec B608 — `weights` is a comma-separated list of floats derived from FTS5_COLUMNS (a static constant), never from user input
            (fts_query, max(retrieve_wide_k, limit)),
        ).fetchall()
        rows = list(rows) + _released_withheld_rows(conn, workspace, fts_query, weights, max(retrieve_wide_k, limit))
    except sqlite3.OperationalError as e:
        _log.warning(
            "fts_query_error_fallback",
            error=str(e),
            query=fts_query,
            msg="FTS5 query failed, falling back to in-memory BM25 scan",
        )
        # Fallback to filesystem scan — results are still valid but may be slower
        from .recall import recall

        fallback_results = recall(
            workspace,
            query,
            limit=limit,
            active_only=active_only,
            graph_boost=graph_boost,
            retrieve_wide_k=retrieve_wide_k,
            rerank=rerank,
            rerank_debug=rerank_debug,
            scoring_instant=scoring_instant,
        )
        for r in fallback_results:
            r["_fallback"] = "bm25_scan"
        return fallback_results

    _scoring_moment = as_utc_datetime(resolve_scoring_instant(scoring_instant))

    results = []
    for row in rows:
        if active_only and row["status"] not in {"active", TaskStatus.TODO, TaskStatus.DOING, "open"}:
            continue

        score = row["bm25_score"]

        # Apply same post-processing as recall.py
        # Recency boost
        block_data = json.loads(row["json_blob"]) if row["json_blob"] else {}
        recency = date_score(block_data, now=_scoring_moment)
        rw = qparams.get("recency_weight", 0.3)
        score *= 1.0 - rw + rw * recency

        # Temporal date boost
        date_boost = qparams.get("date_boost", 1.0)
        if date_boost > 1.0 and row["date"]:
            score *= date_boost

        # Status boost
        if row["status"] == "active":
            score *= 1.2
        elif row["status"] in {TaskStatus.TODO, TaskStatus.DOING}:
            score *= 1.1

        # Priority boost
        priority = block_data.get("Priority", "")
        if priority in ("P0", "P1"):
            score *= 1.1

        # Calibration feedback weight
        if _cal_mgr is not None:
            try:
                cal_weight = _cal_mgr.get_block_weight(row["id"], now=_scoring_moment)
                score *= cal_weight
            except Exception as exc:
                _log.debug("calibration_weight_skipped", block_id=row["id"], error=str(exc))  # graceful degradation; non-fatal

        result = {
            "_id": row["id"],
            "type": row["type"],
            "score": round(score, 4),
            "excerpt": get_excerpt(block_data),
            "speaker": row["speaker"],
            "tags": row["tags"],
            "file": row["file"],
            "line": row["line"],
            "status": row["status"],
        }
        if row["dia_id"]:
            result["DiaID"] = row["dia_id"]
        if row["date"]:
            result["Date"] = row["date"]
        # Pass through provenance fields (Group E) when present
        for _prov_field in PROVENANCE_FIELD_NAMES:
            if block_data.get(_prov_field):
                result[_prov_field] = block_data[_prov_field]
        results.append(result)

    # --- Feature 2: Aggregate fact sub-blocks to parents (small-to-big) ---
    results = _aggregate_facts_to_parents(conn, results, workspace)

    # Graph boost
    if graph_boost and results:
        _apply_graph_boost(conn, results, query_type, workspace)

    # Note: read connection is managed by ConnectionManager — not closed here (#466)

    # Sort by score, then by block ID for deterministic tiebreaking
    results.sort(key=lambda r: (-r["score"], r.get("_id", "")))

    # Dedup
    seen_keys = set()
    deduped = []
    for r in results:
        stable_key = (r.get("file", ""), r.get("line", 0))
        if stable_key != ("", 0) and stable_key in seen_keys:
            continue
        if stable_key != ("", 0):
            seen_keys.add(stable_key)
        dia = r.get("DiaID", "")
        if dia:
            rid = r.get("_id", "")
            prefix = "FACT" if rid.startswith("FACT-") else "DIA" if rid.startswith("DIA-") else rid[:4]
            dia_key = (dia, prefix)
            if dia_key in seen_keys:
                continue
            seen_keys.add(dia_key)
        deduped.append(r)

    # Rerank — cap candidates to prevent latency spikes (#9)
    if rerank and len(deduped) > limit:
        rerank_cap = min(len(deduped), 200)
        deduped = rerank_hits(query, deduped[:rerank_cap], debug=rerank_debug)

    top = deduped[:limit]

    _log.info(
        "query_complete",
        query=query,
        query_type=query_type,
        fts_hits=len(rows),
        results=len(top),
        top_score=top[0]["score"] if top else 0,
    )
    metrics.inc("index_queries")
    return top


def _released_withheld_rows(
    conn: sqlite3.Connection,
    workspace: str,
    fts_query: str,
    weights: str,
    limit: int,
) -> list:
    """Hits for blocks a governance RELEASE has admitted since the last index pass.

    A release names ids in ``decisions/DECISIONS.md``; it does not touch
    the file holding the released block, and it must take effect with no
    reindex — the index anchor is attested and a release must not churn
    it. Those blocks live in ``blocks_fts_withheld``, so this is where
    they come back from.

    They are matched in their own table, which is the point: a withheld
    document must not be a document of the admitted corpus, whatever the
    release set later says, or the corpus statistics would leak it in the
    window before the release. The consequence is that a just-released
    block is scored against the withheld pool until the next index pass
    moves it across — a transient scoring difference for the released
    block alone, never a perturbation of the admitted blocks' scores.

    Costs nothing when nothing is withheld: one indexed probe of an empty
    table, and the release set (a ``stat``, cached) is resolved only if
    that probe finds something. An index built before this table existed
    raises ``OperationalError``, which is caught here rather than
    escaping into the caller's full-scan fallback — such an index still
    carries withheld blocks in ``blocks_fts``, where the release path
    worked without any help from this function.
    """
    try:
        if conn.execute("SELECT 1 FROM blocks_fts_withheld LIMIT 1").fetchone() is None:
            return []
        released = _release_set(workspace)
        if not released:
            return []
        ids = sorted(released)
        placeholders = ",".join("?" for _ in ids)
        return list(
            conn.execute(
                f"""SELECT b.*, f.rank as fts_rank,
                           -bm25(blocks_fts_withheld, {weights}) as bm25_score
                    FROM blocks_fts_withheld f
                    JOIN blocks b ON b.id = f.block_id
                    WHERE blocks_fts_withheld MATCH ? AND b.id IN ({placeholders})
                    ORDER BY bm25_score DESC
                    LIMIT ?""",  # nosec B608 — `weights` is floats from the static FTS5_COLUMNS; `placeholders` is `? * N` with every id bound
                [fts_query, *ids, limit],
            ).fetchall()
        )
    except sqlite3.OperationalError as exc:
        _log.debug("withheld_release_probe_skipped", error=str(exc))
        return []


def _apply_graph_boost(
    conn: sqlite3.Connection,
    results: list[dict],
    query_type: str,
    workspace: str | None = None,
) -> None:
    """Apply cross-reference graph boost to results using xref_edges table.

    ``xref_edges`` spans the whole corpus, withheld blocks included — that
    is deliberate, so a release does not have to rebuild the graph. It
    also means an unfiltered traversal would let a quarantined block both
    RECEIVE a score (and be injected into the result list with its
    excerpt) and PASS one on to its admitted neighbours. The second half
    is the subtler leak: it survives any downstream content filter,
    because what it moves is the ranking of blocks that are allowed to be
    served. So the allow-list runs on every edge destination before a
    boost is computed.
    """
    from .recall import GRAPH_BOOST_FACTOR

    score_by_id = {r["_id"]: r["score"] for r in results}
    result_ids = set(score_by_id.keys())

    hop_decays = [GRAPH_BOOST_FACTOR, GRAPH_BOOST_FACTOR * 0.5]
    if query_type == "multi-hop":
        hop_decays.append(GRAPH_BOOST_FACTOR * 0.25)

    neighbor_scores: dict[str, float] = {}

    for hop, decay in enumerate(hop_decays):
        seed_ids = list(result_ids) if hop == 0 else [nid for nid in neighbor_scores if nid not in result_ids]
        if not seed_ids:
            break

        # Sanitize: only allow string IDs matching block ID format
        seed_ids = [sid for sid in seed_ids if isinstance(sid, str) and len(sid) < 100]
        if not seed_ids:
            break

        placeholders = ",".join("?" for _ in seed_ids)
        edge_rows = conn.execute(
            f"""SELECT e.src AS src, e.dst AS dst, b.status AS dst_status
                FROM xref_edges e JOIN blocks b ON b.id = e.dst
                WHERE e.src IN ({placeholders})""",  # nosec B608 — placeholders is `? * N`, values passed as bind params; seed_ids sanitized above
            seed_ids,
        ).fetchall()
        # ADMISSION. Destinations only: a seed is either an FTS hit (already
        # admitted, since blocks_fts carries the admitted set) or a neighbour
        # this same filter let through on an earlier hop.
        admitted_dst = _admit_ids([(e["dst"], e["dst_status"]) for e in edge_rows], workspace=workspace)
        edges = [e for e in edge_rows if e["dst"] in admitted_dst]

        hop_added = 0
        for edge in edges:
            src, dst = edge["src"], edge["dst"]
            src_score: float = score_by_id.get(src, neighbor_scores.get(src, 0)) or 0.0
            boost = src_score * decay
            if dst not in result_ids:
                if hop_added >= 50:  # Cap neighbors per hop
                    break
                neighbor_scores[dst] = neighbor_scores.get(dst, 0) + boost
                hop_added += 1
            else:
                neighbor_scores[dst] = neighbor_scores.get(dst, 0) + boost * 0.5

    # Apply boosts to existing results
    for r in results:
        if r["_id"] in neighbor_scores:
            r["score"] = round(r["score"] + neighbor_scores[r["_id"]], 4)
            r["via_graph"] = True

    # Add new graph-discovered results. No second admission pass: every id in
    # ``neighbor_scores`` arrived as an edge destination the allow-list above
    # already cleared, off the same ``blocks.status`` column this SELECT reads.
    if neighbor_scores:
        new_ids = [nid for nid in neighbor_scores if nid not in result_ids]
        if new_ids:
            placeholders = ",".join("?" for _ in new_ids)
            rows = conn.execute(
                f"SELECT * FROM blocks WHERE id IN ({placeholders})",  # nosec B608 — placeholders is `? * N`, values passed as bind params
                new_ids,
            ).fetchall()
            for row in rows:
                block_data = json.loads(row["json_blob"]) if row["json_blob"] else {}
                results.append(
                    {
                        "_id": row["id"],
                        "type": row["type"],
                        "score": round(neighbor_scores[row["id"]], 4),
                        "excerpt": get_excerpt(block_data),
                        "speaker": row["speaker"],
                        "tags": row["tags"],
                        "file": row["file"],
                        "line": row["line"],
                        "status": row["status"],
                        "via_graph": True,
                    }
                )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def is_stale(workspace: str) -> bool:
    """Check whether any corpus .md files have changed since last index build.

    Returns True if the index doesn't exist or any file mtime differs from
    the recorded state.  This is a lightweight O(files) check that avoids
    hashing -- suitable for a pre-query gate.
    """
    db = _db_path(workspace)
    if not os.path.isfile(db):
        return True
    try:
        conn = _connect(workspace, readonly=True)
        changed = _get_changed_files(conn, workspace)
        conn.close()
        return len(changed) > 0
    except (OSError, sqlite3.OperationalError):
        return True


def merkle_leaves(workspace: str) -> list[tuple[str, str]]:
    """Return (block_id, content_hash) tuples for Merkle tree construction.

    Used by :func:`mcp_server.verify_merkle` and the standalone
    ``mind-mem-verify`` CLI. Leaves are sorted by block_id so two calls
    against the same index produce the same tree (stable root hash).

    Content hashes live on the ``index_meta`` table (``blocks`` itself
    doesn't carry one — it has ``json_blob`` and friends). The join
    below preserves the invariant that a leaf is only emitted when a
    live ``blocks`` row backs the ``index_meta`` content hash.

    Returns an empty list when the FTS index has not yet been built.
    """
    db_path = _db_path(workspace)
    if not os.path.isfile(db_path):
        return []
    conn = None
    try:
        conn = _connect(workspace, readonly=True)
        # Read-only safe (audit bugs 13 & 14): never run _init_schema on a
        # mode=ro connection. When recall.db exists but the FTS schema has
        # not been built (side-tables only, or a PG-backed workspace whose
        # local cache was never populated), there are no leaves yet.
        if not _index_schema_present(conn):
            return []
        rows = conn.execute(
            """
            SELECT im.block_id AS block_id, im.content_hash AS content_hash
            FROM index_meta im
            JOIN blocks b ON b.id = im.block_id
            WHERE im.content_hash IS NOT NULL AND im.content_hash != ''
            ORDER BY im.block_id
            """
        ).fetchall()
        # De-dupe when the same block_id appears in multiple files.
        seen: set[str] = set()
        leaves: list[tuple[str, str]] = []
        for r in rows:
            bid = r["block_id"]
            if bid in seen:
                continue
            seen.add(bid)
            leaves.append((bid, r["content_hash"]))
        return leaves
    finally:
        if conn is not None:
            conn.close()


def _indexed_block_counts(conn: sqlite3.Connection, workspace: str) -> tuple[int, int]:
    """``(admitted, withheld)`` row counts over the index's ``blocks`` table.

    Read-only: two SELECTs, no DDL, safe on a ``mode=ro`` connection.

    The status vocabulary is not restated here — the distinct statuses are
    grouped in SQL and each is put to :func:`admissibility.is_admissible_status`,
    so a tier that mints a new withheld status is counted as withheld with
    no edit in this module. The release set is read only when something
    actually failed the allow-list, so an all-admitted index pays no probe.
    """
    groups = conn.execute("SELECT status AS status, COUNT(*) AS cnt FROM blocks GROUP BY status").fetchall()
    total = sum(g["cnt"] for g in groups)
    withheld_statuses = [g["status"] for g in groups if not is_admissible_status(g["status"])]
    if not withheld_statuses:
        return total, 0
    placeholders = ",".join("?" for _ in withheld_statuses)
    rows = conn.execute(
        f"SELECT id FROM blocks WHERE status IN ({placeholders})",  # nosec B608 — placeholders is `? * N`, values passed as bind params
        withheld_statuses,
    ).fetchall()
    releases = _release_set(workspace)
    withheld = sum(1 for r in rows if r["id"] not in releases)
    return total - withheld, withheld


def index_status(workspace: str, *, include_withheld: bool = False) -> dict:
    """Return index status: exists, block count, last build time, staleness.

    ``blocks`` counts the **admitted** rows. It used to be
    ``SELECT COUNT(*) FROM blocks``, which moved by one every time a
    quarantined block was indexed — and this number is served to user
    scope by the ``index_stats`` and ``memory_health`` MCP tools, so the
    existence of withheld content was readable straight off a statistic
    by a caller the gate refuses to show that content to. A count that
    tracks withheld blocks is the same disclosure as the blocks.

    *include_withheld* is the governed widening, matching the store read
    seam: an ADMIN caller that is entitled to know asks for it by name
    and gets ``blocks`` over the whole index plus an explicit
    ``withheld`` key. It is never a default, and there is no call site
    for it in the default surface.

    Read-only safe (audit bugs 13 & 14). ``recall.db`` is frequently
    created by side-tables (calibration, retrieval_log) *before*
    ``build_index`` ever runs the FTS schema, and on the Postgres
    backend the FTS schema is often never built locally. In both cases
    the ``blocks``/``meta`` tables are absent. This function opens the DB
    ``mode=ro`` and therefore must never run ``_init_schema`` (a
    ``CREATE TABLE``), which would raise
    ``OperationalError: attempt to write a readonly database``. Instead
    it probes ``sqlite_master`` and reports ``blocks=0`` when the index
    schema has not been built yet.
    """
    db_path = _db_path(workspace)
    if not os.path.isfile(db_path):
        return {"exists": False, "blocks": 0, "stale_files": len(CORPUS_FILES)}

    conn = None
    try:
        conn = _connect(workspace, readonly=True)
        # Do NOT call _init_schema here — the connection is read-only.
        # When recall.db exists but the index schema has not been built
        # (only side-tables present), report "built but empty" without
        # attempting any DDL.
        if not _index_schema_present(conn):
            return {
                "exists": True,
                "blocks": 0,
                "last_build": None,
                "stale_files": len(CORPUS_FILES),
                "db_size_bytes": os.path.getsize(db_path),
                "schema_built": False,
            }

        admitted_count, withheld_count = _indexed_block_counts(conn, workspace)

        last_build = conn.execute("SELECT value FROM meta WHERE key = 'last_build'").fetchone()

        changed = _get_changed_files(conn, workspace)

        status: dict = {
            "exists": True,
            "blocks": admitted_count + withheld_count if include_withheld else admitted_count,
            "last_build": last_build["value"] if last_build else None,
            "stale_files": len(changed),
            "db_size_bytes": os.path.getsize(db_path),
        }
        if include_withheld:
            status["withheld"] = withheld_count
        return status
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Mind Mem SQLite FTS5 Index")
    sub = parser.add_subparsers(dest="command")

    bp = sub.add_parser("build", help="Build or update the index")
    bp.add_argument("--workspace", "-w", default=".", help="Workspace path")
    bp.add_argument("--full", action="store_true", help="Full rebuild (not incremental)")

    qp = sub.add_parser("query", help="Query the index")
    qp.add_argument("--workspace", "-w", default=".", help="Workspace path")
    qp.add_argument("--query", "-q", required=True, help="Search query")
    qp.add_argument("--limit", "-l", type=int, default=10, help="Max results")
    qp.add_argument("--active-only", action="store_true")
    qp.add_argument("--graph", action="store_true")
    qp.add_argument("--json", action="store_true")
    qp.add_argument("--no-rerank", action="store_true")
    qp.add_argument("--rerank-debug", action="store_true")

    sp = sub.add_parser("status", help="Show index status")
    sp.add_argument("--workspace", "-w", default=".", help="Workspace path")

    args = parser.parse_args()

    if args.command == "build":
        ws = os.path.abspath(args.workspace)
        result = build_index(ws, incremental=not args.full)
        print("Index build complete:")
        print(f"  Files checked: {result['files_checked']}")
        print(f"  Files indexed: {result['files_indexed']}")
        print(
            f"  Blocks: {result['blocks_new']} new, {result['blocks_modified']} modified, "
            f"{result['blocks_deleted']} deleted, {result['blocks_unchanged']} unchanged"
        )
        print(f"  Total blocks: {result['total_blocks']}")
        print(f"  Elapsed: {result['elapsed_ms']:.0f}ms")

    elif args.command == "query":
        ws = os.path.abspath(args.workspace)
        results = query_index(
            ws,
            args.query,
            limit=args.limit,
            active_only=args.active_only,
            graph_boost=args.graph,
            rerank=not args.no_rerank,
            rerank_debug=args.rerank_debug,
        )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            if not results:
                print("No results found.")
            else:
                for r in results:
                    graph_tag = " [graph]" if r.get("via_graph") else ""
                    print(f"[{r['score']:.3f}] {r['_id']} ({r['type']}{graph_tag}) — {r['excerpt'][:80]}")
                    print(f"        {r['file']}:{r['line']}")

    elif args.command == "status":
        ws = os.path.abspath(args.workspace)
        status = index_status(ws)
        if not status["exists"]:
            print("No index found. Run 'build' to create one.")
        else:
            print("Index status:")
            print(f"  Blocks: {status['blocks']}")
            print(f"  Last build: {status['last_build']}")
            print(f"  Stale files: {status['stale_files']}")
            print(f"  DB size: {status['db_size_bytes']} bytes")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
