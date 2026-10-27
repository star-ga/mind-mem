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
import atexit
import contextlib
import functools
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from ._conn_lease import ConnLeases
from ._recall_constants import _STOPWORDS
from .admissibility import is_admissible_status, workspace_release_ids
from .block_parser import canonical_day, parse_file
from .block_provenance import PROVENANCE_FIELD_NAMES
from .connection_manager import ConnectionManager
from .corpus_registry import corpus_label_for, discover_corpus_files
from .enums import TaskStatus
from .extractor import extract_facts
from .observability import get_logger, metrics
from .recall import (
    _BLOCK_ID_RE,
    _QUERY_TYPE_PARAMS,
    SEARCH_FIELDS,
    _parse_speaker_from_tags,
    date_score,
    detect_query_type,
    expand_months,
    expand_query,
    get_block_type,
    get_excerpt,
    normalise_tags,
    rerank_hits,
    resolve_expand_mode,
    tokenize,
)
from .recall_supersede import supersede_targets
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


#: Bumped when the FTS table layout changes in a way a legacy database cannot
#: be migrated into incrementally. Stored in ``meta``; a mismatch forces one
#: full rebuild. v2 moved FACT sub-blocks out of ``blocks_fts`` into their own
#: ``blocks_fts_facts`` surface so a derived row stops reweighting its parent.
#: v4 indexes every namespace recall can serve (``shared/``, ``agents/<id>/``,
#: declared custom namespaces), each on its own statistics surface, and the
#: ``supersede_edges`` table recall resolves supersession chains from.
_FTS_SCHEMA_VERSION = "4"


#: How many blocks one dialogue may contribute before later ones are held
#: behind blocks from other dialogues. 0 disables the cap entirely.
#:
#: 1 was chosen by a stratified paired measurement, not by argument -- 12
#: questions from each of LongMemEval-S's six types, the same questions in the
#: same order under each setting:
#:
#:     cap   any@5    all@5    distinct sources/answer
#:     0     91.7%    72.2%    2.96
#:     2     91.7%    75.0%    3.69
#:     1     93.1%    81.9%    5.00
#:
#: The all@5 column is where it matters, because a multi-session question needs
#: several gold sessions on one page and any@5 cannot see that: multi-session
#: went 3/12 to 7/12, temporal-reasoning 9/12 to 10/12, knowledge-update 10/12
#: to 11/12. Overridable so the value stays a measurement.
_MAX_PER_DIALOGUE = int(os.environ.get("MIND_MEM_MAX_PER_DIALOGUE", "1"))


def _dialogue_group(result: dict) -> str | None:
    """The conversation a result belongs to, or None when it belongs to none.

    Uses the product's own ``DiaID`` grammar -- ``D{session}:{turn}`` -- and
    keys on the SESSION half, which is what makes two turns of one conversation
    group together. Returns None when a block carries no DiaID, so a corpus
    that is not conversational is completely unaffected: no DiaID, no grouping,
    no cap.
    """
    dia = result.get("DiaID", "")
    if not dia or not isinstance(dia, str):
        return None
    session = dia.split(":", 1)[0].strip()
    return session or None


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


_leases = ConnLeases()
_pinned = threading.local()


def _evict_conn_managers_locked() -> list[ConnectionManager]:
    """Drop least-recently-used managers past the bound and return them.

    Caller must hold ``_conn_managers_lock`` and, after releasing it, pass each
    victim to ``_leases.close_or_retire`` (which defers the close of a manager
    that is mid-query rather than closing it under the running statement).
    """
    victims: list[ConnectionManager] = []
    while len(_conn_managers) > _CONN_MANAGER_CACHE_MAX:
        victims.append(_conn_managers.pop(next(iter(_conn_managers))))
    return victims


def close_all_conn_managers() -> None:
    """Close and drop every cached manager (teardown / atexit)."""
    with _conn_managers_lock:
        victims = list(_conn_managers.values())
        _conn_managers.clear()
    for mgr in victims:
        _leases.close_or_retire(mgr)


atexit.register(close_all_conn_managers)


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
    pinned: ConnectionManager | None = getattr(_pinned, "mgrs", {}).get(path)
    if pinned is not None and os.path.isfile(path):
        return pinned  # this thread holds a lease on it; do not swap it mid-query
    stale: list[ConnectionManager] = []
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
            # (Retired, not closed outright, if a query is still using it.)
            stale.append(mgr)
            mgr = None
        if mgr is None:
            mgr = ConnectionManager(path)
            _conn_managers[path] = mgr
        else:
            # Touch: most-recently-used goes to the end, so eviction takes
            # the genuinely cold entries rather than whichever was created
            # first.
            _conn_managers[path] = _conn_managers.pop(path)
        stale.extend(_evict_conn_managers_locked())
    for victim in stale:
        _leases.close_or_retire(victim)
    return mgr


@contextlib.contextmanager
def leased_conn_manager(workspace: str) -> Iterator[ConnectionManager]:
    """Yield the workspace's manager, guaranteed not to be closed until exit.

    Use this around any raw ``get_read_connection()`` work: an LRU eviction
    that lands mid-query is deferred to the end of the block instead of
    closing the handle under the running statement (a C-layer use-after-free).
    """
    mgr = _get_conn_manager(workspace)
    path = mgr.db_path
    with _leases.leased(mgr):
        pins = getattr(_pinned, "mgrs", None)
        if pins is None:
            pins = _pinned.mgrs = {}
        prev = pins.get(path)
        pins[path] = mgr
        try:
            yield mgr
        finally:
            if prev is None:
                pins.pop(path, None)
            else:
                pins[path] = prev


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
            hash     TEXT NOT NULL,
            ctime_ns INTEGER,
            ino      INTEGER,
            observed_ns INTEGER
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
    # ``blocks_fts_facts`` / ``_withheld`` mirror the pair above for FACT
    # sub-blocks. They are a SEPARATE statistics surface, not a filing
    # convenience: bm25() draws N and the per-column length averages from the
    # table's own structure record, and no WHERE clause narrows that, so a
    # derived fact row sharing a table with its parent silently reweights every
    # parent in it. Measured on a session-shaped haystack, 334 fact cards beside
    # 48 parents -- 87% of the surface -- pulled the average document length far
    # under the parents' own, normalising each parent at roughly 7.7x avgdl.
    # A fact card is a derived VIEW of a document, not a document of the corpus.
    for _fts_table in (
        "blocks_fts",
        "blocks_fts_withheld",
        "blocks_fts_facts",
        "blocks_fts_facts_withheld",
    ):
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

    # Edit-detection migration: ``(size, mtime_ns)`` cannot see a same-size
    # rewrite whose mtime was put back, nor one that landed in the tick the
    # index observed the file in. ``ctime_ns`` + ``ino`` catch the first (the
    # kernel stamps ctime on every write and ``utime`` cannot set it back; an
    # atomic replace changes the inode); ``observed_ns`` scopes a content
    # hash to files racily close to their observation. Legacy NULL rows are
    # reported changed once, so the next build records the coordinate.
    for column in ("ctime_ns", "ino", "observed_ns"):
        try:
            conn.execute(f"ALTER TABLE file_state ADD COLUMN {column} INTEGER")  # nosec B608 — fixed column names
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Secondary indexes for common query patterns
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_parent_id ON blocks(parent_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_file ON blocks(file)")

    _init_namespace_schema(conn)
    conn.commit()


def _init_namespace_schema(conn: sqlite3.Connection) -> None:
    """Tables for namespace corpora and supersession edges (schema v4).

    ``ns_blocks`` is keyed by ``(file, id)``, not by id: the namespace layout
    keeps a shared copy of a root file under ``shared/<corpus-file>``, so one
    id legitimately exists in several namespaces. Each namespace gets its OWN
    FTS5 table (named in ``ns_fts_tables``) because ``bm25()`` statistics are
    per table and no WHERE clause narrows them: one table for every
    namespace would let an agent's private rows move the scores another
    caller sees. Only ADMITTED blocks are written here.

    ``supersede_edges`` holds one row per (superseder, named id) for every
    admitted superseder in every indexed file, so the query path resolves a
    chain with indexed lookups and never parses the corpus.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ns_blocks (
            id          TEXT NOT NULL,
            type        TEXT NOT NULL,
            file        TEXT NOT NULL,
            namespace   TEXT NOT NULL,
            line        INTEGER NOT NULL DEFAULT 0,
            status      TEXT NOT NULL DEFAULT '',
            date        TEXT NOT NULL DEFAULT '',
            speaker     TEXT NOT NULL DEFAULT '',
            tags        TEXT NOT NULL DEFAULT '',
            dia_id      TEXT NOT NULL DEFAULT '',
            parent_id   TEXT NOT NULL DEFAULT '',
            json_blob   TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (file, id)
        );
        CREATE INDEX IF NOT EXISTS idx_ns_blocks_id ON ns_blocks(id);
        CREATE INDEX IF NOT EXISTS idx_ns_blocks_namespace ON ns_blocks(namespace);

        CREATE TABLE IF NOT EXISTS ns_fts_tables (
            namespace TEXT PRIMARY KEY,
            fts_table TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS supersede_edges (
            superseder_id TEXT NOT NULL,
            target_id     TEXT NOT NULL,
            source_file   TEXT NOT NULL,
            namespace     TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (source_file, superseder_id, target_id)
        );
        CREATE INDEX IF NOT EXISTS idx_supersede_edges_target ON supersede_edges(target_id);
    """)


# ---------------------------------------------------------------------------
# File State Tracking
# ---------------------------------------------------------------------------


_FILE_HASH_CHUNK = 1 << 20  # 1 MiB

#: A file whose last change is within this window of the moment the index
#: observed it is "racily clean": a further write in the same timestamp tick
#: would leave size, mtime and ctime identical, so only its content can say
#: whether it changed. Two seconds covers the coarsest common timestamp
#: granularity (FAT); on ext4/xfs the tick is milliseconds. Only racy files
#: are hashed on the query path.
_RACY_WINDOW_NS = 2_000_000_000


def _file_hash(path: str) -> str:
    """Compute SHA-256 of the entire file (v3.7.0 M8).

    Pre-3.7.0 only hashed the first 64 KiB plus the file size, so a
    file whose size + mtime stayed identical but whose content past
    64 KiB changed (large markdown corpus rewritten in place; an
    editor that truncates-and-rewrites within the same second on
    coarse-mtime filesystems) was treated as unchanged and skipped on
    the next reindex. Reading the whole file is unconditionally
    correct.

    COST, stated because the previous version of this docstring said the
    opposite. The size + ``mtime_ns`` pre-filter in
    :func:`_get_changed_files` does NOT keep the steady state I/O-free: it
    short-circuits only when the metadata *differs*. A corpus where nothing
    changed falls through both cheap checks and reaches this function for
    EVERY file, so the unchanged path is the one that hashes the whole
    corpus. Measured on a 24 MB / 5,000-block workspace: 10.6 ms per
    ``is_stale`` call, ~20% of an indexed query. That is why
    :func:`_get_changed_files` now takes ``verify_hash`` and the query-side
    gate (:func:`is_stale`) passes ``False``; the build path keeps the full
    hash, unconditionally.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_FILE_HASH_CHUNK), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _get_changed_files(
    conn: sqlite3.Connection,
    workspace: str,
    *,
    verify_hash: bool = True,
) -> list[tuple[str, str]]:
    """Return list of (label, rel_path) for corpus files that changed since last index.

    A file is considered changed if:
    - It doesn't exist in file_state table
    - Its size differs from the recorded value
    - Its ``mtime_ns`` differs from the recorded value (or the
      recorded value is NULL — happens once per pre-3.7.0 row after
      the M8 migration, which forces a single rehash)
    - Its full SHA-256 differs (catches the same-size + same-mtime_ns
      case where an in-place edit kept the metadata stable) — this last
      check only when *verify_hash* is true

    ``verify_hash`` splits the two callers that were paying the same price
    for different needs. The BUILD path (:func:`build_index`) keeps the
    default ``True``: it is about to rewrite rows, it runs once per change,
    and a missed in-place edit there would leave a wrong index behind. The
    QUERY path (:func:`is_stale`) passes ``False``: it runs on every recall,
    it only decides whether to *log* staleness / resolve statuses live, and
    the full hash made it linear in corpus BYTES (10.6 ms on 24 MB, ~20% of
    an indexed query — see :func:`_file_hash`).

    RESIDUAL, closed. ``verify_hash=False`` used to report a same-size
    rewrite that kept its nanosecond mtime as unchanged until the next
    build. The row now also records ``ctime_ns`` and the inode (which a
    restored mtime or an atomic replace cannot hide) and the moment the
    file was observed; a file whose last change is within
    :data:`_RACY_WINDOW_NS` of that moment — where a same-tick write could
    leave every coordinate identical — is content-hashed even here. Files
    outside the window, i.e. the steady state, are still never hashed.

    Discovery is :func:`~mind_mem.corpus_registry.discover_corpus_files`
    (I-14): the corpus table PLUS every ``.md`` directly inside a corpus
    directory, which is what ``MarkdownBlockStore`` reads. Indexing the
    table alone left an unnamed corpus-directory block ``get_by_id``-
    readable and absent from every FTS hit.

    A table row that has left the disk is reported as changed by the loop
    below (its ``file_state`` row survives). An *unnamed* file that has
    left the disk cannot be — discovery no longer yields it at all — so
    :func:`_vanished_indexed_files` sweeps ``file_state`` for paths the
    walk no longer produces and reports those too. Without that sweep a
    deleted ``intelligence/ARCHIVE.md`` would keep serving its blocks out
    of ``blocks_fts`` forever.
    """
    changed = []
    ws = os.path.abspath(workspace)

    # Namespace corpora (shared/, agents/<id>/, declared custom) after the
    # root, so the root corpus is checked exactly as before.
    discovered = discover_corpus_files(ws) + _namespace_corpus_files(ws)
    for label, rel_path in discovered:
        full_path = os.path.join(ws, rel_path)
        if not os.path.isfile(full_path):
            # File doesn't exist — check if it was previously indexed
            row = conn.execute("SELECT path FROM file_state WHERE path = ?", (rel_path,)).fetchone()
            if row:
                changed.append((label, rel_path))  # file was deleted
            continue

        stat = os.stat(full_path)
        row = conn.execute(
            "SELECT mtime, mtime_ns, size, hash, ctime_ns, ino, observed_ns FROM file_state WHERE path = ?",
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

        # Same size + same mtime. ``ctime`` and the inode still move on a
        # rewrite whose mtime was restored, or on an atomic replace; a row
        # that predates these columns cannot vouch for the file at all.
        if row["ctime_ns"] is None or row["ino"] is None or row["observed_ns"] is None:
            changed.append((label, rel_path))
            continue
        if stat.st_ctime_ns != row["ctime_ns"] or stat.st_ino != row["ino"]:
            changed.append((label, rel_path))
            continue

        # Every coordinate matches. That proves nothing for a file whose
        # last change is racily close to the moment the index observed it:
        # a write in the same timestamp tick leaves size, mtime and ctime
        # all identical. Only such a file is content-hashed on the query
        # path, so the steady state (files older than the window) hashes
        # nothing — the cost the query path was rid of stays gone.
        racy = max(stat.st_mtime_ns, stat.st_ctime_ns) + _RACY_WINDOW_NS >= row["observed_ns"]
        if not verify_hash and not racy:
            continue
        current_hash = _file_hash(full_path)
        if current_hash != row["hash"]:
            changed.append((label, rel_path))

    changed.extend(_vanished_indexed_files(conn, {rel for _label, rel in discovered}))
    return changed


def _vanished_indexed_files(conn: sqlite3.Connection, discovered: set[str]) -> list[tuple[str, str]]:
    """``(label, rel_path)`` for indexed files discovery no longer produces.

    The deletion half of I-14's widened discovery. A file the corpus
    TABLE names is yielded whether or not it exists, so its removal is
    reported by the existence branch of :func:`_get_changed_files`. A
    file discovered by the *directory walk* is yielded only while it is
    on disk — delete it and the walk simply stops mentioning it, so
    without this sweep its rows sit in ``blocks``/``blocks_fts`` forever
    and recall keeps serving a file that no longer exists.

    Reported as changed rather than deleted here: ``_index_file`` already
    handles a missing path by dropping that file's blocks, so routing the
    vanished path back through it keeps one deletion implementation
    instead of a second one that has to stay in step.

    Sorted so a rebuild is deterministic.
    """
    rows = conn.execute("SELECT path FROM file_state").fetchall()
    stale = sorted({str(row["path"]) for row in rows} - discovered)
    return [(corpus_label_for(rel), rel) for rel in stale]


def _update_file_state(conn: sqlite3.Connection, workspace: str, rel_path: str) -> None:
    """Update file_state for a corpus file."""
    ws = os.path.abspath(workspace)
    full_path = os.path.join(ws, rel_path)

    if not os.path.isfile(full_path):
        conn.execute("DELETE FROM file_state WHERE path = ?", (rel_path,))
        return

    stat = os.stat(full_path)
    h = _file_hash(full_path)
    # Observed AFTER the hash: an edit that lands while hashing has an mtime
    # inside the racy window of this instant, so the query path re-hashes it.
    observed_ns = time.time_ns()
    conn.execute(
        "INSERT OR REPLACE INTO file_state (path, mtime, mtime_ns, size, hash, ctime_ns, ino, observed_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (rel_path, stat.st_mtime, stat.st_mtime_ns, stat.st_size, h, stat.st_ctime_ns, stat.st_ino, observed_ns),
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


def _as_text(value: object) -> str:
    """A field value as index text: list items one per line, anything else via str()."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _extract_fts_fields(block: dict) -> dict:
    """Extract FTS5 column values from a block.

    Every column is bound as text: a list-valued field (``Statement:\n- a\n- b``)
    is indexed one item per line instead of failing the whole build.
    """

    def field(*names: str) -> str:
        for name in names:
            text = _as_text(block.get(name))
            if text:
                return text
        return ""

    return {
        "statement": field("Statement"),
        "title": field("Title", "Name"),
        "name": field("Name"),
        "description": field("Description", "Summary"),
        "tags": normalise_tags(block.get("Tags", "") or block.get("Keywords", "")),
        "context": field("Context", "Rationale"),
        "all_text": " ".join(_as_text(block.get(f)) for f in SEARCH_FIELDS if block.get(f)),
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
        conn.execute("DELETE FROM supersede_edges WHERE source_file = ?", (rel_path,))
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

    # Supersede edges are a property of the whole file's admitted blocks, so
    # they are rewritten for the file every time the file is visited -- the
    # same pass, off the same parse, with no second read.
    _write_supersede_edges(conn, rel_path, "", [blk for bid, (blk, _hash) in current_blocks.items() if bid in admitted_ids])

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
    # ``normalise_tags`` and not ``block.get`` directly: the value is
    # bound as a SQLite parameter two statements down, and a
    # list-valued ``Tags`` -- which the block store renders and the
    # parser reads back as a list -- is not a bindable type.
    tags_str = normalise_tags(block.get("Tags", ""))
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
            canonical_day(block.get("Date", "")),
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
    statement = _as_text(block.get("Statement"))
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
                # NOT the parent's text.
                #
                # d487549 copied the whole parent statement into this card's
                # Context so a card would stay reachable by the language it was
                # extracted from -- a real problem: a turn reading "my internet
                # speed has been really good ... I upgraded to 500 Mbps" yields
                # the card "upgraded to 500 Mbps about three weeks ago", which
                # carries none of "speed", "internet" or "plan".
                #
                # Measured, that cure was far worse. On one LongMemEval haystack
                # the fact surface went from 14,426 indexed characters to
                # 448,559 (31x) and the database from 4 KB to 5.2 MB, because a
                # card's searchable text grew from 37 characters to 1,221 -- the
                # card became a COPY of its parent. Which is also why it stopped
                # working: a card that contains its parent's words can only
                # match when the parent already matched, so the small-to-big
                # injection branch below has nothing left to contribute. On a
                # 488-parent corpus injections went 338 to 0. No recall gain was
                # measured in exchange.
                #
                # The parent's language is already searchable on the parent
                # surface, which is queried separately. A card earns its place by
                # being SMALL and specific; making it big is the one thing that
                # cannot help.
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
                    canonical_day(card.get("date", block_date)),
                    card.get("speaker", ""),
                    fact_tags,
                    block.get("DiaID", ""),
                    bid,
                    json.dumps(fact_block, default=str),
                ),
            )
            fact_fts = _extract_fts_fields(fact_block)
            # The FACT surface, never the parent one -- see the table creation
            # comment: sharing it would put this derived row into the length
            # average and document count that score its own parent.
            fact_fts_table = f"{fts_table}_facts" if fts_table == "blocks_fts" else "blocks_fts_facts_withheld"
            conn.execute(
                f"""INSERT INTO {fact_fts_table} (block_id, statement, title, name,
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
    conn.execute(f"DELETE FROM blocks_fts_facts WHERE block_id IN ({all_ph})", all_ids)  # nosec B608 — same as above
    conn.execute(f"DELETE FROM blocks_fts_facts_withheld WHERE block_id IN ({all_ph})", all_ids)  # nosec B608 — same as above
    conn.execute(
        f"DELETE FROM xref_edges WHERE src IN ({all_ph}) OR dst IN ({all_ph})",  # nosec B608 — same as above
        all_ids + all_ids,
    )


# ---------------------------------------------------------------------------
# Namespace corpora (schema v4)
# ---------------------------------------------------------------------------


def _namespace_config(workspace: str) -> dict:
    """The workspace config recall reads (request-bound when one is bound)."""
    from ._recall_core import _get_config

    cfg = _get_config(workspace)
    return cfg if isinstance(cfg, dict) else {}


def _declared_custom(workspace: str, config: dict | None = None) -> tuple[str, ...]:
    from .namespace_retrieval import declared_custom_namespaces

    try:
        return declared_custom_namespaces(config if config is not None else _namespace_config(workspace))
    except ValueError:
        return ()


def _namespace_roots(workspace: str) -> list[str]:
    """Namespace roots besides the workspace root, in a fixed order.

    ``shared``, each valid ``agents/<id>`` directory, then each declared custom
    namespace -- the roots :func:`mind_mem.namespace_retrieval.admitted_namespace_blocks`
    walks. A root that is a symlink, or resolves anywhere but where it is
    named, is skipped: the same rule that function applies.
    """
    from .namespaces import InvalidAgentIdError, _validate_agent_id

    root = os.path.realpath(workspace)
    roots = ["shared"]
    agents = os.path.join(root, "agents")
    if not os.path.islink(agents) and os.path.isdir(agents):
        for name in sorted(os.listdir(agents)):
            try:
                _validate_agent_id(name)
            except InvalidAgentIdError:
                continue
            roots.append("agents/" + name)
    roots.extend(_declared_custom(workspace))
    kept: list[str] = []
    for namespace in dict.fromkeys(roots):
        path = os.path.join(root, *namespace.split("/"))
        if os.path.islink(path) or not os.path.isdir(path) or os.path.realpath(path) != path:
            continue
        kept.append(namespace)
    return kept


def _namespace_corpus_files(workspace: str) -> list[tuple[str, str]]:
    """``(label, rel_path)`` for every corpus file under a namespace root.

    Discovery under each root is :func:`discover_corpus_files`, so a
    namespace has exactly the corpus-file boundary the workspace root has.
    """
    root = os.path.realpath(workspace)
    out: list[tuple[str, str]] = []
    for namespace in _namespace_roots(workspace):
        base = os.path.join(root, *namespace.split("/"))
        for _label, local in discover_corpus_files(base):
            rel = namespace + "/" + local.replace(os.sep, "/")
            out.append((corpus_label_for(local), rel))
    return out


def _namespace_of(rel_path: str) -> str | None:
    """The namespace a non-root corpus path belongs to, ``None`` for the root corpus."""
    from .namespace_retrieval import namespace_for_path

    namespace = namespace_for_path(rel_path)
    if namespace is None or namespace == "workspace":
        return None
    return namespace


def _ns_fts_table_name(namespace: str) -> str:
    return "blocks_fts_ns_" + hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]


def _ns_fts_table(conn: sqlite3.Connection, namespace: str) -> str:
    """The FTS table for *namespace*, created and registered on first use."""
    row = conn.execute("SELECT fts_table FROM ns_fts_tables WHERE namespace = ?", (namespace,)).fetchone()
    if row is not None:
        return str(row[0])
    table = _ns_fts_table_name(namespace)
    cols = ", ".join(col for col, _ in FTS5_COLUMNS)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5(block_id, {cols}, file UNINDEXED, tokenize='porter unicode61')"  # nosec B608 — table is a fixed prefix plus a hex digest
    )
    conn.execute("INSERT INTO ns_fts_tables (namespace, fts_table) VALUES (?, ?)", (namespace, table))
    return table


def _clear_namespace_index(conn: sqlite3.Connection) -> None:
    """Drop every namespace row, FTS table and supersede edge (full rebuild)."""
    for row in conn.execute("SELECT fts_table FROM ns_fts_tables").fetchall():
        table = str(row[0])
        if re.fullmatch(r"blocks_fts_ns_[0-9a-f]{16}", table):
            conn.execute(f"DROP TABLE IF EXISTS {table}")  # nosec B608 — validated generated name
    conn.execute("DELETE FROM ns_fts_tables")
    conn.execute("DELETE FROM ns_blocks")
    conn.execute("DELETE FROM supersede_edges")


def _write_supersede_edges(conn: sqlite3.Connection, rel_path: str, namespace: str, blocks: list[dict]) -> None:
    """Replace *rel_path*'s supersede edges with those its ADMITTED *blocks* declare."""
    conn.execute("DELETE FROM supersede_edges WHERE source_file = ?", (rel_path,))
    for block in blocks:
        bid = block.get("_id")
        if not isinstance(bid, str) or not bid:
            continue
        for target in supersede_targets(block):
            conn.execute(
                "INSERT OR IGNORE INTO supersede_edges (superseder_id, target_id, source_file, namespace) VALUES (?, ?, ?, ?)",
                (bid, target, rel_path, namespace),
            )


def _index_namespace_file(conn: sqlite3.Connection, workspace: str, rel_path: str, releases: frozenset[str]) -> dict:
    """(Re)index one namespace corpus file: admitted blocks only, whole file at a time.

    Admission is :func:`mind_mem.admissibility.admit_corpus` -- the shared
    egress filter -- so a quarantined or pending block never becomes a row
    any caller can match, count or score against. Namespace files are small
    and change rarely, so the file is replaced wholesale rather than diffed
    block by block.
    """
    from .admissibility import admit_corpus

    namespace = _namespace_of(rel_path) or ""
    counts = {"new": 0, "modified": 0, "deleted": 0, "unchanged": 0, "total": 0}
    table = _ns_fts_table(conn, namespace)
    old = conn.execute("SELECT COUNT(*) FROM ns_blocks WHERE file = ?", (rel_path,)).fetchone()[0]
    conn.execute(f"DELETE FROM {table} WHERE file = ?", (rel_path,))  # nosec B608 — generated table name
    conn.execute("DELETE FROM ns_blocks WHERE file = ?", (rel_path,))
    conn.execute("DELETE FROM supersede_edges WHERE source_file = ?", (rel_path,))
    counts["deleted"] = int(old)

    root = os.path.realpath(workspace)
    full_path = os.path.join(root, *rel_path.split("/"))
    real = os.path.realpath(full_path)
    if os.path.islink(full_path) or not real.startswith(root + os.sep) or not os.path.isfile(real):
        _update_file_state(conn, workspace, rel_path)
        return counts
    try:
        parsed = parse_file(real)
    except (OSError, UnicodeDecodeError, ValueError):
        _update_file_state(conn, workspace, rel_path)
        return counts
    for block in parsed:
        block["_source_file"] = rel_path
    admitted = admit_corpus(parsed, workspace=workspace, releases=releases)

    seen: set[str] = set()
    kept: list[dict] = []
    for block in admitted:
        bid = block.get("_id")
        if not isinstance(bid, str) or not bid or bid in seen:
            continue
        seen.add(bid)
        kept.append(block)
        tags_str = normalise_tags(block.get("Tags", ""))
        conn.execute(
            """INSERT OR REPLACE INTO ns_blocks
               (id, type, file, namespace, line, status, date, speaker, tags, dia_id, parent_id, json_blob)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?)""",
            (
                bid,
                get_block_type(bid),
                rel_path,
                namespace,
                block.get("_line", 0),
                block.get("Status", ""),
                canonical_day(block.get("Date", "")),
                _parse_speaker_from_tags(tags_str),
                tags_str,
                block.get("DiaID", ""),
                json.dumps(block, default=str),
            ),
        )
        fts = _extract_fts_fields(block)
        conn.execute(
            f"""INSERT INTO {table} (block_id, statement, title, name, description, tags, context, all_text, file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",  # nosec B608 — generated table name
            (bid, fts["statement"], fts["title"], fts["name"], fts["description"], fts["tags"], fts["context"], fts["all_text"], rel_path),
        )
    # deferred: namespace blocks get no FACT sub-blocks or xref edges, so small-to-big
    # aggregation and graph boost stay root-only — upgrade path: give each
    # namespace a facts table beside its FTS table and key xref_edges by file.
    _write_supersede_edges(conn, rel_path, namespace, kept)
    counts["new"] = len(kept)
    counts["total"] = len(kept)
    _update_file_state(conn, workspace, rel_path)
    return counts


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
    conn.execute("DELETE FROM blocks_fts_facts")
    conn.execute("DELETE FROM blocks_fts_facts_withheld")
    conn.execute("DELETE FROM xref_edges")
    conn.execute("DELETE FROM index_meta")
    conn.execute("DELETE FROM file_state")
    _clear_namespace_index(conn)

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
        if bid in admitted_ids:
            for target in supersede_targets(block):
                conn.execute(
                    "INSERT OR IGNORE INTO supersede_edges (superseder_id, target_id, source_file, namespace) VALUES (?, ?, ?, '')",
                    (bid, target, rel_path),
                )
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
    incrementally via :func:`~mind_mem.corpus_registry.discover_corpus_files`
    + :func:`parse_file`. For a
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

            # Collect all block IDs for xref resolution. I-14: the same
            # discovery the store reads, so an xref that points into an
            # unnamed corpus-directory file resolves.
            corpus_rows = discover_corpus_files(ws)
            all_block_ids = set()
            for label, rel_path in corpus_rows:
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
            # A database built before the fact surface was split still holds
            # FACT rows inside ``blocks_fts``, where they go on distorting the
            # statistics of every parent scored against it. Nothing about a
            # file's bytes changes when the schema does, so incremental change
            # detection would never revisit them and the old layout would
            # survive indefinitely. One forced full rebuild moves them.
            _fts_ver = conn.execute("SELECT value FROM meta WHERE key = 'fts_schema_version'").fetchone()
            if (_fts_ver["value"] if _fts_ver else None) != _FTS_SCHEMA_VERSION:
                if incremental:
                    _log.info(
                        "fts_schema_migration",
                        workspace=ws,
                        from_version=(_fts_ver["value"] if _fts_ver else None),
                        to_version=_FTS_SCHEMA_VERSION,
                        msg="fact rows move to their own statistics surface; forcing one full rebuild",
                    )
                incremental = False

            ns_rows = _namespace_corpus_files(ws)

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
                    _log.info("release_set_changed", workspace=ws, files=len(corpus_rows))
                    changed = list(corpus_rows) + ns_rows
            else:
                changed = list(corpus_rows) + ns_rows
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
                conn.execute("DELETE FROM blocks_fts_facts")
                conn.execute("DELETE FROM blocks_fts_facts_withheld")
                conn.execute("DELETE FROM xref_edges")
                _clear_namespace_index(conn)

            total_blocks = 0
            total_new = 0
            total_modified = 0
            total_deleted = 0
            total_unchanged = 0
            for label, rel_path in changed:
                if _namespace_of(rel_path) is not None:
                    counts = _index_namespace_file(conn, workspace, rel_path, releases)
                else:
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
            # Stamped only after a successful build, so a run that dies partway
            # is retried rather than recorded as migrated.
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("fts_schema_version", _FTS_SCHEMA_VERSION),
            )

            conn.commit()

            elapsed = (datetime.now() - start).total_seconds() * 1000
            summary = {
                "files_checked": len(corpus_rows) + len(ns_rows),
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


#: Bounded tie-break applied to a parent that a fact card also matched.
#: Deliberately multiplicative-and-tiny rather than the old ``0.8f + 0.2p``
#: blend: once facts are scored on their own surface their numbers are on a
#: different scale entirely (a fact card is tens of characters, a session parent
#: thousands), so blending them re-imports the cross-scale error the split
#: exists to remove. This can reorder parents that are already near-tied on
#: their own surface and cannot move one past a materially better parent.
#: Overridable so the value can be chosen by a paired sweep rather than by
#: argument; the default is what that sweep selected.
_FACT_TIEBREAK = float(os.environ.get("MIND_MEM_FACT_TIEBREAK", "1.02"))


def _aggregate_facts_to_parents(
    conn: sqlite3.Connection,
    results: list[dict],
    workspace: str | None = None,
    *,
    fts_query: str | None = None,
    weights: str | None = None,
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
            parent_sc = r.get("score", 0)
            # A bounded tie-break on the parent's OWN score. The fact score is
            # deliberately not mixed in: it is measured against a different
            # corpus and comparing the two numerically is the error this split
            # removes.
            r["score"] = round(parent_sc * _FACT_TIEBREAK, 4)
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

        # Score the injected parents on the PARENT surface, not by rescaling a
        # fact score. Every term of a fact card is present in the parent it was
        # extracted from, so a parent reached this way always matches the same
        # query -- it simply fell outside the fetch window. Asking the parent
        # surface for its real bm25 keeps every score in this result set
        # commensurable; the old ``fact_score * 0.8`` inserted a number drawn
        # from a different corpus and ordered it against parent scores.
        parent_bm25: dict[str, float] = {}
        if fts_query and weights and admitted:
            ph = ",".join("?" for _ in admitted)
            try:
                for prow in conn.execute(
                    f"""SELECT block_id, -bm25(blocks_fts, {weights}) AS s
                        FROM blocks_fts
                        WHERE blocks_fts MATCH ? AND block_id IN ({ph})""",  # nosec B608 — `weights` derives from the static FTS5_COLUMNS; ids are bind params
                    (fts_query, *admitted),
                ).fetchall():
                    parent_bm25[prow["block_id"]] = float(prow["s"])
            except sqlite3.OperationalError:
                # A malformed MATCH here must not lose the injection; fall back
                # to the floor below rather than dropping the parent entirely.
                parent_bm25 = {}

        # Floor for a parent the surface could not score: rank it below every
        # parent that WAS scored, so an injection can never outrank a real hit.
        _floor = min(parent_bm25.values(), default=0.0) - 1e-6

        for row in rows:
            if row["id"] not in admitted:
                continue
            block_data = json.loads(row["json_blob"]) if row["json_blob"] else {}
            regular.append(
                {
                    "_id": row["id"],
                    "type": row["type"],
                    "score": round(parent_bm25.get(row["id"], _floor), 4),
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


# ---------------------------------------------------------------------------
# The caller's view of the index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CallerView:
    """What one caller may be served from the index, resolved once per query.

    ``root_files`` is ``None`` when every root-table row is readable (the
    unbound caller, or a bound one whose ACL covers every indexed root file),
    else the readable root files -- possibly none. ``namespaces`` lists the
    namespace FTS tables the caller reads, each with its readable files
    (``None`` = all of them). The SQL that fetches candidates is narrowed by
    exactly these, so an unreadable row never takes a fetch slot, a rerank
    slot or a score.
    """

    root_files: tuple[str, ...] | None
    namespaces: tuple[tuple[str, str, tuple[str, ...] | None], ...]

    def readable(self, namespace: str, file: str) -> bool:
        if not namespace:
            return self.root_files is None or file in self.root_files
        for name, _table, files in self.namespaces:
            if name == namespace:
                return files is None or file in files
        return False


_UNRESTRICTED_VIEW = _CallerView(None, ())


def _file_filter(files: tuple[str, ...] | None) -> tuple[str, tuple[str, ...]]:
    """SQL narrowing ``b.file`` to *files*; empty when unrestricted."""
    if files is None:
        return "", ()
    return " AND b.file IN (SELECT value FROM json_each(?))", (json.dumps(list(files)),)


def _ns_registry(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """``(namespace, fts_table)`` rows; empty on an index built before schema v4."""
    try:
        rows = conn.execute("SELECT namespace, fts_table FROM ns_fts_tables ORDER BY namespace").fetchall()
    except sqlite3.OperationalError:
        return []
    return [(str(r[0]), str(r[1])) for r in rows if re.fullmatch(r"blocks_fts_ns_[0-9a-f]{16}", str(r[1]))]


def _caller_view(conn: sqlite3.Connection, workspace: str, agent_id: str | None) -> _CallerView:
    """Resolve what *agent_id* may read from this index.

    UNBOUND (no principal): the root table, unnarrowed, plus every declared
    custom namespace -- the set the scan leg serves an unbound caller.
    ``shared/`` and ``agents/<id>/`` are served to a bound principal only,
    exactly as on the scan leg.

    BOUND: every indexed file the namespace ACL grants, decided by
    :func:`mind_mem._recall_core._indexed_hit_is_readable` (the lexical path
    AND its resolved real path), and kept only where the namespace's
    retrieval declaration allows search. An agent id the ACL authority
    rejects reads nothing.
    """
    from .namespace_retrieval import namespace_search_allowed

    registry = _ns_registry(conn)
    config = _namespace_config(workspace)
    if not agent_id:
        if not registry:
            return _UNRESTRICTED_VIEW
        custom = set(_declared_custom(workspace, config))
        custom_views = tuple(
            (name, table, None)
            for name, table in registry
            if name in custom and namespace_search_allowed(f"{name}/decisions/DECISIONS.md", config)
        )
        return _CallerView(None, custom_views)

    from ._recall_core import _indexed_hit_is_readable
    from .namespaces import InvalidAgentIdError, NamespaceManager

    try:
        manager = NamespaceManager(workspace, agent_id=agent_id)
    except InvalidAgentIdError:
        return _CallerView((), ())

    def _readable(file: str) -> bool:
        return _indexed_hit_is_readable(workspace, {"file": file}, manager)

    root_all = [str(r[0]) for r in conn.execute("SELECT DISTINCT file FROM blocks").fetchall()]
    root_ok = tuple(sorted(f for f in root_all if _readable(f)))
    root_files = None if len(root_ok) == len(set(root_all)) else root_ok
    namespaces: list[tuple[str, str, tuple[str, ...] | None]] = []
    for name, table in registry:
        files = [str(r[0]) for r in conn.execute("SELECT DISTINCT file FROM ns_blocks WHERE namespace = ?", (name,)).fetchall()]
        ok = tuple(sorted(f for f in files if _readable(f) and namespace_search_allowed(f, config)))
        if ok:
            namespaces.append((name, table, ok))
    return _CallerView(root_files, tuple(namespaces))


def _index_superseders_of(conn: sqlite3.Connection, view: _CallerView):  # type: ignore[no-untyped-def]
    """``superseders_of`` for :func:`resolve_heads`, answered from ``supersede_edges``.

    An edge counts only when the caller may read the file the superseder
    lives in. Edges were written from admitted superseders only.
    """

    def lookup(ids: list[str]) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            try:
                rows = conn.execute(
                    "SELECT target_id, superseder_id, source_file, namespace FROM supersede_edges "
                    "WHERE target_id IN (SELECT value FROM json_each(?))",
                    (json.dumps(chunk),),
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
            for row in rows:
                if view.readable(str(row[3] or ""), str(row[2])):
                    found.setdefault(str(row[0]), []).append(str(row[1]))
        return found

    return lookup


def _row_to_hit(row: sqlite3.Row, score: float) -> dict:
    """The hit shape :func:`query_index` returns, for a row pulled in by id."""
    block_data = json.loads(row["json_blob"]) if row["json_blob"] else {}
    hit = {
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
        hit["DiaID"] = row["dia_id"]
    if row["date"]:
        hit["Date"] = row["date"]
    return hit


def _index_head_fetcher(conn: sqlite3.Connection, workspace: str, view: _CallerView):  # type: ignore[no-untyped-def]
    """Fetch a chain head the query did not match, if the caller may read it and it is admitted."""

    def fetch(block_id: str) -> dict | None:
        try:
            rows = list(conn.execute("SELECT *, '' AS namespace FROM blocks WHERE id = ? AND parent_id = ''", (block_id,)).fetchall())
            rows += list(conn.execute("SELECT * FROM ns_blocks WHERE id = ? ORDER BY file", (block_id,)).fetchall())
        except sqlite3.OperationalError:
            return None
        for row in rows:
            if not view.readable(str(row["namespace"] or ""), str(row["file"])):
                continue
            if not _admit_ids([(row["id"], row["status"])], workspace=workspace):
                continue
            return _row_to_hit(row, 0.0)
        return None

    return fetch


def _apply_index_supersession(conn: sqlite3.Connection, workspace: str, hits: list[dict], view: _CallerView) -> list[dict]:
    """Put each retired hit's live chain head directly above it (see :mod:`mind_mem.recall_supersede`)."""
    if not hits:
        return hits
    from .recall_supersede import order_by_supersession, resolve_heads

    ids = [h["_id"] for h in hits if isinstance(h.get("_id"), str)]
    heads = resolve_heads(ids, _index_superseders_of(conn, view))
    if not heads:
        return hits
    return order_by_supersession(hits, heads, _index_head_fetcher(conn, workspace, view))


def _supersession_view(workspace: str, hits: list[dict], agent_id: str | None = None) -> list[dict]:
    """Apply index supersession to *hits* assembled by another leg (fusion, MCP).

    Answers from the index without parsing the corpus. With no index, or one
    built before schema v4, *hits* are returned unchanged.
    """
    if not hits or not _index_present(workspace):
        return hits
    try:
        with leased_conn_manager(workspace) as mgr:
            conn = mgr.get_read_connection()
            conn.row_factory = sqlite3.Row
            view = _caller_view(conn, workspace, agent_id)
            return _apply_index_supersession(conn, workspace, hits, view)
    except sqlite3.Error as exc:
        _log.debug("supersession_view_skipped", error=str(exc))
        return hits


def _query_index_impl(
    workspace: str,
    query: str,
    limit: int = 10,
    active_only: bool = False,
    graph_boost: bool = False,
    retrieve_wide_k: int = 200,
    rerank: bool = True,
    rerank_debug: bool = False,
    scoring_instant: date | None = None,
    *,
    since: str | None = None,
    until: str | None = None,
    return_k: int | None = None,
    agent_id: str | None = None,
) -> list[dict]:
    """Query the FTS5 index. Returns ranked results matching recall() format.

    ``agent_id`` is the caller's namespace principal. The index itself is not
    filtered by it (callers apply the ACL to the hits); it exists so that when
    this function falls back to the filesystem scan -- no index, or a database
    without the FTS schema -- the scan runs AS that principal and walks only
    the namespaces it may read. A fallback that dropped the principal answered
    a bound caller from the root corpus alone.

    Falls back to filesystem scan (recall.recall()) if index doesn't exist.

    ``scoring_instant`` is the UTC date the recency ramp and the calibration
    window score against — the same seam ``recall()`` carries, threaded here so
    the FTS5 leg is reproducible on exactly the same terms as the scan leg
    rather than reading its own clock. ``None`` resolves to today in UTC.

    ``since`` / ``until`` / ``return_k`` are this leg's half of the recall
    filter push-down. ``recall()``'s date bounds map onto a real column
    (``blocks.date``), so they ride the FTS5 ``SELECT`` itself and shape which
    candidates are generated at all. Its other three filters — lifecycle,
    event id, maturity — live inside ``json_blob`` with no column to test, so
    they cannot be pushed; ``return_k`` is the answer for those. It says how
    many ranked hits to hand back (``None`` = ``limit``, the historical
    behaviour), and ``recall()`` sets it to the wide retrieval width whenever
    any filter is set so that ``_apply_post_filters`` cuts to ``limit`` AFTER
    filtering rather than filtering a list already cut to ``limit``. The two
    together are why a filtered query stopped answering "nothing" when its
    matches happened to rank below the top-k.

    None of the three is set on an unfiltered query, and each is inert when
    unset: the emitted SQL, its bound parameters and the returned slice are
    then character-for-character the ones this function always used.

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
                # Same argument for the date bounds: the scan leg has its own
                # push-down, and a fallback that dropped them would answer a
                # bounded query from an unbounded pool.
                since=since,
                until=until,
                agent_id=agent_id,
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

    # Local import: `_recall_core` imports from this module, so a module-level
    # import here would close the cycle. `_get_config` is mtime-cached, so this
    # costs one stat per query, not a parse.
    from ._recall_core import _get_config

    expand_mode = resolve_expand_mode(qparams, _get_config(workspace).get("recall", {}))
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
    # The RAW query words go into MATCH alongside the processed tokens.
    #
    # This index is built with FTS5's real Porter stemmer
    # (tokenize='porter unicode61'), but ``tokenize()`` above has already run
    # mind-mem's own simplified stemmer over the query, so FTS5 then stems an
    # already-stemmed token a second time and the two token spaces stop
    # agreeing. Measured over 470 LongMemEval-S questions: 165 of 1592 distinct
    # query words land on a term the index does not contain, affecting 332 of
    # 4420 served query tokens. Two shapes, both from the same cause -- the
    # index space is Porter-over-raw-text with no lemma table:
    #   over-truncation   speed->spe, creamer->cream, sister->sist, denver->denv
    #   irregular lemma   bought->buy, got->get, spent->spend
    # Worked case: for "What speed is my new internet plan?", MATCH "speed"
    # returns the gold block at rank 1 while MATCH "spe" returns NOTHING.
    #
    # The raw words are ADDED rather than substituted. Dropping the processed
    # tokens outright measured net-positive on the affected questions but lost
    # one on an unaffected control, because the lemma and month expansions do
    # real work; a union keeps both token spaces reachable and lets bm25 decide.
    _raw_terms = [t for t in re.findall(r"[a-z0-9_]+", query.lower()) if t not in _STOPWORDS and len(t) > 1 and _FTS5_SAFE.match(t)]
    _match_terms: list[str] = []
    for _t in list(query_tokens) + _raw_terms:
        if _FTS5_SAFE.match(_t) and _t not in _match_terms:
            _match_terms.append(_t)
    fts_query = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in _match_terms)
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
        # admitted set only.
        #
        # It does NOT yet put this leg's IDF and length-average on the same
        # corpus the in-memory scan leg computes over, and an earlier version
        # of this comment claimed that it did. FACT sub-blocks are inserted
        # into this same table (see ``_insert_block`` above), while the
        # in-memory leg holds no reference to them at all — so the two legs
        # score against different corpora. Measured on a session-shaped
        # LongMemEval haystack: 48 parents alongside 334 fact cards, which
        # drags the table's average document length far below the parents'
        # own, so bm25 length-normalisation sees each parent as several times
        # avgdl. Whether the fact layer should get its own statistics surface
        # is settled for 5.1 under a paired scorecard; it is recorded here so
        # the comment stops asserting a property the code does not have.
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
        date_sql, date_params = _date_predicate(since, until)
        # Adaptive over-fetch. Widening only the RETURNED slice is not enough
        # when the filter the caller still has to apply is one this leg cannot
        # evaluate: the candidates it wants may never have been fetched. The
        # measured shape — 200 strongly-matching blocks that fail the filter
        # ahead of 30 weakly-matching blocks that pass it — puts every survivor
        # outside a 200-row fetch, so the widened slice comes back full of
        # rejects. The fetch itself therefore widens by the same predicate.
        #
        # Stated honestly, because it is a heuristic and not a guarantee: this
        # widens the window, it does not prove completeness. What IS exact is
        # ``date_sql`` — a bound this leg can evaluate is answered by the
        # engine, not by fetching more rows and hoping. The multiplier is
        # capped so a filtered query cannot turn into an unbounded scan.
        fetch_k = max(retrieve_wide_k, limit)
        if return_k is not None:
            fetch_k = min(max(fetch_k, return_k) * _FILTERED_OVERFETCH, _FILTERED_FETCH_CAP)
        # The caller's view narrows the candidate SQL itself. An unbound caller
        # on a root-only corpus gets an empty fragment, so the statement and
        # its parameters are exactly the historical ones.
        view = _caller_view(conn, workspace, agent_id)
        root_sql, root_params = _file_filter(view.root_files)
        rows = []
        fact_rows = []
        if view.root_files != ():
            rows = conn.execute(
                f"""SELECT b.*, f.rank as fts_rank,
                           -bm25(blocks_fts, {weights}) as bm25_score
                    FROM blocks_fts f
                    JOIN blocks b ON b.id = f.block_id
                    WHERE blocks_fts MATCH ?{date_sql}{root_sql}
                    ORDER BY bm25_score DESC
                    LIMIT ?""",  # nosec B608 — `weights` is a comma-separated list of floats derived from FTS5_COLUMNS (a static constant), never from user input; `date_sql` / `root_sql` are fixed fragments whose values are bound parameters
                (fts_query, *date_params, *root_params, fetch_k),
            ).fetchall()
            # The FACT surface is queried SEPARATELY so each side's bm25 is computed
            # over its own corpus. Fact scores are on a different scale from parent
            # scores by construction -- a fact card is tens of characters where a
            # session parent is thousands -- so they are never compared numerically
            # against parents. They identify WHICH parents matched; those parents are
            # then scored on the PARENT surface in _aggregate_facts_to_parents.
            fact_rows = conn.execute(
                f"""SELECT b.*, f.rank as fts_rank,
                           -bm25(blocks_fts_facts, {weights}) as bm25_score
                    FROM blocks_fts_facts f
                    JOIN blocks b ON b.id = f.block_id
                    WHERE blocks_fts_facts MATCH ?{date_sql}{root_sql}
                    ORDER BY bm25_score DESC
                    LIMIT ?""",  # nosec B608 -- same provenance as the parent query above
                (fts_query, *date_params, *root_params, fetch_k),
            ).fetchall()
            rows = _with_live_supplement(
                conn, "blocks_fts", rows, fts_query, weights, fetch_k, date_sql, date_params, root_sql, root_params
            )
            fact_rows = _with_live_supplement(
                conn, "blocks_fts_facts", fact_rows, fts_query, weights, fetch_k, date_sql, date_params, root_sql, root_params
            )
            rows = (
                list(rows)
                + list(fact_rows)
                + _released_withheld_rows(
                    conn,
                    workspace,
                    fts_query,
                    weights,
                    fetch_k,
                    date_sql=date_sql + root_sql,
                    date_params=(*date_params, *root_params),
                )
            )
        # Each namespace the caller reads, on its own statistics surface.
        for _namespace, ns_table, ns_files in view.namespaces:
            ns_sql, ns_params = _file_filter(ns_files)
            rows += conn.execute(
                f"""SELECT b.*, f.rank as fts_rank,
                           -bm25({ns_table}, {weights}) as bm25_score
                    FROM {ns_table} f
                    JOIN ns_blocks b ON b.file = f.file AND b.id = f.block_id
                    WHERE {ns_table} MATCH ?{date_sql}{ns_sql}
                    ORDER BY bm25_score DESC
                    LIMIT ?""",  # nosec B608 — ns_table is a validated generated name; weights / fragments as above
                (fts_query, *date_params, *ns_params, fetch_k),
            ).fetchall()
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
            since=since,
            until=until,
            agent_id=agent_id,
        )
        for r in fallback_results:
            r["_fallback"] = "bm25_scan"
        return fallback_results

    _scoring_moment = as_utc_datetime(resolve_scoring_instant(scoring_instant))

    # Calibration weights for the whole candidate list in one batched pass.
    # This was a statement PER ROW inside the loop below while the batch form
    # sat unused on the manager: 2.2 ms of the ~52 ms query at 200 candidates,
    # 58.3 ms at 5,000. Same rows, same ``_compute_weight``, same 1.0 for a
    # block with no feedback — latency only, ordering untouched. ``rows`` is
    # bounded by ``max(retrieve_wide_k, limit)``, and the helper chunks anyway.
    _cal_weights: dict[str, float] = {}
    if _cal_mgr is not None and rows:
        try:
            from ._recall_core import batch_calibration_weights

            _cal_weights = batch_calibration_weights(_cal_mgr, [row["id"] for row in rows], now=_scoring_moment)
        except Exception as exc:
            _log.debug("calibration_weight_skipped", blocks=len(rows), error=str(exc))  # graceful degradation; non-fatal

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

        # Calibration feedback weight (resolved above, in one batch). A row
        # absent from the map gets 1.0 — the same neutral multiplier the
        # per-row lookup returned for a block with no feedback, and the same
        # one it returned when the lookup itself failed.
        if _cal_weights:
            score *= _cal_weights.get(row["id"], 1.0)

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
        # These filter fields live in json_blob rather than dedicated SQLite
        # columns. Keep indexed hits aligned with the scan path so the shared
        # funnel does not treat known metadata as absent.
        for _filter_field in ("Lifecycle", "EventId", "Maturity"):
            if block_data.get(_filter_field):
                result[_filter_field] = block_data[_filter_field]
        # Pass through provenance fields (Group E) when present
        for _prov_field in PROVENANCE_FIELD_NAMES:
            if block_data.get(_prov_field):
                result[_prov_field] = block_data[_prov_field]
        results.append(result)

    # --- Feature 2: Aggregate fact sub-blocks to parents (small-to-big) ---
    results = _aggregate_facts_to_parents(conn, results, workspace, fts_query=fts_query, weights=weights)

    # Graph boost
    if graph_boost and results:
        _apply_graph_boost(conn, results, query_type, workspace, view=view)

    # Note: read connection is managed by ConnectionManager — not closed here (#466)

    # Sort by score, then by block ID for deterministic tiebreaking
    results.sort(key=lambda r: (-r["score"], r.get("_id", "")))

    # Dedup, then bound how much of the answer any ONE dialogue may occupy.
    #
    # The dedup below collapses exact repeats: same file+line, or the same
    # DiaID under the same id prefix. It cannot collapse two DIFFERENT turns of
    # one conversation, because a DiaID is ``D{session}:{turn}`` and those two
    # turns carry different DiaIDs -- which is correct for dedup and useless
    # for diversity. Measured on a per-turn-chunked conversational corpus, 157
    # of 300 top-5 slots went to turns of a dialogue already represented, and
    # 18 of 60 queries came back with every one of their five slots drawn from
    # a single conversation. A memory answer built from five fragments of one
    # conversation is worth much less than five sources, and the caller cannot
    # recover the sources that were crowded out.
    #
    # So a per-dialogue cap is applied AFTER ranking and BEFORE truncation:
    # the best-scoring blocks of a dialogue are kept in order, and once a
    # dialogue has contributed its allowance the rest are held back so a
    # different dialogue can be seen. Nothing is discarded that would otherwise
    # have been returned unless a lower-ranked block from another dialogue
    # takes its place.
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
    #
    # ``rerank_hits`` returns only what it was given, so this cap is also a
    # truncation of the pool. At the historical 200 it would have discarded
    # the whole over-fetched tail — the candidates a filtered query widened
    # the fetch to reach — before ``return_k`` ever got to slice it. The cap
    # therefore rises to the pool the caller asked for, and stays exactly 200
    # on every unfiltered query, which is every query that does not set
    # ``return_k``.
    if rerank and len(deduped) > limit:
        rerank_cap = min(len(deduped), 200 if return_k is None else max(200, return_k))
        # Dead-status rows do not count against the cap (see _live_aware_window).
        from ._recall_core import _live_aware_window

        deduped = rerank_hits(query, _live_aware_window(deduped, rerank_cap), debug=rerank_debug)

    # Bound how much of the answer any ONE dialogue may occupy.
    #
    # This runs AFTER the reranker on purpose. An earlier revision applied it
    # inside the dedup loop above, where it worked and was then silently undone:
    # rerank_hits reorders the whole pool it is handed, so held-back blocks
    # simply came back. Diversity is a property of the ORDER that is finally
    # returned, so it has to be the last thing applied to that order.
    #
    # Measured on a per-turn-chunked conversational corpus: 157 of 300 top-5
    # slots went to turns of a dialogue already represented, and 18 of 60
    # queries came back with all five slots drawn from ONE conversation. Five
    # fragments of one conversation are worth much less to a caller than five
    # sources, and the sources crowded out are unrecoverable.
    #
    # Nothing is discarded -- a capped block is moved behind the blocks that
    # earned a place, so a caller asking for more still receives it. A corpus
    # with no DiaID has no groups and is untouched.
    if _MAX_PER_DIALOGUE > 0:
        kept: list[dict] = []
        deferred: list[dict] = []
        per_dialogue: dict[str, int] = {}
        for r in deduped:
            group = _dialogue_group(r)
            if group is None:
                kept.append(r)
                continue
            n = per_dialogue.get(group, 0)
            if n >= _MAX_PER_DIALOGUE:
                deferred.append(r)
                continue
            per_dialogue[group] = n + 1
            kept.append(r)
        deduped = kept + deferred

    # ``return_k`` widens this slice — and only this slice — when the caller
    # has a filter left to apply. It never narrows: ``limit`` is the floor, so
    # a caller can only ever be handed at least what it asked for.
    # Superseded/revoked rows drop below every live row before the cut, so a
    # dead block that out-scores its replacement cannot take the replacement's
    # slot at a small ``limit``. The pool here is the wide ``fetch_k`` one.
    # Local import: ``_recall_core`` imports from this module.
    from ._recall_core import _rank_replaced_below_live

    deduped = _rank_replaced_below_live(deduped)
    # A block an admitted, caller-readable block supersedes keeps its slot,
    # with the live head of its chain directly above it -- pulled in when the
    # query did not match it -- so a superseded memory is never returned
    # instead of its superseder at any limit.
    deduped = _apply_index_supersession(conn, workspace, deduped, view)
    top = deduped[: limit if return_k is None else max(return_k, limit)]

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


@functools.wraps(_query_index_impl)
def query_index(workspace: str, *args: Any, **kwargs: Any) -> list[dict]:
    # Lease the workspace's manager for the whole query so an LRU eviction on
    # another thread cannot close the raw read connection under it.
    with leased_conn_manager(workspace):
        return _query_index_impl(workspace, *args, **kwargs)


#: How much wider the FTS fetch goes when the caller still holds a filter this
#: leg cannot evaluate in SQL (lifecycle / event id / maturity all live inside
#: ``json_blob``). Applied only when ``query_index`` is given a ``return_k``,
#: so an unfiltered query fetches exactly ``max(retrieve_wide_k, limit)`` rows
#: as it always has.
_FILTERED_OVERFETCH: int = 5

#: Hard ceiling on that widened fetch. Over-fetch trades latency for recall on
#: a filtered query; without a ceiling a small ``limit`` with a rare filter
#: would degenerate into scanning the index.
_FILTERED_FETCH_CAP: int = 5000


def _date_predicate(since: str | None, until: str | None) -> tuple[str, tuple[str, ...]]:
    """Compile ``recall``'s date bounds into a SQL fragment plus its parameters.

    The pushed-down twin of ``_recall_core._in_date_range``, and deliberately
    a transcription of it rather than an interpretation:

    * no bound at all → empty fragment, so the SQL is byte-for-byte the one
      an unfiltered query has always emitted;
    * a bound present but empty (``""``) is "open on that side" there, so it
      contributes no comparison here either;
    * a block with no date cannot satisfy a bounded query there, so
      ``b.date <> ''`` is emitted the moment either bound is supplied.

    The comparisons themselves line up because SQLite's default BINARY
    collation orders ISO-8601 text exactly as Python's ``<`` / ``>`` do. Both
    bounds travel as bound parameters; nothing from the caller is ever
    interpolated into the statement.
    """
    if since is None and until is None:
        return "", ()
    fragment = " AND b.date <> ''"
    params: list[str] = []
    if since:
        fragment += " AND b.date >= ?"
        params.append(since)
    if until:
        fragment += " AND b.date <= ?"
        params.append(until)
    return fragment, tuple(params)


def _with_live_supplement(
    conn: sqlite3.Connection,
    fts_table: str,
    rows: list[sqlite3.Row],
    fts_query: str,
    weights: str,
    fetch_k: int,
    date_sql: str,
    date_params: tuple[str, ...],
    file_sql: str = "",
    file_params: tuple[str, ...] = (),
) -> list[sqlite3.Row]:
    """*rows* plus the live rows a saturated fetch left out because dead rows filled it.

    The FTS fetch is cut at ``fetch_k`` before any demotion can run, so once
    ``fetch_k`` superseded/revoked rows out-score a live replacement the
    replacement is never fetched and no later demotion can bring it back.
    When the fetch came back full AND holds a dead row, a second fetch over
    live rows only adds every live row a live-only fetch of ``fetch_k`` would
    have held; rows already present are not repeated. An unsaturated fetch, or
    one with no dead row, is returned untouched (no extra query runs).
    """
    from ._recall_core import _REPLACED_STATUSES

    def _dead(row: sqlite3.Row) -> bool:
        return str(row["status"] or "").strip().lower() in _REPLACED_STATUSES

    if len(rows) < fetch_k or not any(_dead(r) for r in rows):
        return rows
    dead = sorted(_REPLACED_STATUSES)
    placeholders = ",".join("?" for _ in dead)
    live_rows = conn.execute(
        f"""SELECT b.*, f.rank as fts_rank,
                   -bm25({fts_table}, {weights}) as bm25_score
            FROM {fts_table} f
            JOIN blocks b ON b.id = f.block_id
            WHERE {fts_table} MATCH ?{date_sql}{file_sql}
              AND LOWER(TRIM(COALESCE(b.status, ''))) NOT IN ({placeholders})
            ORDER BY bm25_score DESC
            LIMIT ?""",  # nosec B608 -- fts_table is one of two module constants; weights/date_sql/file_sql as in query_index; values are bound
        (fts_query, *date_params, *file_params, *dead, fetch_k),
    ).fetchall()
    seen = {r["id"] for r in rows}
    return list(rows) + [r for r in live_rows if r["id"] not in seen]


def _released_withheld_rows(
    conn: sqlite3.Connection,
    workspace: str,
    fts_query: str,
    weights: str,
    limit: int,
    *,
    date_sql: str = "",
    date_params: tuple[str, ...] = (),
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
                    WHERE blocks_fts_withheld MATCH ? AND b.id IN ({placeholders}){date_sql}
                    ORDER BY bm25_score DESC
                    LIMIT ?""",  # nosec B608 — `weights` is floats from the static FTS5_COLUMNS; `placeholders` is `? * N` with every id bound; `date_sql` is a fixed fragment whose bounds are bound parameters
                [fts_query, *ids, *date_params, limit],
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
    *,
    view: _CallerView | None = None,
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
            f"""SELECT e.src AS src, e.dst AS dst, b.status AS dst_status, b.file AS dst_file
                FROM xref_edges e JOIN blocks b ON b.id = e.dst
                WHERE e.src IN ({placeholders})""",  # nosec B608 — placeholders is `? * N`, values passed as bind params; seed_ids sanitized above
            seed_ids,
        ).fetchall()
        # ADMISSION. Destinations only: a seed is either an FTS hit (already
        # admitted, since blocks_fts carries the admitted set) or a neighbour
        # this same filter let through on an earlier hop. The caller's view
        # too: a neighbour the caller cannot read neither receives nor passes
        # on a boost.
        admitted_dst = _admit_ids([(e["dst"], e["dst_status"]) for e in edge_rows], workspace=workspace)
        edges = [e for e in edge_rows if e["dst"] in admitted_dst and (view is None or view.readable("", str(e["dst_file"])))]

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


#: Workspaces this process has already tried to auto-build. A build that
#: fails (unreadable corpus, read-only volume, a lock we lost) must not be
#: retried on every subsequent query — "on first recall" is once, not always.
_AUTO_BUILD_ATTEMPTED: set[str] = set()
_AUTO_BUILD_LOCK = threading.Lock()


def _index_present(workspace: str) -> bool:
    """Does *workspace* have a usable FTS index (schema, not merely a file)?

    See :func:`ensure_index` for why the file is not the question.
    """
    if not os.path.isfile(_db_path(workspace)):
        return False
    conn = None
    try:
        conn = _connect(workspace, readonly=True)
        return _index_schema_present(conn)
    except (OSError, sqlite3.Error):
        return False
    finally:
        if conn is not None:
            conn.close()


def ensure_index(workspace: str, *, enabled: bool) -> dict | None:
    """Build the FTS5 index on first recall when it is absent. Opt-in.

    ``docs/performance-tuning.md`` has promised "The FTS5 index is built
    automatically on first recall" for as long as it has existed, and nothing
    in the package ever built one: a default install answered every query
    with the O(corpus) scan. Measured on a 24 MB / 5,000-block workspace,
    that is 1,928 ms per query against 51.9 ms indexed — 37x, paid by the
    first thing anyone evaluating this product does.

    WHY THIS IS OPT-IN, and must stay opt-in until row 9's paired scorecard
    runs. Building the index does not merely make the existing path faster —
    it CHANGES WHICH LEG SERVES. Both this module's caller
    (``hybrid_recall.HybridBackend._bm25_search_raw``) and the MCP ``recall``
    tool dispatch on "does recall.db exist", so creating one moves a default
    workspace from the in-memory BM25F scan to FTS5 bm25(), and those two
    rank differently. This project has already shipped a silently degraded
    recall path once. Latency wins are free; ranking changes are not, so the
    machinery lands here and the default does not move.

    Args:
        workspace: Workspace root.
        enabled: The resolved ``recall.auto_build_index`` flag. ``False``
            returns immediately — no ``stat``, no config read, no work — so
            the default path pays one boolean test for the whole feature.

    "ABSENT" is decided by the FTS SCHEMA, not by the file. ``recall.db`` is
    shared with :class:`~mind_mem.calibration.CalibrationManager`, whose
    constructor runs on the recall path and creates the file with a
    ``calibration_feedback`` table and nothing else — so on a default install
    the database exists from the first query while ``blocks``/``blocks_fts``
    never do, ``query_index`` raises ``OperationalError`` and silently falls
    back to the scan. A file-existence probe here would therefore report an
    index that is not there and build nothing, forever.

    Returns:
        The :func:`build_index` summary when a build ran, else ``None``
        (feature off, index already present, already attempted this process,
        or the build failed — the failure is logged, never raised: a recall
        must not fail because an optional index could not be created).
    """
    if not enabled:
        return None
    ws = os.path.abspath(workspace)
    if _index_present(ws):
        return None
    with _AUTO_BUILD_LOCK:
        if ws in _AUTO_BUILD_ATTEMPTED:
            return None
        _AUTO_BUILD_ATTEMPTED.add(ws)
    try:
        summary = build_index(ws, incremental=False)
    except Exception as exc:
        _log.warning("auto_build_index_failed", workspace=ws, error=str(exc))
        return None
    _log.info(
        "auto_build_index",
        workspace=ws,
        blocks_indexed=summary.get("blocks_indexed", 0),
        elapsed_ms=summary.get("elapsed_ms", 0),
        note="recall.auto_build_index is on; subsequent recalls take the FTS leg",
    )
    metrics.inc("recall_index_auto_built")
    return summary


def is_stale(workspace: str, *, verify_hash: bool = False) -> bool:
    """Check whether any corpus .md files have changed since last index build.

    Returns True if the index doesn't exist, or any corpus file's size or
    ``mtime_ns`` differs from the recorded state.

    This docstring used to claim the check "avoids hashing". It did not:
    the size + mtime pre-filter short-circuits only on a file that CHANGED,
    so a corpus where nothing changed — the steady state, and the state
    every query is in — fell through to a full SHA-256 of every corpus
    file. Measured at 10.6 ms on a 24 MB / 5,000-block workspace, ~20% of
    an indexed query, and linear in corpus bytes.

    It is now true by construction: ``verify_hash`` defaults to ``False``,
    so this really is an O(files) metadata check, and the full hash stays
    where it belongs — on the build path, which passes ``True``. A caller
    that wants the stronger (and much more expensive) verdict asks for it
    explicitly. See :func:`_get_changed_files` for the residual that
    default costs.
    """
    db = _db_path(workspace)
    if not os.path.isfile(db):
        return True
    try:
        conn = _connect(workspace, readonly=True)
        changed = _get_changed_files(conn, workspace, verify_hash=verify_hash)
        conn.close()
        return len(changed) > 0
    except (OSError, sqlite3.OperationalError):
        return True


def merkle_leaves(workspace: str, *, admitted_only: bool = False) -> list[tuple[str, str]]:
    """Return (block_id, content_hash) tuples for Merkle tree construction.

    *admitted_only* is for a caller that SHIPS the leaf set rather than
    attesting it — ``build_core`` packs these ids into a portable bundle.
    The integrity anchor (the default) covers every indexed block, withheld
    ones included, so it does not churn on a release; a bundle must carry
    only what recall may serve, so its rows go through the shared admission
    filter on the LIVE status (the index column is a cache that can be
    stale in the fail-open direction — see ``admissibility.live_statuses``).

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
            SELECT im.block_id AS block_id, im.content_hash AS content_hash, b.status AS status
            FROM index_meta im
            JOIN blocks b ON b.id = im.block_id
            WHERE im.content_hash IS NOT NULL AND im.content_hash != ''
            ORDER BY im.block_id
            """
        ).fetchall()
        admitted: set[str] | None = None
        if admitted_only:
            admitted = _admitted_leaf_ids(workspace, [(r["block_id"], r["status"]) for r in rows])
        # De-dupe when the same block_id appears in multiple files.
        seen: set[str] = set()
        leaves: list[tuple[str, str]] = []
        for r in rows:
            bid = r["block_id"]
            if bid in seen or (admitted is not None and bid not in admitted):
                continue
            seen.add(bid)
            leaves.append((bid, r["content_hash"]))
        return leaves
    finally:
        if conn is not None:
            conn.close()


def _admitted_leaf_ids(workspace: str, pairs: list[tuple[str, str]]) -> set[str]:
    """Ids among ``(block_id, cached status)`` *pairs* the shared filter admits.

    The cached ``status`` column is refreshed from the corpus when the index
    is stale, then the whole decision — allow-list, governed releases,
    revoked credentials — is :func:`admissibility.admit_workspace_blocks`'s.
    """
    from .admissibility import admit_workspace_blocks, live_statuses, with_live_statuses

    rows = with_live_statuses([{"_id": bid, "Status": status} for bid, status in pairs], live_statuses(workspace), status_key="Status")
    return {str(row["_id"]) for row in admit_workspace_blocks(rows, workspace=workspace)}


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
        return {"exists": False, "blocks": 0, "stale_files": len(discover_corpus_files(workspace))}

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
                "stale_files": len(discover_corpus_files(workspace)),
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
