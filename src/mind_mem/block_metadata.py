"""mind-mem A-MEM — auto-evolving block metadata.

Tracks access patterns, evolves keywords, and computes importance scores
for memory blocks. All data stored in SQLite block_meta table.

Uses ConnectionManager (#466) for connection pooling with read/write
separation — reads use thread-local connections, writes use a single
serialized connection, both in WAL mode.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone

from .block_provenance import PROVENANCE_FIELDS, clean_provenance_value
from .connection_manager import ConnectionManager
from .observability import get_logger

_log = get_logger("block_metadata")

#: Ids per ``IN (...)`` batch in :meth:`BlockMetadataManager.get_importance_boosts`.
#: SQLite's ``SQLITE_MAX_VARIABLE_NUMBER`` is 32766 on 3.32+ but 999 on older
#: builds; 900 stays under the oldest limit we can be compiled against.
_IMPORTANCE_BATCH_SIZE = 900

# Provenance columns (roadmap Group E + T-001) — snake_case, all nullable
# TEXT so the migration is purely additive: existing rows read back as NULL
# and existing DBs are upgraded in place via idempotent ALTER TABLE.
_PROVENANCE_COLUMNS: tuple[str, ...] = tuple(PROVENANCE_FIELDS.keys())

#: Workspace-relative directory holding this manager's SQLite store.
#: Created on first use by :class:`BlockMetadataManager` and scaffolded by
#: ``init_workspace`` -- it used to be neither, so the store was never
#: created on any workspace and every telemetry write silently went
#: nowhere.
BLOCK_META_DIR = ".mind-mem"

#: File name of the store inside :data:`BLOCK_META_DIR`.
BLOCK_META_FILENAME = "block_meta.db"

#: The ONE importance scale. ``importance`` is a *rerank multiplier*: recall
#: multiplies a block's score by it, so 1.0 is neutral, the floor damps and
#: the ceiling boosts. Every other consumer converts from this scale rather
#: than inventing its own -- the consolidation planner used to read this
#: column against a ``[0, 1]`` threshold of 0.25, a value the writer's clamp
#: makes unreachable, so the planner could never emit a single transition.
IMPORTANCE_FLOOR = 0.8
IMPORTANCE_CEILING = 1.5

#: Weights of the three importance terms. Named so the planner and the
#: writer cannot drift apart: both go through :func:`compute_importance`.
_FREQ_WEIGHT = 0.4
_RECENCY_WEIGHT = 0.4
_CONNECTION_WEIGHT = 0.2
_RAW_GAIN = 0.35

#: Recency score used when a block has no usable ``last_accessed``.
_NEUTRAL_RECENCY = 0.5


def block_meta_db_path(workspace: str) -> str:
    """Canonical location of the block-metadata store for *workspace*.

    Single source of truth for the path. Three different literals used to
    name three different files -- the recall writer wrote
    ``<ws>/.mind-mem/block_meta.db``, two MCP tools read
    ``<ws>/memory/block_meta.db``, and the consolidation planner read the
    ``block_meta`` table inside ``<ws>/.mind-mem-index/recall.db``. No two
    of them ever saw the same rows.
    """
    return os.path.join(workspace, BLOCK_META_DIR, BLOCK_META_FILENAME)


def compute_importance(
    *,
    access_count: int,
    last_accessed: str | None,
    connection_count: int,
    now: datetime,
    decay_days: int = 30,
) -> float:
    """Importance for one block, on the :data:`IMPORTANCE_FLOOR` scale.

    Pure: no clock, no DB, no I/O -- *now* is injected so a caller can
    reproduce a score exactly. This is the only definition of the formula;
    :meth:`BlockMetadataManager.update_importance` and the consolidation
    planner both call it, so a block's importance means the same thing on
    the ranking path and on the forgetting path.

    A block with no telemetry at all (never accessed, no connections)
    scores at the bottom of the band rather than in the middle: never
    having been read is evidence about value, and the previous reader-side
    default of 0.5 -- on a scale the writer cannot even reach -- is what
    made the planner's mark threshold unmeetable.
    """
    freq_score = math.log(max(access_count, 0) + 1)

    recency_score = _NEUTRAL_RECENCY
    if last_accessed:
        try:
            last_dt = datetime.fromisoformat(last_accessed)
            days_since = max((now - last_dt).total_seconds() / 86400, 0)
            recency_score = math.exp(-days_since / max(decay_days, 1))
        except (ValueError, TypeError):
            recency_score = _NEUTRAL_RECENCY

    conn_score = math.log(max(connection_count, 0) + 1)

    raw = _FREQ_WEIGHT * freq_score + _RECENCY_WEIGHT * recency_score + _CONNECTION_WEIGHT * conn_score
    return max(IMPORTANCE_FLOOR, min(IMPORTANCE_CEILING, IMPORTANCE_FLOOR + raw * _RAW_GAIN))


def keep_value(importance: float) -> float:
    """Convert a stored importance multiplier to a ``[0, 1]`` keep-value.

    :class:`mind_mem.cognitive_forget.BlockCognition` contracts on ``[0, 1]``
    and rejects anything outside it; this is the one conversion between that
    contract and the persisted scale. Linear and monotone, so the ordering
    the reranker sees and the ordering the planner sees are the same
    ordering.
    """
    span = IMPORTANCE_CEILING - IMPORTANCE_FLOOR
    return max(0.0, min(1.0, (float(importance) - IMPORTANCE_FLOOR) / span))


class BlockMetadataManager:
    """Tracks access patterns, evolves keywords, computes importance.

    Thread-safe: writes are serialized via ConnectionManager.write_lock,
    reads use per-thread connections (#32, #466).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS block_meta (
        id TEXT PRIMARY KEY,
        importance REAL DEFAULT 1.0,
        access_count INTEGER DEFAULT 0,
        last_accessed TEXT,
        keywords TEXT DEFAULT '',
        connections TEXT DEFAULT '',
        actor_id TEXT,
        actor_role TEXT,
        session_id TEXT,
        tool_id TEXT,
        purpose TEXT,
        content_source TEXT
    );
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_store_dir(db_path)
        self._conn_mgr = ConnectionManager(db_path)
        self._lock = threading.RLock()
        self._ensure_table()

    @staticmethod
    def _ensure_store_dir(db_path: str) -> None:
        """Create the store's directory on first use.

        ``init_workspace`` scaffolds :data:`BLOCK_META_DIR` for new
        workspaces, but no existing workspace will ever be re-inited, and
        every caller of this manager used to require the directory to
        already exist -- so on every workspace ever created the store was
        absent, the manager was never built, and ``record_access`` /
        ``evolve_keywords`` never ran once. Creating it here, at the layer
        that owns the file, is the same pattern ``sqlite_index._connect``
        and ``extraction_feedback._save`` already use for their own state
        directories.

        Degrades silently on an unwritable path: the rest of this manager
        answers neutrally when the DB is unavailable, and a constructor
        that raised where the old one did not would be a new failure mode
        rather than a fix.
        """
        directory = os.path.dirname(db_path)
        if not directory:
            return
        try:
            os.makedirs(directory, mode=0o700, exist_ok=True)
        except OSError:
            pass  # Graceful degradation, matching the DB-unavailable paths below

    def _ensure_table(self) -> None:
        """Create block_meta table if it doesn't exist; add missing columns.

        The provenance columns (Group E) are added via idempotent
        ``ALTER TABLE ... ADD COLUMN`` for databases created before the
        columns existed — same zero-downtime pattern as
        ``block_lineage.ensure_lineage_schema``.
        """
        with self._lock:
            try:
                with self._conn_mgr.write_lock:
                    conn = self._conn_mgr.get_write_connection()
                    conn.execute(self.SCHEMA)
                    cols = {row[1] for row in conn.execute("PRAGMA table_info(block_meta)").fetchall()}
                    for col in _PROVENANCE_COLUMNS:
                        if col not in cols:
                            conn.execute(f"ALTER TABLE block_meta ADD COLUMN {col} TEXT")  # nosec B608 — col from the module-level _PROVENANCE_COLUMNS constant, not user input
                    conn.commit()
            except (sqlite3.Error, ValueError):
                pass  # Graceful degradation if DB unavailable

    def record_access(self, block_ids: list[str], query: str = "", *, now: datetime | None = None) -> None:
        """Count an access for each block, and record their co-occurrence.

        *now* is the instant to stamp ``last_accessed`` with. It is a
        parameter and NOT a clock read for a load-bearing reason: the only
        caller on the recall path is ``_recall_core``, and recall is
        contractually clock-free — deterministic given (corpus, config,
        scoring_instant), with ``mind_mem.scoring_instant`` the single
        sanctioned boundary read, taken once, before ranking. A
        ``datetime.now()`` in here is a second read, after ranking, inside
        the deterministic core; ``tests/_recall_clock_sentinel.py`` sees it
        and is right to.

        With *now* omitted the access is still counted and the co-occurrence
        still recorded — ``access_count`` carries the larger of the two
        importance terms — but ``last_accessed`` is left exactly as it was.
        Inventing an instant would put a hidden clock on the caller's path,
        and a stamp nobody asked for is worse than an absent one: the
        consolidation planner treats a missing ``last_accessed`` as neutral
        recency and falls back to the block's creation date for staleness.

        Wired 5.0.2: ``_recall_core.recall`` now passes ``now=_scoring_moment``
        -- the instant it already resolved at its own clock boundary -- at the
        A-MEM record-access site, so the recency half of this telemetry is live
        on the recall path with no second clock read. Direct callers that omit
        *now* still get the access count and the co-occurrence edge.
        """
        if not block_ids:
            return
        _log.debug("record_access", block_count=len(block_ids))
        stamp = now.isoformat() if now is not None else None
        with self._lock:
            try:
                with self._conn_mgr.write_lock:
                    conn = self._conn_mgr.get_write_connection()
                    for bid in block_ids:
                        if stamp is None:
                            conn.execute(
                                """INSERT INTO block_meta (id, access_count)
                                VALUES (?, 1)
                                ON CONFLICT(id) DO UPDATE SET
                                    access_count = access_count + 1""",
                                (bid,),
                            )
                            continue
                        conn.execute(
                            """INSERT INTO block_meta (id, access_count, last_accessed)
                            VALUES (?, 1, ?)
                            ON CONFLICT(id) DO UPDATE SET
                                access_count = access_count + 1,
                                last_accessed = ?""",
                            (bid, stamp, stamp),
                        )
                    # Record co-occurrence pairs for connection tracking
                    if len(block_ids) > 1:
                        for i, bid in enumerate(block_ids):
                            others = [b for j, b in enumerate(block_ids) if j != i]
                            row = conn.execute("SELECT connections FROM block_meta WHERE id = ?", (bid,)).fetchone()
                            if row and row[0]:
                                existing = set(json.loads(row[0])) if row[0] else set()
                            else:
                                existing = set()
                            existing.update(others[:10])  # Cap connections
                            conn.execute(
                                "UPDATE block_meta SET connections = ? WHERE id = ?",
                                (json.dumps(list(existing)[:50]), bid),
                            )
                    conn.commit()
            except (sqlite3.Error, json.JSONDecodeError):
                pass  # Graceful degradation

    def update_importance(self, block_id: str, decay_days: int = 30) -> float:
        """Recalculate importance from access frequency + recency + connections.
        Returns importance score in [0.8, 1.5] range."""
        with self._lock:
            try:
                # Read current values
                rconn = self._conn_mgr.get_read_connection()
                row = rconn.execute(
                    "SELECT access_count, last_accessed, connections FROM block_meta WHERE id = ?",
                    (block_id,),
                ).fetchone()
                if not row:
                    return 1.0

                access_count, last_accessed, connections_json = row

                connections = json.loads(connections_json) if connections_json else []
                importance = compute_importance(
                    access_count=access_count,
                    last_accessed=last_accessed,
                    connection_count=len(connections),
                    now=datetime.now(timezone.utc),
                    decay_days=decay_days,
                )
                _log.debug("update_importance", block_id=block_id, importance=round(importance, 3))

                # Update stored importance (write path)
                with self._conn_mgr.write_lock:
                    wconn = self._conn_mgr.get_write_connection()
                    wconn.execute(
                        "UPDATE block_meta SET importance = ? WHERE id = ?",
                        (importance, block_id),
                    )
                    wconn.commit()

                return importance
            except (sqlite3.Error, json.JSONDecodeError, ValueError):
                return 1.0

    def get_importance_boost(self, block_id: str) -> float:
        """Returns [0.8, 1.5] multiplier for reranking.

        One block, one statement. On a ranking path that scores a whole
        candidate list, call :meth:`get_importance_boosts` instead — the
        recall scan leg was issuing this once per candidate (17.6 ms per
        query at 5,000 candidates) with no batch form to reach for.
        """
        try:
            conn = self._conn_mgr.get_read_connection()
            row = conn.execute("SELECT importance FROM block_meta WHERE id = ?", (block_id,)).fetchone()
            return row[0] if row else 1.0
        except (sqlite3.Error, TypeError):
            return 1.0

    def get_importance_boosts(self, block_ids: list[str]) -> dict[str, float]:
        """``{block_id: boost}`` for many blocks, in chunked batch queries.

        The batch twin of :meth:`get_importance_boost`, and deliberately
        identical to it in VALUE: the same column, the same rows, and the
        same 1.0 for a block with no ``block_meta`` row. Only the statement
        count changes — one query per :data:`_IMPORTANCE_BATCH_SIZE` ids
        instead of one per id.

        Chunked rather than trusting the host's ``SQLITE_MAX_VARIABLE_NUMBER``
        (999 on pre-3.32 builds, 32766 after), because a recall scan can
        present tens of thousands of candidates at once.

        A chunk that fails is absorbed the way the per-id call absorbs a
        failure: its ids are simply absent from the returned map, and the
        caller's ``.get(bid, 1.0)`` supplies the same neutral multiplier the
        single-block form would have returned.
        """
        if not block_ids:
            return {}
        out: dict[str, float] = {}
        try:
            conn = self._conn_mgr.get_read_connection()
        except sqlite3.Error:
            return out
        for start in range(0, len(block_ids), _IMPORTANCE_BATCH_SIZE):
            chunk = block_ids[start : start + _IMPORTANCE_BATCH_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            try:
                rows = conn.execute(
                    f"SELECT id, importance FROM block_meta WHERE id IN ({placeholders})",  # nosec B608 — `placeholders` is `?`*N; every id is a bind parameter
                    chunk,
                ).fetchall()
            except (sqlite3.Error, TypeError):
                continue
            for row in rows:
                if row[1] is not None:
                    out[row[0]] = row[1]
        return out

    def evolve_keywords(self, block_id: str, query_tokens: list[str], block_content: str = "", max_keywords: int = 20) -> None:
        """Add query tokens found in block content to block's keyword set."""
        if not query_tokens:
            return
        with self._lock:
            try:
                # Read current keywords
                rconn = self._conn_mgr.get_read_connection()
                row = rconn.execute("SELECT keywords FROM block_meta WHERE id = ?", (block_id,)).fetchone()

                existing_kw = set()
                if row and row[0]:
                    existing_kw = set(row[0].split(",")) if row[0] else set()

                content_lower = block_content.lower()
                new_kw = set()
                for token in query_tokens:
                    if token.lower() in content_lower:
                        new_kw.add(token.lower())

                combined = existing_kw | new_kw
                # Cap at max_keywords
                kw_list = sorted(combined)[:max_keywords]
                kw_str = ",".join(kw_list)

                # Write updated keywords
                with self._conn_mgr.write_lock:
                    wconn = self._conn_mgr.get_write_connection()
                    wconn.execute(
                        """INSERT INTO block_meta (id, keywords)
                        VALUES (?, ?)
                        ON CONFLICT(id) DO UPDATE SET keywords = ?""",
                        (block_id, kw_str, kw_str),
                    )
                    wconn.commit()
            except (sqlite3.Error, json.JSONDecodeError):
                pass

    def get_co_occurring_blocks(self, block_id: str, limit: int = 5) -> list[str]:
        """Blocks that frequently appear together in results."""
        try:
            conn = self._conn_mgr.get_read_connection()
            row = conn.execute("SELECT connections FROM block_meta WHERE id = ?", (block_id,)).fetchone()
            if row and row[0]:
                connections: list[str] = json.loads(row[0])
                return connections[:limit]
            return []
        except (sqlite3.Error, json.JSONDecodeError):
            return []

    def set_provenance(
        self,
        block_id: str,
        *,
        actor_id: str | None = None,
        actor_role: str | None = None,
        session_id: str | None = None,
        tool_id: str | None = None,
        purpose: str | None = None,
        content_source: str | None = None,
    ) -> bool:
        """Record provenance for a block (Group E + T-001). All optional.

        Only the fields provided (non-None, non-blank after sanitization)
        are written; existing values for omitted fields are preserved.
        Values are single-line by contract and capped at
        :data:`~mind_mem.block_provenance.MAX_PROVENANCE_VALUE_LEN` chars.

        *content_source* is vocabulary-bound
        (:data:`~mind_mem.block_provenance.CONTENT_SOURCES`) and rejected
        loudly. That is deliberately the ONE exception to this manager's
        graceful-degradation contract: swallowing a bad trust tag the way
        a DB error is swallowed would leave the caller believing a source
        class was recorded when none was.

        Returns True when a write happened, False when nothing was
        provided or the DB is unavailable (graceful degradation, matching
        the rest of this manager).

        Raises:
            ValueError: *content_source* is outside the vocabulary. Raised
                before any DB work, so nothing partial is written.
        """
        provided = {
            "actor_id": actor_id,
            "actor_role": actor_role,
            "session_id": session_id,
            "tool_id": tool_id,
            "purpose": purpose,
            "content_source": content_source,
        }
        updates: dict[str, str] = {}
        for col in _PROVENANCE_COLUMNS:
            raw = provided[col]
            if raw is None:
                continue
            value = clean_provenance_value(col, str(raw))
            if value:
                updates[col] = value
        if not updates:
            return False
        set_clause = ", ".join(f"{col} = ?" for col in updates)
        with self._lock:
            try:
                with self._conn_mgr.write_lock:
                    conn = self._conn_mgr.get_write_connection()
                    conn.execute("INSERT OR IGNORE INTO block_meta (id) VALUES (?)", (block_id,))
                    conn.execute(
                        f"UPDATE block_meta SET {set_clause} WHERE id = ?",  # nosec B608 — set_clause built from _PROVENANCE_COLUMNS constant; values bound as params
                        (*updates.values(), block_id),
                    )
                    conn.commit()
                _log.debug("set_provenance", block_id=block_id, fields=sorted(updates))
                return True
            except sqlite3.Error:
                return False  # Graceful degradation

    def get_provenance(self, block_id: str) -> dict[str, str]:
        """Return recorded provenance for a block; ``{}`` when none.

        Keys are the snake_case caller-facing names (``actor_id``,
        ``actor_role``, ``session_id``, ``tool_id``, ``purpose``,
        ``content_source``); only non-null, non-empty fields are included.
        """
        cols = ", ".join(_PROVENANCE_COLUMNS)
        try:
            conn = self._conn_mgr.get_read_connection()
            row = conn.execute(
                f"SELECT {cols} FROM block_meta WHERE id = ?",  # nosec B608 — cols from the _PROVENANCE_COLUMNS constant
                (block_id,),
            ).fetchone()
            if not row:
                return {}
            return {col: value for col, value in zip(_PROVENANCE_COLUMNS, row) if value}
        except sqlite3.Error:
            return {}

    def close(self) -> None:
        """Close the underlying ConnectionManager."""
        self._conn_mgr.close()
