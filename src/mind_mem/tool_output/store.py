"""Tool-output store — full text out-of-context, keyed by handle (mind-mem §5).

A large tool log does NOT belong in the embedded ``blocks`` table (it would
pollute recall + burn embeddings). It lives in a dedicated ``tool_outputs`` sibling
table, keyed by a content-addressed handle, so:

    store_and_summarize(text, source, exit_code) -> {handle, summary, line_count}
    recall_output(handle)                        -> the full stored text

Backend: reuses the EXISTING Postgres connection when configured (no new DB — the
sibling table lives in the same schema via the block-store connection helper), and
falls back to a local SQLite file otherwise (and in tests), so the capability works
everywhere without a live Postgres. The full text is stored, never summarized away.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from .summarize import make_handle, summarize

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS tool_outputs (
    handle      TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    exit_code   INTEGER,
    ts          TEXT NOT NULL,
    full_text   TEXT NOT NULL,
    summary     TEXT NOT NULL,
    line_count  INTEGER NOT NULL,
    byte_count  INTEGER NOT NULL
)
"""

# Retention defaults — the table is BOUNDED so an agent that stores every test run
# can't grow it without limit. Newest ``max_rows`` are kept (insertion order via
# rowid); ``max_store_bytes`` caps a single stored blob (a runaway GB log is stored
# truncated with an EXPLICIT marker — the summary is still computed on the full text
# so failures are never missed). Both are config, not autonomous policy.
DEFAULT_MAX_ROWS = 500
DEFAULT_MAX_STORE_BYTES = 32 * 1024 * 1024  # 32 MiB


@dataclass(frozen=True)
class StoreResult:
    handle: str
    summary: str
    line_count: int
    failure_lines: int
    dropped_lines: int
    stored_bytes: int = 0
    truncated_store: bool = False


def _default_sqlite_path() -> str:
    root = os.environ.get("MIND_MEM_WORKSPACE") or os.path.expanduser("~/.mind-mem")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "tool_outputs.db")


class ToolOutputStore:
    """Handle → full-text store. SQLite by default; ``backend='postgres'`` reuses
    the mind-mem block-store connection (same DB, sibling table)."""

    def __init__(
        self,
        sqlite_path: str | None = None,
        *,
        backend: str = "sqlite",
        max_rows: int = DEFAULT_MAX_ROWS,
        max_store_bytes: int = DEFAULT_MAX_STORE_BYTES,
    ):
        self.backend = backend
        self.max_rows = max_rows
        self.max_store_bytes = max_store_bytes
        self._sqlite_path = sqlite_path or _default_sqlite_path()
        if backend == "sqlite":
            self._init_sqlite()
        # postgres init is lazy (first store) so importing never requires psycopg.

    @staticmethod
    def _cap_store(text: str, max_bytes: int) -> tuple[str, bool]:
        """Cap the stored blob at ``max_bytes`` (utf-8) with an explicit marker.
        The summary is computed on the FULL text upstream, so a truncated STORE
        never hides a failure from the summary — only the recall tail is bounded."""
        raw = text.encode("utf-8")
        if len(raw) <= max_bytes:
            return text, False
        head = raw[:max_bytes].decode("utf-8", "ignore")
        return head + f"\n…[stored blob truncated: {len(raw) - max_bytes} bytes dropped]\n", True

    @contextmanager
    def _sqlite(self, *, row_factory: type[sqlite3.Row] | None = None) -> Iterator[sqlite3.Connection]:
        """Open the SQLite file, commit-or-rollback, then CLOSE it.

        Every SQLite statement below goes through here rather than through
        ``with sqlite3.connect(...) as con``, because a bare sqlite3
        connection used as a context manager commits (or, on an exception,
        rolls back) and then **leaves the handle open** — its ``__exit__``
        documents exactly that and nothing more.

        Refcounting does not clean up after it either: a
        ``sqlite3.Connection`` holds its prepared-statement cache and that
        cache holds the connection back, so every connection sits in a
        reference cycle and is reclaimed only if and when the cyclic
        collector happens to run. Until then this process keeps an open
        descriptor on ``tool_outputs.db`` *and* on its ``-wal`` / ``-shm``
        sidecars. ``store_and_summarize`` is called once per captured tool
        run, so the count grows with the workload.

        On Windows those open handles also make ``os.unlink`` / ``rmdir``
        fail, so a workspace holding a tool-output store could not be
        deleted. Closing is likewise what lets SQLite checkpoint and remove
        the sidecars.

        Transaction semantics are unchanged: the inner ``with con`` still
        commits on success and rolls back on an exception, and it does so
        *before* the close. ``close()`` on its own never commits, so the
        ordering cannot turn a rollback into a commit.

        The Postgres branch deliberately does NOT route through here — a
        psycopg3 connection's ``__exit__`` closes as well as commits, so
        ``with self._pg() as con`` is already leak-free.
        """
        con = sqlite3.connect(self._sqlite_path)
        try:
            if row_factory is not None:
                con.row_factory = row_factory
            with con:
                yield con
        finally:
            con.close()

    def _init_sqlite(self) -> None:
        with self._sqlite() as con:
            con.execute(_SQLITE_DDL)

    # ── Postgres path (reuses the existing block-store connection) ─────────────
    def _pg(self):
        from mind_mem.block_store_postgres import _require_psycopg  # existing helper

        psycopg, _ = _require_psycopg()
        dsn = os.environ.get("MIND_MEM_BLOCK_STORE") or os.environ.get("MIND_MEM_PG_DSN")
        if not dsn:
            raise RuntimeError("postgres backend needs MIND_MEM_BLOCK_STORE / MIND_MEM_PG_DSN")
        return psycopg.connect(dsn)

    def _pg_init(self, con) -> None:
        con.execute(
            "CREATE TABLE IF NOT EXISTS tool_outputs ("
            "handle TEXT PRIMARY KEY, source TEXT NOT NULL, exit_code INTEGER, "
            "ts TIMESTAMPTZ NOT NULL DEFAULT now(), full_text TEXT NOT NULL, "
            "summary TEXT NOT NULL, line_count INTEGER NOT NULL, byte_count INTEGER NOT NULL)"
        )

    # ── public API ────────────────────────────────────────────────────────────
    def store_and_summarize(self, text: str, source: str = "", exit_code: int | None = None, *, ts: str = "") -> StoreResult:
        """Store the FULL text out-of-context; return only {handle, summary, …}.

        Idempotent: the handle is content-addressed, so re-storing identical output
        overwrites the same row (no duplicates). ``ts`` is passed in (never a clock
        here) so the summary/handle stay deterministic; storage stamps it only as
        metadata, never part of the summary or handle. Left empty, the row is
        stamped with the storage clock (SQLite writes the empty string; Postgres
        applies the column's ``now()`` default). On the Postgres backend the
        column is ``TIMESTAMPTZ``, so a supplied ``ts`` must be a timestamp
        literal the database can parse — an unparsable one fails loudly rather
        than being dropped for a wall clock.
        """
        # Summarize the FULL text FIRST (failures surfaced regardless of store cap),
        # then cap what we persist so a runaway log can't fill the disk.
        s = summarize(text, source=source, exit_code=exit_code)
        handle = make_handle(text, source)
        stored, truncated = self._cap_store(text, self.max_store_bytes)
        stored_bytes = len(stored.encode("utf-8"))
        if self.backend == "postgres":
            with self._pg() as con:
                self._pg_init(con)
                # ``ts`` is written explicitly (as NULL → the column's
                # ``now()`` default only when the caller passed none), or a
                # replay pinning ``ts`` would silently get a wall clock here
                # while the SQLite branch honoured it — two workspaces that
                # ARE identical would diff. Every mutable column is refreshed
                # on conflict, matching SQLite's INSERT OR REPLACE: a
                # re-store of byte-identical output under a different exit
                # code must not keep the stale one (the handle hashes only
                # source ‖ text), and ``ts`` must move or the row never
                # refreshes its recency for ``_evict_pg``'s ORDER BY ts.
                con.execute(
                    "INSERT INTO tool_outputs (handle, source, exit_code, ts, full_text, "
                    "summary, line_count, byte_count) "
                    "VALUES (%s,%s,%s,COALESCE(%s::timestamptz, now()),%s,%s,%s,%s) "
                    "ON CONFLICT (handle) DO UPDATE SET source=EXCLUDED.source, "
                    "exit_code=EXCLUDED.exit_code, ts=EXCLUDED.ts, "
                    "full_text=EXCLUDED.full_text, summary=EXCLUDED.summary, "
                    "line_count=EXCLUDED.line_count, byte_count=EXCLUDED.byte_count",
                    (handle, source, exit_code, ts or None, stored, s.summary, s.line_count, s.byte_count),
                )
                self._evict_pg(con)
        else:
            with self._sqlite() as con:
                con.execute(
                    "INSERT OR REPLACE INTO tool_outputs "
                    "(handle, source, exit_code, ts, full_text, summary, line_count, byte_count) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (handle, source, exit_code, ts, stored, s.summary, s.line_count, s.byte_count),
                )
                self._evict_sqlite(con)
        return StoreResult(
            handle=handle,
            summary=s.summary,
            line_count=s.line_count,
            failure_lines=s.failure_lines,
            dropped_lines=s.dropped_lines,
            stored_bytes=stored_bytes,
            truncated_store=truncated,
        )

    def _evict_sqlite(self, con) -> int:
        """Keep only the newest ``max_rows`` (by insertion rowid). Returns #evicted."""
        if self.max_rows <= 0:
            return 0
        cur = con.execute(
            "DELETE FROM tool_outputs WHERE handle IN (SELECT handle FROM tool_outputs ORDER BY rowid DESC LIMIT -1 OFFSET ?)",
            (self.max_rows,),
        )
        return cur.rowcount or 0

    def _evict_pg(self, con) -> int:
        """Keep only the newest ``max_rows`` (by ``ts``). Returns #evicted."""
        if self.max_rows <= 0:
            return 0
        cur = con.execute(
            "DELETE FROM tool_outputs WHERE ctid IN (SELECT ctid FROM tool_outputs ORDER BY ts DESC OFFSET %s)",
            (self.max_rows,),
        )
        return getattr(cur, "rowcount", 0) or 0

    def gc(self) -> int:
        """Force retention now (evict beyond ``max_rows``). Returns #evicted.

        Runs on whichever backend is configured. It used to return 0 for a
        non-sqlite backend without evicting anything — and 0 is the
        success-shaped answer meaning "nothing needed evicting", so an
        operator scripting retention was told it had run when it had not.
        """
        if self.backend == "postgres":
            with self._pg() as con:
                self._pg_init(con)
                return self._evict_pg(con)
        with self._sqlite() as con:
            return self._evict_sqlite(con)

    def recall_output(self, handle: str) -> str | None:
        """Return the FULL stored text for ``handle``, or ``None`` if unknown."""
        if self.backend == "postgres":
            with self._pg() as con:
                self._pg_init(con)
                row = con.execute("SELECT full_text FROM tool_outputs WHERE handle=%s", (handle,)).fetchone()
        else:
            with self._sqlite() as con:
                row = con.execute("SELECT full_text FROM tool_outputs WHERE handle=?", (handle,)).fetchone()
        # ``fetchone`` materialises the row, so reading it after the
        # connection closes is safe (no lazy cursor is held).
        return row[0] if row else None

    _META_COLUMNS = ("handle", "source", "exit_code", "ts", "line_count", "byte_count")

    def meta(self, handle: str) -> dict | None:
        """Stored metadata for ``handle``, or ``None`` when it is unknown.

        ``None`` means exactly one thing — no such handle. It used to also
        mean "this backend does not implement meta", so a caller could not
        tell a missing row from an unsupported backend and reported a
        stored output as missing.
        """
        cols = ", ".join(self._META_COLUMNS)
        if self.backend == "postgres":
            with self._pg() as con:
                self._pg_init(con)
                row = con.execute(f"SELECT {cols} FROM tool_outputs WHERE handle=%s", (handle,)).fetchone()
            # psycopg returns a plain tuple unless a row factory is set;
            # ``ts`` comes back as a datetime, rendered as its ISO form so
            # the shape matches the SQLite branch's text column.
            if row is None:
                return None
            record = dict(zip(self._META_COLUMNS, row))
            if record.get("ts") is not None and not isinstance(record["ts"], str):
                record["ts"] = record["ts"].isoformat()
            return record
        with self._sqlite(row_factory=sqlite3.Row) as con:
            row = con.execute(f"SELECT {cols} FROM tool_outputs WHERE handle=?", (handle,)).fetchone()
        return dict(row) if row else None


__all__ = ["ToolOutputStore", "StoreResult", "summarize", "make_handle"]
