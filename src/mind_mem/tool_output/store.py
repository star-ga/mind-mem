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

    def __init__(self, sqlite_path: str | None = None, *, backend: str = "sqlite",
                 max_rows: int = DEFAULT_MAX_ROWS,
                 max_store_bytes: int = DEFAULT_MAX_STORE_BYTES):
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

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self._sqlite_path) as con:
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
    def store_and_summarize(self, text: str, source: str = "",
                            exit_code: int | None = None, *, ts: str = "") -> StoreResult:
        """Store the FULL text out-of-context; return only {handle, summary, …}.

        Idempotent: the handle is content-addressed, so re-storing identical output
        overwrites the same row (no duplicates). ``ts`` is passed in (never a clock
        here) so the summary/handle stay deterministic; storage stamps it only as
        opaque metadata, never part of the summary or handle.
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
                con.execute(
                    "INSERT INTO tool_outputs (handle, source, exit_code, full_text, "
                    "summary, line_count, byte_count) VALUES (%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (handle) DO UPDATE SET full_text=EXCLUDED.full_text, "
                    "summary=EXCLUDED.summary",
                    (handle, source, exit_code, stored, s.summary, s.line_count, s.byte_count),
                )
                self._evict_pg(con)
        else:
            with sqlite3.connect(self._sqlite_path) as con:
                con.execute(
                    "INSERT OR REPLACE INTO tool_outputs "
                    "(handle, source, exit_code, ts, full_text, summary, line_count, byte_count) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (handle, source, exit_code, ts, stored, s.summary, s.line_count, s.byte_count),
                )
                self._evict_sqlite(con)
        return StoreResult(handle=handle, summary=s.summary, line_count=s.line_count,
                           failure_lines=s.failure_lines, dropped_lines=s.dropped_lines,
                           stored_bytes=stored_bytes, truncated_store=truncated)

    def _evict_sqlite(self, con) -> int:
        """Keep only the newest ``max_rows`` (by insertion rowid). Returns #evicted."""
        if self.max_rows <= 0:
            return 0
        cur = con.execute(
            "DELETE FROM tool_outputs WHERE handle IN "
            "(SELECT handle FROM tool_outputs ORDER BY rowid DESC LIMIT -1 OFFSET ?)",
            (self.max_rows,))
        return cur.rowcount or 0

    def _evict_pg(self, con) -> None:
        if self.max_rows <= 0:
            return
        con.execute(
            "DELETE FROM tool_outputs WHERE ctid IN "
            "(SELECT ctid FROM tool_outputs ORDER BY ts DESC OFFSET %s)",
            (self.max_rows,))

    def gc(self) -> int:
        """Force retention now (evict beyond ``max_rows``). Returns #evicted."""
        if self.backend != "sqlite":
            return 0
        with sqlite3.connect(self._sqlite_path) as con:
            return self._evict_sqlite(con)

    def recall_output(self, handle: str) -> str | None:
        """Return the FULL stored text for ``handle``, or ``None`` if unknown."""
        if self.backend == "postgres":
            with self._pg() as con:
                self._pg_init(con)
                row = con.execute(
                    "SELECT full_text FROM tool_outputs WHERE handle=%s", (handle,)).fetchone()
        else:
            with sqlite3.connect(self._sqlite_path) as con:
                row = con.execute(
                    "SELECT full_text FROM tool_outputs WHERE handle=?", (handle,)).fetchone()
        return row[0] if row else None

    def meta(self, handle: str) -> dict | None:
        if self.backend != "sqlite":
            return None
        with sqlite3.connect(self._sqlite_path) as con:
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT handle, source, exit_code, ts, line_count, "
                              "byte_count FROM tool_outputs WHERE handle=?", (handle,)).fetchone()
        return dict(row) if row else None


__all__ = ["ToolOutputStore", "StoreResult", "summarize", "make_handle"]
