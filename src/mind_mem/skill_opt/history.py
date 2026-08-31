# Copyright 2026 STARGA, Inc.
"""SQLite-backed optimization run history and lineage tracking."""

from __future__ import annotations

import json
import os
import sqlite3
import weakref
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Optional

#: Terminal status for a run whose store went away before ``complete_run``.
#: Distinct from ``"running"`` (in flight) and from ``"completed"``.
INTERRUPTED_STATUS = "interrupted"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS optimization_runs (
    run_id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT DEFAULT 'running',
    overall_score_before REAL,
    overall_score_after REAL,
    mutation_accepted INTEGER DEFAULT 0,
    config_json TEXT
);
CREATE TABLE IF NOT EXISTS test_results (
    result_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES optimization_runs(run_id),
    test_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    model TEXT NOT NULL,
    output TEXT NOT NULL,
    latency_ms REAL,
    timestamp TEXT
);
CREATE TABLE IF NOT EXISTS critique_reports (
    critique_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES optimization_runs(run_id),
    test_id TEXT NOT NULL,
    critic_model TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    overall_score REAL NOT NULL,
    failure_modes_json TEXT,
    timestamp TEXT
);
CREATE TABLE IF NOT EXISTS mutations (
    mutation_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES optimization_runs(run_id),
    skill_id TEXT NOT NULL,
    proposed_content TEXT NOT NULL,
    rationale TEXT NOT NULL,
    score_before REAL,
    score_after REAL,
    governance_signal_id TEXT,
    status TEXT DEFAULT 'proposed'
);
CREATE INDEX IF NOT EXISTS idx_runs_skill ON optimization_runs(skill_id);
CREATE INDEX IF NOT EXISTS idx_mutations_skill ON mutations(skill_id);
"""


def _finalize_store(conn: sqlite3.Connection, open_runs: set[str]) -> None:
    """Close *conn*, first stamping a terminal status on unfinished runs.

    ``start_run`` commits ``status='running'`` before any work happens, and
    the work can raise. Without this, a store that goes away between
    ``start_run`` and ``complete_run`` leaves the row saying "running"
    forever, so a later reader cannot tell a live run from a dead one.

    Runs at garbage-collection time *and* at interpreter exit (that is why
    this is a :func:`weakref.finalize` callback and not ``__del__``), which
    covers the case that actually bites: a CLI process dying on an uncaught
    exception mid-run, where no ``finally`` at the call site ever executes.
    Everything is defensive — a finalizer must not raise, least of all
    during shutdown.
    """
    try:
        if open_runs:
            now = datetime.now(timezone.utc).isoformat()
            for run_id in sorted(open_runs):
                conn.execute(
                    "UPDATE optimization_runs SET status=?, completed_at=? WHERE run_id=? AND status='running'",
                    (INTERRUPTED_STATUS, now, run_id),
                )
            conn.commit()
            open_runs.clear()
    except Exception:  # nosec B110 — finalizers must never raise
        pass
    try:
        conn.close()
    except Exception:  # nosec B110 — finalizers must never raise
        pass


class HistoryStore:
    """Persistent storage for optimization runs, results, and mutations.

    Owns a long-lived SQLite connection, so use it as a context manager —
    ``with HistoryStore(path) as store:`` — or call :meth:`close` from a
    ``finally``. A store that is dropped without either still finalises
    itself (see :func:`_finalize_store`): the connection is closed and any
    run started but never completed is stamped
    ``status='interrupted'`` rather than being left claiming to be running.
    """

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # Runs this store opened and has not closed. Handed to the
        # finalizer by value so the callback holds no reference to self —
        # a finalizer that kept the object alive would never fire.
        self._open_runs: set[str] = set()
        # Registered against *self*: the finalize object holds strong refs to
        # its arguments, so registering against the connection would keep that
        # connection alive forever and the callback would only ever fire at
        # interpreter exit. Weak on self, strong on what it needs to clean up.
        self._finalizer = weakref.finalize(self, _finalize_store, self._conn, self._open_runs)

    def close(self) -> None:
        """Close the connection. Idempotent; safe to call after a drop."""
        self._finalizer()

    def __enter__(self) -> HistoryStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def start_run(
        self,
        run_id: str,
        skill_id: str,
        content_hash: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO optimization_runs (run_id, skill_id, content_hash, started_at, config_json) VALUES (?, ?, ?, ?, ?)",
            (run_id, skill_id, content_hash, now, json.dumps(config or {})),
        )
        self._conn.commit()
        # Tracked from the moment the 'running' row is durable, so a crash
        # anywhere in the work that follows still resolves to a terminal
        # status when the store is finalised.
        self._open_runs.add(run_id)

    def complete_run(
        self,
        run_id: str,
        status: str = "completed",
        score_before: float = 0.0,
        score_after: float = 0.0,
        mutation_accepted: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE optimization_runs SET completed_at=?, status=?, overall_score_before=?, overall_score_after=?, mutation_accepted=? WHERE run_id=?",
            (now, status, score_before, score_after, int(mutation_accepted), run_id),
        )
        self._conn.commit()
        self._open_runs.discard(run_id)

    def store_test_result(
        self,
        result_id: str,
        run_id: str,
        test_id: str,
        skill_id: str,
        model: str,
        output: str,
        latency_ms: float,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO test_results (result_id, run_id, test_id, skill_id, model, output, latency_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (result_id, run_id, test_id, skill_id, model, output, latency_ms, now),
        )
        self._conn.commit()

    def store_critique(
        self,
        critique_id: str,
        run_id: str,
        test_id: str,
        critic_model: str,
        scores: dict[str, float],
        overall_score: float,
        failure_modes: list[str] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO critique_reports (critique_id, run_id, test_id, critic_model, scores_json, overall_score, failure_modes_json, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (critique_id, run_id, test_id, critic_model, json.dumps(scores), overall_score, json.dumps(failure_modes or []), now),
        )
        self._conn.commit()

    def store_mutation(
        self,
        run_id: str,
        mutation_id: str,
        skill_id: str,
        proposed_content: str,
        rationale: str,
        score_before: float = 0.0,
        score_after: float = 0.0,
        governance_signal_id: str = "",
        status: str = "proposed",
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO mutations (mutation_id, run_id, skill_id, proposed_content, rationale, score_before, score_after, governance_signal_id, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mutation_id, run_id, skill_id, proposed_content, rationale, score_before, score_after, governance_signal_id, status),
        )
        self._conn.commit()

    def update_mutation_status(self, mutation_id: str, status: str, signal_id: str = "") -> None:
        updates = "status=?"
        params: list[Any] = [status]
        if signal_id:
            updates += ", governance_signal_id=?"
            params.append(signal_id)
        params.append(mutation_id)
        self._conn.execute(f"UPDATE mutations SET {updates} WHERE mutation_id=?", params)  # nosec B608 — `updates` is built from a fixed set of column names ("status", "governance_signal_id"); no user input interpolated
        self._conn.commit()

    def get_run_history(self, skill_id: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM optimization_runs WHERE skill_id=? ORDER BY started_at DESC LIMIT ?",
            (skill_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_score(self, skill_id: str) -> Optional[float]:
        row = self._conn.execute(
            "SELECT overall_score_after FROM optimization_runs WHERE skill_id=? AND status='completed' ORDER BY completed_at DESC LIMIT 1",
            (skill_id,),
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def get_mutation(self, mutation_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM mutations WHERE mutation_id=?", (mutation_id,)).fetchone()
        return dict(row) if row else None

    def get_pending_mutations(self, skill_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM mutations WHERE skill_id=? AND status='proposed' ORDER BY score_after DESC",
            (skill_id,),
        ).fetchall()
        return [dict(r) for r in rows]
