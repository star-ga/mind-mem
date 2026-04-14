# Copyright 2026 STARGA, Inc.
"""SQLite-backed optimization run history and lineage tracking."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional


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


class HistoryStore:
    """Persistent storage for optimization runs, results, and mutations."""

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

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
        self._conn.execute(f"UPDATE mutations SET {updates} WHERE mutation_id=?", params)
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
