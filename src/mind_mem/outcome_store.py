"""Outcome-attribution persistence over the calibration store.

The SQL half of :mod:`mind_mem.outcome_attribution`. These are plain
functions over the **same** :class:`~mind_mem.connection_manager.ConnectionManager`
that :class:`mind_mem.calibration.CalibrationManager` already owns — same
``.mind-mem-index/recall.db``, same schema script — split out only to keep
``calibration.py`` inside the module size budget.
:class:`~mind_mem.calibration.CalibrationManager` exposes each of them as a
method; nothing here opens a second store.

Determinism: every read is unwindowed (no clock, no rolling window), so the
counts the recall validity gate scores on are a pure function of stored
state. The one clock value written (``recorded_at``) is provenance and is
injectable.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from .calibration import _MAX_OUTCOME_PAGE, _OUTCOME_TO_FEEDBACK, _init_calibration_schema
from .connection_manager import ConnectionManager
from .observability import get_logger, metrics
from .outcome_attribution import (
    MIN_OUTCOME_EVIDENCE,
    OutcomeSignal,
    bounded_field,
    canonical_outcome_id,
    normalize_block_ids,
    validate_outcome,
)

_log = get_logger("outcome_store")

__all__ = [
    "get_outcome_signals",
    "get_outcome_stats",
    "list_outcomes",
    "record_outcome",
]


def record_outcome(
    mgr: ConnectionManager,
    block_ids: Iterable[str],
    outcome: str,
    *,
    query_id: str = "",
    task_id: str = "",
    actor_id: str = "",
    session_id: str = "",
    tool_id: str = "",
    evidence: str = "",
    recorded_at: str | None = None,
    project_to_calibration: bool = False,
) -> dict[str, Any]:
    """Record whether acting on ``block_ids`` actually worked.

    Appends to ``recall_outcome``. With ``project_to_calibration`` the
    same verdicts are additionally written into ``calibration_feedback``
    (success -> accepted, failure -> rejected, neutral -> ignored) so
    the pre-existing per-block weight loop sees utility too.

    That projection is **opt-in** on purpose: ``_recall_core`` applies
    calibration weights unconditionally, so projecting by default would
    move recall scores for callers who never enabled the validity-gate
    outcome signal. Default-off keeps flag-off byte-identical; the
    deterministic, unwindowed ``recall_outcome`` counts are what the
    validity gate reads either way.

    Idempotent: the outcome id is the SHA-256 of the canonical payload,
    so a replay conflicts on the primary key and changes nothing —
    including the originally stored ``recorded_at``.

    Never mutates block content: the only writes are to this sidecar
    index database. Corpus changes go through ``propose_update``.
    """
    verdict = validate_outcome(outcome)
    ids = normalize_block_ids(block_ids)
    query_id = bounded_field("query_id", query_id)
    task_id = bounded_field("task_id", task_id)
    actor_id = bounded_field("actor_id", actor_id)
    session_id = bounded_field("session_id", session_id)
    tool_id = bounded_field("tool_id", tool_id)
    evidence = bounded_field("evidence", evidence)

    outcome_id, payload_hash = canonical_outcome_id(
        ids,
        verdict,
        task_id=task_id,
        actor_id=actor_id,
        session_id=session_id,
        tool_id=tool_id,
        evidence=evidence,
    )
    stamp = recorded_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    feedback = _OUTCOME_TO_FEEDBACK[verdict]
    projected_query_id = f"outcome:{outcome_id}"
    recorded = 0
    projected = 0

    with mgr.write_lock:
        conn = mgr.get_write_connection()
        conn.row_factory = sqlite3.Row
        _init_calibration_schema(conn)

        for bid in ids:
            cur = conn.execute(
                """INSERT INTO recall_outcome
                   (outcome_id, block_id, outcome, query_id, task_id, actor_id,
                    session_id, tool_id, evidence, payload_hash, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(outcome_id, block_id) DO NOTHING""",
                (
                    outcome_id,
                    bid,
                    verdict,
                    query_id,
                    task_id,
                    actor_id,
                    session_id,
                    tool_id,
                    evidence,
                    payload_hash,
                    stamp,
                ),
            )
            recorded += max(0, cur.rowcount)

            # Opt-in projection into the pre-existing feedback loop.
            if project_to_calibration:
                projected += max(
                    0,
                    conn.execute(
                        """INSERT INTO calibration_feedback
                           (query_id, query_text, query_type, block_id, feedback, created_at)
                           VALUES (?, '', 'OUTCOME', ?, ?, ?)
                           ON CONFLICT(query_id, block_id, feedback) DO NOTHING""",
                        (projected_query_id, bid, feedback, stamp),
                    ).rowcount,
                )

        conn.commit()

        stored = conn.execute(
            """SELECT recorded_at FROM recall_outcome
               WHERE outcome_id = ? ORDER BY block_id LIMIT 1""",
            (outcome_id,),
        ).fetchone()

    effective_stamp = stored["recorded_at"] if stored else stamp
    metrics.inc("outcome_recorded", recorded)
    _log.info(
        "outcome_recorded",
        outcome_id=outcome_id,
        outcome=verdict,
        blocks=len(ids),
        recorded=recorded,
    )

    return {
        "outcome_id": outcome_id,
        "payload_hash": payload_hash,
        "outcome": verdict,
        "block_ids": list(ids),
        "blocks": len(ids),
        "recorded": recorded,
        "duplicate": len(ids) - recorded,
        "idempotent": recorded == 0,
        "recorded_at": effective_stamp,
        "projected": projected,
        "calibration_feedback": feedback if project_to_calibration else "",
        "query_id": query_id,
        "task_id": task_id,
        "actor_id": actor_id,
        "session_id": session_id,
        "tool_id": tool_id,
        "evidence": evidence,
    }


def get_outcome_signals(mgr: ConnectionManager, block_ids: list[str]) -> dict[str, OutcomeSignal]:
    """Batch per-block utility evidence, keyed by block id.

    Unwindowed on purpose: this feeds the validity gate, whose scored
    path must contain no clock. Blocks with no attributed outcome are
    simply absent (the caller reads absence as neutral).
    """
    if not block_ids:
        return {}

    conn = mgr.get_read_connection()
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in block_ids)
    try:
        rows = conn.execute(
            f"""SELECT block_id, outcome, COUNT(*) AS cnt
                FROM recall_outcome
                WHERE block_id IN ({placeholders})
                GROUP BY block_id, outcome""",  # nosec B608 — placeholders is `? * N`; every id is a bind param
            list(block_ids),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = counts.setdefault(row["block_id"], {})
        bucket[row["outcome"]] = row["cnt"]

    return {
        bid: OutcomeSignal(
            block_id=bid,
            success=bucket.get("success", 0),
            failure=bucket.get("failure", 0),
            neutral=bucket.get("neutral", 0),
        )
        for bid, bucket in counts.items()
    }


def list_outcomes(
    mgr: ConnectionManager,
    block_id: str = "",
    task_id: str = "",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recorded outcomes with full provenance, newest first.

    Ordering is fully specified — ``recorded_at``, then ``outcome_id``,
    then ``block_id`` — so two identical stores list identically.
    """
    limit = max(1, min(int(limit), _MAX_OUTCOME_PAGE))
    clauses: list[str] = []
    params: list[Any] = []
    if block_id:
        clauses.append("block_id = ?")
        params.append(bounded_field("block_id", block_id))
    if task_id:
        clauses.append("task_id = ?")
        params.append(bounded_field("task_id", task_id))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    conn = mgr.get_read_connection()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""SELECT outcome_id, block_id, outcome, query_id, task_id, actor_id,
                       session_id, tool_id, evidence, payload_hash, recorded_at
                FROM recall_outcome
                {where}
                ORDER BY recorded_at DESC, outcome_id ASC, block_id ASC
                LIMIT ?""",  # nosec B608 — `where` is built from fixed literals; all values are bind params
            [*params, limit],
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(row) for row in rows]


def get_outcome_stats(mgr: ConnectionManager, top_n: int = 20) -> dict[str, Any]:
    """Utility health report — totals, corroborated blocks, implicated blocks."""
    top_n = max(1, min(int(top_n), _MAX_OUTCOME_PAGE))
    conn = mgr.get_read_connection()
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """SELECT block_id, outcome, COUNT(*) AS cnt
               FROM recall_outcome GROUP BY block_id, outcome"""
        ).fetchall()
        reports = conn.execute("SELECT COUNT(DISTINCT outcome_id) AS cnt FROM recall_outcome").fetchone()["cnt"]
    except (sqlite3.OperationalError, TypeError):
        return {
            "total_outcomes": 0,
            "unique_reports": 0,
            "unique_blocks": 0,
            "corroborated": [],
            "implicated": [],
            "message": "No outcome data available.",
        }

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row["block_id"], {})[row["outcome"]] = row["cnt"]

    signals = [
        OutcomeSignal(
            block_id=bid,
            success=bucket.get("success", 0),
            failure=bucket.get("failure", 0),
            neutral=bucket.get("neutral", 0),
        )
        for bid, bucket in counts.items()
    ]

    corroborated = sorted(
        (s for s in signals if s.corroborated),
        key=lambda s: (-s.success, s.block_id),
    )[:top_n]
    implicated = sorted(
        (s for s in signals if s.factor < 1.0),
        key=lambda s: (s.factor, s.block_id),
    )[:top_n]

    return {
        "total_outcomes": sum(s.total for s in signals),
        "unique_reports": reports,
        "unique_blocks": len(signals),
        "min_evidence": MIN_OUTCOME_EVIDENCE,
        "corroborated": [s.as_dict() for s in corroborated],
        "implicated": [s.as_dict() for s in implicated],
    }
