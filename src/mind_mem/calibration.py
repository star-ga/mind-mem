"""Calibration feedback loop — track retrieval quality and adjust block ranking.

Stores per-block and per-query-type calibration data in the same SQLite
database used by the FTS5 index.  Rolling 30-day windows prevent stale
feedback from dominating scores.

Outcome attribution (``recall_outcome``) shares this store: it records
whether *acting* on a block worked, with unwindowed counts so the recall
validity gate can read it without a clock on the scored path.

Calibration weights are multiplicative factors in [0.5, 1.5]:
  - Blocks with consistent positive feedback are boosted (>1.0)
  - Blocks with consistent negative feedback are demoted (<1.0)
  - Blocks with no feedback data default to 1.0

Time-relative by design — read this before reasoning about reproducibility
-------------------------------------------------------------------------
The calibration weight is multiplied into the recall score unconditionally,
and it is computed over a **rolling 30-day window** (``CALIBRATION_WINDOW_DAYS``)
anchored on the current instant. That is the point of the feature: feedback
older than the window stops counting, so stale opinions decay out.

The consequence must be stated plainly rather than left implicit: because
rows age out of the window as time passes, a recall score that includes a
calibration weight is reproducible only **for a fixed "as-of" instant**, not
across arbitrary wall-clock time. Re-running the same query a month later
against a byte-identical corpus can legitimately return a different score.

What *is* guaranteed is machine-independence: the window boundary is
computed in UTC (:meth:`CalibrationManager._cutoff_date`), and the stored
timestamps it is compared against are UTC too (``created_at`` defaults to
SQLite's UTC ``strftime(..., 'now')``). Two hosts in different timezones
evaluating the same instant therefore cut the window at the same point.
``_cutoff_date`` accepts an injected ``now`` so a replay can pin the
as-of instant explicitly. Recency scoring in ``_recall_scoring`` follows
the same UTC rule.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .connection_manager import ConnectionManager
from .observability import get_logger, metrics
from .outcome_attribution import (
    OUTCOME_FAILURE,
    OUTCOME_NEUTRAL,
    OUTCOME_SUCCESS,
    OutcomeSignal,
)

_log = get_logger("calibration")

# Rolling window in days for calibration scoring
CALIBRATION_WINDOW_DAYS = 30

# Calibration weight range (clamped)
MIN_CALIBRATION_WEIGHT = 0.5
MAX_CALIBRATION_WEIGHT = 1.5

# Minimum feedback events before calibration kicks in for a block
MIN_FEEDBACK_THRESHOLD = 3

# DB location relative to workspace (same directory as FTS5 index)
_DB_REL_PATH = ".mind-mem-index/recall.db"

# Outcome verdict -> the calibration_feedback row it projects to, so utility
# evidence also moves the pre-existing per-block calibration weight.
_OUTCOME_TO_FEEDBACK: dict[str, str] = {
    OUTCOME_SUCCESS: "accepted",
    OUTCOME_FAILURE: "rejected",
    OUTCOME_NEUTRAL: "ignored",
}

# Hard ceiling on rows returned by an outcome listing/report.
_MAX_OUTCOME_PAGE = 500


def _db_path(workspace: str) -> str:
    """Return absolute path to the index database."""
    return os.path.join(os.path.abspath(workspace), _DB_REL_PATH)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CALIBRATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS calibration_feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id    TEXT NOT NULL,
    query_text  TEXT NOT NULL DEFAULT '',
    query_type  TEXT NOT NULL DEFAULT '',
    block_id    TEXT NOT NULL,
    feedback    TEXT NOT NULL CHECK(feedback IN ('accepted', 'rejected', 'ignored')),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(query_id, block_id, feedback)
);

CREATE INDEX IF NOT EXISTS idx_cal_block_id
    ON calibration_feedback(block_id);
CREATE INDEX IF NOT EXISTS idx_cal_query_id
    ON calibration_feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_cal_created_at
    ON calibration_feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_cal_query_type
    ON calibration_feedback(query_type);

-- Outcome attribution (utility, not quality): did acting on this block
-- actually work?  Sibling table in the SAME store as calibration_feedback;
-- rows are optionally projected into calibration_feedback so the
-- pre-existing weight loop can see utility too.  `outcome_id` is the
-- SHA-256-derived idempotency key, so replaying a report is a no-op.
CREATE TABLE IF NOT EXISTS recall_outcome (
    outcome_id   TEXT NOT NULL,
    block_id     TEXT NOT NULL,
    outcome      TEXT NOT NULL CHECK(outcome IN ('success', 'failure', 'neutral')),
    query_id     TEXT NOT NULL DEFAULT '',
    task_id      TEXT NOT NULL DEFAULT '',
    actor_id     TEXT NOT NULL DEFAULT '',
    session_id   TEXT NOT NULL DEFAULT '',
    tool_id      TEXT NOT NULL DEFAULT '',
    evidence     TEXT NOT NULL DEFAULT '',
    payload_hash TEXT NOT NULL,
    recorded_at  TEXT NOT NULL,
    PRIMARY KEY (outcome_id, block_id)
);

CREATE INDEX IF NOT EXISTS idx_outcome_block_id
    ON recall_outcome(block_id);
CREATE INDEX IF NOT EXISTS idx_outcome_verdict
    ON recall_outcome(block_id, outcome);
CREATE INDEX IF NOT EXISTS idx_outcome_task_id
    ON recall_outcome(task_id);
"""


def _init_calibration_schema(conn: sqlite3.Connection) -> None:
    """Create calibration tables if they don't exist."""
    conn.executescript(_CALIBRATION_SCHEMA)
    conn.commit()


# ---------------------------------------------------------------------------
# Query ID generation
# ---------------------------------------------------------------------------


def make_query_id(query: str) -> str:
    """Generate a deterministic query ID from query text + timestamp.

    Format: ``cal-<sha256_prefix>-<epoch_ms>`` to allow both dedup and
    time-ordering.
    """
    h = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    ts = int(time.time() * 1000)
    return f"cal-{h}-{ts}"


# ---------------------------------------------------------------------------
# Calibration Manager
# ---------------------------------------------------------------------------

_conn_managers: dict[str, ConnectionManager] = {}
_conn_managers_lock = __import__("threading").Lock()


def _get_conn_manager(workspace: str) -> ConnectionManager:
    """Return a shared ConnectionManager for the workspace DB."""
    path = _db_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _conn_managers_lock:
        mgr = _conn_managers.get(path)
        if mgr is None:
            mgr = ConnectionManager(path)
            _conn_managers[path] = mgr
    return mgr


@dataclass
class CalibrationScore:
    """Per-block calibration data."""

    block_id: str
    accepted: int
    rejected: int
    ignored: int
    weight: float


class CalibrationManager:
    """Track retrieval quality and compute calibration weights.

    Operates on the same SQLite database as the FTS5 index (``recall.db``),
    adding a ``calibration_feedback`` table for feedback events.
    """

    def __init__(self, workspace: str) -> None:
        self._workspace = os.path.abspath(workspace)
        self._mgr = _get_conn_manager(self._workspace)
        # Ensure schema exists
        with self._mgr.write_lock:
            conn = self._mgr.get_write_connection()
            conn.row_factory = sqlite3.Row
            _init_calibration_schema(conn)

    # -----------------------------------------------------------------------
    # Record feedback
    # -----------------------------------------------------------------------

    def record_feedback(
        self,
        query_id: str,
        block_ids_useful: list[str],
        block_ids_not_useful: list[str],
        feedback_type: str,
        query_text: str = "",
        query_type: str = "",
    ) -> dict[str, Any]:
        """Record user feedback for a retrieval query.

        Args:
            query_id: Unique identifier for the query (from recall result).
            block_ids_useful: Block IDs that were useful.
            block_ids_not_useful: Block IDs that were not useful.
            feedback_type: One of "accepted", "rejected", "ignored".
            query_text: Original query text (for analytics).
            query_type: Intent type (WHAT, WHEN, WHO, HOW, etc.).

        Returns:
            Summary dict with counts of recorded feedback.
        """
        if feedback_type not in ("accepted", "rejected", "ignored"):
            raise ValueError(f"Invalid feedback_type: {feedback_type}")

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        recorded = 0

        with self._mgr.write_lock:
            conn = self._mgr.get_write_connection()
            conn.row_factory = sqlite3.Row
            _init_calibration_schema(conn)

            # Record useful blocks as "accepted"
            for bid in block_ids_useful:
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO calibration_feedback
                           (query_id, query_text, query_type, block_id, feedback, created_at)
                           VALUES (?, ?, ?, ?, 'accepted', ?)""",
                        (query_id, query_text, query_type, bid, now),
                    )
                    recorded += 1
                except sqlite3.IntegrityError:
                    pass  # Duplicate — skip

            # Record not-useful blocks as "rejected"
            for bid in block_ids_not_useful:
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO calibration_feedback
                           (query_id, query_text, query_type, block_id, feedback, created_at)
                           VALUES (?, ?, ?, ?, 'rejected', ?)""",
                        (query_id, query_text, query_type, bid, now),
                    )
                    recorded += 1
                except sqlite3.IntegrityError:
                    pass

            # If feedback_type is "ignored", record all blocks as ignored
            if feedback_type == "ignored":
                all_blocks = list(set(block_ids_useful + block_ids_not_useful))
                for bid in all_blocks:
                    try:
                        conn.execute(
                            """INSERT OR REPLACE INTO calibration_feedback
                               (query_id, query_text, query_type, block_id, feedback, created_at)
                               VALUES (?, ?, ?, ?, 'ignored', ?)""",
                            (query_id, query_text, query_type, bid, now),
                        )
                        recorded += 1
                    except sqlite3.IntegrityError:
                        pass

            conn.commit()

        metrics.inc("calibration_feedback_recorded", recorded)
        _log.info(
            "feedback_recorded",
            query_id=query_id,
            useful=len(block_ids_useful),
            not_useful=len(block_ids_not_useful),
            feedback_type=feedback_type,
            recorded=recorded,
        )

        return {
            "query_id": query_id,
            "feedback_type": feedback_type,
            "useful_count": len(block_ids_useful),
            "not_useful_count": len(block_ids_not_useful),
            "recorded": recorded,
        }

    # -----------------------------------------------------------------------
    # Outcome attribution — utility, not quality
    # -----------------------------------------------------------------------

    def record_outcome(
        self,
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
        """Record whether acting on ``block_ids`` actually worked. Idempotent.

        See :func:`mind_mem.outcome_store.record_outcome`.
        """
        from .outcome_store import record_outcome

        return record_outcome(
            self._mgr,
            block_ids,
            outcome,
            query_id=query_id,
            task_id=task_id,
            actor_id=actor_id,
            session_id=session_id,
            tool_id=tool_id,
            evidence=evidence,
            recorded_at=recorded_at,
            project_to_calibration=project_to_calibration,
        )

    def get_outcome_signals(self, block_ids: list[str]) -> dict[str, OutcomeSignal]:
        """Batch per-block utility evidence — unwindowed, clock-free."""
        from .outcome_store import get_outcome_signals

        return get_outcome_signals(self._mgr, block_ids)

    def list_outcomes(self, block_id: str = "", task_id: str = "", limit: int = 100) -> list[dict[str, Any]]:
        """Recorded outcomes with full provenance, newest first."""
        from .outcome_store import list_outcomes

        return list_outcomes(self._mgr, block_id=block_id, task_id=task_id, limit=limit)

    def get_outcome_stats(self, top_n: int = 20) -> dict[str, Any]:
        """Utility health report — corroborated vs. implicated blocks."""
        from .outcome_store import get_outcome_stats

        return get_outcome_stats(self._mgr, top_n=top_n)

    # -----------------------------------------------------------------------
    # Compute calibration weights
    # -----------------------------------------------------------------------

    def _cutoff_date(self, now: datetime | None = None) -> str:
        """Return the ISO-8601 UTC timestamp at which the rolling window opens.

        The boundary is UTC-anchored so two hosts in different timezones cut
        the window at the same point, and it is compared against ``created_at``
        values that SQLite also writes in UTC. The window itself is
        deliberately time-relative — see the module docstring: recall scores
        that include a calibration weight are reproducible for a fixed
        as-of instant, not across arbitrary wall-clock time.

        Args:
            now: Optional as-of instant. Defaults to UTC now; a naive value
                is interpreted as UTC rather than local time.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        cutoff = now.timestamp() - (CALIBRATION_WINDOW_DAYS * 86400)
        return datetime.fromtimestamp(cutoff, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def get_block_weight(self, block_id: str) -> float:
        """Compute calibration weight for a single block.

        Returns a float in [MIN_CALIBRATION_WEIGHT, MAX_CALIBRATION_WEIGHT].
        Defaults to 1.0 if insufficient data.
        """
        cutoff = self._cutoff_date()
        conn = self._mgr.get_read_connection()
        conn.row_factory = sqlite3.Row

        try:
            rows = conn.execute(
                """SELECT feedback, COUNT(*) as cnt
                   FROM calibration_feedback
                   WHERE block_id = ? AND created_at >= ?
                   GROUP BY feedback""",
                (block_id, cutoff),
            ).fetchall()
        except sqlite3.OperationalError:
            return 1.0

        counts = {"accepted": 0, "rejected": 0, "ignored": 0}
        for row in rows:
            counts[row["feedback"]] = row["cnt"]

        return _compute_weight(counts["accepted"], counts["rejected"], counts["ignored"])

    def get_block_weights(self, block_ids: list[str]) -> dict[str, float]:
        """Compute calibration weights for multiple blocks in one query.

        Returns {block_id: weight} dict. Missing blocks default to 1.0.
        """
        if not block_ids:
            return {}

        cutoff = self._cutoff_date()
        conn = self._mgr.get_read_connection()
        conn.row_factory = sqlite3.Row

        placeholders = ",".join("?" for _ in block_ids)
        try:
            rows = conn.execute(
                f"""SELECT block_id, feedback, COUNT(*) as cnt
                    FROM calibration_feedback
                    WHERE block_id IN ({placeholders}) AND created_at >= ?
                    GROUP BY block_id, feedback""",  # nosec B608 — placeholders is `? * N`, all block_ids passed as bind params; no user input in query string
                [*block_ids, cutoff],
            ).fetchall()
        except sqlite3.OperationalError:
            return {bid: 1.0 for bid in block_ids}

        # Aggregate per block
        block_counts: dict[str, dict[str, int]] = {}
        for row in rows:
            bid = row["block_id"]
            if bid not in block_counts:
                block_counts[bid] = {"accepted": 0, "rejected": 0, "ignored": 0}
            block_counts[bid][row["feedback"]] = row["cnt"]

        result = {}
        for bid in block_ids:
            if bid in block_counts:
                c = block_counts[bid]
                result[bid] = _compute_weight(c["accepted"], c["rejected"], c["ignored"])
            else:
                result[bid] = 1.0

        return result

    # -----------------------------------------------------------------------
    # Per-query-type accuracy
    # -----------------------------------------------------------------------

    def get_query_type_accuracy(self) -> dict[str, dict[str, Any]]:
        """Compute per-query-type accuracy rates over the rolling window.

        Returns {query_type: {total, accepted, rejected, ignored, accuracy}}.
        """
        cutoff = self._cutoff_date()
        conn = self._mgr.get_read_connection()
        conn.row_factory = sqlite3.Row

        try:
            rows = conn.execute(
                """SELECT query_type, feedback, COUNT(*) as cnt
                   FROM calibration_feedback
                   WHERE created_at >= ? AND query_type != ''
                   GROUP BY query_type, feedback""",
                (cutoff,),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}

        stats: dict[str, dict[str, int]] = {}
        for row in rows:
            qt = row["query_type"]
            if qt not in stats:
                stats[qt] = {"accepted": 0, "rejected": 0, "ignored": 0}
            stats[qt][row["feedback"]] = row["cnt"]

        result = {}
        for qt, counts in stats.items():
            total = counts["accepted"] + counts["rejected"] + counts["ignored"]
            accuracy = counts["accepted"] / total if total > 0 else 0.0
            result[qt] = {
                "total": total,
                "accepted": counts["accepted"],
                "rejected": counts["rejected"],
                "ignored": counts["ignored"],
                "accuracy": round(accuracy, 4),
            }

        return result

    # -----------------------------------------------------------------------
    # Stats / diagnostics
    # -----------------------------------------------------------------------

    def get_calibration_stats(self, top_n: int = 20) -> dict[str, Any]:
        """Return calibration health report.

        Includes per-block scores (top boosted + top demoted),
        per-query-type accuracy, and overall health metrics.
        """
        cutoff = self._cutoff_date()
        conn = self._mgr.get_read_connection()
        conn.row_factory = sqlite3.Row

        # Overall counts
        try:
            total_row = conn.execute(
                """SELECT COUNT(*) as cnt FROM calibration_feedback
                   WHERE created_at >= ?""",
                (cutoff,),
            ).fetchone()
            total_feedback = total_row["cnt"] if total_row else 0
        except sqlite3.OperationalError:
            return {
                "total_feedback": 0,
                "window_days": CALIBRATION_WINDOW_DAYS,
                "message": "No calibration data available.",
            }

        # Per-block aggregation
        try:
            block_rows = conn.execute(
                """SELECT block_id, feedback, COUNT(*) as cnt
                   FROM calibration_feedback
                   WHERE created_at >= ?
                   GROUP BY block_id, feedback""",
                (cutoff,),
            ).fetchall()
        except sqlite3.OperationalError:
            block_rows = []

        block_counts: dict[str, dict[str, int]] = {}
        for row in block_rows:
            bid = row["block_id"]
            if bid not in block_counts:
                block_counts[bid] = {"accepted": 0, "rejected": 0, "ignored": 0}
            block_counts[bid][row["feedback"]] = row["cnt"]

        # Compute per-block scores
        block_scores: list[CalibrationScore] = []
        for bid, counts in block_counts.items():
            weight = _compute_weight(counts["accepted"], counts["rejected"], counts["ignored"])
            block_scores.append(
                CalibrationScore(
                    block_id=bid,
                    accepted=counts["accepted"],
                    rejected=counts["rejected"],
                    ignored=counts["ignored"],
                    weight=weight,
                )
            )

        # Sort: top boosted and top demoted
        block_scores.sort(key=lambda s: s.weight, reverse=True)
        top_boosted = [
            {
                "block_id": s.block_id,
                "weight": s.weight,
                "accepted": s.accepted,
                "rejected": s.rejected,
                "ignored": s.ignored,
            }
            for s in block_scores[:top_n]
            if s.weight > 1.0
        ]
        top_demoted = [
            {
                "block_id": s.block_id,
                "weight": s.weight,
                "accepted": s.accepted,
                "rejected": s.rejected,
                "ignored": s.ignored,
            }
            for s in reversed(block_scores)
            if s.weight < 1.0
        ][:top_n]

        # Query type accuracy
        query_type_accuracy = self.get_query_type_accuracy()

        # Unique queries and blocks
        try:
            unique_queries = conn.execute(
                """SELECT COUNT(DISTINCT query_id) as cnt
                   FROM calibration_feedback WHERE created_at >= ?""",
                (cutoff,),
            ).fetchone()["cnt"]
            unique_blocks = conn.execute(
                """SELECT COUNT(DISTINCT block_id) as cnt
                   FROM calibration_feedback WHERE created_at >= ?""",
                (cutoff,),
            ).fetchone()["cnt"]
        except sqlite3.OperationalError:
            unique_queries = 0
            unique_blocks = 0

        # Overall acceptance rate
        try:
            accept_row = conn.execute(
                """SELECT COUNT(*) as cnt FROM calibration_feedback
                   WHERE created_at >= ? AND feedback = 'accepted'""",
                (cutoff,),
            ).fetchone()
            accept_count = accept_row["cnt"] if accept_row else 0
        except sqlite3.OperationalError:
            accept_count = 0

        overall_accuracy = accept_count / total_feedback if total_feedback > 0 else 0.0

        return {
            "window_days": CALIBRATION_WINDOW_DAYS,
            "total_feedback": total_feedback,
            "unique_queries": unique_queries,
            "unique_blocks": unique_blocks,
            "overall_accuracy": round(overall_accuracy, 4),
            "top_boosted": top_boosted,
            "top_demoted": top_demoted,
            "query_type_accuracy": query_type_accuracy,
        }


# ---------------------------------------------------------------------------
# Weight computation (pure function)
# ---------------------------------------------------------------------------


def _compute_weight(accepted: int, rejected: int, ignored: int) -> float:
    """Compute calibration weight from feedback counts.

    Uses a Bayesian-inspired formula:
      - ratio = (accepted + 1) / (accepted + rejected + 2)  (Laplace smoothing)
      - weight = 0.5 + ratio (maps [0, 1] -> [0.5, 1.5])

    Ignored feedback counts as mild negative (0.3x weight of rejection)
    to demote blocks users consistently skip.

    Returns 1.0 if total feedback is below MIN_FEEDBACK_THRESHOLD.
    """
    total = accepted + rejected + ignored
    if total < MIN_FEEDBACK_THRESHOLD:
        return 1.0

    # Ignored counts as partial rejection (users skipped this block)
    effective_rejected = rejected + ignored * 0.3
    effective_total = accepted + effective_rejected

    if effective_total <= 0:
        return 1.0

    # Laplace-smoothed ratio
    ratio = (accepted + 1) / (effective_total + 2)

    # Map [0, 1] -> [MIN_CALIBRATION_WEIGHT, MAX_CALIBRATION_WEIGHT]
    weight = MIN_CALIBRATION_WEIGHT + ratio * (MAX_CALIBRATION_WEIGHT - MIN_CALIBRATION_WEIGHT)
    return round(max(MIN_CALIBRATION_WEIGHT, min(MAX_CALIBRATION_WEIGHT, weight)), 4)
