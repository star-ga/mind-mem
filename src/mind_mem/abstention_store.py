# Copyright 2026 STARGA, Inc.
"""Abstention as a prediction the existing calibration loop can score.

A verdict that nobody ever checks is decoration. ``outcome_attribution``
already projects "did acting on this recall work?" back into the calibration
store; this module makes the abstention that was *shown at recall time* a thing
that loop can be right or wrong about -- which is what makes the statistic
falsifiable instead of decorative.

No parallel subsystem
---------------------
* **Same store.** The prediction is a sibling table (``recall_abstention``) in
  the ``recall.db`` that already holds ``calibration_feedback`` and
  ``recall_outcome``. The join is a single SQL statement; there is no second
  database and no second identity.
* **Same join key.** A prediction is keyed by the run identity the recall
  envelope already publishes (``attestation.query_id``, the value a client
  already passes to ``report_outcome``). ``report_outcome(run_id=...)`` and the
  legacy ``query_id=`` label land in the same physical column, so both join.
* **Same bound.** One reporter is one vote per (run, verdict); a thousand
  distinct reports from one actor read as one. The scored path is pure
  arithmetic over stored counts -- no clock, no window, no randomness -- and
  uses the loop's own :data:`~mind_mem.outcome_attribution.MIN_OUTCOME_EVIDENCE`
  as the evidence floor.

What the join says
------------------
For each band (``act`` / ``proceed_with_caution`` / ``do_not_act``) it counts
how the work that followed actually went. A discriminating abstention is one
where recalls it called ``act`` succeed more often than the ones it called
``do_not_act``. The verdict can come out ``inverted`` (confident recalls fail
*more*) and the ``silent`` flag exposes the pairing the roadmap names: failures
accumulating while the abstention never fires. Both are reported, never hidden.

Stratified by intended use and by score ``basis`` (``rrf`` rank-derived scores
mean something different from a raw BM25 gap), so a threshold that is wrong for
one regime is visible as such rather than averaged away.

Governance
----------
This module writes ONLY the sidecar prediction table -- never block content, and
never anything the ranking reads. The table is created by the first prediction,
so a workspace that never asks for a verdict has no such table, no write and no
cost. A (run, use) already recorded is detected by a READ and left alone:
``recall.db`` is part of the recall cache's corpus stamp, and rewriting an
existing row on every replay would retire every cache entry for nothing.
A recorded prediction is immutable (first write wins), so what was shown at
recall time is what gets scored.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any

from .connection_manager import ConnectionManager
from .observability import get_logger
from .outcome_attribution import MIN_OUTCOME_EVIDENCE
from .recall_abstention import BAND_ACT, BAND_DO_NOT_ACT, BANDS, INTENDED_USES, assess, resolve_policy

_log = get_logger("abstention_store")

__all__ = [
    "abstention_stats",
    "insert_prediction",
    "judge",
    "judge_served",
    "record_assessment",
    "record_prediction",
    "stats_from_manager",
]

#: A run identity is the 64-character lowercase hex ``recall_digests.run_id``.
_RUN_ID = re.compile(r"^[0-9a-f]{64}$")
_MAX_BASIS = 64
_MAX_STAMP = 32

_SCHEMA = (
    """
CREATE TABLE IF NOT EXISTS recall_abstention (
    run_id       TEXT NOT NULL,
    intended_use TEXT NOT NULL,
    band         TEXT NOT NULL,
    separation   REAL NOT NULL,
    candidates   INTEGER NOT NULL,
    basis        TEXT NOT NULL DEFAULT '',
    recorded_at  TEXT NOT NULL,
    PRIMARY KEY (run_id, intended_use)
)
""",
    "CREATE INDEX IF NOT EXISTS idx_abstention_run ON recall_abstention(run_id)",
)

#: Verdicts :func:`judge` can return.
VERDICT_INSUFFICIENT = "insufficient_evidence"
VERDICT_DISCRIMINATING = "discriminating"
VERDICT_NOT_DISCRIMINATING = "not_discriminating"
VERDICT_INVERTED = "inverted"


# ---------------------------------------------------------------------------
# Validation (boundary)
# ---------------------------------------------------------------------------


def _validated(run_id: Any, assessment: Mapping[str, Any], recorded_at: Any) -> tuple[str, str, str, float, int, str, str]:
    """``(run_id, use, band, separation, candidates, basis, stamp)`` or ``ValueError``."""
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
        raise ValueError("run_id must be a 64-character lowercase hexadecimal run identity")
    if not isinstance(assessment, Mapping):
        raise ValueError("assessment must be a mapping")
    use = assessment.get("intended_use")
    if use not in INTENDED_USES:
        raise ValueError(f"assessment.intended_use must be one of {list(INTENDED_USES)}, got {use!r}")
    band = assessment.get("band")
    if band not in BANDS:
        raise ValueError(f"assessment.band must be one of {list(BANDS)}, got {band!r}")
    separation = assessment.get("separation")
    if isinstance(separation, bool) or not isinstance(separation, (int, float)) or not math.isfinite(float(separation)):
        raise ValueError("assessment.separation must be a finite number")
    if not 0.0 <= float(separation) <= 1.0:
        raise ValueError("assessment.separation must be in [0, 1]")
    candidates = assessment.get("candidates")
    if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 0:
        raise ValueError("assessment.candidates must be a non-negative integer")
    basis = assessment.get("basis", "")
    if not isinstance(basis, str) or len(basis) > _MAX_BASIS:
        raise ValueError(f"assessment.basis must be a string of at most {_MAX_BASIS} characters")
    if not isinstance(recorded_at, str) or not recorded_at or len(recorded_at) > _MAX_STAMP:
        raise ValueError(f"recorded_at must be a non-empty string of at most {_MAX_STAMP} characters")
    return run_id, str(use), str(band), float(separation), int(candidates), basis, recorded_at


# ---------------------------------------------------------------------------
# Writing the prediction
# ---------------------------------------------------------------------------


def insert_prediction(mgr: ConnectionManager, run_id: str, assessment: Mapping[str, Any], recorded_at: str) -> bool:
    """Record one prediction over *mgr*. ``True`` when a new row was written.

    Idempotent and first-write-wins. The existence check is a READ so a replay
    of a recorded (run, use) never touches the database file (see the module
    docstring). Sidecar only: this writes no block and nothing recall reads.
    """
    row = _validated(run_id, assessment, recorded_at)
    read = mgr.get_read_connection()
    try:
        if read.execute("SELECT 1 FROM recall_abstention WHERE run_id = ? AND intended_use = ?", (row[0], row[1])).fetchone():
            return False
    except sqlite3.OperationalError:
        pass  # no table yet -- the first prediction creates it below
    with mgr.write_lock:
        conn = mgr.get_write_connection()
        for statement in _SCHEMA:
            conn.execute(statement)
        cur = conn.execute(
            """INSERT INTO recall_abstention
               (run_id, intended_use, band, separation, candidates, basis, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, intended_use) DO NOTHING""",
            row,
        )
        conn.commit()
        return max(0, cur.rowcount) == 1


def record_prediction(workspace: str, run_id: str, assessment: Mapping[str, Any], recorded_at: str) -> bool:
    """Record the abstention shown for a run, in the workspace's calibration store.

    Validates BEFORE touching the store, so a rejected prediction leaves no
    trace. ``recorded_at`` is provenance only -- callers pass the recall's
    injected scoring instant, so recording reads no clock.
    """
    _validated(run_id, assessment, recorded_at)
    from .calibration import CalibrationManager

    return bool(CalibrationManager(workspace).record_abstention(run_id, assessment, recorded_at))


def record_assessment(workspace: str, assessment: Mapping[str, Any], attestation: Any) -> str | None:
    """Record a door's verdict against the run its attestation publishes.

    Returns ``None`` on success AND when there is nothing to join (no
    attestation, or one that carries no run identity -- a derivation that
    failed): a row keyed on junk would never join an outcome. Returns a
    human-readable warning when a recordable prediction could not be written,
    so a door can surface it; recall itself is never failed by the sidecar.
    """
    if not isinstance(attestation, Mapping):
        return None
    run_id = attestation.get("query_id")
    stamp = attestation.get("scoring_instant")
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id) or not isinstance(stamp, str) or not stamp:
        return None
    try:
        record_prediction(workspace, run_id, assessment, stamp)
    except Exception as exc:  # noqa: BLE001 -- a sidecar write must never cost the answer
        _log.warning("abstention_prediction_not_recorded", error=str(exc))
        return f"Abstention prediction not recorded: {exc}"
    return None


def judge_served(
    workspace: str,
    results: Any,
    *,
    intended_use: str | None,
    recall_cfg: Any,
) -> tuple[dict[str, Any], tuple[str, ...], str | None]:
    """Verdict + prediction record for a door that serves via ``mind_mem.recall.recall``.

    The CLI and the stdlib HTTP route reach the engine through the serving entry
    rather than through the MCP handler, so they share this one helper instead
    of each re-deriving the policy, the basis label and the recording rule.
    Returns ``(verdict, policy_warnings, record_problem)``: the warnings belong
    in the door's ``config_warnings``, the problem (``None`` when the
    prediction was written or there was nothing to join) in its ``warnings``.

    *results* is the ``ServedResults`` the serving entry returned; its
    ``.attestation`` carries the run identity the prediction is keyed on.
    """
    section = recall_cfg if isinstance(recall_cfg, Mapping) else None
    policy = resolve_policy(section)
    backend = section.get("backend") if section is not None else None
    verdict = assess(results, intended_use=intended_use, policy=policy, basis=backend if isinstance(backend, str) and backend else "scan")
    problem = record_assessment(workspace, verdict, getattr(results, "attestation", None))
    return verdict, policy.warnings, problem


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _rate(success: int, failure: int) -> float | None:
    decisive = success + failure
    return round(success / decisive, 4) if decisive else None


def _band_cell(counts: Mapping[str, int]) -> dict[str, Any]:
    success = int(counts.get("success", 0))
    failure = int(counts.get("failure", 0))
    return {
        "success": success,
        "failure": failure,
        "neutral": int(counts.get("neutral", 0)),
        "votes": success + failure,
        "success_rate": _rate(success, failure),
    }


def _empty_bands() -> dict[str, dict[str, int]]:
    return {band: {"success": 0, "failure": 0, "neutral": 0} for band in BANDS}


def judge(band_counts: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """Is the abstention discriminating? Pure arithmetic over integer counts.

    Compares the success rate of recalls called ``act`` with those called
    ``do_not_act``; ``proceed_with_caution`` makes no claim and neither rescues
    nor sinks the comparison. Each side needs at least
    :data:`~mind_mem.outcome_attribution.MIN_OUTCOME_EVIDENCE` decisive votes,
    the same floor the outcome loop itself applies.

    ``silent`` is the pairing the roadmap names: at least that many failures,
    none of which the abstention flagged (all landed under ``act``).
    """
    act = band_counts.get(BAND_ACT, {})
    stop = band_counts.get(BAND_DO_NOT_ACT, {})
    act_s, act_f = int(act.get("success", 0)), int(act.get("failure", 0))
    stop_s, stop_f = int(stop.get("success", 0)), int(stop.get("failure", 0))
    total_failures = sum(int(cell.get("failure", 0)) for cell in band_counts.values())
    silent = total_failures >= MIN_OUTCOME_EVIDENCE and total_failures == act_f
    if act_s + act_f < MIN_OUTCOME_EVIDENCE or stop_s + stop_f < MIN_OUTCOME_EVIDENCE:
        verdict = VERDICT_INSUFFICIENT
    else:
        act_rate = act_s / (act_s + act_f)
        stop_rate = stop_s / (stop_s + stop_f)
        if act_rate > stop_rate:
            verdict = VERDICT_DISCRIMINATING
        elif act_rate < stop_rate:
            verdict = VERDICT_INVERTED
        else:
            verdict = VERDICT_NOT_DISCRIMINATING
    return {
        "verdict": verdict,
        "silent": silent,
        "overconfident_failures": act_f,
        "min_evidence": MIN_OUTCOME_EVIDENCE,
    }


def _group(rows: Iterable[tuple[str, str, str, str, int]], position: int) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        key = str(row[position])
        grouped.setdefault(key, _empty_bands())[row[0]][row[3]] += int(row[4])
    return {key: {band: _band_cell(cells) for band, cells in bands.items()} for key, bands in sorted(grouped.items())}


def stats_from_manager(mgr: ConnectionManager) -> dict[str, Any]:
    """The abstention scoring report over *mgr*'s store. Read-only, clock-free."""
    conn = mgr.get_read_connection()
    try:
        predictions = int(conn.execute("SELECT COUNT(*) FROM recall_abstention").fetchone()[0])
    except sqlite3.OperationalError:
        predictions = 0
    rows: list[tuple[str, str, str, str, int]] = []
    scored = 0
    if predictions:
        try:
            # One vote per (run, reporter, verdict): DISTINCT is the bound.
            rows = [
                (str(r[0]), str(r[1]), str(r[2]), str(r[3]), int(r[4]))
                for r in conn.execute(
                    """SELECT p.band, p.intended_use, p.basis, o.outcome, COUNT(*)
                       FROM recall_abstention p
                       JOIN (SELECT DISTINCT query_id AS run_id, actor_id, outcome
                             FROM recall_outcome WHERE query_id != '') o
                         ON o.run_id = p.run_id
                       GROUP BY p.band, p.intended_use, p.basis, o.outcome"""
                ).fetchall()
            ]
            scored = int(
                conn.execute(
                    """SELECT COUNT(*) FROM recall_abstention p
                       WHERE EXISTS (SELECT 1 FROM recall_outcome o
                                     WHERE o.query_id = p.run_id AND o.outcome IN ('success', 'failure'))"""
                ).fetchone()[0]
            )
        except sqlite3.OperationalError:
            rows, scored = [], 0
    totals = _empty_bands()
    for band, _use, _basis, outcome, count in rows:
        totals[band][outcome] += count
    return {
        "predictions": predictions,
        "scored_predictions": scored,
        "min_evidence": MIN_OUTCOME_EVIDENCE,
        "bands": {band: _band_cell(totals[band]) for band in BANDS},
        "by_use": _group(rows, 1),
        "by_basis": _group(rows, 2),
        "calibration": judge(totals),
    }


def abstention_stats(workspace: str) -> dict[str, Any]:
    """The abstention scoring report for a workspace (see :func:`stats_from_manager`)."""
    from .calibration import CalibrationManager

    report: dict[str, Any] = CalibrationManager(workspace).get_abstention_stats()
    return report
