"""Abuse bounds for outcome attribution — one reporter, one vote.

Companion gate to ``tests/test_outcome_attribution.py``, which locks in the
*behaviour*. This file locks in the **limits**, both of which are stated
contracts rather than emergent properties:

  * the opt-in ``calibration_feedback`` projection is keyed on the reporting
    ``actor_id``, so a caller filing a thousand distinct reports about one
    block moves its calibration weight by exactly one vote's worth — the same
    as filing one;
  * the counts the validity gate scores on **saturate** at twice
    ``MIN_OUTCOME_EVIDENCE``, so past six reports of a verdict, reporting
    harder buys nothing;
  * neither limit changes anything at honest volumes, and neither touches the
    deterministic, unwindowed ``recall_outcome`` path the gate reads.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

import pytest

from mind_mem._recall_core import recall
from mind_mem.calibration import CalibrationManager
from mind_mem.init_workspace import init
from mind_mem.outcome_attribution import (
    MIN_OUTCOME_EVIDENCE,
    outcome_factor,
    report_outcome,
)
from mind_mem.outcome_store import _ANONYMOUS_ACTOR, _OUTCOME_COUNT_CAP

_QUERY = "outcome bounds fixture widget rollout"

_DECISIONS_BODY = """
[D-20260501-001]
Date: 2026-05-01
Status: active
Scope: global
Statement: Outcome bounds fixture widget rollout entry Alpha
Rationale: Outcome bounds regression fixture
Tags: outcome-bounds

[D-20260501-002]
Date: 2026-05-01
Status: active
Scope: global
Statement: Outcome bounds fixture widget rollout entry Beta
Rationale: Outcome bounds regression fixture
Tags: outcome-bounds

[D-20260501-003]
Date: 2026-05-01
Status: active
Scope: global
Statement: Outcome bounds fixture widget rollout entry Gamma
Rationale: Outcome bounds regression fixture
Tags: outcome-bounds
"""

_ID_A = "D-20260501-001"
_ID_B = "D-20260501-002"
_ID_C = "D-20260501-003"

#: How many forged reports one actor files in the abuse tests.
_FORGED = 100


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def ws(tmp_path) -> str:
    """A seeded workspace with three clean, active fixture blocks."""
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    init(workspace)
    with open(os.path.join(workspace, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as fh:
        fh.write(_DECISIONS_BODY)
    return workspace


def _in_window_stamp() -> str:
    """A timestamp inside the 30-day calibration window.

    The *projection* weight is windowed on ``created_at``, so a fixed past
    date would silently drop every row out of scope and make these assertions
    vacuous. The scored ``recall_outcome`` path stays unwindowed either way.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _enable_gate(workspace: str, *, outcome: bool) -> None:
    """Flip the validity gate on, optionally with the outcome sub-flag."""
    cfg_path = os.path.join(workspace, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    gate: dict[str, object] = {"enabled": True}
    if outcome:
        gate["outcome_attribution"] = {"enabled": True}
    cfg["recall"]["validity_gate"] = gate
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    bumped = os.path.getmtime(cfg_path) + 5.0
    os.utime(cfg_path, (bumped, bumped))


def _feedback_rows(workspace: str, block_id: str) -> list[tuple[str, str]]:
    """``(query_id, feedback)`` projection rows for one block, ordered."""
    db = os.path.join(workspace, ".mind-mem-index", "recall.db")
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            """SELECT query_id, feedback FROM calibration_feedback
               WHERE block_id = ? ORDER BY query_id, feedback""",
            (block_id,),
        ).fetchall()
    finally:
        conn.close()


def _flood(
    workspace: str,
    block_id: str,
    verdict: str,
    count: int,
    *,
    actor_id: str = "pumper",
    project: bool = True,
    prefix: str = "forge",
) -> None:
    """File ``count`` *distinct* reports — distinct payloads, one reporter."""
    stamp = _in_window_stamp()
    for i in range(count):
        report_outcome(
            workspace,
            [block_id],
            verdict,
            task_id=f"{prefix}-{i}",
            actor_id=actor_id,
            recorded_at=stamp,
            project_to_calibration=project,
        )


def _score(results: list[dict], block_id: str) -> float:
    return next(hit["score"] for hit in results if hit["_id"] == block_id)


def _tuples(results: list[dict]) -> list[tuple[str, float]]:
    return [(r["_id"], r["score"]) for r in results]


# ---------------------------------------------------------------------------
# 1. The projection path — one reporter is worth one vote
# ---------------------------------------------------------------------------


def test_forged_successes_cannot_pump_the_projection_path(ws: str) -> None:
    """100 distinct forged successes buy exactly what one honest one buys."""
    baseline = _score(recall(ws, _QUERY, limit=10), _ID_B)
    assert CalibrationManager(ws).get_block_weight(_ID_B) == 1.0

    # One honest report from a different reporter, for the comparison bound.
    _flood(ws, _ID_C, "success", 1, actor_id="honest", prefix="honest")
    honest_weight = CalibrationManager(ws).get_block_weight(_ID_C)

    _flood(ws, _ID_B, "success", _FORGED)

    # Exactly one projected vote, not _FORGED of them.
    assert _feedback_rows(ws, _ID_B) == [("outcome:pumper", "accepted")]
    # ...and that single vote is under MIN_FEEDBACK_THRESHOLD, so the weight
    # does not move at all: 100 forged successes == 1 honest success == 1.0.
    assert CalibrationManager(ws).get_block_weight(_ID_B) == honest_weight == 1.0
    assert _score(recall(ws, _QUERY, limit=10), _ID_B) == baseline


def test_forged_failures_cannot_bury_via_the_projection_path(ws: str) -> None:
    """100 distinct forged failures likewise collapse to a single vote."""
    baseline = _score(recall(ws, _QUERY, limit=10), _ID_B)

    _flood(ws, _ID_C, "failure", 1, actor_id="honest", prefix="honest")
    honest_weight = CalibrationManager(ws).get_block_weight(_ID_C)

    _flood(ws, _ID_B, "failure", _FORGED)

    assert _feedback_rows(ws, _ID_B) == [("outcome:pumper", "rejected")]
    assert CalibrationManager(ws).get_block_weight(_ID_B) == honest_weight == 1.0
    assert _score(recall(ws, _QUERY, limit=10), _ID_B) == baseline


def test_one_reporter_is_capped_at_three_projected_votes(ws: str) -> None:
    """A reporter's entire projection budget is one row per verdict.

    Filing all three verdicts many times each yields three rows and a weight
    of 0.9651 — the furthest a single reporter can move any block, and it can
    only move it *down*: there is no way for one reporter to exceed 1.0.
    """
    for verdict in ("success", "failure", "neutral"):
        _flood(ws, _ID_B, verdict, 40, prefix=verdict)

    assert _feedback_rows(ws, _ID_B) == [
        ("outcome:pumper", "accepted"),
        ("outcome:pumper", "ignored"),
        ("outcome:pumper", "rejected"),
    ]
    assert CalibrationManager(ws).get_block_weight(_ID_B) == 0.9651


def test_distinct_reporters_still_vote_independently(ws: str) -> None:
    """The fix bounds one caller, it does not mute honest corroboration."""
    for actor in ("alice", "bob", "carol"):
        _flood(ws, _ID_B, "success", 1, actor_id=actor, prefix=actor)

    assert _feedback_rows(ws, _ID_B) == [
        ("outcome:alice", "accepted"),
        ("outcome:bob", "accepted"),
        ("outcome:carol", "accepted"),
    ]
    # Three independent reporters clear MIN_FEEDBACK_THRESHOLD and do lift it.
    assert CalibrationManager(ws).get_block_weight(_ID_B) == 1.3


def test_unattributed_reports_share_one_anonymous_vote(ws: str) -> None:
    """A blank ``actor_id`` gets the documented stable fallback, not a pass."""
    _flood(ws, _ID_B, "failure", 25, actor_id="", prefix="anon")

    assert _feedback_rows(ws, _ID_B) == [(f"outcome:{_ANONYMOUS_ACTOR}", "rejected")]
    assert CalibrationManager(ws).get_block_weight(_ID_B) == 1.0


def test_projection_reports_zero_after_the_first_vote(ws: str) -> None:
    """``projected`` is honest: 1 on the first report, 0 on every later one."""
    stamp = _in_window_stamp()
    first = report_outcome(
        ws,
        [_ID_B],
        "success",
        task_id="p-0",
        actor_id="ci",
        recorded_at=stamp,
        project_to_calibration=True,
    )
    later = report_outcome(
        ws,
        [_ID_B],
        "success",
        task_id="p-1",
        actor_id="ci",
        recorded_at=stamp,
        project_to_calibration=True,
    )
    assert first["projected"] == 1
    assert later["projected"] == 0  # distinct report, vote already cast
    assert first["outcome_id"] != later["outcome_id"]
    assert later["recorded"] == 1  # the deterministic path still records it


# ---------------------------------------------------------------------------
# 2. The scored path — counts saturate by contract
# ---------------------------------------------------------------------------


def test_outcome_counts_saturate_at_twice_min_evidence(ws: str) -> None:
    """1000-reports-reads-as-6 is the invariant; here 60 reads as 6."""
    assert _OUTCOME_COUNT_CAP == 2 * MIN_OUTCOME_EVIDENCE == 6

    _flood(ws, _ID_B, "failure", 60, project=False, prefix="fail")
    _flood(ws, _ID_B, "success", 60, project=False, prefix="pass")

    signal = CalibrationManager(ws).get_outcome_signals([_ID_B])[_ID_B]
    assert signal.failure == _OUTCOME_COUNT_CAP
    assert signal.success == _OUTCOME_COUNT_CAP

    # The operator health report is NOT clamped — true volume stays visible.
    stats = CalibrationManager(ws).get_outcome_stats()
    assert stats["total_outcomes"] == 120


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 6])
def test_saturation_is_a_no_op_at_honest_volumes(ws: str, count: int) -> None:
    """Any count at or below the cap is returned exactly as stored."""
    _flood(ws, _ID_B, "failure", count, project=False, prefix="honest")
    signal = CalibrationManager(ws).get_outcome_signals([_ID_B])[_ID_B]
    assert signal.failure == count
    assert signal.success == 0


def test_saturation_pins_the_gate_at_the_cap(ws: str) -> None:
    """Past the cap the gate stops moving — and the ranking never moved.

    Three honest failures already demote; forging 47 more walks the utility
    factor down to its saturated value and no further, and because the gate's
    demotion is a fixed multiplier rather than a function of the factor, the
    scores an operator actually sees are identical throughout.
    """
    _enable_gate(ws, outcome=True)
    _flood(ws, _ID_B, "failure", MIN_OUTCOME_EVIDENCE, project=False, prefix="honest")

    honest = recall(ws, _QUERY, limit=10)
    demoted = next(h for h in honest if h["_id"] == _ID_B)
    assert demoted["validity"]["outcome"] == 0.6
    assert demoted["_validity_demoted"] is True

    _flood(ws, _ID_B, "failure", 47, project=False, prefix="forge")  # 50 stored
    at_cap = recall(ws, _QUERY, limit=10)
    capped = next(h for h in at_cap if h["_id"] == _ID_B)
    assert capped["validity"]["outcome"] == outcome_factor(0, _OUTCOME_COUNT_CAP) == 0.5625

    _flood(ws, _ID_B, "failure", 44, project=False, prefix="more")  # 94 stored
    saturated = recall(ws, _QUERY, limit=10)
    assert [(h["_id"], h["score"], h["validity"]) for h in saturated] == [(h["_id"], h["score"], h["validity"]) for h in at_cap]

    # The visible ranking is unchanged from the honest three onward.
    assert _tuples(honest) == _tuples(at_cap) == _tuples(saturated)

    clean = next(h for h in saturated if h["_id"] == _ID_A)
    assert clean["validity"]["outcome"] == 1.0
    assert "_validity_demoted" not in clean


# ---------------------------------------------------------------------------
# 3. Exact replay is still a no-op on both paths
# ---------------------------------------------------------------------------


def test_exact_replay_stays_idempotent_on_both_paths(ws: str) -> None:
    """Replaying one identical report 100 times writes one row, once."""
    stamp = _in_window_stamp()
    kwargs = dict(
        task_id="build-9",
        actor_id="ci",
        evidence="green",
        recorded_at=stamp,
        project_to_calibration=True,
    )
    first = report_outcome(ws, [_ID_B], "success", **kwargs)
    for _ in range(_FORGED - 1):
        replay = report_outcome(ws, [_ID_B], "success", **kwargs)
        assert replay["outcome_id"] == first["outcome_id"]
        assert replay["idempotent"] is True
        assert replay["recorded"] == 0

    cal = CalibrationManager(ws)
    assert len(cal.list_outcomes(block_id=_ID_B)) == 1
    assert cal.get_outcome_signals([_ID_B])[_ID_B].success == 1
    assert _feedback_rows(ws, _ID_B) == [("outcome:ci", "accepted")]
    assert cal.get_block_weight(_ID_B) == 1.0
