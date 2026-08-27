"""Regression gate for outcome attribution — did the memory actually help?

Locks in the acceptance contract:

  * outcomes are recorded **with provenance** and are **queryable**;
  * a block repeatedly implicated in FAILED outcomes is **demoted
    deterministically** by the recall validity gate;
  * a block in successful outcomes is **corroborated**;
  * outcome recording **never mutates block content** (governed write only);
  * replaying the same outcome twice is **idempotent**;
  * **flag-off is byte-identical** — including when outcomes already exist
    in the store.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import pytest

from mind_mem._recall_core import recall
from mind_mem.calibration import CalibrationManager
from mind_mem.init_workspace import init
from mind_mem.outcome_attribution import (
    MIN_OUTCOME_EVIDENCE,
    OUTCOME_FLOOR,
    OutcomeSignal,
    canonical_outcome_id,
    is_corroborated,
    normalize_block_ids,
    outcome_factor,
    outcome_proposal,
    outcome_stats,
    report_outcome,
    validate_outcome,
)
from mind_mem.validity_gate import validity_components

_QUERY = "outcome attribution fixture widget rollout"

_DECISIONS_BODY = """
[D-20260401-001]
Date: 2026-04-01
Status: active
Scope: global
Statement: Outcome attribution fixture widget rollout entry Alpha
Rationale: Outcome attribution regression fixture
Tags: outcome-fixture

[D-20260401-002]
Date: 2026-04-01
Status: active
Scope: global
Statement: Outcome attribution fixture widget rollout entry Beta
Rationale: Outcome attribution regression fixture
Tags: outcome-fixture

[D-20260401-003]
Date: 2026-04-01
Status: active
Scope: global
Statement: Outcome attribution fixture widget rollout entry Gamma
Rationale: Outcome attribution regression fixture
Tags: outcome-fixture
"""

_ID_A = "D-20260401-001"
_ID_B = "D-20260401-002"
_ID_C = "D-20260401-003"

_STAMP = "2026-04-02T09:00:00Z"


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


def _enable_gate(workspace: str, *, outcome: bool) -> None:
    """Flip the validity gate on (optionally with the outcome sub-flag).

    Forces a distinct config mtime so ``_recall_core``'s mtime-cached
    config reload notices the rewrite inside one test process.
    """
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


def _tuples(results: list[dict]) -> list[tuple[str, float]]:
    return [(r["_id"], r["score"]) for r in results]


def _by_id(results: list[dict]) -> dict[str, dict]:
    return {r["_id"]: r for r in results}


def _corpus_digest(workspace: str) -> str:
    """SHA-256 over every Markdown file in the workspace, path-ordered."""
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(workspace):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if not name.endswith(".md"):
                continue
            path = os.path.join(root, name)
            digest.update(os.path.relpath(path, workspace).encode("utf-8"))
            with open(path, "rb") as fh:
                digest.update(fh.read())
    return digest.hexdigest()


def _record_failures(workspace: str, block_id: str, count: int, **kw) -> None:
    for i in range(count):
        report_outcome(
            workspace,
            [block_id],
            "failure",
            task_id=f"build-{i}",
            recorded_at=_STAMP,
            **kw,
        )


# ---------------------------------------------------------------------------
# 1. Pure math — deterministic, no clock, no learned parameters
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("success", "failure", "expected"),
    [
        (0, 0, 1.0),  # no evidence at all -> neutral
        (9, 0, 1.0),  # successes never debit
        (0, 1, 1.0),  # one bad day is not a pattern
        (0, 2, 1.0),  # still under MIN_OUTCOME_EVIDENCE
        (0, 3, 0.6),  # three failures -> demotable
        (0, 5, 0.5714),
        (1, 2, 0.7),
        (2, 1, 0.8),  # majority-success stays at the threshold, not below
        (10, 1, 0.9231),
    ],
)
def test_outcome_factor_table(success: int, failure: int, expected: float) -> None:
    assert outcome_factor(success, failure) == expected


@pytest.mark.unit
def test_outcome_factor_is_bounded_and_pure() -> None:
    for success in range(0, 12):
        for failure in range(0, 12):
            value = outcome_factor(success, failure)
            assert OUTCOME_FLOOR <= value <= 1.0
            assert value == outcome_factor(success, failure)


@pytest.mark.unit
def test_corroboration_needs_a_pattern_of_successes() -> None:
    assert not is_corroborated(MIN_OUTCOME_EVIDENCE - 1, 0)
    assert is_corroborated(MIN_OUTCOME_EVIDENCE, 0)
    assert not is_corroborated(3, 3)
    assert is_corroborated(4, 3)


@pytest.mark.unit
def test_outcome_identity_is_order_independent_and_content_bound() -> None:
    first = canonical_outcome_id(normalize_block_ids(["B", "A"]), "failure", task_id="t")
    second = canonical_outcome_id(normalize_block_ids(["A", "B", "A"]), "failure", task_id="t")
    assert first == second
    assert first[0].startswith("out-")
    assert len(first[1]) == 64
    assert canonical_outcome_id(("A", "B"), "success", task_id="t") != first


@pytest.mark.unit
def test_boundary_validation() -> None:
    assert validate_outcome(" Success ") == "success"
    with pytest.raises(ValueError):
        validate_outcome("worked-ish")
    with pytest.raises(ValueError):
        normalize_block_ids([])
    with pytest.raises(ValueError):
        normalize_block_ids("D-20260401-001")  # a string is not a list of ids
    with pytest.raises(ValueError):
        normalize_block_ids([f"id-{i}" for i in range(300)])


# ---------------------------------------------------------------------------
# 2. Recording — provenance, queryability, idempotency, governed write
# ---------------------------------------------------------------------------


def test_outcomes_are_recorded_with_provenance_and_queryable(ws: str) -> None:
    result = report_outcome(
        ws,
        [_ID_B, _ID_A],
        "failure",
        query_id="q-42",
        task_id="fix-lint-gate",
        actor_id="ci-runner",
        session_id="s-7",
        tool_id="pytest",
        evidence="tests/test_lint.py :: 3 failed",
        recorded_at=_STAMP,
    )

    assert result["outcome"] == "failure"
    assert result["block_ids"] == [_ID_A, _ID_B]  # canonical (sorted) order
    assert result["recorded"] == 2
    assert result["idempotent"] is False
    assert len(result["payload_hash"]) == 64
    assert result["recorded_at"] == _STAMP

    rows = CalibrationManager(ws).list_outcomes(block_id=_ID_B)
    assert len(rows) == 1
    row = rows[0]
    assert row["outcome"] == "failure"
    assert row["query_id"] == "q-42"
    assert row["task_id"] == "fix-lint-gate"
    assert row["actor_id"] == "ci-runner"
    assert row["session_id"] == "s-7"
    assert row["tool_id"] == "pytest"
    assert row["evidence"] == "tests/test_lint.py :: 3 failed"
    assert row["payload_hash"] == result["payload_hash"]
    assert row["recorded_at"] == _STAMP

    by_task = CalibrationManager(ws).list_outcomes(task_id="fix-lint-gate")
    assert {r["block_id"] for r in by_task} == {_ID_A, _ID_B}

    stats = outcome_stats(ws)
    assert stats["total_outcomes"] == 2
    assert stats["unique_reports"] == 1
    assert stats["unique_blocks"] == 2


def test_replaying_the_same_outcome_is_idempotent(ws: str) -> None:
    kwargs = dict(task_id="build-9", actor_id="ci", evidence="green", recorded_at=_STAMP)
    first = report_outcome(ws, [_ID_B], "success", **kwargs)
    second = report_outcome(ws, [_ID_B], "success", **kwargs)
    # Same verdict replayed with a *later* clock still collapses.
    third = report_outcome(ws, [_ID_B], "success", task_id="build-9", actor_id="ci", evidence="green", recorded_at="2027-01-01T00:00:00Z")

    assert first["outcome_id"] == second["outcome_id"] == third["outcome_id"]
    assert first["recorded"] == 1
    assert second["recorded"] == 0 and second["idempotent"] is True
    assert third["recorded"] == 0 and third["idempotent"] is True
    assert third["recorded_at"] == _STAMP  # original provenance survives

    cal = CalibrationManager(ws)
    assert len(cal.list_outcomes(block_id=_ID_B)) == 1
    assert cal.get_outcome_signals([_ID_B])[_ID_B] == OutcomeSignal(_ID_B, success=1)


def test_recording_never_mutates_block_content(ws: str) -> None:
    before = _corpus_digest(ws)
    _record_failures(ws, _ID_B, 4)
    report_outcome(ws, [_ID_A], "success", recorded_at=_STAMP)
    assert _corpus_digest(ws) == before

    # The governed route is a *proposal*, not a write.
    signal = CalibrationManager(ws).get_outcome_signals([_ID_B])[_ID_B]
    proposal = outcome_proposal(signal, task_id="fix-lint-gate")
    assert proposal["block_type"] == "decision"
    assert _ID_B in proposal["statement"]
    assert len(proposal["rationale"].strip()) >= 8
    assert _corpus_digest(ws) == before


def test_calibration_projection_is_opt_in(ws: str) -> None:
    db = os.path.join(ws, ".mind-mem-index", "recall.db")

    def _feedback_rows() -> list[tuple]:
        conn = sqlite3.connect(db)
        try:
            return conn.execute("SELECT block_id, feedback FROM calibration_feedback ORDER BY block_id, feedback").fetchall()
        finally:
            conn.close()

    report_outcome(ws, [_ID_B], "failure", task_id="t1", recorded_at=_STAMP)
    assert _feedback_rows() == []

    out = report_outcome(ws, [_ID_B], "failure", task_id="t2", recorded_at=_STAMP, project_to_calibration=True)
    assert out["projected"] == 1
    assert out["calibration_feedback"] == "rejected"
    assert (_ID_B, "rejected") in _feedback_rows()

    # Replaying a projected outcome adds no second feedback row.
    report_outcome(ws, [_ID_B], "failure", task_id="t2", recorded_at=_STAMP, project_to_calibration=True)
    assert _feedback_rows().count((_ID_B, "rejected")) == 1


# ---------------------------------------------------------------------------
# 3. Validity-gate integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validity_components_default_is_the_untouched_four_criteria() -> None:
    hit = {"_id": _ID_A, "fusion_sources": ["bm25"]}
    base = validity_components(hit, set(), {})
    assert set(base) == {"corroboration", "status", "contradiction", "staleness", "score"}
    assert base["corroboration"] == 0.5
    assert base["score"] == round(0.25 * (0.5 + 1.0 + 1.0 + 1.0), 4)


@pytest.mark.unit
def test_successful_outcomes_corroborate_a_single_source_hit() -> None:
    hit = {"_id": _ID_A, "fusion_sources": ["bm25"]}
    signals = {_ID_A: OutcomeSignal(_ID_A, success=3)}
    scored = validity_components(hit, set(), {}, signals)
    assert scored["corroboration"] == 1.0  # lifted from 0.5 by real-world success
    assert scored["outcome"] == 1.0
    assert scored["score"] == 1.0


@pytest.mark.unit
def test_failed_outcomes_debit_and_absence_stays_neutral() -> None:
    hit = {"_id": _ID_B}
    assert validity_components(hit, set(), {}, {})["score"] == 1.0
    failed = validity_components(hit, set(), {}, {_ID_B: OutcomeSignal(_ID_B, failure=3)})
    assert failed["outcome"] == 0.6
    assert failed["score"] == 0.6


def test_repeated_failures_demote_deterministically_via_recall(ws: str) -> None:
    baseline = recall(ws, _QUERY, limit=10)
    assert len(baseline) >= 3, [r["_id"] for r in baseline]
    baseline_scores = dict(_tuples(baseline))

    _record_failures(ws, _ID_B, MIN_OUTCOME_EVIDENCE)
    report_outcome(ws, [_ID_C], "success", task_id="s1", recorded_at=_STAMP)
    report_outcome(ws, [_ID_C], "success", task_id="s2", recorded_at=_STAMP)
    report_outcome(ws, [_ID_C], "success", task_id="s3", recorded_at=_STAMP)

    _enable_gate(ws, outcome=True)
    on_1 = recall(ws, _QUERY, limit=10)
    hits = _by_id(on_1)

    demoted = hits[_ID_B]
    assert demoted["validity"]["outcome"] == 0.6
    assert demoted["validity"]["score"] == 0.6
    assert demoted["_validity_demoted"] is True
    assert demoted["score"] == round(baseline_scores[_ID_B] * 0.5, 4)

    for clean_id in (_ID_A, _ID_C):
        clean = hits[clean_id]
        assert clean["validity"]["outcome"] == 1.0
        assert clean["validity"]["score"] == 1.0
        assert "_validity_demoted" not in clean
        assert clean["score"] == baseline_scores[clean_id]
        assert clean["score"] > demoted["score"]

    # Determinism: two more flag-on runs are byte-identical.
    def _ordered(results: list[dict]) -> list[tuple[str, float, dict]]:
        return [(r["_id"], r["score"], r["validity"]) for r in results]

    assert _ordered(on_1) == _ordered(recall(ws, _QUERY, limit=10))
    assert _ordered(on_1) == _ordered(recall(ws, _QUERY, limit=10))


def test_flag_off_is_byte_identical_even_with_outcomes_recorded(ws: str) -> None:
    baseline = recall(ws, _QUERY, limit=10)
    baseline_tuples = _tuples(baseline)
    for hit in baseline:
        assert "validity" not in hit

    # Record a damning failure history for B — with the sub-flag off it must
    # not move a single score.
    _record_failures(ws, _ID_B, 6)
    assert _tuples(recall(ws, _QUERY, limit=10)) == baseline_tuples

    # Validity gate ON, outcome sub-flag absent: four-criteria output only.
    _enable_gate(ws, outcome=False)
    gate_only = recall(ws, _QUERY, limit=10)
    assert _tuples(gate_only) == baseline_tuples
    for hit in gate_only:
        assert "outcome" not in hit["validity"]
        assert set(hit["validity"]) == {
            "corroboration",
            "status",
            "contradiction",
            "staleness",
            "score",
        }
        assert "_validity_demoted" not in hit


# ---------------------------------------------------------------------------
# 4. MCP surface
# ---------------------------------------------------------------------------


def test_mcp_report_outcome_and_outcome_stats(ws: str) -> None:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools.calibration import outcome_stats as mcp_outcome_stats
    from mind_mem.mcp.tools.calibration import report_outcome as mcp_report_outcome

    with use_workspace(ws):
        first = json.loads(
            mcp_report_outcome(
                block_ids=[_ID_B],
                outcome="failure",
                task_id="ci-1",
                evidence="3 failed",
            )
        )
        assert first["status"] == "recorded"
        assert first["recorded"] == 1

        replay = json.loads(
            mcp_report_outcome(
                block_ids=[_ID_B],
                outcome="failure",
                task_id="ci-1",
                evidence="3 failed",
            )
        )
        assert replay["outcome_id"] == first["outcome_id"]
        assert replay["idempotent"] is True

        bad = json.loads(mcp_report_outcome(block_ids=[_ID_B], outcome="maybe"))
        assert "error" in bad and "success" in bad["error"]

        empty = json.loads(mcp_report_outcome(block_ids=[], outcome="failure"))
        assert "error" in empty

        listing = json.loads(mcp_outcome_stats(block_id=_ID_B))
        assert listing["count"] == 1
        assert listing["outcomes"][0]["task_id"] == "ci-1"

        health = json.loads(mcp_outcome_stats())
        assert health["total_outcomes"] == 1
        assert health["unique_blocks"] == 1
