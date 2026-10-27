# Copyright 2026 STARGA, Inc.
"""Abstention as a PREDICTION the existing calibration loop can score.

A verdict nobody ever checks is decoration. ``outcome_attribution`` already
projects "did acting on this recall work?" back into the calibration store;
this module's job is to make the abstention that was shown at recall time
something that loop can be right or wrong about:

* the verdict is recorded against the run's published identity (the
  ``attestation.query_id`` a client already passes to ``report_outcome``),
  as a sibling table in the SAME store -- no second database, no second join
  key;
* the scoring is a pure function of integer counts -- no clock, no window;
* one reporter is one vote per (run, verdict), exactly the bound the outcome
  projection already applies;
* nothing here is read by recall: the prediction describes a ranking and can
  never feed back into one.

The tests below drive the REAL path end to end -- MCP recall (ranker stubbed
to known scores) -> served ledger -> ``report_outcome(run_id=...)`` -> stats --
because a scoring function fed hand-built rows proves nothing about whether the
rows it expects are ever written.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from tests._abstention_fixture import fake_engine, make_workspace

CLEAR = [10.0, 1.0]  # separation 0.9  -> act for every use
FLAT = [5.0, 4.9, 4.8]  # separation 0.02 -> do_not_act for every use
MIDDLING = [10.0, 5.5, 1.0]  # separation 0.45 -> caution for answer


@pytest.fixture()
def world(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    import mind_mem.mcp.tools.recall as mcp_recall
    from mind_mem import recall_cache

    ws = make_workspace(str(tmp_path))
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    recall_cache.reset_singleton()

    class World:
        pass

    w = World()
    w.ws = ws  # type: ignore[attr-defined]
    w.door = mcp_recall  # type: ignore[attr-defined]
    w.monkeypatch = monkeypatch  # type: ignore[attr-defined]

    def serve(query: str, scores: list[float], use: str = "answer", basis: str = "sqlite") -> str:
        monkeypatch.setattr(mcp_recall, "_recall_impl_uncached", fake_engine(scores, backend=basis))
        env = json.loads(mcp_recall._recall_impl(query, limit=10, backend="bm25", intended_use=use))
        return str(env["attestation"]["query_id"])

    w.serve = serve  # type: ignore[attr-defined]
    try:
        yield w
    finally:
        recall_cache.reset_singleton()


def _outcome(ws: str, run_id: str, verdict: str, *, actor: str = "alice", evidence: str = "") -> dict[str, Any]:
    from mind_mem.outcome_attribution import report_outcome

    return report_outcome(ws, None, verdict, run_id=run_id, actor_id=actor, evidence=evidence)


def _stats(ws: str) -> dict[str, Any]:
    from mind_mem.abstention_store import abstention_stats

    return abstention_stats(ws)


def _tables(ws: str) -> set[str]:
    import os

    conn = sqlite3.connect(os.path.join(ws, ".mind-mem-index", "recall.db"))
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_table_is_created_by_the_first_prediction_and_not_before(tmp_path: Any) -> None:
    from mind_mem.abstention_store import record_prediction
    from mind_mem.calibration import CalibrationManager
    from mind_mem.recall_abstention import assess

    ws = make_workspace(str(tmp_path))
    CalibrationManager(ws)  # the loop's own store is initialised...
    tables = _tables(ws)
    assert "recall_outcome" in tables  # positive control: the calibration schema ran
    assert "recall_abstention" not in tables  # ...and the new table does not exist yet

    verdict = assess([{"_id": "a", "score": 9.0}, {"_id": "b", "score": 1.0}], intended_use="answer")
    assert record_prediction(ws, "a" * 64, verdict, "2026-10-09") is True
    assert "recall_abstention" in _tables(ws)


def test_record_is_idempotent_and_first_write_wins(tmp_path: Any) -> None:
    from mind_mem.abstention_store import abstention_stats, record_prediction
    from mind_mem.recall_abstention import assess

    ws = make_workspace(str(tmp_path))
    strong = assess([{"_id": "a", "score": 9.0}, {"_id": "b", "score": 1.0}], intended_use="answer")
    weak = assess([{"_id": "a", "score": 5.0}, {"_id": "b", "score": 4.9}], intended_use="answer")
    assert record_prediction(ws, "b" * 64, strong, "2026-10-09") is True
    assert record_prediction(ws, "b" * 64, weak, "2026-10-10") is False
    assert abstention_stats(ws)["predictions"] == 1
    conn = sqlite3.connect(f"{ws}/.mind-mem-index/recall.db")
    try:
        row = conn.execute("SELECT band, recorded_at FROM recall_abstention").fetchone()
    finally:
        conn.close()
    assert row == ("act", "2026-10-09")


def test_a_replayed_prediction_is_detected_by_a_read_and_takes_no_write_connection(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``recall.db`` is part of the recall cache's corpus stamp, so replaying a
    recorded (run, use) must not even open a write transaction on it."""
    from mind_mem.abstention_store import insert_prediction, record_prediction
    from mind_mem.calibration import _get_conn_manager
    from mind_mem.recall_abstention import assess

    ws = make_workspace(str(tmp_path))
    verdict = assess([{"_id": "a", "score": 9.0}, {"_id": "b", "score": 1.0}], intended_use="answer")
    assert record_prediction(ws, "c" * 64, verdict, "2026-10-09") is True
    mgr = _get_conn_manager(ws)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("took the write connection")

    monkeypatch.setattr(mgr, "get_write_connection", boom)
    assert insert_prediction(mgr, "c" * 64, verdict, "2026-10-09") is False
    # Positive control: a genuinely new run DOES need the write connection.
    with pytest.raises(AssertionError, match="write connection"):
        insert_prediction(mgr, "d" * 64, verdict, "2026-10-09")


def test_concurrent_recording_is_exactly_once(tmp_path: Any) -> None:
    """Eight threads racing to record the same (run, use) write one row and
    exactly one of them is told it won; eight distinct runs write eight."""
    from concurrent.futures import ThreadPoolExecutor

    from mind_mem.abstention_store import abstention_stats, record_prediction
    from mind_mem.recall_abstention import assess

    ws = make_workspace(str(tmp_path))
    verdict = assess([{"_id": "a", "score": 9.0}, {"_id": "b", "score": 1.0}], intended_use="answer")
    with ThreadPoolExecutor(max_workers=8) as pool:
        same = list(pool.map(lambda _i: record_prediction(ws, "e" * 64, verdict, "2026-10-09"), range(8)))
    assert same.count(True) == 1 and same.count(False) == 7
    assert abstention_stats(ws)["predictions"] == 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        distinct = list(pool.map(lambda i: record_prediction(ws, f"{i:064x}", verdict, "2026-10-09"), range(1, 9)))
    assert all(distinct)
    assert abstention_stats(ws)["predictions"] == 9


def test_a_lost_race_cannot_overwrite_the_recorded_prediction(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The existence check is a READ and so can lose a race. When it does, the
    INSERT's own conflict clause is what keeps the first prediction -- simulated
    deterministically by making the read miss a row that is really there."""
    from mind_mem.abstention_store import insert_prediction, record_prediction
    from mind_mem.calibration import _get_conn_manager
    from mind_mem.recall_abstention import assess

    ws = make_workspace(str(tmp_path))
    strong = assess([{"_id": "a", "score": 9.0}, {"_id": "b", "score": 1.0}], intended_use="answer")
    weak = assess([{"_id": "a", "score": 5.0}, {"_id": "b", "score": 4.9}], intended_use="answer")
    assert record_prediction(ws, "f" * 64, strong, "2026-10-09") is True
    mgr = _get_conn_manager(ws)

    class _MissesTheRow:
        def execute(self, *_a: Any, **_k: Any) -> Any:
            class _Empty:
                def fetchone(self) -> None:
                    return None

            return _Empty()

    monkeypatch.setattr(mgr, "get_read_connection", lambda: _MissesTheRow())
    assert insert_prediction(mgr, "f" * 64, weak, "2026-10-10") is False
    monkeypatch.undo()
    conn = sqlite3.connect(f"{ws}/.mind-mem-index/recall.db")
    try:
        assert conn.execute("SELECT band, separation, recorded_at FROM recall_abstention").fetchall() == [
            ("act", strong["separation"], "2026-10-09")
        ]
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("run_id", "patch"),
    [
        ("short", {}),
        ("G" * 64, {}),
        ("a" * 64, {"band": "maybe"}),
        ("a" * 64, {"intended_use": "yolo"}),
        ("a" * 64, {"separation": float("nan")}),
        ("a" * 64, {"separation": 1.5}),
        ("a" * 64, {"candidates": -1}),
        ("a" * 64, {"candidates": True}),
        ("a" * 64, {"basis": "x" * 500}),
    ],
)
def test_record_validates_at_the_boundary_and_leaves_no_trace(tmp_path: Any, run_id: str, patch: dict[str, Any]) -> None:
    from mind_mem.abstention_store import record_prediction
    from mind_mem.recall_abstention import assess

    ws = make_workspace(str(tmp_path))
    good = assess([{"_id": "a", "score": 9.0}, {"_id": "b", "score": 1.0}], intended_use="answer")
    with pytest.raises(ValueError):
        record_prediction(ws, run_id, {**good, **patch}, "2026-10-09")
    assert "recall_abstention" not in _tables(ws)  # rejected BEFORE the table is made


# ---------------------------------------------------------------------------
# Scoring, end to end
# ---------------------------------------------------------------------------


def test_predictions_without_outcomes_are_counted_but_not_scored(world: Any) -> None:
    for index in range(3):
        world.serve(f"unreported {index}", CLEAR)
    stats = _stats(world.ws)
    assert stats["predictions"] == 3
    assert stats["scored_predictions"] == 0
    assert stats["calibration"]["verdict"] == "insufficient_evidence"
    assert all(stats["bands"][band]["votes"] == 0 for band in stats["bands"])


def test_a_discriminating_abstention_is_recognised(world: Any) -> None:
    acts = [world.serve(f"clear {i}", CLEAR) for i in range(4)]
    stops = [world.serve(f"flat {i}", FLAT) for i in range(4)]
    for run in acts:
        _outcome(world.ws, run, "success")
    for run in stops:
        _outcome(world.ws, run, "failure")
    stats = _stats(world.ws)
    assert stats["bands"]["act"] == {"success": 4, "failure": 0, "neutral": 0, "votes": 4, "success_rate": 1.0}
    assert stats["bands"]["do_not_act"]["failure"] == 4
    assert stats["bands"]["do_not_act"]["success_rate"] == 0.0
    assert stats["calibration"]["verdict"] == "discriminating"
    assert stats["calibration"]["overconfident_failures"] == 0
    assert stats["scored_predictions"] == 8


def test_an_inverted_abstention_is_flagged_not_hidden(world: Any) -> None:
    """If confident recalls fail MORE than the ones it told callers to stop on,
    the signal is worse than none -- the loop must say so."""
    acts = [world.serve(f"clear {i}", CLEAR) for i in range(4)]
    stops = [world.serve(f"flat {i}", FLAT) for i in range(4)]
    for run in acts:
        _outcome(world.ws, run, "failure")
    for run in stops:
        _outcome(world.ws, run, "success")
    stats = _stats(world.ws)
    assert stats["calibration"]["verdict"] == "inverted"
    assert stats["calibration"]["overconfident_failures"] == 4


def test_a_non_discriminating_abstention_is_flagged(world: Any) -> None:
    runs = [world.serve(f"clear {i}", CLEAR) for i in range(4)] + [world.serve(f"flat {i}", FLAT) for i in range(4)]
    for run in runs:
        _outcome(world.ws, run, "success")
    assert _stats(world.ws)["calibration"]["verdict"] == "not_discriminating"


def test_too_little_evidence_is_not_a_verdict(world: Any) -> None:
    acts = [world.serve(f"clear {i}", CLEAR) for i in range(2)]
    stops = [world.serve(f"flat {i}", FLAT) for i in range(4)]
    for run in acts:
        _outcome(world.ws, run, "success")
    for run in stops:
        _outcome(world.ws, run, "failure")
    stats = _stats(world.ws)
    assert stats["bands"]["act"]["votes"] == 2  # below MIN_OUTCOME_EVIDENCE
    assert stats["calibration"]["verdict"] == "insufficient_evidence"


def test_silent_abstention_is_exposed_when_failures_pile_up_under_act(world: Any) -> None:
    """The roadmap's pairing: an abstention that never fires while failures
    accumulate is a calibration bug this join must expose."""
    for index in range(3):
        _outcome(world.ws, world.serve(f"confident {index}", CLEAR), "failure")
    calibration = _stats(world.ws)["calibration"]
    assert calibration["silent"] is True
    # Positive control: a failure that abstention DID flag clears the condition.
    for index in range(3):
        _outcome(world.ws, world.serve(f"flagged {index}", FLAT), "failure")
    assert _stats(world.ws)["calibration"]["silent"] is False


def test_one_reporter_is_one_vote_per_run_and_verdict(world: Any) -> None:
    """Bounded influence: filing many distinct reports cannot manufacture
    evidence, exactly as the calibration projection already guarantees."""
    run = world.serve("padded", CLEAR)
    for index in range(40):
        _outcome(world.ws, run, "success", actor="mallory", evidence=f"distinct report {index}")
    assert _stats(world.ws)["bands"]["act"]["votes"] == 1
    # Positive control: a DIFFERENT reporter is a second vote, so the bound is
    # per-reporter and not a blanket cap that would mask a broken join.
    _outcome(world.ws, run, "success", actor="bob")
    assert _stats(world.ws)["bands"]["act"]["votes"] == 2
    # Unattributed reports share one anonymous vote.
    for index in range(10):
        _outcome(world.ws, run, "success", actor="", evidence=f"anon {index}")
    assert _stats(world.ws)["bands"]["act"]["votes"] == 3


def test_neutral_outcomes_are_counted_but_never_move_a_rate(world: Any) -> None:
    run = world.serve("neutral", CLEAR)
    _outcome(world.ws, run, "success", actor="a")
    _outcome(world.ws, run, "neutral", actor="b")
    act = _stats(world.ws)["bands"]["act"]
    assert (act["success"], act["failure"], act["neutral"]) == (1, 0, 1)
    assert act["success_rate"] == 1.0


def test_an_outcome_for_an_unpredicted_run_is_ignored_not_an_error(world: Any) -> None:
    unpredicted = world.serve("plain", CLEAR, use="")  # no intended use -> no prediction
    predicted = world.serve("asked", CLEAR)
    _outcome(world.ws, unpredicted, "failure")
    _outcome(world.ws, predicted, "success")
    stats = _stats(world.ws)
    assert stats["predictions"] == 1
    assert stats["bands"]["act"]["votes"] == 1


def test_legacy_query_id_label_joins_like_run_id(world: Any) -> None:
    """The documented client contract is ``report_outcome(query_id=attestation.query_id)``;
    ``run_id=`` only adds ledger validation. Both land in the same column."""
    from mind_mem.outcome_attribution import report_outcome

    run = world.serve("legacy", CLEAR)
    report_outcome(world.ws, ["D-20260801-001"], "success", query_id=run, actor_id="legacy")
    assert _stats(world.ws)["bands"]["act"]["votes"] == 1


def test_stratified_by_use_and_basis(world: Any) -> None:
    summary = world.serve("by use", CLEAR, use="summary")
    write = world.serve("by use", CLEAR, use="governed_write")  # same run, second use
    rrf = world.serve("by basis", CLEAR, use="answer", basis="hybrid")
    assert summary == write
    _outcome(world.ws, summary, "success")
    stats = _stats(world.ws)
    assert set(stats["by_use"]) >= {"summary", "governed_write"}
    assert stats["by_use"]["summary"]["act"]["votes"] == 1
    assert stats["by_use"]["governed_write"]["act"]["votes"] == 1
    # A run recalled under two intended uses carries one prediction PER use, and
    # each is scored against the run's outcomes (the report does not say which use
    # the work was for) -- so the unstratified view counts that vote twice.
    assert stats["by_basis"]["sqlite"]["act"]["votes"] == 2
    assert rrf != summary
    # A stratum appears once it has votes: an unreported run is not a stratum.
    assert "hybrid" not in stats["by_basis"]
    _outcome(world.ws, rrf, "failure")
    assert _stats(world.ws)["by_basis"]["hybrid"]["act"]["failure"] == 1


def test_stats_are_deterministic_and_clock_free(world: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    runs = [world.serve(f"d{i}", CLEAR) for i in range(3)]
    for run in runs:
        _outcome(world.ws, run, "success")
    first = _stats(world.ws)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the scoring path read a clock")

    monkeypatch.setattr(time, "time", boom)
    monkeypatch.setattr(time, "time_ns", boom)
    assert _stats(world.ws) == first


# ---------------------------------------------------------------------------
# Pure judgement
# ---------------------------------------------------------------------------


def _counts(act: tuple[int, int], caution: tuple[int, int], stop: tuple[int, int]) -> dict[str, dict[str, int]]:
    return {
        "act": {"success": act[0], "failure": act[1], "neutral": 0},
        "proceed_with_caution": {"success": caution[0], "failure": caution[1], "neutral": 0},
        "do_not_act": {"success": stop[0], "failure": stop[1], "neutral": 0},
    }


@pytest.mark.parametrize(
    ("counts", "verdict"),
    [
        (_counts((5, 0), (0, 0), (0, 5)), "discriminating"),
        (_counts((4, 1), (0, 0), (1, 4)), "discriminating"),
        (_counts((3, 2), (0, 0), (3, 2)), "not_discriminating"),
        (_counts((1, 4), (0, 0), (4, 1)), "inverted"),
        (_counts((2, 0), (0, 0), (0, 9)), "insufficient_evidence"),
        (_counts((9, 0), (0, 0), (0, 2)), "insufficient_evidence"),
        (_counts((0, 0), (0, 0), (0, 0)), "insufficient_evidence"),
        # caution votes never rescue or sink the comparison
        (_counts((5, 0), (0, 40), (0, 5)), "discriminating"),
    ],
)
def test_judgement_table(counts: dict[str, dict[str, int]], verdict: str) -> None:
    from mind_mem.abstention_store import judge

    assert judge(counts)["verdict"] == verdict


def test_judgement_threshold_is_the_loops_own_minimum_evidence() -> None:
    from mind_mem.abstention_store import judge
    from mind_mem.outcome_attribution import MIN_OUTCOME_EVIDENCE

    below = _counts((MIN_OUTCOME_EVIDENCE - 1, 0), (0, 0), (0, MIN_OUTCOME_EVIDENCE))
    at = _counts((MIN_OUTCOME_EVIDENCE, 0), (0, 0), (0, MIN_OUTCOME_EVIDENCE))
    assert judge(below)["verdict"] == "insufficient_evidence"
    assert judge(at)["verdict"] == "discriminating"


# ---------------------------------------------------------------------------
# Wired into the EXISTING loop's report
# ---------------------------------------------------------------------------


def test_outcome_stats_is_unchanged_without_predictions(world: Any) -> None:
    """Flag off: the health report is exactly the pre-feature shape."""
    from mind_mem.calibration import CalibrationManager

    _outcome_plain = CalibrationManager(world.ws).record_outcome(["D-20260801-001"], "success", actor_id="x")
    assert _outcome_plain["recorded"] == 1
    report = CalibrationManager(world.ws).get_outcome_stats()
    assert "abstention" not in report
    assert report["total_outcomes"] == 1  # positive control: the report is a real one


def test_outcome_stats_carries_the_abstention_section_once_predictions_exist(world: Any) -> None:
    from mind_mem.calibration import CalibrationManager

    run = world.serve("surfaced", CLEAR)
    _outcome(world.ws, run, "success")
    report = CalibrationManager(world.ws).get_outcome_stats()
    assert report["abstention"]["predictions"] == 1
    assert report["abstention"]["bands"]["act"]["votes"] == 1
    assert report["total_outcomes"] >= 1


def test_calibration_stats_carries_the_same_section_only_when_it_exists(world: Any) -> None:
    """``calibration_stats`` is the loop's own health report; abstention is a
    prediction of that loop, so it shows there too -- and the pre-feature report
    is untouched until a verdict has been recorded."""
    from mind_mem.calibration import CalibrationManager

    before = CalibrationManager(world.ws).get_calibration_stats()
    assert "abstention" not in before
    assert "total_feedback" in before  # positive control: a real report
    run = world.serve("calibration surface", CLEAR)
    _outcome(world.ws, run, "success")
    after = CalibrationManager(world.ws).get_calibration_stats()
    assert after["abstention"]["predictions"] == 1
    assert {k: v for k, v in after.items() if k != "abstention"} == before


def test_the_mcp_outcome_stats_tool_surfaces_it(world: Any) -> None:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools.calibration import outcome_stats

    run = world.serve("via tool", CLEAR)
    _outcome(world.ws, run, "success")
    with use_workspace(world.ws):
        payload = json.loads(outcome_stats())
    assert payload["abstention"]["predictions"] == 1
    assert payload["abstention"]["calibration"]["verdict"] == "insufficient_evidence"


def test_recall_never_reads_the_prediction_table(world: Any) -> None:
    """The prediction describes a ranking; it must never feed back into one.
    Poisoning every stored row cannot change what recall serves."""
    world.monkeypatch.setattr(world.door, "_recall_impl_uncached", fake_engine([9.0, 3.0, 1.0]))
    before = json.loads(world.door._recall_impl("stable", limit=10, backend="bm25", intended_use="answer"))
    conn = sqlite3.connect(f"{world.ws}/.mind-mem-index/recall.db")
    try:
        assert conn.execute("SELECT COUNT(*) FROM recall_abstention").fetchone()[0] == 1  # positive control
        conn.execute("UPDATE recall_abstention SET band='do_not_act', separation=0.0")
        conn.commit()
    finally:
        conn.close()
    after = json.loads(world.door._recall_impl("stable", limit=10, backend="bm25", intended_use="answer"))
    assert after["results"] == before["results"]
    assert after["abstention"] == before["abstention"]  # computed from scores, not read back
