"""Every leg of the recall scoring path, proven to read no clock.

``tests/test_recall_determinism.py`` proves the *product* property — same
corpus, same config, same ``scoring_instant``, same answer. This file proves
the *structural* one that keeps it true: no code on the scoring path reads a
wall clock, on any backend, whether or not the read would have been noticed.

That distinction is not academic. A first version of this guard broke the clock
accessors, ran one recall on one backend, and asserted the call completed. It
passed with **eight** separate pieces of the threading reverted, because:

* ``recall()`` degrades rather than fails — the calibration weight, the
  validity gate, the trust-signal load and every multi-hop sub-query are each
  wrapped in ``except Exception``, which swallowed the guard's own alarm and
  turned it into a log line; and
* one recall on the scan backend never executes the FTS, hybrid or dense legs
  at all, so four more reverts were simply never run.

Both holes are closed here by construction rather than by vigilance. The
signal is a ``BaseException`` that also *records* itself before raising
(:class:`~_recall_clock_sentinel.ClockSentinel`), and the coverage comes from
:func:`~_recall_clock_sentinel.clock_census`, a ``sys.setprofile`` observer that
sees every ``datetime.now`` / ``date.today`` executed anywhere inside
``mind_mem`` — no accessor list, nothing to keep in sync, and nothing an
``except`` clause can hide from it. Each test below drives one real leg and
asserts the census stayed empty.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone

import pytest
from _recall_clock_sentinel import (
    ClockRead,
    clock_census,
    install_clock_sentinel,
    seed_calibration_feedback,
    write_workspace,
)

from mind_mem import _recall_core, calibration, hybrid_recall, recall_vector, sqlite_index

INSTANT = date(2026, 8, 27)

# Stamped inside the 30-day calibration window of INSTANT and outside the
# window of an instant a few months later, so the calibration leg is a live
# ranking input rather than a constant 1.0.
FEEDBACK_STAMP = "2026-08-20T00:00:00Z"

QUERY = "retrieval rollout determinism"
TEMPORAL_QUERY = "when in the last 7 days was the retrieval rollout note recorded"
MULTI_HOP_QUERY = "why did the retrieval rollout note land and who approved the retrieval rollout"

BLOCKS = (
    ("D-20260827-001", "retrieval rollout determinism notes shipped", "2026-08-27"),
    ("D-20260823-002", "retrieval rollout determinism notes reviewed", "2026-08-23"),
    ("D-20260819-003", "retrieval rollout determinism notes drafted", "2026-08-19"),
)

# Turns on every clock-sensitive leg that is off by default. Each of these was
# a place the threading could be reverted without any test noticing.
ALL_LEGS_ON = {
    "recall": {
        "validity_gate": {"enabled": True, "provenance_class": {"enabled": True}},
        "temporal_hard_filter": True,
    },
    "retrieval": {
        "temporal_decay_hot_path": True,
        "trust_scores": {"enabled": True, "use_calibration": True},
    },
}


@pytest.fixture
def seeded_workspace(tmp_path):
    """A workspace with content, calibration feedback and every leg enabled."""
    ws = str(tmp_path)
    write_workspace(ws, BLOCKS)
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump(ALL_LEGS_ON, fh)
    seed_calibration_feedback(ws, tuple(b[0] for b in BLOCKS), stamped=FEEDBACK_STAMP)
    return ws


# ---------------------------------------------------------------------------
# The instruments themselves must be able to fail
# ---------------------------------------------------------------------------


def test_census_sees_an_unguarded_clock_read(tmp_path) -> None:
    """Guard the guard: the census fires on a real, unallowlisted read."""
    mgr = calibration.CalibrationManager(str(tmp_path))
    with clock_census() as census:
        mgr._cutoff_date()  # no injected instant -> reads the wall clock
    assert census.reads, "the census observed nothing while a clock was being read"
    assert any("_cutoff_date" in read for read in census.reads)


def test_default_instant_run_reads_only_the_sanctioned_boundary(tmp_path) -> None:
    """A run that omits the instant reads exactly one clock: the resolver's."""
    ws = str(tmp_path)
    write_workspace(ws, BLOCKS)
    with clock_census(allow_boundary_read=True) as permissive:
        _recall_core.recall(ws, QUERY)
    permissive.assert_clock_free()

    with clock_census() as strict:
        _recall_core.recall(ws, QUERY)
    assert strict.reads, "no boundary read observed — the allowance above proves nothing"
    assert all("scoring_instant.py" in read for read in strict.reads), strict.reads


def test_sentinel_signal_escapes_the_degradation_handlers() -> None:
    """``ClockRead`` is not an ``Exception``, which is the whole point.

    The predecessor of this guard raised ``AssertionError``. ``recall()``'s
    ``except Exception`` handlers caught it, logged a warning and carried on,
    so reverting the threading left the suite green while the alarm fired.
    """
    assert not issubclass(ClockRead, Exception)

    def _degrading_leg() -> str:
        try:
            raise ClockRead("boom")
        except Exception:  # noqa: BLE001 — reproducing the production handler shape
            return "swallowed"

    with pytest.raises(ClockRead):
        _degrading_leg()


def test_sentinel_records_the_read_even_if_something_catches_it() -> None:
    """Belt and braces: ``.reads`` is appended *before* the raise."""
    with pytest.MonkeyPatch.context() as mp:
        sentinel = install_clock_sentinel(mp)
        try:
            calibration.datetime.now(timezone.utc)
        except BaseException:  # noqa: BLE001 — the hypothetical over-broad handler
            pass
        assert sentinel.reads, "a swallowed clock read left no trace"
        with pytest.raises(AssertionError, match="read a clock"):
            sentinel.assert_clock_free()


# ---------------------------------------------------------------------------
# One test per leg. Each drives a real backend; the census does the judging.
# ---------------------------------------------------------------------------


def test_scan_leg_reads_no_clock(seeded_workspace) -> None:
    """The full-scan path: recency ramp, calibration weight, temporal filter, validity gate."""
    with clock_census() as census:
        hits = _recall_core.recall(seeded_workspace, TEMPORAL_QUERY, scoring_instant=INSTANT)
    assert hits, "fixture served nothing — the guard would be vacuous"
    census.assert_clock_free()


def test_multi_hop_decomposition_reads_no_clock(seeded_workspace) -> None:
    """Each sub-query re-enters ``recall``; the instant must ride along."""
    with clock_census() as census:
        hits = _recall_core.recall(seeded_workspace, MULTI_HOP_QUERY, scoring_instant=INSTANT)
    assert hits, "every sub-query died silently"
    census.assert_clock_free()


def test_fts_index_leg_reads_no_clock(seeded_workspace) -> None:
    """The FTS5 path has its own recency ramp and its own calibration lookup."""
    sqlite_index.build_index(seeded_workspace)
    with clock_census() as census:
        hits = sqlite_index.query_index(seeded_workspace, QUERY, limit=5, scoring_instant=INSTANT)
    assert hits, "the FTS index served nothing — the guard would be vacuous"
    assert not any(h.get("_fallback") for h in hits), "fell back to scan; the FTS leg was not exercised"
    census.assert_clock_free()


def test_index_missing_fallback_reads_no_clock(tmp_path) -> None:
    """No index yet: ``query_index`` delegates to the scan, instant and all.

    Deliberately *not* the seeded workspace — the calibration store shares
    ``.mind-mem-index/recall.db`` with the FTS index, so seeding feedback
    creates the file and this fallback would never be reached.
    """
    ws = str(tmp_path)
    write_workspace(ws, BLOCKS)
    assert not os.path.isfile(sqlite_index._db_path(ws)), "an index exists; the fallback would not run"
    with clock_census() as census:
        hits = sqlite_index.query_index(ws, QUERY, limit=5, scoring_instant=INSTANT)
    assert hits, "the fallback served nothing — the guard would be vacuous"
    census.assert_clock_free()


def test_fts_failure_fallback_reads_no_clock(seeded_workspace) -> None:
    """A broken FTS table drops to the scan; that fallback carries the instant too."""
    sqlite_index.build_index(seeded_workspace)
    conn = sqlite_index._connect(seeded_workspace)
    conn.execute("DROP TABLE IF EXISTS blocks_fts")
    conn.commit()
    with clock_census() as census:
        hits = sqlite_index.query_index(seeded_workspace, QUERY, limit=5, scoring_instant=INSTANT)
    assert hits, "the fallback served nothing — the guard would be vacuous"
    assert any(h.get("_fallback") == "bm25_scan" for h in hits), "the FTS-failure fallback did not run"
    census.assert_clock_free()


def test_hybrid_bm25_only_leg_reads_no_clock(seeded_workspace) -> None:
    """The BM25-only branch hands off to the lexical leg and returns early.

    It is a *separate* code path from the fusion branch below — it returns
    before the post-fusion stages — so it carries its own hand-off of the
    instant and needs its own guard.
    """
    backend = hybrid_recall.HybridBackend({**ALL_LEGS_ON, "vector_enabled": False})
    with clock_census() as census:
        hits = backend.search(QUERY, seeded_workspace, limit=5, scoring_instant=INSTANT)
    assert hits, "the hybrid backend served nothing — the guard would be vacuous"
    census.assert_clock_free()


def test_hybrid_fusion_leg_reads_no_clock(seeded_workspace, monkeypatch) -> None:
    """The two-leg fusion path: half-life decay and the trust-score annotation.

    Both of those post-fusion stages sit *after* the BM25-only branch's early
    return, so they are unreachable without an available dense leg. The dense
    leg is stubbed to an empty result rather than skipped: that drives the real
    fusion, decay and trust-score code with no optional model dependency, which
    is exactly the stretch of code being guarded.
    """
    backend = hybrid_recall.HybridBackend({**ALL_LEGS_ON, "vector_enabled": True})
    monkeypatch.setattr(backend, "_vector_available", True)

    # The stub stands in for the model, but it still has to *witness* the
    # hand-off: the census cannot observe a clock read inside code it replaced,
    # so the contract that the dense leg is handed the instant is asserted here
    # instead of being quietly assumed.
    handed: list[object] = []
    monkeypatch.setattr(backend, "_vector_search", lambda *_a, **kw: handed.append(kw.get("scoring_instant")) or [])

    with clock_census() as census:
        hits = backend.search(QUERY, seeded_workspace, limit=5, scoring_instant=INSTANT)
    assert hits, "the hybrid backend served nothing — the guard would be vacuous"
    assert any(h.get("_temporal_decay") for h in hits), "the decay leg did not run — the guard would be vacuous"
    assert handed == [INSTANT], f"the dense leg was not handed the run's instant: {handed}"
    census.assert_clock_free()


def test_temporal_decay_leg_reads_no_clock() -> None:
    """Drive the half-life decay directly, so the assertion cannot miss it."""
    backend = hybrid_recall.HybridBackend({"retrieval": {"temporal_decay_hot_path": True}, "vector_enabled": False})
    hits = [{"_id": bid, "score": 1.0, "Date": when} for bid, _, when in BLOCKS]
    with clock_census() as census:
        decayed = backend._maybe_temporal_decay(hits, scoring_instant=INSTANT)
    assert [h["score"] for h in decayed] != [1.0, 1.0, 1.0], "decay did not run — the guard would be vacuous"
    census.assert_clock_free()


def test_dense_local_leg_reads_no_clock(tmp_path, monkeypatch) -> None:
    """The local dense index multiplies in its own recency boost."""
    ws = str(tmp_path)
    backend = recall_vector.VectorBackend({"provider": "local"})
    index_dir = os.path.join(ws, backend.index_path)
    os.makedirs(index_dir, exist_ok=True)
    payload = {
        "blocks": [
            {"_id": bid, "type": "decision", "date": when, "status": "active", "excerpt": text, "file": "f", "line": 1}
            for bid, text, when in BLOCKS
        ],
        "embeddings": [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
    }
    with open(os.path.join(index_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    # A stub embedder keeps the leg reachable without the optional model
    # dependency; the recency multiply under test is downstream of it.
    monkeypatch.setattr(backend, "embed", lambda texts: [[1.0, 0.0] for _ in texts])

    with clock_census() as census:
        hits = backend._search_local(ws, QUERY, 10, False, scoring_instant=INSTANT)
    assert hits, "the dense leg served nothing — the guard would be vacuous"
    census.assert_clock_free()


def test_validity_gate_leg_reads_no_clock(seeded_workspace) -> None:
    """The gate's provenance component reads the rolling calibration window."""
    from mind_mem.validity_gate import apply_validity_gate

    hits = [{"_id": bid, "score": 1.0} for bid, _, _ in BLOCKS]
    with clock_census() as census:
        apply_validity_gate(hits, seeded_workspace, ALL_LEGS_ON["recall"], scoring_instant=INSTANT)
    assert all("validity" in h for h in hits), "the gate did not run — the guard would be vacuous"
    census.assert_clock_free()


# ---------------------------------------------------------------------------
# The recency layer is parameterised, not deleted
# ---------------------------------------------------------------------------


def test_every_guarded_leg_still_moves_with_the_instant(seeded_workspace) -> None:
    """A clock-free path that ignored the instant would pass every test above.

    Two instants either side of the calibration window and far enough apart to
    move the recency ramp must produce two different rankings — otherwise the
    seam has flattened recency instead of parameterising it.
    """
    near = _recall_core.recall(seeded_workspace, QUERY, scoring_instant=INSTANT)
    far = _recall_core.recall(seeded_workspace, QUERY, scoring_instant=date(2027, 8, 27))
    near_scores = [(h["_id"], round(float(h["score"]), 6)) for h in near]
    far_scores = [(h["_id"], round(float(h["score"]), 6)) for h in far]
    assert near_scores != far_scores, "the instant no longer changes anything — recency was flattened"


def test_calibration_window_is_the_instants_window_not_todays(seeded_workspace) -> None:
    """The rolling window opens at the injected instant, not at the wall clock."""
    mgr = calibration.CalibrationManager(seeded_workspace)
    inside = mgr.get_block_weight(BLOCKS[0][0], now=datetime(2026, 8, 27, tzinfo=timezone.utc))
    outside = mgr.get_block_weight(BLOCKS[0][0], now=datetime(2027, 8, 27, tzinfo=timezone.utc))
    assert inside != outside, "the seeded feedback is not inside exactly one of the two windows"
    assert outside == 1.0, "feedback outside the window still counted"
