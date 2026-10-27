"""Dead-status demotion keeps every recall invariant it sits next to.

Regression: the first cut of the superseded fix reordered hits without
touching their scores, so a superseded block with score 0.6973 was served
after a live hit scoring 0.0804 (breaking the score-descending contract), and
it always built a new list (breaking carrier identity when nothing changed).
The demotion now works on the SCORE: a dead hit that would sit at or above a
live hit is re-scored strictly below the lowest live score, the original
score is recorded in a marker, and a pool that needs no change is returned as
the very same object.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from test_recall_superseded_leg_cut import NEW, OLD, QUERY, _seed

from mind_mem._recall_core import _rank_replaced_below_live

MARKER = "_replaced_demoted_from"


def _hit(bid: str, score: float, status: str = "active") -> dict:
    return {"_id": bid, "score": score, "status": status}


def _scores(hits: list[dict]) -> list[float]:
    return [float(h.get("score", 0.0)) for h in hits]


def _assert_descending(hits: list[dict]) -> None:
    scores = _scores(hits)
    assert scores == sorted(scores, reverse=True), scores


# ---------------------------------------------------------------------------
# The helper's contract
# ---------------------------------------------------------------------------


def test_no_dead_hit_returns_the_same_list_object() -> None:
    hits = [_hit("a", 0.9), _hit("b", 0.5)]
    assert _rank_replaced_below_live(hits) is hits


def test_dead_hit_already_below_live_returns_the_same_list_object() -> None:
    hits = [_hit("a", 0.9), _hit("b", 0.5), _hit("c", 0.1, "superseded")]
    before = copy.deepcopy(hits)
    assert _rank_replaced_below_live(hits) is hits
    assert hits == before


def test_only_dead_hits_returns_the_same_list_object() -> None:
    hits = [_hit("a", 0.9, "superseded"), _hit("b", 0.5, "revoked")]
    assert _rank_replaced_below_live(hits) is hits


def test_dead_hit_is_demoted_by_score_below_the_lowest_live_score() -> None:
    hits = [
        _hit("dead", 2.6061, "superseded"),
        _hit("live1", 0.6973),
        _hit("live2", 0.0804),
    ]
    out = _rank_replaced_below_live(hits)
    assert [h["_id"] for h in out] == ["live1", "live2", "dead"]
    _assert_descending(out)
    dead = out[-1]
    assert dead["score"] < 0.0804
    assert dead[MARKER] == 2.6061


def test_demotion_does_not_mutate_the_input() -> None:
    hits = [_hit("dead", 0.9, "superseded"), _hit("live", 0.5)]
    before = copy.deepcopy(hits)
    _rank_replaced_below_live(hits)
    assert hits == before


def test_tie_with_the_lowest_live_score_is_demoted() -> None:
    """Strictly below: a tie broken by id could otherwise put the dead hit first."""
    hits = [_hit("live", 0.5), _hit("a-dead", 0.5, "superseded")]
    out = _rank_replaced_below_live(hits)
    assert [h["_id"] for h in out] == ["live", "a-dead"]
    assert out[1]["score"] < 0.5


def test_several_dead_hits_stay_below_every_live_hit_and_descending() -> None:
    hits = [
        _hit("d1", 3.0, "superseded"),
        _hit("l1", 2.0),
        _hit("d2", 1.5, "revoked"),
        _hit("l2", 1.0),
        _hit("d3", 0.2, "deprecated"),
    ]
    out = _rank_replaced_below_live(hits)
    ids = [h["_id"] for h in out]
    assert ids[:2] == ["l1", "l2"]
    assert sorted(ids[2:]) == ["d1", "d2", "d3"]
    _assert_descending(out)
    assert all(h["score"] < 1.0 for h in out[2:])
    # Demoted hits keep their relative order (d1 out-scored d2).
    assert ids.index("d1") < ids.index("d2")


def test_demoted_score_is_never_negative() -> None:
    out = _rank_replaced_below_live([_hit("dead", 5.0, "superseded"), _hit("live", 0.0001)])
    assert 0.0 <= out[-1]["score"] < 0.0001


def test_a_second_demotion_keeps_the_original_score_in_the_marker() -> None:
    once = _rank_replaced_below_live([_hit("dead", 0.9, "superseded"), _hit("live", 0.5)])
    boosted = [once[0], {**once[1], "score": 0.95}]
    twice = _rank_replaced_below_live(boosted)
    assert twice[-1][MARKER] == 0.9
    assert twice[-1]["score"] < 0.5


def test_a_score_anchored_on_a_boosted_live_hit_carries_the_boost_flag() -> None:
    """The demoted score derives from the lowest live score; its provenance travels with it."""
    boosted = {**_hit("live", 0.17), "_co_retrieval_boost": True}
    out = _rank_replaced_below_live([_hit("dead", 0.9, "superseded"), _hit("top", 2.0), boosted])
    assert out[-1]["_co_retrieval_boost"] is True
    plain = _rank_replaced_below_live([_hit("dead", 0.9, "superseded"), _hit("top", 2.0), _hit("live", 0.17)])
    assert "_co_retrieval_boost" not in plain[-1]


def test_demotion_is_deterministic() -> None:
    hits = [_hit("dead", 0.9, "superseded"), _hit("live", 0.5), _hit("x", 0.4)]
    assert _rank_replaced_below_live(copy.deepcopy(hits)) == _rank_replaced_below_live(copy.deepcopy(hits))


def test_status_key_case_is_honoured() -> None:
    hits = [{"_id": "dead", "score": 0.9, "Status": "Superseded"}, _hit("live", 0.5)]
    out = _rank_replaced_below_live(hits)
    assert [h["_id"] for h in out] == ["live", "dead"]
    _assert_descending(out)


# ---------------------------------------------------------------------------
# Every leg, end to end: score-descending AND live replacement first
# ---------------------------------------------------------------------------


def _legs(ws_idx: str, ws_scan: str, limit: int) -> dict[str, list[dict]]:
    from mind_mem._recall_core import recall
    from mind_mem.hybrid_recall import HybridBackend
    from mind_mem.sqlite_index import query_index

    return {
        "sqlite": list(query_index(ws_idx, QUERY, limit=limit)),
        "scan": list(recall(ws_scan, QUERY, limit=limit)),
        "hybrid": list(HybridBackend(config={"vector_enabled": False}).search(QUERY, ws_idx, limit=limit)),
    }


@pytest.mark.parametrize("limit", [1, 2, 3, 10])
def test_every_leg_is_score_descending_with_live_first(limit: int, tmp_path: Any) -> None:
    ws_idx = _seed(tmp_path, f"idx-{limit}", index=True)
    ws_scan = _seed(tmp_path, f"scan-{limit}", index=False)
    for leg, hits in _legs(ws_idx, ws_scan, limit).items():
        ids = [str(h.get("_id")) for h in hits]
        assert NEW in ids, (leg, ids)
        if OLD in ids:
            assert ids.index(NEW) < ids.index(OLD), (leg, ids)
            old, new = hits[ids.index(OLD)]["score"], hits[ids.index(NEW)]["score"]
            # Strictly below a positive live score; a live 0.0 can only be tied.
            assert old < new or old == new == 0.0, (leg, old, new)
        scores = _scores(hits)
        assert scores == sorted(scores, reverse=True), (leg, scores)


@pytest.mark.parametrize("limit", [1, 3])
def test_mcp_recall_is_score_descending_with_live_first(limit: int, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import recall as mcp_recall

    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    for backend, index in (("bm25", True), ("bm25", False), ("auto", True)):
        ws = _seed(tmp_path, f"mcp-{backend}-{index}-{limit}", index=index)
        with use_workspace(ws):
            payload = json.loads(mcp_recall.recall(QUERY, limit=limit, backend=backend))
        hits = payload.get("results", [])
        ids = [str(h.get("_id") or h.get("id")) for h in hits]
        assert ids and ids[0] == NEW, (backend, index, ids)
        scores = [float(h.get("score", 0.0)) for h in hits]
        assert scores == sorted(scores, reverse=True), (backend, index, scores)
