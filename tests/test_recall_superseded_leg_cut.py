"""A superseded block never displaces its live replacement at ANY limit.

Reproduction: after the Ilona call moved from Tuesday to Thursday, MCP
``recall(limit=1..3)`` returned only the superseded Tuesday block. The
demotion of dead statuses ran in the funnel, after every leg had already cut
its own list to ``limit``, so the live replacement was never in the pool the
demotion re-ordered. These tests seed a real workspace and assert, on every
leg and at every small limit, that the live replacement is served and ranks
above the block it replaced.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from mind_mem.init_workspace import init
from mind_mem.sqlite_index import build_index

QUERY = "Ilona call Tuesday"
OLD = "D-20261006-001"
NEW = "D-20261008-001"

#: The superseded block repeats every query term, so on raw lexical score it
#: outranks its replacement -- the shape that exposed the defect.
_OLD_BLOCK = (
    f"[{OLD}]\n"
    "Type: Decision\n"
    "Statement: Ilona call is on Tuesday. The Ilona call Tuesday slot is 10:00, Ilona call every Tuesday.\n"
    "Date: 2026-10-06\n"
    "Status: superseded\n"
    f"SupersededBy: {NEW}\n\n---\n\n"
)
_NEW_BLOCK = (
    f"[{NEW}]\n"
    "Type: Decision\n"
    "Statement: Ilona call moved to Thursday at 10:00.\n"
    "Date: 2026-10-08\n"
    "Status: active\n"
    f"Supersedes: {OLD}\n\n---\n\n"
)


def _seed(tmp_path: Any, name: str, *, index: bool) -> str:
    ws = str(tmp_path / name)
    os.makedirs(ws)
    init(ws)
    path = os.path.join(ws, "decisions", "DECISIONS.md")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n" + _OLD_BLOCK + _NEW_BLOCK)
    with open(os.path.join(ws, "mind-mem.json"), encoding="utf-8") as handle:
        cfg = json.load(handle)
    recall_cfg = cfg.get("recall") if isinstance(cfg.get("recall"), dict) else {}
    recall_cfg["vector_enabled"] = False
    cfg["recall"] = recall_cfg
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle)
    if index:
        build_index(ws)
    return ws


def _ids(hits: Any) -> list[str]:
    return [str(h.get("_id") or h.get("id") or "") for h in hits]


def _assert_live_first(ids: list[str], limit: int) -> None:
    assert len(ids) <= limit, ids
    assert NEW in ids, f"live replacement missing at limit={limit}: {ids}"
    if OLD in ids:
        assert ids.index(NEW) < ids.index(OLD), ids


def test_control_the_superseded_block_outscores_its_replacement(tmp_path: Any) -> None:
    """Positive control: without demotion the dead block IS the lexical #1.

    The replacement also names the dead block in ``Supersedes``, so the
    link-supersession order is switched off too: both are demotions, and the
    control is about the score order underneath them.
    """
    from mind_mem.sqlite_index import query_index

    ws = _seed(tmp_path, "control", index=True)
    from unittest.mock import patch

    with (
        patch("mind_mem._recall_core._rank_replaced_below_live", side_effect=lambda h: list(h)),
        patch("mind_mem.sqlite_index._apply_index_supersession", side_effect=lambda _c, _w, h, _v: list(h)),
    ):
        raw = _ids(query_index(ws, QUERY, limit=10))
    assert raw and raw[0] == OLD, raw
    assert NEW in raw, raw


# ---------------------------------------------------------------------------
# MCP surface, end to end
# ---------------------------------------------------------------------------

_MCP_LEGS = [
    pytest.param("bm25", True, id="sqlite"),
    pytest.param("bm25", False, id="scan"),
    pytest.param("auto", True, id="hybrid"),
]


@pytest.mark.parametrize("backend,index", _MCP_LEGS)
@pytest.mark.parametrize("limit", [1, 2, 3])
def test_mcp_recall_serves_the_live_replacement(
    backend: str, index: bool, limit: int, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import recall as mcp_recall

    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    ws = _seed(tmp_path, f"mcp-{backend}-{index}-{limit}", index=index)
    with use_workspace(ws):
        raw = mcp_recall.recall(QUERY, limit=limit, backend=backend)
    payload = json.loads(raw)
    assert "error" not in payload, payload
    ids = _ids(payload.get("results", []))
    _assert_live_first(ids, limit)


# ---------------------------------------------------------------------------
# Each leg directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_query_index_serves_the_live_replacement(limit: int, tmp_path: Any) -> None:
    from mind_mem.sqlite_index import query_index

    ws = _seed(tmp_path, f"qi-{limit}", index=True)
    _assert_live_first(_ids(query_index(ws, QUERY, limit=limit)), limit)


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_scan_recall_serves_the_live_replacement(limit: int, tmp_path: Any) -> None:
    from mind_mem._recall_core import recall

    ws = _seed(tmp_path, f"scan-{limit}", index=False)
    _assert_live_first(_ids(recall(ws, QUERY, limit=limit)), limit)


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_hybrid_backend_serves_the_live_replacement(limit: int, tmp_path: Any) -> None:
    from mind_mem.hybrid_recall import HybridBackend

    ws = _seed(tmp_path, f"hb-{limit}", index=True)
    backend = HybridBackend(config={"vector_enabled": False})
    _assert_live_first(_ids(backend.search(QUERY, ws, limit=limit)), limit)


@pytest.mark.parametrize("limit", [1, 2])
def test_hybrid_fused_leg_serves_the_live_replacement(limit: int, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Vector leg ranks the dead block first too: the fused cut must still demote it."""
    from mind_mem.hybrid_recall import HybridBackend

    ws = _seed(tmp_path, f"fused-{limit}", index=True)
    backend = HybridBackend(config={"vector_enabled": True, "provider": "local"})
    monkeypatch.setattr(
        backend,
        "_vector_search",
        lambda *a, **k: [{"_id": OLD, "score": 0.99, "status": "superseded"}],
    )
    _assert_live_first(_ids(backend.search(QUERY, ws, limit=limit)), limit)


def test_dead_blocks_still_served_when_room(tmp_path: Any) -> None:
    """History is kept: with room for both, the superseded block is still returned, second."""
    from mind_mem.sqlite_index import query_index

    ws = _seed(tmp_path, "room", index=True)
    ids = _ids(query_index(ws, QUERY, limit=5))
    assert ids.index(NEW) < ids.index(OLD), ids


@pytest.mark.parametrize("limit", [1, 2])
def test_cross_encoder_cut_keeps_the_live_replacement(limit: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reranker returns only its own top-k: a dead block it scores high must not take the live slot."""
    from mind_mem import cross_encoder_reranker, rerank_ensemble
    from mind_mem.hybrid_recall import HybridBackend

    asked: list[int] = []

    class _FakeCE:
        @staticmethod
        def is_available() -> bool:
            return True

        def rerank(self, query: str, candidates: list[dict], top_k: int = 10, blend_weight: float = 0.6) -> list[dict]:
            asked.append(top_k)
            ranked = sorted(candidates, key=lambda r: r["_id"] != OLD)  # the model loves the dead block
            return ranked[:top_k]

    monkeypatch.setattr(rerank_ensemble, "create_ensemble", lambda cfg: None)
    monkeypatch.setattr(cross_encoder_reranker, "CrossEncoderReranker", _FakeCE)
    pool = [
        {"_id": OLD, "score": 9.0, "status": "superseded"},
        {"_id": NEW, "score": 4.0, "status": "active"},
        {"_id": "D-other", "score": 1.0, "status": "active"},
    ]
    backend = HybridBackend(config={"vector_enabled": False, "cross_encoder": {"top_k": limit}})
    out = backend._maybe_cross_encoder_rerank(QUERY, pool, limit, rerank_depth=10, ce_active=True)
    _assert_live_first(_ids(out), limit)
    assert asked == [limit + 1], asked  # bounded: one extra slot per dead candidate


def test_cross_encoder_request_is_unchanged_without_dead_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_mem import cross_encoder_reranker, rerank_ensemble
    from mind_mem.hybrid_recall import HybridBackend

    asked: list[int] = []

    class _FakeCE:
        @staticmethod
        def is_available() -> bool:
            return True

        def rerank(self, query: str, candidates: list[dict], top_k: int = 10, blend_weight: float = 0.6) -> list[dict]:
            asked.append(top_k)
            return list(reversed(candidates))[:top_k]

    monkeypatch.setattr(rerank_ensemble, "create_ensemble", lambda cfg: None)
    monkeypatch.setattr(cross_encoder_reranker, "CrossEncoderReranker", _FakeCE)
    pool = [{"_id": f"D-{i}", "score": float(10 - i), "status": "active"} for i in range(4)]
    backend = HybridBackend(config={"vector_enabled": False, "cross_encoder": {"top_k": 2}})
    out = backend._maybe_cross_encoder_rerank(QUERY, pool, 2, rerank_depth=10, ce_active=True)
    assert _ids(out) == ["D-3", "D-2"]
    assert asked == [2]


def _h(bid: str, score: float, status: str = "active") -> dict:
    return {"_id": bid, "score": score, "status": status}


def test_live_aware_knee_is_the_plain_knee_without_dead_hits() -> None:
    from mind_mem._recall_core import _knee_cutoff_live_aware, knee_cutoff

    hits = [_h("A", 9.0), _h("B", 8.5), _h("C", 0.2), _h("D", 0.1)]
    for floor in (1, 2, 3):
        assert _knee_cutoff_live_aware(hits, min_results=floor, min_score=0.0) == knee_cutoff(hits, min_results=floor, min_score=0.0)


def test_live_aware_knee_keeps_a_replacement_behind_a_dead_top_hit() -> None:
    from mind_mem._recall_core import _knee_cutoff_live_aware

    out = _knee_cutoff_live_aware([_h(OLD, 9.0, "superseded"), _h(NEW, 4.0)], min_results=1, min_score=0.0)
    assert _ids(out) == [OLD, NEW]


def test_live_aware_knee_does_not_admit_live_noise() -> None:
    """A dead hit the floor kept below live hits must not drag live noise past the knee."""
    from mind_mem._recall_core import _knee_cutoff_live_aware

    hits = [_h("A", 4.8), _h("C", 4.8), _h("B", 4.0, "deprecated"), _h("NOISE", 0.14)]
    assert _ids(_knee_cutoff_live_aware(hits, min_results=3, min_score=0.0)) == ["A", "C", "B"]
