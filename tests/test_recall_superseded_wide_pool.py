"""A superseded block never displaces its live replacement, whatever the pool size.

Two shapes the earlier per-leg demotion could not reach, because each leg cut
a pool BEFORE the demotion ran:

1. Hybrid fused path with a reranker active: the reranker's candidate slice
   (``rerank_depth``, 5 at ``limit=1``) was taken from a fused pool that had
   not been demoted yet. A vector leg that ranks the superseded versions first
   pushed the live replacement out of that slice, so the reranker never saw it.
2. Every leg: the wide candidate pool is cut at ``retrieve_wide_k`` (200). Once
   200 superseded blocks out-score the live replacement, the replacement is
   outside the pool and is missing at every limit, ``limit=1`` included.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from mind_mem.init_workspace import init
from mind_mem.sqlite_index import build_index

QUERY = "Ilona call Tuesday"
LIVE = "D-20261030-001"


def _block(bid: str, statement: str, date: str, status: str, extra: str = "") -> str:
    return f"[{bid}]\nType: Decision\nStatement: {statement}\nDate: {date}\nStatus: {status}\n{extra}\n---\n\n"


def _seed(tmp_path: Any, name: str, blocks: list[str], *, index: bool, recall_patch: dict | None = None) -> str:
    ws = str(tmp_path / name)
    os.makedirs(ws)
    init(ws)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write("\n" + "".join(blocks))
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    recall_cfg = cfg.get("recall") if isinstance(cfg.get("recall"), dict) else {}
    recall_cfg["vector_enabled"] = False
    recall_cfg.update(recall_patch or {})
    cfg["recall"] = recall_cfg
    with open(cfg_path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle)
    if index:
        build_index(ws)
    return ws


def _mcp_ids(ws: str, limit: int, backend: str) -> list[str]:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import recall as mcp_recall

    with use_workspace(ws):
        raw = mcp_recall.recall(QUERY, limit=limit, backend=backend)
    payload = json.loads(raw)
    assert "error" not in payload, payload
    return [str(h.get("_id") or h.get("id") or "") for h in payload.get("results", [])]


def _assert_live_first(ids: list[str], dead: list[str], limit: int) -> None:
    assert len(ids) <= limit, ids
    assert LIVE in ids, f"live replacement missing at limit={limit}: {ids}"
    for d in dead:
        if d in ids:
            assert ids.index(LIVE) < ids.index(d), ids


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")


# ---------------------------------------------------------------------------
# 1. Hybrid fused path, reranker active, vector leg ranks the dead hits first
# ---------------------------------------------------------------------------


class _FakeCE:
    """Neutral reranker: orders by the score it is handed, returns its top-k."""

    calls: list[tuple[int, int, list[str]]] = []

    @staticmethod
    def is_available() -> bool:
        return True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def rerank(self, query: str, cands: list[dict], top_k: int = 10, blend_weight: float = 0.6) -> list[dict]:
        _FakeCE.calls.append((len(cands), top_k, [str(c.get("_id")) for c in cands]))
        ordered = sorted(cands, key=lambda r: r.get("score", 0), reverse=True)
        return [dict(r) for r in ordered[:top_k]]


def _patch_hybrid_vector(monkeypatch: pytest.MonkeyPatch, dead: list[str], *, live_in_vector: bool) -> None:
    from mind_mem import cross_encoder_reranker, rerank_ensemble
    from mind_mem import hybrid_recall as hr

    monkeypatch.setattr(cross_encoder_reranker, "CrossEncoderReranker", _FakeCE)
    monkeypatch.setattr(rerank_ensemble, "create_ensemble", lambda cfg: None)

    def fake_vector(self: Any, query: str, workspace: str, limit: int = 200, active_only: bool = False, **kw: Any) -> list[dict]:
        hits = [
            {
                "_id": d,
                "score": 0.9 - 0.01 * i,
                "status": "superseded",
                "type": "decision",
                "excerpt": QUERY,
                "file": "decisions/DECISIONS.md",
                "line": 10 + i,
            }
            for i, d in enumerate(dead)
        ]
        if live_in_vector:
            hits.append(
                {
                    "_id": LIVE,
                    "score": 0.1,
                    "status": "active",
                    "type": "decision",
                    "excerpt": "Ilona call moved",
                    "file": "decisions/DECISIONS.md",
                    "line": 999,
                }
            )
        return hits

    monkeypatch.setattr(hr.HybridBackend, "_vector_search", fake_vector)
    orig_init = hr.HybridBackend.__init__

    def init_with_vector(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        self._vector_available = True
        self.vector_enabled = True

    monkeypatch.setattr(hr.HybridBackend, "__init__", init_with_vector)


def _dead_versions(n: int) -> tuple[list[str], list[str]]:
    dead = [f"D-202610{i + 1:02d}-001" for i in range(n)]
    blocks = [
        _block(
            d,
            "Ilona call is on Tuesday. Ilona call Tuesday at 10. Ilona call every Tuesday.",
            f"2026-10-{i + 1:02d}",
            "superseded",
            f"SupersededBy: {LIVE}",
        )
        for i, d in enumerate(dead)
    ]
    blocks.append(_block(LIVE, "Ilona call moved to Thursday.", "2026-10-30", "active"))
    return dead, blocks


@pytest.mark.parametrize("ce_enabled", [False, True], ids=["ce-off-control", "ce-on"])
@pytest.mark.parametrize("n_dead,live_in_vector", [(20, True), (6, False)])
@pytest.mark.parametrize("limit", [1, 2])
def test_hybrid_reranker_slice_keeps_the_live_replacement(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    ce_enabled: bool,
    n_dead: int,
    live_in_vector: bool,
    limit: int,
) -> None:
    dead, blocks = _dead_versions(n_dead)
    _patch_hybrid_vector(monkeypatch, dead, live_in_vector=live_in_vector)
    ws = _seed(
        tmp_path,
        "ws",
        blocks,
        index=True,
        recall_patch={"cross_encoder": {"enabled": ce_enabled}, "vector_enabled": True},
    )
    _FakeCE.calls = []
    ids = _mcp_ids(ws, limit, "hybrid")
    if ce_enabled:
        # Positive control: the reranker actually ran on this request.
        assert _FakeCE.calls, "reranker never ran; the test would prove nothing"
        assert LIVE in _FakeCE.calls[0][2], f"live replacement absent from the reranker slice: {_FakeCE.calls[0]}"
    _assert_live_first(ids, dead, limit)


def test_hybrid_reranker_failure_fallback_keeps_the_live_replacement(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reranker that raises falls back to the candidate slice, which must hold the replacement."""
    from mind_mem import cross_encoder_reranker

    dead, blocks = _dead_versions(20)
    _patch_hybrid_vector(monkeypatch, dead, live_in_vector=True)

    class _Broken(_FakeCE):
        def rerank(self, *args: Any, **kwargs: Any) -> list[dict]:
            _FakeCE.calls.append((0, 0, []))
            raise RuntimeError("model crashed")

    monkeypatch.setattr(cross_encoder_reranker, "CrossEncoderReranker", _Broken)
    ws = _seed(tmp_path, "ws", blocks, index=True, recall_patch={"cross_encoder": {"enabled": True}, "vector_enabled": True})
    _FakeCE.calls = []
    ids = _mcp_ids(ws, 1, "hybrid")
    assert _FakeCE.calls, "reranker never ran; the fallback was not exercised"
    _assert_live_first(ids, dead, 1)


# ---------------------------------------------------------------------------
# 2. Wide-pool cut: 200+ superseded blocks out-score the replacement
# ---------------------------------------------------------------------------


def _many_dead(n: int) -> tuple[list[str], list[str]]:
    dead = [f"D-2026{(i // 28) % 9 + 1:02d}{i % 28 + 1:02d}-{i:03d}" for i in range(n)]
    blocks = [
        _block(
            d,
            "Ilona call Tuesday. Ilona call Tuesday. Ilona call Tuesday.",
            f"2026-0{(i // 28) % 9 + 1}-{i % 28 + 1:02d}",
            "superseded",
            "SupersededBy: " + LIVE,
        )
        for i, d in enumerate(dead)
    ]
    blocks.append(_block(LIVE, "Ilona call moved to Thursday.", "2026-10-30", "active"))
    return dead, blocks


_WIDE_LEGS = [
    pytest.param("bm25", True, id="sqlite"),
    pytest.param("bm25", False, id="scan"),
    pytest.param("hybrid", True, id="hybrid"),
]


@pytest.mark.parametrize("backend,index", _WIDE_LEGS)
@pytest.mark.parametrize("n_dead", [200, 210])
def test_wide_pool_overflow_keeps_the_live_replacement(tmp_path: Any, backend: str, index: bool, n_dead: int) -> None:
    dead, blocks = _many_dead(n_dead)
    ws = _seed(tmp_path, "ws", blocks, index=index)
    for limit in (1, 3):
        ids = _mcp_ids(ws, limit, backend)
        _assert_live_first(ids, dead, limit)


@pytest.mark.parametrize("backend,index", _WIDE_LEGS)
def test_wide_pool_overflow_with_raised_wide_k(tmp_path: Any, backend: str, index: bool) -> None:
    dead, blocks = _many_dead(210)
    ws = _seed(tmp_path, "ws", blocks, index=index, recall_patch={"retrieve_wide_k": 400})
    _assert_live_first(_mcp_ids(ws, 1, backend), dead, 1)


def test_sqlite_query_index_direct_overflow(tmp_path: Any) -> None:
    from mind_mem.sqlite_index import query_index

    dead, blocks = _many_dead(205)
    ws = _seed(tmp_path, "ws", blocks, index=True)
    ids = [str(h.get("_id")) for h in query_index(ws, QUERY, limit=1)]
    assert ids == [LIVE], ids


def test_scan_recall_direct_overflow(tmp_path: Any) -> None:
    from mind_mem.recall import recall

    dead, blocks = _many_dead(205)
    ws = _seed(tmp_path, "ws", blocks, index=False)
    ids = [str(h.get("_id")) for h in recall(ws, QUERY, limit=1)]
    assert ids == [LIVE], ids


# ---------------------------------------------------------------------------
# _live_aware_window: inert when nothing is dead, a superset when something is
# ---------------------------------------------------------------------------


def _hit(i: int, status: str = "active") -> dict:
    return {"_id": f"H{i:03d}", "score": 100.0 - i, "status": status}


def test_live_aware_window_is_plain_slice_without_dead_hits() -> None:
    from mind_mem._recall_core import _live_aware_window

    hits = [_hit(i) for i in range(10)]
    window = _live_aware_window(hits, 4)
    assert window == hits[:4]
    assert all(a is b for a, b in zip(window, hits))


def test_live_aware_window_short_pool_is_untouched() -> None:
    from mind_mem._recall_core import _live_aware_window

    hits = [_hit(0, "superseded"), _hit(1)]
    assert _live_aware_window(hits, 5) == hits


def test_live_aware_window_adds_one_live_hit_per_dead_hit_in_order() -> None:
    from mind_mem._recall_core import _live_aware_window

    hits = [_hit(0, "superseded"), _hit(1), _hit(2, "revoked"), _hit(3), _hit(4, "superseded"), _hit(5), _hit(6)]
    window = _live_aware_window(hits, 3)
    # Everything the plain slice kept, in order, plus the next two LIVE hits.
    assert [h["_id"] for h in window] == ["H000", "H001", "H002", "H003", "H005"]
    assert all(any(w is h for h in hits) for w in window)
