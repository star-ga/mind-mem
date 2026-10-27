"""The scan leg's cross-encoder cut must not let a superseded block take its replacement's slot.

A reranker returns only its own ``top_k``. On the scan leg that request was
``limit``, so at ``limit=1`` a superseded block the reranker kept on top was the
only hit handed to the later demotion, and its live replacement was already
gone. These tests use a NEUTRAL fake reranker (score order kept, truncated to
``top_k``) so the truncation alone is what is under test.
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

_BLOCKS = (
    f"[{OLD}]\nType: Decision\n"
    "Statement: Ilona call is on Tuesday. Ilona call Tuesday at 10. Ilona call every Tuesday.\n"
    f"Date: 2026-10-06\nStatus: superseded\nSupersededBy: {NEW}\n\n---\n\n"
    f"[{NEW}]\nType: Decision\nStatement: Ilona call moved to Thursday.\n"
    f"Date: 2026-10-08\nStatus: active\nSupersedes: {OLD}\n\n---\n\n"
)


class _NeutralCE:
    """Keeps score order and returns only ``top_k`` -- the truncation a real reranker does."""

    requests: list[int] = []

    @staticmethod
    def is_available() -> bool:
        return True

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10, blend_weight: float = 0.6) -> list[dict]:
        _NeutralCE.requests.append(top_k)
        ordered = sorted(candidates, key=lambda r: r.get("score", 0), reverse=True)
        return [dict(r) for r in ordered[:top_k]]


@pytest.fixture
def neutral_ce(monkeypatch: pytest.MonkeyPatch) -> type[_NeutralCE]:
    from mind_mem import cross_encoder_reranker

    _NeutralCE.requests = []
    monkeypatch.setattr(cross_encoder_reranker, "CrossEncoderReranker", _NeutralCE)
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    return _NeutralCE


def _seed(tmp_path: Any, name: str, *, index: bool, blocks: str = _BLOCKS) -> str:
    ws = str(tmp_path / name)
    os.makedirs(ws)
    init(ws)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write("\n" + blocks)
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    recall_cfg = cfg.get("recall") if isinstance(cfg.get("recall"), dict) else {}
    recall_cfg["vector_enabled"] = False
    recall_cfg["cross_encoder"] = {"enabled": True}
    cfg["recall"] = recall_cfg
    with open(cfg_path, "w", encoding="utf-8") as handle:
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


@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("limit", [1, 2, 3])
def test_scan_cross_encoder_keeps_the_live_replacement(neutral_ce: type[_NeutralCE], index: bool, limit: int, tmp_path: Any) -> None:
    from mind_mem.recall import recall

    ws = _seed(tmp_path, f"scan-ce-{index}-{limit}", index=index)
    _assert_live_first(_ids(recall(ws, QUERY, limit=limit)), limit)
    assert neutral_ce.requests, "positive control: the cross-encoder must actually have run"


@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("limit", [1, 2])
def test_mcp_bm25_cross_encoder_keeps_the_live_replacement(neutral_ce: type[_NeutralCE], index: bool, limit: int, tmp_path: Any) -> None:
    from mind_mem.mcp.infra.workspace import use_workspace
    from mind_mem.mcp.tools import recall as mcp_recall

    ws = _seed(tmp_path, f"mcp-ce-{index}-{limit}", index=index)
    with use_workspace(ws):
        payload = json.loads(mcp_recall.recall(QUERY, limit=limit, backend="bm25"))
    assert "error" not in payload, payload
    _assert_live_first(_ids(payload.get("results", [])), limit)


def test_scan_cross_encoder_request_is_unchanged_without_dead_candidates(neutral_ce: type[_NeutralCE], tmp_path: Any) -> None:
    """Inert default: with no dead candidate the reranker is asked for exactly ``limit``."""
    from mind_mem._recall_core import recall

    live_only = (
        "[D-20261008-001]\nType: Decision\nStatement: Ilona call moved to Thursday.\n"
        "Date: 2026-10-08\nStatus: active\n\n---\n\n"
        "[D-20261008-002]\nType: Decision\nStatement: Ilona call Tuesday standup notes.\n"
        "Date: 2026-10-08\nStatus: active\n\n---\n\n"
    )
    ws = _seed(tmp_path, "live-only", index=False, blocks=live_only)
    recall(ws, QUERY, limit=1)
    assert neutral_ce.requests == [1], neutral_ce.requests


def test_scan_cross_encoder_is_deterministic(neutral_ce: type[_NeutralCE], tmp_path: Any) -> None:
    from mind_mem._recall_core import recall

    ws = _seed(tmp_path, "det", index=False)
    first = recall(ws, QUERY, limit=2)
    second = recall(ws, QUERY, limit=2)
    assert _ids(first) == _ids(second)
    assert [h["score"] for h in first] == [h["score"] for h in second]
    scores = [h["score"] for h in first]
    assert scores == sorted(scores, reverse=True), scores
