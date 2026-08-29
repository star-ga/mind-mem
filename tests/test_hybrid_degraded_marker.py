"""Tests for the in-band recall degradation marker (Task 2).

Silent degradation was the bug: a "hybrid" recall that quietly served
BM25-only because the vector leg was unavailable / timed out / failed left
the caller unable to tell. These tests pin the ``.degraded`` marker on
``HybridBackend.search()`` and the ``degraded`` field in the MCP recall
envelope.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from mind_mem.hybrid_recall import (
    HybridBackend,
    RecallResults,
    VectorLegError,
    _as_results,
    _union_degraded,
)


def _make_workspace() -> str:
    ws = tempfile.mkdtemp(prefix="mm_deg_")
    os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as f:
        f.write(
            "# DECISIONS\n\n---\n\n"
            "[D-20260101-001]\nStatement: The capital of France is Paris.\nStatus: active\n\n---\n\n"
            "[D-20260101-002]\nStatement: Python is used for data science.\nStatus: active\n"
        )
    return ws


# --------------------------------------------------------------------------
# RecallResults container
# --------------------------------------------------------------------------


def test_recall_results_is_a_list_and_carries_marker():
    rr = _as_results([{"doc_id": "a"}], {"leg": "vector", "reason": "error"})
    assert isinstance(rr, list)
    assert rr[0]["doc_id"] == "a"
    assert rr.degraded == {"leg": "vector", "reason": "error"}


def test_plain_list_getattr_degraded_defaults_none():
    assert getattr([], "degraded", None) is None
    assert RecallResults([]).degraded is None


# --------------------------------------------------------------------------
# search() degradation paths
# --------------------------------------------------------------------------


def test_bm25_config_is_not_degraded():
    ws = _make_workspace()
    hb = HybridBackend({})  # vector never requested
    results = hb.search("capital of France", ws, limit=5)
    assert getattr(results, "degraded", None) is None


def test_vector_requested_but_unavailable_is_marked():
    ws = _make_workspace()
    hb = HybridBackend({"vector_enabled": True})
    # Force the "requested but backend unavailable" condition.
    hb._vector_available = False
    results = hb.search("capital of France", ws, limit=5)
    assert results.degraded == {"leg": "vector", "reason": "unavailable"}
    # still returns BM25 results, not empty
    assert len(results) >= 1


def test_vector_leg_failure_degrades_and_marks(monkeypatch):
    ws = _make_workspace()
    hb = HybridBackend({"vector_enabled": True})
    hb._vector_available = True  # get past the early return into the fused path

    def _boom(*a, **k):
        raise VectorLegError("import_failed")

    monkeypatch.setattr(hb, "_vector_search", _boom)
    results = hb.search("capital of France", ws, limit=5)
    assert results.degraded == {"leg": "vector", "reason": "import_failed"}
    # BM25 leg still produced results (degrade, don't crash / empty)
    assert len(results) >= 1


def test_vector_leg_timeout_degrades_and_marks(monkeypatch):
    import time as _time

    ws = _make_workspace()
    hb = HybridBackend({"vector_enabled": True})
    hb._vector_available = True
    monkeypatch.setattr(hb, "_vector_deadline_seconds", lambda: 0.05)

    def _slow(*a, **k):
        _time.sleep(0.5)
        return []

    monkeypatch.setattr(hb, "_vector_search", _slow)
    results = hb.search("capital of France", ws, limit=5)
    assert results.degraded == {"leg": "vector", "reason": "deadline_exceeded"}


def test_healthy_vector_leg_not_marked(monkeypatch):
    ws = _make_workspace()
    hb = HybridBackend({"vector_enabled": True})
    hb._vector_available = True
    # Vector leg runs fine and simply returns nothing — NOT a degradation.
    monkeypatch.setattr(hb, "_vector_search", lambda *a, **k: [])
    results = hb.search("capital of France", ws, limit=5)
    assert getattr(results, "degraded", None) is None


def test_vector_search_raises_typed_error_on_failure(monkeypatch):
    """_vector_search must raise VectorLegError (not swallow to []) so the
    caller can distinguish a failed vector leg from a genuinely-empty one."""
    ws = _make_workspace()
    hb = HybridBackend({"vector_enabled": True})

    import mind_mem.recall_vector as rv

    def _boom(*a, **k):
        raise RuntimeError("embed service down")

    # search_batch is the first path _vector_search tries.
    monkeypatch.setattr(rv, "search_batch", _boom, raising=False)
    with pytest.raises(VectorLegError) as exc:
        hb._vector_search("q", ws)
    assert exc.value.reason == "error"


# --------------------------------------------------------------------------
# MCP recall envelope surfaces the marker
# --------------------------------------------------------------------------


def test_mcp_recall_envelope_surfaces_degraded(monkeypatch):
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = _make_workspace()
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)

    class _FakeHB:
        @staticmethod
        def from_config(config):
            return _FakeHB()

        def search(self, query, workspace, limit=10, active_only=False, **kwargs):
            return _as_results(
                [{"_id": "D-1", "score": 1.0}],
                {"leg": "vector", "reason": "deadline_exceeded"},
            )

    # Patch the symbol imported inside the hybrid branch.
    import mind_mem.hybrid_recall as hr

    monkeypatch.setattr(hr, "HybridBackend", _FakeHB)

    raw = mcp_recall._recall_impl_uncached("capital of France", limit=5, backend="hybrid")
    envelope = json.loads(raw)
    assert envelope.get("degraded") == {"leg": "vector", "reason": "deadline_exceeded"}
    assert any("BM25-only" in w for w in envelope.get("warnings", []))


# --------------------------------------------------------------------------
# Multi-query degraded propagation (_union_degraded + _search_expanded)
# --------------------------------------------------------------------------


def test_union_degraded_none_when_no_variant_degraded():
    assert _union_degraded([None, None], total=2) is None
    assert _union_degraded([], total=0) is None


def test_union_degraded_single_variant():
    out = _union_degraded([None, {"leg": "vector", "reason": "error"}, None], total=3)
    assert out == {
        "leg": "vector",
        "reason": "error",
        "variants_degraded": "1",
        "variants_total": "3",
    }


def test_union_degraded_dedups_and_sorts_reasons():
    out = _union_degraded(
        [
            {"leg": "vector", "reason": "deadline_exceeded"},
            {"leg": "vector", "reason": "unavailable"},
            {"leg": "vector", "reason": "unavailable"},
        ],
        total=3,
    )
    assert out["leg"] == "vector"
    assert out["reason"] == "deadline_exceeded,unavailable"  # sorted + deduped
    assert out["variants_degraded"] == "3"
    assert out["variants_total"] == "3"


def test_search_expanded_marks_combined_when_any_variant_degraded(monkeypatch):
    """ANY degraded variant marks the fused multi-query result."""
    ws = _make_workspace()
    hb = HybridBackend({"vector_enabled": True})

    def _fake(query, workspace, *a, **k):
        if query == "variant-bad":
            return _as_results(
                [{"_id": "D-2", "score": 0.5}],
                {"leg": "vector", "reason": "deadline_exceeded"},
            )
        return _as_results([{"_id": "D-1", "score": 0.9}], None)

    monkeypatch.setattr(hb, "search", _fake)
    out = hb._search_expanded(
        queries=["variant-ok", "variant-bad", "variant-ok-2"],
        workspace=ws,
        limit=5,
    )
    assert isinstance(out, RecallResults)
    assert out.degraded is not None
    assert out.degraded["leg"] == "vector"
    assert out.degraded["reason"] == "deadline_exceeded"
    assert out.degraded["variants_degraded"] == "1"
    assert out.degraded["variants_total"] == "3"
    # Fused results are still returned — degrade, don't drop.
    assert len(out) >= 1


def test_search_expanded_not_marked_when_all_variants_healthy(monkeypatch):
    ws = _make_workspace()
    hb = HybridBackend({"vector_enabled": True})
    monkeypatch.setattr(
        hb,
        "search",
        lambda query, workspace, *a, **k: _as_results([{"_id": "D-1", "score": 1.0}], None),
    )
    out = hb._search_expanded(queries=["a", "b", "c"], workspace=ws, limit=5)
    assert getattr(out, "degraded", None) is None


def test_search_expanded_unions_distinct_variant_reasons(monkeypatch):
    ws = _make_workspace()
    hb = HybridBackend({"vector_enabled": True})
    markers = {
        "q1": {"leg": "vector", "reason": "unavailable"},
        "q2": {"leg": "vector", "reason": "import_failed"},
    }

    def _fake(query, workspace, *a, **k):
        return _as_results([{"_id": query, "score": 1.0}], markers.get(query))

    monkeypatch.setattr(hb, "search", _fake)
    out = hb._search_expanded(queries=["q1", "q2"], workspace=ws, limit=5)
    assert out.degraded["reason"] == "import_failed,unavailable"
    assert out.degraded["variants_degraded"] == "2"
    assert out.degraded["variants_total"] == "2"


def test_search_expanded_single_variant_branch_propagates(monkeypatch):
    """The len<=1 inline branch also propagates the marker."""
    ws = _make_workspace()
    hb = HybridBackend({"vector_enabled": True})
    monkeypatch.setattr(
        hb,
        "search",
        lambda query, workspace, *a, **k: _as_results([{"_id": "D-1", "score": 1.0}], {"leg": "vector", "reason": "error"}),
    )
    out = hb._search_expanded(queries=["only"], workspace=ws, limit=5)
    assert out.degraded is not None
    assert out.degraded["reason"] == "error"
    assert out.degraded["variants_degraded"] == "1"
    assert out.degraded["variants_total"] == "1"


def test_expansion_end_to_end_propagates_degraded(monkeypatch):
    """Real search() -> expansion -> per-variant recursion: a failing
    vector leg on every variant surfaces as a combined degraded marker on
    the fused result (the single-query 419bee5 fix is no longer lost on
    the multi-query path)."""
    import mind_mem.query_expansion as qe

    ws = _make_workspace()
    hb = HybridBackend({"vector_enabled": True, "query_expansion": {"enabled": True}})
    hb._vector_available = True  # reach the fused path in each recursion

    # Expand into 3 variants.
    monkeypatch.setattr(qe, "expand_queries", lambda query, config=None: [query, query + " alt", query + " alt2"])

    # Every variant's vector leg fails -> each single-query result degrades.
    def _boom(*a, **k):
        raise VectorLegError("import_failed")

    monkeypatch.setattr(hb, "_vector_search", _boom)

    results = hb.search("capital of France", ws, limit=5)
    assert getattr(results, "degraded", None) is not None
    assert results.degraded["leg"] == "vector"
    assert results.degraded["reason"] == "import_failed"
    assert results.degraded["variants_total"] == "3"
    # BM25 leg still produced fused results across variants.
    assert len(results) >= 1
