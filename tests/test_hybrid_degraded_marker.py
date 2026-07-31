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

        def search(self, query, workspace, limit=10, active_only=False):
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
