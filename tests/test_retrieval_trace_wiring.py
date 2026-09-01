"""``retrieval_trace`` wired into the live recall pipeline.

The module has always been importable; what it lacked was a caller. These
tests pin the wiring itself:

* the conditional retrieval features in :class:`~mind_mem.hybrid_recall.HybridBackend`
  record a step each into the trace open for the request,
* the summary rides back out on ``RecallResults.trace`` and into the MCP recall
  envelope,
* and with ``recall.retrieval.trace_attribution`` off — the default — no trace
  is ever opened, nothing records, and the ranked output is the same list it
  was before the wiring existed.

The load-bearing assertion is :func:`test_graph_expanded_query_records_graph_expand_step`:
a query that really does get graph-expanded comes back with a ``graph_expand``
step whose ``added_count`` is the number of blocks the walk appended. Delete the
``_step`` wrapper in ``_maybe_graph_expand`` and it fails.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from mind_mem.hybrid_recall import HybridBackend
from mind_mem.retrieval_trace import current_trace

SEED_ID = "D-20260101-001"
NEIGHBOUR_ID = "D-20260101-002"
QUERY = "capital of France"


def _make_workspace(trace_attribution: bool) -> str:
    """A two-block workspace where one block cross-references the other.

    BM25 matches only the seed; the neighbour is reachable *only* through the
    xref walk, so its presence in the result set is proof graph expansion ran.
    """
    ws = tempfile.mkdtemp(prefix="mm_trace_wiring_")
    os.makedirs(os.path.join(ws, "decisions"), exist_ok=True)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as f:
        f.write(
            "# DECISIONS\n\n---\n\n"
            f"[{SEED_ID}]\n"
            f"Statement: The capital of France is Paris, see {NEIGHBOUR_ID}.\n"
            "Status: active\n\n---\n\n"
            f"[{NEIGHBOUR_ID}]\n"
            "Statement: Neighbour block, reachable only by the cross-reference walk.\n"
            "Status: active\n"
        )
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as f:
        json.dump({"recall": _recall_config(trace_attribution)}, f)
    return ws


def _recall_config(trace_attribution: bool) -> dict:
    return {
        "vector_enabled": True,
        "retrieval": {
            "multi_hop": {"enabled": True},
            "trace_attribution": trace_attribution,
        },
    }


def _backend(trace_attribution: bool) -> HybridBackend:
    """A backend on the FUSED path with a silent vector leg.

    ``_maybe_graph_expand`` only runs post-fusion, so the BM25-only
    early-return branch never reaches it. Forcing ``_vector_available`` and
    stubbing the leg to return nothing is what the existing degraded-marker
    tests do; it keeps the real ``search`` pipeline intact.
    """
    hb = HybridBackend(_recall_config(trace_attribution))
    hb._vector_available = True
    hb._vector_search = lambda *a, **k: []  # type: ignore[method-assign]
    return hb


def _steps(results: object) -> list[dict]:
    trace = getattr(results, "trace", None)
    assert trace is not None, "expected an attribution trace on the results"
    steps = trace["steps"]
    assert isinstance(steps, list)
    return steps


def _step_named(results: object, feature: str) -> dict:
    matches = [s for s in _steps(results) if s["feature"] == feature]
    assert matches, f"no {feature!r} step in {[s['feature'] for s in _steps(results)]}"
    return matches[0]


# --------------------------------------------------------------------------
# Flag ON — the features actually record what they contributed
# --------------------------------------------------------------------------


def test_graph_expanded_query_records_graph_expand_step() -> None:
    """The definition of working: a graph-expanded query reports the expansion.

    Not "a trace object exists" — the step has to carry the real count of
    blocks the walk appended, and the neighbour it appended has to be in the
    served results.
    """
    ws = _make_workspace(trace_attribution=True)
    results = _backend(trace_attribution=True).search(QUERY, ws, limit=5)

    ids = [r.get("_id") for r in results]
    assert SEED_ID in ids
    assert NEIGHBOUR_ID in ids, "graph expansion did not run — test setup is wrong"

    step = _step_named(results, "graph_expand")
    assert step["added_count"] > 0
    assert step["added_count"] == len(ids) - 1  # exactly the one walked neighbour
    assert step["latency_ms"] >= 0
    assert step["metadata"]["max_hops"] == 2


def test_trace_summary_carries_query_and_total_latency() -> None:
    ws = _make_workspace(trace_attribution=True)
    results = _backend(trace_attribution=True).search(QUERY, ws, limit=5)
    trace = results.trace
    assert trace["query"] == QUERY
    assert trace["total_latency_ms"] >= 0


def test_other_enabled_hooks_also_record() -> None:
    """graph_expand is not special-cased — the sibling hooks record too."""
    ws = _make_workspace(trace_attribution=True)
    results = _backend(trace_attribution=True).search(QUERY, ws, limit=5)
    features = [s["feature"] for s in _steps(results)]
    # entity_prefetch auto-enables alongside the multi-hop walk.
    assert "entity_prefetch" in features
    # A hook that never fired must not invent a step for itself.
    assert "kg_expand" not in features


def test_disabled_feature_records_no_step() -> None:
    """Trace on, expansion off — the trace exists and is honestly empty."""
    ws = _make_workspace(trace_attribution=True)
    hb = HybridBackend(
        {
            "vector_enabled": True,
            "retrieval": {
                "multi_hop": {"enabled": False, "auto_enable": False},
                "entity_prefetch": {"enabled": False},
                "trace_attribution": True,
            },
        }
    )
    hb._vector_available = True
    hb._vector_search = lambda *a, **k: []  # type: ignore[method-assign]
    results = hb.search(QUERY, ws, limit=5)
    assert "graph_expand" not in [s["feature"] for s in _steps(results)]
    assert NEIGHBOUR_ID not in [r.get("_id") for r in results]


# --------------------------------------------------------------------------
# Flag OFF — unchanged, and not merely "trace is None"
# --------------------------------------------------------------------------


def test_flag_off_attaches_no_trace() -> None:
    ws = _make_workspace(trace_attribution=False)
    results = _backend(trace_attribution=False).search(QUERY, ws, limit=5)
    assert getattr(results, "trace", None) is None


def test_flag_off_opens_no_trace_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-cost when off: the feature helpers see NO active trace.

    ``retrieval_trace.step`` reads the clock twice and emits a debug record on
    every call. With the flag off none of that may happen, so the assertion is
    on the ContextVar the ``_step`` shim consults — not on the output.
    """
    import mind_mem.graph_recall as graph_recall

    seen: list[object] = []
    real = graph_recall.graph_expand

    def _spy(*args, **kwargs):
        seen.append(current_trace())
        return real(*args, **kwargs)

    monkeypatch.setattr(graph_recall, "graph_expand", _spy)

    ws = _make_workspace(trace_attribution=False)
    _backend(trace_attribution=False).search(QUERY, ws, limit=5)
    assert seen, "graph_expand never ran — the spy proved nothing"
    assert seen == [None] * len(seen)

    seen.clear()
    ws_on = _make_workspace(trace_attribution=True)
    _backend(trace_attribution=True).search(QUERY, ws_on, limit=5)
    assert seen and all(t is not None for t in seen)


def test_flag_off_ranking_is_identical_to_flag_on() -> None:
    """Attribution is observation: turning it on cannot move a result."""
    ws_off = _make_workspace(trace_attribution=False)
    ws_on = _make_workspace(trace_attribution=True)
    off = _backend(trace_attribution=False).search(QUERY, ws_off, limit=5)
    on = _backend(trace_attribution=True).search(QUERY, ws_on, limit=5)

    def _ranking(rows: list[dict]) -> list[tuple[str, float]]:
        return [(str(r.get("_id")), round(float(r.get("score", 0.0) or 0.0), 9)) for r in rows]

    assert _ranking(list(off)) == _ranking(list(on))
    assert getattr(off, "degraded", None) == getattr(on, "degraded", None)


def test_empty_query_still_short_circuits_with_flag_on() -> None:
    ws = _make_workspace(trace_attribution=True)
    assert list(_backend(trace_attribution=True).search("   ", ws, limit=5)) == []


# --------------------------------------------------------------------------
# MCP recall envelope
# --------------------------------------------------------------------------


def _patch_backend_onto_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the MCP hybrid branch through a fused-path backend.

    ``from_config`` resolves ``HybridBackend`` as a module global, so patching
    the global is enough for the subclass to build itself.
    """
    import mind_mem.hybrid_recall as hr

    class _FusedBackend(hr.HybridBackend):
        def __init__(self, config=None):
            super().__init__(config)
            self._vector_available = True

        def _vector_search(self, *a, **k):
            return []

    monkeypatch.setattr(hr, "HybridBackend", _FusedBackend)


def test_mcp_envelope_surfaces_the_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = _make_workspace(trace_attribution=True)
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    _patch_backend_onto_mcp(monkeypatch)

    envelope = json.loads(mcp_recall._recall_impl_uncached(QUERY, limit=5, backend="hybrid"))
    steps = envelope["trace"]["steps"]
    graph = [s for s in steps if s["feature"] == "graph_expand"]
    assert graph and graph[0]["added_count"] > 0


def test_mcp_envelope_has_no_trace_key_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = _make_workspace(trace_attribution=False)
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    _patch_backend_onto_mcp(monkeypatch)

    envelope = json.loads(mcp_recall._recall_impl_uncached(QUERY, limit=5, backend="hybrid"))
    assert "trace" not in envelope


# --------------------------------------------------------------------------
# The trace never comes out of the recall cache
# --------------------------------------------------------------------------


def test_trace_attribution_enabled_reads_the_recall_section() -> None:
    from mind_mem.mcp.tools.recall import _trace_attribution_enabled

    assert _trace_attribution_enabled({"recall": {"retrieval": {"trace_attribution": True}}}) is True
    assert _trace_attribution_enabled({"recall": {"retrieval": {}}}) is False
    assert _trace_attribution_enabled({"recall": "not-a-dict"}) is False
    assert _trace_attribution_enabled({}) is False
    assert _trace_attribution_enabled(None) is False


def test_tracing_bypasses_the_recall_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache hit runs no features, so a cached trace would be a lie."""
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = _make_workspace(trace_attribution=True)
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)

    calls: list[str] = []

    def _counting(query, **kwargs):
        calls.append(query)
        return json.dumps({"results": [], "count": 0})

    monkeypatch.setattr(mcp_recall, "_recall_impl_uncached", _counting)

    mcp_recall._recall_impl(QUERY, limit=5, backend="hybrid")
    mcp_recall._recall_impl(QUERY, limit=5, backend="hybrid")
    assert len(calls) == 2, "second identical query was served from cache while tracing"


def test_cache_still_used_when_tracing_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bypass is scoped to the flag — the default path keeps its cache."""
    import mind_mem.mcp.tools.recall as mcp_recall

    ws = _make_workspace(trace_attribution=False)
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)

    calls: list[str] = []

    def _counting(query, **kwargs):
        calls.append(query)
        return json.dumps({"results": [], "count": 0})

    monkeypatch.setattr(mcp_recall, "_recall_impl_uncached", _counting)

    mcp_recall._recall_impl(QUERY, limit=5, backend="hybrid")
    mcp_recall._recall_impl(QUERY, limit=5, backend="hybrid")
    assert len(calls) == 1, "recall cache stopped serving the default (untraced) path"
