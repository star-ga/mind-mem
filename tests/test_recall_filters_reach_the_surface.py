# Copyright 2026 STARGA, Inc.
"""The engine's five recall filters must be reachable from the MCP surface.

BTT-010 recorded ``lifecycle`` as NOT_STARTED. The engine has had it for a
while, along with four siblings: ``_recall_core.recall()`` accepts ``since``,
``until``, ``lifecycle``, ``event_id`` and ``min_maturity``, and
``_apply_lifecycle_filter`` (``_recall_core.py:578``) implements the filtering.

MEASURED before this change, on 28b9d7f1:

    grep -c lifecycle src/mind_mem/mcp/tools/recall.py  -> 0
    grep -c lifecycle src/mind_mem/api/rest.py          -> 0
    the only lifecycle= callers in src/ were compaction.py and consolidation.py,
    neither of them a recall surface.

So the capability was built, tested and unreachable: no MCP tool, REST route or
CLI command could express it. This is a wiring gap, not a missing feature, and
the fix is to connect it rather than to remove it.

THE TRAP THESE CONTROLS EXIST FOR. ``_recall_impl_ranked`` serves from
``recall_cache`` on a key built by ``make_cache_key``. Threading a filter to the
engine WITHOUT adding it to that key makes a filtered query collide with the
unfiltered one and be answered from its cached envelope -- the filter would
appear to work in a unit test and silently do nothing in a warm process.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile

import pytest

from mind_mem.recall_cache import make_cache_key

FILTERS = ("since", "until", "lifecycle", "event_id", "min_maturity")


@pytest.fixture
def indexed_ws(monkeypatch):
    """An INDEXED temp workspace, pinned as the surface's workspace.

    These controls previously read whatever workspace the ambient environment
    resolved to. That made them order-dependent: they passed alone and failed
    inside the wider selection, because a workspace with no FTS database sends
    ``_recall_impl_uncached`` down the scan fallback and ``fts_query`` is never
    called at all. A control whose outcome depends on which tests ran before it
    is not measuring the thing it names.
    """
    from mind_mem.mcp.tools import recall as r
    from mind_mem.sqlite_index import build_index

    ws = tempfile.mkdtemp(prefix="mm-filters-")
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump({"cache": {"enabled": False}}, fh)
    dec = os.path.join(ws, "decisions")
    os.makedirs(dec, exist_ok=True)
    with open(os.path.join(dec, "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write("[D-1]\nStatement: The vendor quoted a March price.\nStatus: active\n")
    build_index(ws, incremental=False)
    monkeypatch.setattr(r, "_workspace", lambda: ws)
    return ws


def test_the_engine_really_does_accept_all_five():
    from mind_mem._recall_core import recall as engine_recall

    params = inspect.signature(engine_recall).parameters
    for name in FILTERS:
        assert name in params, f"engine lost {name}"


def test_every_filter_is_expressible_on_the_public_tool():
    from mind_mem.mcp.tools.public import recall as public_recall

    params = inspect.signature(public_recall).parameters
    missing = [f for f in FILTERS if f not in params]
    assert not missing, f"unreachable from the public MCP tool: {missing}"


def test_every_filter_is_expressible_on_the_impl_chain():
    from mind_mem.mcp.tools import recall as r

    for fn in (r._recall_impl, r._recall_impl_ranked, r._recall_impl_uncached):
        params = inspect.signature(fn).parameters
        missing = [f for f in FILTERS if f not in params]
        assert not missing, f"{fn.__name__} drops {missing}; the chain breaks before the engine"


def test_a_filtered_query_does_not_collide_with_the_unfiltered_one():
    """The cache-key trap. Same query, different filter -> different key."""
    base = make_cache_key("q", namespace="ws", limit=10, backend="auto")
    for name in FILTERS:
        value = 0.5 if name == "min_maturity" else "x"
        keyed = make_cache_key("q", namespace="ws", limit=10, backend="auto", filters={name: value})
        assert keyed != base, f"{name} is absent from the cache key: a filtered query would be answered from the unfiltered entry"


def test_two_different_values_of_one_filter_do_not_collide():
    a = make_cache_key("q", namespace="ws", filters={"lifecycle": "durable"})
    b = make_cache_key("q", namespace="ws", filters={"lifecycle": "ephemeral"})
    assert a != b


def test_absent_filters_leave_the_key_byte_identical():
    """Default callers must not have their cache invalidated by this change."""
    assert make_cache_key("q", namespace="ws", limit=10, backend="auto") == make_cache_key(
        "q", namespace="ws", limit=10, backend="auto", filters=None
    )
    assert make_cache_key("q", namespace="ws", limit=10, backend="auto") == make_cache_key(
        "q", namespace="ws", limit=10, backend="auto", filters={}
    )


def test_filters_reach_the_post_filter_funnel(monkeypatch, indexed_ws):
    """Dispatch proof: the values must ARRIVE somewhere that applies them.

    The first version of this control spied on ``recall_engine`` and failed --
    correctly. ``recall_engine`` is only the SCAN FALLBACK, taken when no FTS
    index exists; on any indexed workspace the leg is ``fts_query``. Spying on
    the fallback would have proved nothing about the path real deployments use.

    The funnel is the right seam: it is where every leg's results converge and
    the only place the filter contract is stated.
    """
    from mind_mem._recall_core import _apply_post_filters as real_funnel
    from mind_mem.mcp.tools import recall as r

    seen: dict = {}

    def spy(hits, **kw):
        seen.update(kw)
        return real_funnel(hits, **kw)

    # Feed the leg real hits: the surface funnel is guarded by ``if results:``,
    # so an empty result set would skip it and the spy would record nothing --
    # a vacuous pass. This is the positive control, inlined.
    monkeypatch.setattr(
        r,
        "fts_query",
        lambda ws, query, **kw: [{"id": "D-1", "Statement": "vendor quote", "status": "active", "score": 1.0}],
        raising=False,
    )
    monkeypatch.setattr("mind_mem._recall_core._apply_post_filters", spy)
    r._recall_impl_uncached(
        "vendor quote",
        limit=5,
        backend="bm25",
        since="2026-01-01",
        until="2026-12-31",
        lifecycle="durable",
        event_id="E-1",
        min_maturity=0.25,
    )
    assert seen, "the surface never funnelled its results; the filters are dropped"
    assert seen.get("lifecycle") == "durable", f"lifecycle never reached the funnel; got {sorted(seen)}"
    assert seen.get("since") == "2026-01-01"
    assert seen.get("until") == "2026-12-31"
    assert seen.get("event_id") == "E-1"
    assert seen.get("min_maturity") == 0.25


def test_the_widened_pool_is_only_taken_when_a_filter_is_set(monkeypatch, indexed_ws):
    """An unfiltered call must not silently start retrieving 4x as much."""
    from mind_mem.mcp.tools import recall as r

    seen: list = []

    def spy_fts(ws, query, **kw):
        seen.append(kw.get("limit"))
        return []

    monkeypatch.setattr(r, "fts_query", spy_fts, raising=False)
    r._recall_impl_uncached("vendor quote", limit=5, backend="bm25")
    r._recall_impl_uncached("vendor quote", limit=5, backend="bm25", lifecycle="durable")
    assert len(seen) == 2, f"fts_query was not the leg taken on both calls: {seen}"
    assert seen[0] == 5, f"unfiltered call widened its pool: {seen[0]}"
    assert seen[1] > 5, f"filtered call did not widen its pool: {seen[1]}"
