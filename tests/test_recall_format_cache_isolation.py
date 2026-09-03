"""``format`` is not in the recall-cache key, so it must not be applied inside it.

``recall_cache.make_cache_key`` digests exactly
``(query, namespace, limit, backend, active_only, scoring_instant, index_anchor)``.
``format`` is absent. Building the ``format="bundle"`` envelope inside the
cached region therefore stored one caller's chosen SHAPE under a key the other
shape hashes to identically, and every later caller inside the TTL window was
served the wrong envelope — a bundle client receiving a raw blocks list with no
``facts``/``relations``/``timeline``, or a blocks client receiving a bundle.

The fix moves the conversion post-cache, onto the same rail the attestation and
explain annotations already run on. These tests pin that: the format the caller
asked for is the format the caller gets, in either order, with the cache on.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

QUERY = "pineapple protocol rollout"


def _workspace(tmp_path: Any) -> str:
    ws = os.path.join(str(tmp_path), "ws")
    for d in ("decisions", "tasks", "entities", "intelligence", "memory"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w", encoding="utf-8") as fh:
        fh.write(f"[D-20260829-001]\nStatement: The {QUERY} decision is approved.\nDate: 2026-08-29\nStatus: active\n\n---\n\n")
    with open(os.path.join(ws, "mind-mem.json"), "w", encoding="utf-8") as fh:
        json.dump({"recall": {"vector_enabled": False, "provider": "local"}}, fh)
    return ws


@pytest.fixture()
def cached_recall_module(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The MCP recall surface pointed at a temp workspace, cache ENABLED.

    The cache is the whole point of the test, so it is switched on explicitly
    rather than left to the (already-on) default — a later default flip must
    not silently turn this into a no-op.
    """
    import mind_mem.mcp.tools.recall as mcp_recall
    from mind_mem import recall_cache

    ws = _workspace(tmp_path)
    monkeypatch.setattr(mcp_recall, "_workspace", lambda: ws)
    monkeypatch.setattr(
        mcp_recall,
        "_load_config",
        lambda _ws: {"cache": {"enabled": True, "ttl_seconds": 3600}},
    )
    recall_cache.reset_singleton()
    try:
        yield mcp_recall
    finally:
        recall_cache.reset_singleton()


def _shape(raw: str) -> str:
    """Name the envelope shape from its own keys, not from what we hoped for."""
    parsed = json.loads(raw)
    assert isinstance(parsed, dict), raw
    if "facts" in parsed and "relations" in parsed:
        return "bundle"
    if "results" in parsed:
        return "blocks"
    return f"unknown:{sorted(parsed)}"


def test_bundle_after_blocks_is_not_served_the_cached_blocks_envelope(cached_recall_module: Any) -> None:
    """blocks first, bundle second — the second caller must still get a bundle."""
    first = cached_recall_module._recall_impl(QUERY, limit=5, backend="bm25", format="blocks")
    assert _shape(first) == "blocks"

    second = cached_recall_module._recall_impl(QUERY, limit=5, backend="bm25", format="bundle")
    assert _shape(second) == "bundle"
    parsed = json.loads(second)
    assert parsed["query"] == QUERY
    assert "timeline" in parsed and "entities" in parsed and "source_blocks" in parsed


def test_blocks_after_bundle_is_not_served_the_cached_bundle_envelope(cached_recall_module: Any) -> None:
    """bundle first, blocks second — the reverse direction of the same key collision."""
    first = cached_recall_module._recall_impl(QUERY, limit=5, backend="bm25", format="bundle")
    assert _shape(first) == "bundle"

    second = cached_recall_module._recall_impl(QUERY, limit=5, backend="bm25", format="blocks")
    assert _shape(second) == "blocks"


def test_the_cached_payload_is_the_blocks_shape_regardless_of_the_first_caller(
    cached_recall_module: Any,
    tmp_path: Any,
) -> None:
    """Exactly one shape is stored, and it is the un-narrowed one.

    Asserted against the cache itself rather than through a second recall, so
    the test names the MECHANISM (what got stored) and not merely a symptom
    that some other layer could coincidentally repair.
    """
    # datetime.UTC is 3.11+, but requires-python is >=3.10 and the matrix
    # runs 3.10. timezone.utc is the same instant on every supported version.
    from datetime import datetime, timezone

    from mind_mem.prefetch import chain_head
    from mind_mem.recall_cache import get_cache, make_cache_key

    # The instant is resolved inside _recall_impl, so it is bracketed here
    # rather than guessed: sampling only after the call would miss the key by a
    # day on a run that crosses UTC midnight.
    before = datetime.now(timezone.utc).date().isoformat()
    cached_recall_module._recall_impl(QUERY, limit=5, backend="bm25", format="bundle")
    after = datetime.now(timezone.utc).date().isoformat()

    # The key also carries the governed-ledger head, which is what makes a cached
    # answer belong to one corpus state. Read through the same resolver the
    # recall path used, so this reconstructs the real key rather than a
    # hand-rolled guess at it.
    anchor = chain_head(cached_recall_module._workspace())
    cache = get_cache(None)
    stored = None
    for instant in dict.fromkeys((before, after)):
        key = make_cache_key(
            QUERY,
            limit=5,
            backend="bm25",
            active_only=False,
            scoring_instant=instant,
            index_anchor=anchor,
        )
        stored = stored or cache.get(key)
    assert stored is not None, "cache should have been populated on the miss"
    assert _shape(stored) == "blocks", "the cache must hold the format-independent envelope"


def test_bundle_and_blocks_agree_on_the_underlying_result_set(cached_recall_module: Any) -> None:
    """Sharing one cached retrieval must not change WHICH blocks are returned."""
    blocks = json.loads(cached_recall_module._recall_impl(QUERY, limit=5, backend="bm25", format="blocks"))
    bundle = json.loads(cached_recall_module._recall_impl(QUERY, limit=5, backend="bm25", format="bundle"))

    block_ids = [r.get("_id") for r in blocks["results"]]
    bundle_ids = [r.get("_id") for r in bundle["source_blocks"]]
    assert block_ids == bundle_ids
