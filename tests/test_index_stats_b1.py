# Copyright 2026 STARGA, Inc.
"""Ensure v2.0.0b1 prefix-cache + prefetch stats reach the MCP index_stats envelope."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def minimal_workspace(monkeypatch):
    tmp = tempfile.TemporaryDirectory()
    ws = Path(tmp.name)
    (ws / "decisions").mkdir()
    (ws / "tasks").mkdir()
    (ws / "entities").mkdir()
    (ws / "intelligence").mkdir()
    (ws / "memory").mkdir()
    (ws / "mind-mem.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    yield ws
    tmp.cleanup()


def _call_tool(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
    raw = fn(**kwargs)
    return json.loads(raw) if isinstance(raw, str) else raw


def test_index_stats_surfaces_prefix_cache_stats(minimal_workspace):
    from mind_mem import mcp_server, prefix_cache

    prefix_cache.reset_all()
    cache = prefix_cache.get_cache("test_ns_unique_b1")
    cache.put("prefix", {"q": "x"}, "value")
    cache.get("prefix", {"q": "x"})  # record a hit
    cache.get("prefix", {"q": "miss"})  # record a miss

    stats = _call_tool(mcp_server.index_stats)
    assert "prefix_caches" in stats
    namespaces = {entry["namespace"] for entry in stats["prefix_caches"]}
    assert "test_ns_unique_b1" in namespaces
    target = next(
        entry for entry in stats["prefix_caches"]
        if entry["namespace"] == "test_ns_unique_b1"
    )
    assert target["hits"] >= 1
    assert target["misses"] >= 1


def test_index_stats_surfaces_prefetch_stats(minimal_workspace):
    from mind_mem import mcp_server, speculative_prefetch

    speculative_prefetch.reset_default_predictor()
    pred = speculative_prefetch.get_default_predictor()
    pred.observe("observed query", ["B-001", "B-002"])

    stats = _call_tool(mcp_server.index_stats)
    assert "speculative_prefetch" in stats
    prefetch = stats["speculative_prefetch"]
    assert prefetch["signatures"] >= 1
    assert prefetch["observations"] >= 2
    assert "hit_rate" in prefetch
