"""Tests for graph_boost recall parameter."""

from __future__ import annotations

import os

import pytest

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


@pytest.fixture
def ws(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "graph_test.md")
    with open(blocks_md, "w") as f:
        f.write("[GR-001]\nType: Decision\nStatement: Graph boost test alpha\nReferences: GR-002\n\n")
        f.write("[GR-002]\nType: Decision\nStatement: Graph boost test beta\nReferences: GR-001\n\n")
        f.write("[GR-003]\nType: Decision\nStatement: Unrelated block gamma\n\n")
    return ws


def test_graph_boost_enabled(ws):
    """graph_boost=True doesn't crash."""
    results = recall(ws, "graph boost test", limit=5, graph_boost=True)
    assert isinstance(results, list)


def test_graph_boost_disabled(ws):
    """graph_boost=False returns normal results."""
    results = recall(ws, "graph boost test", limit=5, graph_boost=False)
    assert isinstance(results, list)


def test_graph_boost_default(ws):
    """Default graph_boost works."""
    results = recall(ws, "graph boost test", limit=5)
    assert isinstance(results, list)
