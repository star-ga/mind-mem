"""Tests for graph_boost recall parameter."""

from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "graph_test.md")
    with open(blocks_md, "w") as f:
        f.write("[GR-001]\nType: Decision\nStatement: Graph boost test alpha\nReferences: GR-002\n\n")
        f.write("[GR-002]\nType: Decision\nStatement: Graph boost test beta\nReferences: GR-001\n\n")
        f.write("[GR-003]\nType: Decision\nStatement: Unrelated block gamma\n\n")
    return ws


def test_graph_boost_enabled():
    """graph_boost=True doesn't crash."""
    ws = _make_workspace()
    results = recall(ws, "graph boost test", limit=5, graph_boost=True)
    assert isinstance(results, list)


def test_graph_boost_disabled():
    """graph_boost=False returns normal results."""
    ws = _make_workspace()
    results = recall(ws, "graph boost test", limit=5, graph_boost=False)
    assert isinstance(results, list)


def test_graph_boost_default():
    """Default graph_boost works."""
    ws = _make_workspace()
    results = recall(ws, "graph boost test", limit=5)
    assert isinstance(results, list)
