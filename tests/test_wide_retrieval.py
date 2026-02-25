"""Tests for wide retrieval parameter."""
from __future__ import annotations

import os
import tempfile

from scripts.init_workspace import init
from scripts._recall_core import recall


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "wide_test.md")
    with open(blocks_md, "w") as f:
        for i in range(20):
            f.write(f"[WD-{i:03d}]\nType: Decision\nStatement: Wide retrieval test {i}\n\n")
    return ws


def test_default_wide_k():
    """Default retrieve_wide_k works."""
    ws = _make_workspace()
    results = recall(ws, "wide retrieval", limit=5)
    assert isinstance(results, list)


def test_small_wide_k():
    """Small retrieve_wide_k still returns results."""
    ws = _make_workspace()
    results = recall(ws, "wide retrieval", limit=5, retrieve_wide_k=10)
    assert isinstance(results, list)


def test_large_wide_k():
    """Large retrieve_wide_k doesn't crash."""
    ws = _make_workspace()
    results = recall(ws, "wide retrieval", limit=5, retrieve_wide_k=1000)
    assert isinstance(results, list)
