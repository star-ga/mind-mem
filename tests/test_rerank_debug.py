"""Tests for rerank debug mode."""
from __future__ import annotations

import os
import tempfile

from scripts.init_workspace import init
from scripts._recall_core import recall


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "rerank_test.md")
    with open(blocks_md, "w") as f:
        for i in range(5):
            f.write(f"[RR-{i:03d}]\nType: Decision\nStatement: Rerank test {i}\n\n")
    return ws


def test_rerank_enabled():
    """Reranking enabled doesn't crash."""
    ws = _make_workspace()
    results = recall(ws, "rerank test", limit=5, rerank=True)
    assert isinstance(results, list)


def test_rerank_disabled():
    """Reranking disabled returns results."""
    ws = _make_workspace()
    results = recall(ws, "rerank test", limit=5, rerank=False)
    assert isinstance(results, list)


def test_rerank_debug_mode():
    """Debug mode doesn't crash."""
    ws = _make_workspace()
    results = recall(ws, "rerank test", limit=5, rerank=True, rerank_debug=True)
    assert isinstance(results, list)
