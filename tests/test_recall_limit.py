"""Tests for recall limit parameter behavior."""
from __future__ import annotations

import os
import tempfile

from scripts.init_workspace import init
from scripts._recall_core import recall


def _make_workspace(n_blocks=20):
    ws = tempfile.mkdtemp()
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "limit_test.md")
    with open(blocks_md, "w") as f:
        for i in range(n_blocks):
            f.write(f"[LIM-{i:03d}]\nType: Decision\nStatement: Test block {i} about limiting\n\n")
    return ws


def test_limit_1():
    """limit=1 returns at most 1 result."""
    ws = _make_workspace()
    results = recall(ws, "test block limiting", limit=1)
    assert len(results) <= 1


def test_limit_5():
    """limit=5 returns at most 5 results."""
    ws = _make_workspace()
    results = recall(ws, "test block limiting", limit=5)
    assert len(results) <= 5


def test_limit_larger_than_blocks():
    """limit larger than block count returns all blocks."""
    ws = _make_workspace(n_blocks=3)
    results = recall(ws, "test block limiting", limit=100)
    assert len(results) <= 3


def test_limit_default():
    """Default limit (10) is applied."""
    ws = _make_workspace()
    results = recall(ws, "test block limiting")
    assert len(results) <= 10
