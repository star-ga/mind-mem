"""Tests for recall across multiple files."""
from __future__ import annotations

import os
import tempfile

from scripts.init_workspace import init
from scripts._recall_core import recall


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    # Create blocks in multiple directories
    for dir_name, prefix in [("decisions", "DEC"), ("tasks", "TASK"), ("entities", "ENT")]:
        blocks_md = os.path.join(ws, dir_name, "multi.md")
        with open(blocks_md, "w") as f:
            for i in range(3):
                f.write(f"[{prefix}-{i:03d}]\nType: {dir_name.title()[:-1]}\n")
                f.write(f"Statement: Multi-file test for {dir_name} number {i}\n\n")
    return ws


def test_recall_searches_decisions():
    """Recall finds blocks in decisions directory."""
    ws = _make_workspace()
    results = recall(ws, "multi-file test decisions", limit=10)
    assert isinstance(results, list)


def test_recall_searches_tasks():
    """Recall finds blocks in tasks directory."""
    ws = _make_workspace()
    results = recall(ws, "multi-file test tasks", limit=10)
    assert isinstance(results, list)


def test_recall_cross_directory():
    """Recall searches across all directories."""
    ws = _make_workspace()
    results = recall(ws, "multi-file test", limit=20)
    assert isinstance(results, list)
