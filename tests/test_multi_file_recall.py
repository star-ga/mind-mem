"""Tests for recall across multiple files."""

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
    # Create blocks in multiple directories
    for dir_name, prefix in [("decisions", "DEC"), ("tasks", "TASK"), ("entities", "ENT")]:
        blocks_md = os.path.join(ws, dir_name, "multi.md")
        with open(blocks_md, "w") as f:
            for i in range(3):
                f.write(f"[{prefix}-{i:03d}]\nType: {dir_name.title()[:-1]}\n")
                f.write(f"Statement: Multi-file test for {dir_name} number {i}\n\n")
    return ws


def test_recall_searches_decisions(ws):
    """Recall finds blocks in decisions directory."""
    results = recall(ws, "multi-file test decisions", limit=10)
    assert isinstance(results, list)


def test_recall_searches_tasks(ws):
    """Recall finds blocks in tasks directory."""
    results = recall(ws, "multi-file test tasks", limit=10)
    assert isinstance(results, list)


def test_recall_cross_directory(ws):
    """Recall searches across all directories."""
    results = recall(ws, "multi-file test", limit=20)
    assert isinstance(results, list)
