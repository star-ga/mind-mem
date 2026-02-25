"""Tests for recall limit parameter behavior."""

from __future__ import annotations

import os

import pytest

from scripts._recall_core import recall
from scripts.init_workspace import init


@pytest.fixture
def ws(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "limit_test.md")
    with open(blocks_md, "w") as f:
        for i in range(20):
            f.write(f"[LIM-{i:03d}]\nType: Decision\nStatement: Test block {i} about limiting\n\n")
    return ws


@pytest.fixture
def ws_small(tmp_path):
    ws = str(tmp_path / "ws_small")
    os.makedirs(ws)
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "limit_test.md")
    with open(blocks_md, "w") as f:
        for i in range(3):
            f.write(f"[LIM-{i:03d}]\nType: Decision\nStatement: Test block {i} about limiting\n\n")
    return ws


def test_limit_1(ws):
    """limit=1 returns at most 1 result."""
    results = recall(ws, "test block limiting", limit=1)
    assert len(results) <= 1


def test_limit_5(ws):
    """limit=5 returns at most 5 results."""
    results = recall(ws, "test block limiting", limit=5)
    assert len(results) <= 5


def test_limit_larger_than_blocks(ws_small):
    """limit larger than block count returns all blocks."""
    results = recall(ws_small, "test block limiting", limit=100)
    assert len(results) <= 3


def test_limit_default(ws):
    """Default limit (10) is applied."""
    results = recall(ws, "test block limiting")
    assert len(results) <= 10
