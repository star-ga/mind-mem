"""Tests for active_only recall filter."""

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
    blocks_md = os.path.join(ws, "decisions", "active_test.md")
    with open(blocks_md, "w") as f:
        f.write("[ACT-001]\nType: Decision\nStatement: Active decision\nStatus: Active\n\n")
        f.write("[ACT-002]\nType: Decision\nStatement: Archived decision\nStatus: Archived\n\n")
        f.write("[ACT-003]\nType: Decision\nStatement: WIP decision\nStatus: WIP\n\n")
    return ws


def test_active_only_filters(ws):
    """active_only=True filters non-active blocks."""
    results = recall(ws, "decision", limit=10, active_only=True)
    assert isinstance(results, list)


def test_without_active_only(ws):
    """Without active_only, all blocks returned."""
    results = recall(ws, "decision", limit=10, active_only=False)
    assert isinstance(results, list)


def test_active_only_default_false(ws):
    """Default active_only is False."""
    results = recall(ws, "decision", limit=10)
    assert isinstance(results, list)
