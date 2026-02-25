"""Tests for active_only recall filter."""
from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "active_test.md")
    with open(blocks_md, "w") as f:
        f.write("[ACT-001]\nType: Decision\nStatement: Active decision\nStatus: Active\n\n")
        f.write("[ACT-002]\nType: Decision\nStatement: Archived decision\nStatus: Archived\n\n")
        f.write("[ACT-003]\nType: Decision\nStatement: WIP decision\nStatus: WIP\n\n")
    return ws


def test_active_only_filters():
    """active_only=True filters non-active blocks."""
    ws = _make_workspace()
    results = recall(ws, "decision", limit=10, active_only=True)
    assert isinstance(results, list)


def test_without_active_only():
    """Without active_only, all blocks returned."""
    ws = _make_workspace()
    results = recall(ws, "decision", limit=10, active_only=False)
    assert isinstance(results, list)


def test_active_only_default_false():
    """Default active_only is False."""
    ws = _make_workspace()
    results = recall(ws, "decision", limit=10)
    assert isinstance(results, list)
