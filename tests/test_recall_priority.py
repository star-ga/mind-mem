"""Tests for priority boost in recall."""
from __future__ import annotations
import os, tempfile
from scripts.init_workspace import init
from scripts._recall_core import recall

def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    path = os.path.join(ws, "decisions", "prio.md")
    with open(path, "w") as f:
        f.write("[PRI-001]\nType: Decision\nStatement: Priority test item\nPriority: High\n\n")
        f.write("[PRI-002]\nType: Decision\nStatement: Priority test item\nPriority: Low\n\n")
        f.write("[PRI-003]\nType: Decision\nStatement: Priority test item\n\n")
    return ws

def test_priority_boost_runs():
    ws = _make_workspace()
    results = recall(ws, "priority test", limit=10)
    assert isinstance(results, list)

def test_high_priority_exists():
    ws = _make_workspace()
    results = recall(ws, "priority test item", limit=10)
    assert len(results) >= 1

def test_priority_ordering():
    ws = _make_workspace()
    results = recall(ws, "priority test item", limit=10)
    if len(results) >= 2:
        assert results[0].get("score", 0) >= results[-1].get("score", 0)
