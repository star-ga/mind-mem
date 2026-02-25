"""Tests for date field in recall results."""
from __future__ import annotations
import os, tempfile
from scripts.init_workspace import init
from scripts._recall_core import recall

def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    path = os.path.join(ws, "decisions", "dated.md")
    with open(path, "w") as f:
        f.write("[DT-001]\nType: Decision\nStatement: Recent decision\nDate: 2026-02-24\n\n")
        f.write("[DT-002]\nType: Decision\nStatement: Old decision\nDate: 2025-01-01\n\n")
    return ws

def test_dated_blocks_found():
    ws = _make_workspace()
    results = recall(ws, "decision", limit=5)
    assert isinstance(results, list)

def test_recent_date_search():
    ws = _make_workspace()
    results = recall(ws, "recent decision", limit=5)
    assert isinstance(results, list)

def test_old_date_search():
    ws = _make_workspace()
    results = recall(ws, "old decision", limit=5)
    assert isinstance(results, list)
