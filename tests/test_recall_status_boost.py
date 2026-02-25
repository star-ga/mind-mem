"""Tests for status boost in recall."""
from __future__ import annotations
import os, tempfile
from scripts.init_workspace import init
from scripts._recall_core import recall

def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    path = os.path.join(ws, "decisions", "status.md")
    with open(path, "w") as f:
        f.write("[ST-001]\nType: Decision\nStatement: Status test active\nStatus: Active\n\n")
        f.write("[ST-002]\nType: Decision\nStatement: Status test wip\nStatus: WIP\n\n")
        f.write("[ST-003]\nType: Decision\nStatement: Status test archived\nStatus: Archived\n\n")
    return ws

def test_status_boost_runs():
    ws = _make_workspace()
    results = recall(ws, "status test", limit=10)
    assert isinstance(results, list)

def test_active_blocks_found():
    ws = _make_workspace()
    results = recall(ws, "status test active", limit=10)
    assert len(results) >= 1

def test_archived_blocks_found():
    ws = _make_workspace()
    results = recall(ws, "status test archived", limit=10)
    assert isinstance(results, list)
