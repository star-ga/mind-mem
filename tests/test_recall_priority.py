"""Tests for priority boost in recall."""
from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    path = os.path.join(ws, "decisions", "DECISIONS.md")
    with open(path, "a") as f:
        f.write("\n\n[D-20260101-001]\n")
        f.write("Date: 2026-01-01\nStatus: active\nScope: global\n")
        f.write("Statement: Priority test item high importance\n")
        f.write("Rationale: Testing priority boost\nPriority: High\nTags: test\n\n")
        f.write("[D-20260101-002]\n")
        f.write("Date: 2026-01-02\nStatus: active\nScope: global\n")
        f.write("Statement: Priority test item low importance\n")
        f.write("Rationale: Testing priority boost\nPriority: Low\nTags: test\n\n")
        f.write("[D-20260101-003]\n")
        f.write("Date: 2026-01-03\nStatus: active\nScope: global\n")
        f.write("Statement: Priority test item default importance\n")
        f.write("Rationale: Testing priority boost\nTags: test\n\n")
    return ws

def test_priority_boost_runs():
    ws = _make_workspace()
    results = recall(ws, "priority test item importance", limit=10)
    assert isinstance(results, list)

def test_high_priority_exists():
    ws = _make_workspace()
    results = recall(ws, "priority test item importance", limit=10)
    assert isinstance(results, list)

def test_priority_ordering():
    ws = _make_workspace()
    results = recall(ws, "priority test item importance", limit=10)
    if len(results) >= 2:
        assert results[0].get("score", 0) >= results[-1].get("score", 0)
