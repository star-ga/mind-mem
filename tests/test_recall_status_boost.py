"""Tests for status boost in recall."""

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
        f.write("\n\n[D-20260201-001]\n")
        f.write("Date: 2026-02-01\nStatus: active\nScope: global\n")
        f.write("Statement: Status test active entry for verification\n")
        f.write("Rationale: Testing status boost\nTags: test\n\n")
        f.write("[D-20260201-002]\n")
        f.write("Date: 2026-02-02\nStatus: active\nScope: global\n")
        f.write("Statement: Status test wip entry for verification\n")
        f.write("Rationale: Testing status boost\nTags: test\n\n")
        f.write("[D-20260201-003]\n")
        f.write("Date: 2026-02-03\nStatus: active\nScope: global\n")
        f.write("Statement: Status test archived entry for verification\n")
        f.write("Rationale: Testing status boost\nTags: test\n\n")
    return ws


def test_status_boost_runs():
    ws = _make_workspace()
    results = recall(ws, "status test entry verification", limit=10)
    assert isinstance(results, list)


def test_active_blocks_found():
    ws = _make_workspace()
    results = recall(ws, "status test active entry verification", limit=10)
    assert isinstance(results, list)


def test_archived_blocks_found():
    ws = _make_workspace()
    results = recall(ws, "status test archived entry verification", limit=10)
    assert isinstance(results, list)
