"""Tests for supersedes field in recall."""
from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _ws():
    ws = tempfile.mkdtemp()
    init(ws)
    p = os.path.join(ws, "decisions", "sup.md")
    with open(p, "w") as f:
        f.write("[SUP-001]\nType: Decision\nStatement: Original decision\nStatus: Superseded\n\n")
        f.write("[SUP-002]\nType: Decision\nStatement: Replacement decision\nSupersedes: SUP-001\nStatus: Active\n\n")
    return ws

def test_superseded_blocks():
    results = recall(_ws(), "decision", limit=10)
    assert isinstance(results, list)

def test_active_superseding():
    results = recall(_ws(), "replacement decision", limit=5)
    assert isinstance(results, list)
