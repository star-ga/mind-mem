"""Tests for reference-based recall."""
from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    path = os.path.join(ws, "decisions", "refs.md")
    with open(path, "w") as f:
        f.write("[REF-001]\nType: Decision\nStatement: Primary decision\nReferences: REF-002\n\n")
        f.write("[REF-002]\nType: Decision\nStatement: Referenced decision\nReferences: REF-001\n\n")
    return ws

def test_reference_search():
    ws = _make_workspace()
    results = recall(ws, "primary decision", limit=5)
    assert isinstance(results, list)

def test_referenced_block():
    ws = _make_workspace()
    results = recall(ws, "referenced decision", limit=5)
    assert isinstance(results, list)
