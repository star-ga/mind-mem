"""Tests for reference-based recall."""

from __future__ import annotations

import os
import tempfile

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


def _make_workspace():
    td = tempfile.TemporaryDirectory()
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    path = os.path.join(ws, "decisions", "refs.md")
    with open(path, "w") as f:
        f.write("[REF-001]\nType: Decision\nStatement: Primary decision\nReferences: REF-002\n\n")
        f.write("[REF-002]\nType: Decision\nStatement: Referenced decision\nReferences: REF-001\n\n")
    return ws, td


def test_reference_search():
    ws, td = _make_workspace()
    try:
        results = recall(ws, "primary decision", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_referenced_block():
    ws, td = _make_workspace()
    try:
        results = recall(ws, "referenced decision", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()
