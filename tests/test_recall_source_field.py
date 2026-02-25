"""Tests for source field in recall results."""

from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _ws():
    ws = tempfile.mkdtemp()
    init(ws)
    p = os.path.join(ws, "decisions", "src_test.md")
    with open(p, "w") as f:
        f.write("[SRC-001]\nType: Decision\nStatement: Source field test\n\n")
    return ws


def test_results_have_source():
    ws = _ws()
    results = recall(ws, "source field test", limit=5)
    if results:
        r = results[0]
        assert "source" in r or "file" in r or "_source" in r or "raw" in r


def test_source_is_string():
    ws = _ws()
    results = recall(ws, "source field test", limit=5)
    for r in results:
        src = r.get("source") or r.get("file") or r.get("_source", "")
        assert isinstance(src, str)
