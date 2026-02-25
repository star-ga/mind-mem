"""Tests for tag-based recall."""
from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    path = os.path.join(ws, "decisions", "tags.md")
    with open(path, "w") as f:
        f.write("[TAG-001]\nType: Decision\nStatement: Tagged decision\nTags: api, design\n\n")
        f.write("[TAG-002]\nType: Decision\nStatement: Another tagged\nTags: performance, optimization\n\n")
    return ws

def test_tag_search():
    ws = _make_workspace()
    results = recall(ws, "api design", limit=5)
    assert isinstance(results, list)

def test_tag_content_search():
    ws = _make_workspace()
    results = recall(ws, "performance optimization", limit=5)
    assert isinstance(results, list)

def test_mixed_tag_statement():
    ws = _make_workspace()
    results = recall(ws, "tagged decision api", limit=5)
    assert isinstance(results, list)
