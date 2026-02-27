"""Tests for tag-based recall."""

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
    path = os.path.join(ws, "decisions", "tags.md")
    with open(path, "w") as f:
        f.write("[TAG-001]\nType: Decision\nStatement: Tagged decision\nTags: api, design\n\n")
        f.write("[TAG-002]\nType: Decision\nStatement: Another tagged\nTags: performance, optimization\n\n")
    return ws, td


def test_tag_search():
    ws, td = _make_workspace()
    try:
        results = recall(ws, "api design", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_tag_content_search():
    ws, td = _make_workspace()
    try:
        results = recall(ws, "performance optimization", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_mixed_tag_statement():
    ws, td = _make_workspace()
    try:
        results = recall(ws, "tagged decision api", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()
