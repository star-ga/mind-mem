"""Tests for wide retrieval parameter."""

from __future__ import annotations

import os
import tempfile

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


def _make_workspace():
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "wide_test.md")
    with open(blocks_md, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(f"[WD-{i:03d}]\nType: Decision\nStatement: Wide retrieval test {i}\n\n")
    return ws, td


def test_default_wide_k():
    """Default retrieve_wide_k works."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "wide retrieval", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_small_wide_k():
    """Small retrieve_wide_k still returns results."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "wide retrieval", limit=5, retrieve_wide_k=10)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_large_wide_k():
    """Large retrieve_wide_k doesn't crash."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "wide retrieval", limit=5, retrieve_wide_k=1000)
        assert isinstance(results, list)
    finally:
        td.cleanup()
