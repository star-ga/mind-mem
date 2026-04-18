"""Tests for hybrid search functionality."""

from __future__ import annotations

import os
import tempfile

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


def _make_workspace():
    """Create a test workspace with sample blocks."""
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "hybrid.md")
    with open(blocks_md, "w") as f:
        f.write("[HYB-001]\nType: Decision\nStatement: Use BM25 for text retrieval\n\n")
        f.write("[HYB-002]\nType: Decision\nStatement: Vector search for semantic matching\n\n")
        f.write("[HYB-003]\nType: Decision\nStatement: RRF fusion combines both scores\n\n")
    return ws, td


def test_recall_returns_list():
    """Recall always returns a list."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "text retrieval", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_recall_respects_limit():
    """Results count doesn't exceed limit."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "decision", limit=2)
        assert len(results) <= 2
    finally:
        td.cleanup()


def test_recall_results_have_id():
    """Each result has an id field."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "BM25 retrieval", limit=5)
        for r in results:
            assert "id" in r or "block_id" in r or "ID" in r.get("raw", {})
    finally:
        td.cleanup()


def test_recall_relevance():
    """More relevant results rank higher."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "BM25 text retrieval", limit=5)
        if len(results) >= 2:
            # First result should have higher or equal score
            scores = [r.get("score", 0) for r in results]
            assert scores == sorted(scores, reverse=True)
    finally:
        td.cleanup()


def test_recall_no_duplicates():
    """No duplicate block IDs in results."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "decision retrieval", limit=10)
        ids = [r.get("id") or r.get("block_id") for r in results]
        ids = [i for i in ids if i is not None]
        assert len(ids) == len(set(ids))
    finally:
        td.cleanup()
