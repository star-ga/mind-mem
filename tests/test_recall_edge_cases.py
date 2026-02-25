"""Edge case tests for recall engine."""

from __future__ import annotations

import os

import pytest

from scripts._recall_core import recall
from scripts.init_workspace import init


@pytest.fixture
def ws(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init(ws)
    # Create a test block file
    blocks_md = os.path.join(ws, "decisions", "edge.md")
    with open(blocks_md, "w") as f:
        f.write("[EDGE-001]\nType: Decision\nStatement: Empty query test\n\n")
        f.write("[EDGE-002]\nType: Decision\nStatement: Special chars !@#$%^&*()\n\n")
        f.write("[EDGE-003]\nType: Decision\nStatement: Very long " + "word " * 200 + "\n\n")
    return ws


def test_empty_query(ws):
    """Empty query returns empty results."""
    results = recall(ws, "", limit=5)
    assert results == []


def test_whitespace_query(ws):
    """Whitespace-only query returns empty results."""
    results = recall(ws, "   ", limit=5)
    assert results == []


def test_special_chars_query(ws):
    """Special characters in query don't crash."""
    results = recall(ws, "!@#$%^&*()", limit=5)
    assert isinstance(results, list)


def test_very_long_query(ws):
    """Very long query doesn't crash."""
    long_q = "test " * 500
    results = recall(ws, long_q, limit=5)
    assert isinstance(results, list)


def test_limit_zero(ws):
    """Limit of zero returns empty list."""
    results = recall(ws, "decision", limit=0)
    assert results == []


def test_limit_negative(ws):
    """Negative limit returns empty list."""
    results = recall(ws, "decision", limit=-1)
    assert results == []


def test_nonexistent_workspace():
    """Non-existent workspace doesn't crash."""
    results = recall("/nonexistent/path/xyz", "test query", limit=5)
    assert isinstance(results, list)


def test_unicode_query(ws):
    """Unicode query doesn't crash."""
    results = recall(ws, "日本語テスト", limit=5)
    assert isinstance(results, list)


def test_numeric_query(ws):
    """Pure numeric query works."""
    results = recall(ws, "12345", limit=5)
    assert isinstance(results, list)


def test_single_char_query(ws):
    """Single character query works."""
    results = recall(ws, "a", limit=5)
    assert isinstance(results, list)
