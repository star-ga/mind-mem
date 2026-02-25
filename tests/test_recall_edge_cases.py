"""Edge case tests for recall engine."""
from __future__ import annotations

import os
import tempfile

import pytest

from scripts._recall_core import recall
from scripts.init_workspace import init


@pytest.fixture
def workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    # Create a test block file
    blocks_md = os.path.join(ws, "decisions", "edge.md")
    with open(blocks_md, "w") as f:
        f.write("[EDGE-001]\nType: Decision\nStatement: Empty query test\n\n")
        f.write("[EDGE-002]\nType: Decision\nStatement: Special chars !@#$%^&*()\n\n")
        f.write("[EDGE-003]\nType: Decision\nStatement: Very long " + "word " * 200 + "\n\n")
    return ws


def test_empty_query(workspace):
    """Empty query returns empty results."""
    results = recall(workspace, "", limit=5)
    assert results == []


def test_whitespace_query(workspace):
    """Whitespace-only query returns empty results."""
    results = recall(workspace, "   ", limit=5)
    assert results == []


def test_special_chars_query(workspace):
    """Special characters in query don't crash."""
    results = recall(workspace, "!@#$%^&*()", limit=5)
    assert isinstance(results, list)


def test_very_long_query(workspace):
    """Very long query doesn't crash."""
    long_q = "test " * 500
    results = recall(workspace, long_q, limit=5)
    assert isinstance(results, list)


def test_limit_zero(workspace):
    """Limit of zero returns empty list."""
    results = recall(workspace, "decision", limit=0)
    assert results == []


def test_limit_negative(workspace):
    """Negative limit returns empty list."""
    results = recall(workspace, "decision", limit=-1)
    assert results == []


def test_nonexistent_workspace():
    """Non-existent workspace doesn't crash."""
    results = recall("/nonexistent/path/xyz", "test query", limit=5)
    assert isinstance(results, list)


def test_unicode_query(workspace):
    """Unicode query doesn't crash."""
    results = recall(workspace, "日本語テスト", limit=5)
    assert isinstance(results, list)


def test_numeric_query(workspace):
    """Pure numeric query works."""
    results = recall(workspace, "12345", limit=5)
    assert isinstance(results, list)


def test_single_char_query(workspace):
    """Single character query works."""
    results = recall(workspace, "a", limit=5)
    assert isinstance(results, list)
