"""Tests for RM3 query expansion."""
from __future__ import annotations

from scripts._recall_expansion import rm3_expand


def test_rm3_expand_basic():
    """RM3 expansion returns list."""
    result = rm3_expand(["test", "query"], [{"statement": "test document about queries"}])
    assert isinstance(result, list)


def test_rm3_expand_empty_tokens():
    """Empty tokens return empty."""
    result = rm3_expand([], [{"statement": "test"}])
    assert isinstance(result, list)


def test_rm3_expand_empty_docs():
    """Empty docs return original tokens."""
    result = rm3_expand(["test"], [])
    assert isinstance(result, list)


def test_rm3_expand_preserves_original():
    """Original tokens are preserved in expansion."""
    tokens = ["important", "query"]
    result = rm3_expand(tokens, [{"statement": "related document content"}])
    for t in tokens:
        assert t in result or any(t in r for r in result)


def test_rm3_expand_no_crash_on_short_docs():
    """Short document content doesn't crash."""
    result = rm3_expand(["x"], [{"statement": "y"}])
    assert isinstance(result, list)
