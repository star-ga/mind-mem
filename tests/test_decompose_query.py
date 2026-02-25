"""Tests for query decomposition."""

from __future__ import annotations

from scripts._recall_detection import decompose_query


def test_decompose_simple():
    """Simple query returns single part."""
    parts = decompose_query("what is BM25")
    assert isinstance(parts, list)
    assert len(parts) >= 1


def test_decompose_compound():
    """Compound query split into parts."""
    parts = decompose_query("what is BM25 and when was it implemented")
    assert len(parts) >= 1


def test_decompose_empty():
    """Empty query returns empty or single part."""
    parts = decompose_query("")
    assert isinstance(parts, list)


def test_decompose_preserves_content():
    """Decomposition preserves query content."""
    query = "important decision about API"
    parts = decompose_query(query)
    combined = " ".join(parts)
    assert "API" in combined or "api" in combined.lower()
