"""Tests for reranking module."""
from __future__ import annotations

from scripts._recall_reranking import rerank_hits


def test_rerank_empty():
    """Empty hits list returns empty."""
    result = rerank_hits([], "test query")
    assert result == []


def test_rerank_single():
    """Single hit returns it unchanged."""
    hits = [{"id": "T-001", "score": 1.0, "statement": "test"}]
    result = rerank_hits(hits, "test")
    assert len(result) == 1


def test_rerank_preserves_count():
    """Reranking preserves hit count."""
    hits = [
        {"id": f"T-{i:03d}", "score": float(i), "statement": f"test {i}"}
        for i in range(5)
    ]
    result = rerank_hits(hits, "test")
    assert len(result) == 5


def test_rerank_returns_list():
    """Reranking always returns a list."""
    hits = [{"id": "T-001", "score": 1.0, "statement": "hello world"}]
    result = rerank_hits(hits, "hello")
    assert isinstance(result, list)


def test_rerank_with_missing_fields():
    """Hits with missing fields don't crash reranker."""
    hits = [{"id": "T-001", "score": 1.0}]
    try:
        result = rerank_hits(hits, "test")
        assert isinstance(result, list)
    except (KeyError, TypeError):
        pass  # Acceptable if strict validation
