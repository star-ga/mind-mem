"""Tests for reranking module."""
from __future__ import annotations
from scripts._recall_reranking import rerank_hits

def test_rerank_empty():
    result = rerank_hits("test query", [])
    assert result == []

def test_rerank_single():
    hits = [{"_id": "T-001", "score": 1.0, "statement": "test"}]
    result = rerank_hits("test", hits)
    assert len(result) == 1

def test_rerank_preserves_count():
    hits = [
        {"_id": f"T-{i:03d}", "score": float(i), "statement": f"test {i}"}
        for i in range(5)
    ]
    result = rerank_hits("test", hits)
    assert len(result) == 5

def test_rerank_returns_list():
    hits = [{"_id": "T-001", "score": 1.0, "statement": "hello world"}]
    result = rerank_hits("hello", hits)
    assert isinstance(result, list)

def test_rerank_with_missing_fields():
    hits = [{"_id": "T-001", "score": 1.0}]
    try:
        result = rerank_hits("test", hits)
        assert isinstance(result, list)
    except (KeyError, TypeError):
        pass
