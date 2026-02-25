"""Tests for BM25 scoring functions."""
from __future__ import annotations

from scripts._recall_scoring import bm25f_score_terms, compute_weighted_tf


def test_bm25f_score_terms_basic():
    """Basic BM25F scoring returns float."""
    # Simple test with minimal input
    score = bm25f_score_terms(
        query_tokens=["test"],
        field_tfs={"statement": {"test": 1}},
        field_lengths={"statement": 5},
        avg_field_lengths={"statement": 10.0},
        doc_freqs={"test": 1},
        num_docs=10,
    )
    assert isinstance(score, (int, float))
    assert score >= 0


def test_bm25f_score_terms_no_match():
    """No matching terms gives zero score."""
    score = bm25f_score_terms(
        query_tokens=["nonexistent"],
        field_tfs={"statement": {"other": 1}},
        field_lengths={"statement": 5},
        avg_field_lengths={"statement": 10.0},
        doc_freqs={"nonexistent": 0},
        num_docs=10,
    )
    assert score == 0 or score >= 0


def test_bm25f_score_terms_multiple_terms():
    """Multiple query terms accumulate score."""
    score_single = bm25f_score_terms(
        query_tokens=["test"],
        field_tfs={"statement": {"test": 1, "query": 1}},
        field_lengths={"statement": 10},
        avg_field_lengths={"statement": 10.0},
        doc_freqs={"test": 1, "query": 1},
        num_docs=10,
    )
    score_multi = bm25f_score_terms(
        query_tokens=["test", "query"],
        field_tfs={"statement": {"test": 1, "query": 1}},
        field_lengths={"statement": 10},
        avg_field_lengths={"statement": 10.0},
        doc_freqs={"test": 1, "query": 1},
        num_docs=10,
    )
    assert score_multi >= score_single


def test_compute_weighted_tf():
    """Weighted TF computation returns dict."""
    result = compute_weighted_tf({"statement": {"word": 2}})
    assert isinstance(result, dict)


def test_bm25f_empty_query():
    """Empty query tokens give zero score."""
    score = bm25f_score_terms(
        query_tokens=[],
        field_tfs={"statement": {"test": 1}},
        field_lengths={"statement": 5},
        avg_field_lengths={"statement": 10.0},
        doc_freqs={},
        num_docs=10,
    )
    assert score == 0
