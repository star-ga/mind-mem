"""Tests for BM25 scoring functions."""
from __future__ import annotations
from collections import Counter
from scripts._recall_scoring import bm25f_score_terms, compute_weighted_tf

def test_bm25f_score_terms_basic():
    score = bm25f_score_terms(
        query_terms=["test"],
        weighted_tf=Counter({"test": 1.0}),
        wdl=5.0,
        idf_cache={"test": 1.5},
        avg_wdl=10.0,
    )
    assert isinstance(score, (int, float))
    assert score >= 0

def test_bm25f_score_terms_no_match():
    score = bm25f_score_terms(
        query_terms=["nonexistent"],
        weighted_tf=Counter({"other": 1.0}),
        wdl=5.0,
        idf_cache={"nonexistent": 0.0},
        avg_wdl=10.0,
    )
    assert score >= 0

def test_bm25f_score_terms_multiple_terms():
    score_single = bm25f_score_terms(["test"], Counter({"test": 1.0, "query": 1.0}), 10.0, {"test": 1.5, "query": 1.5}, 10.0)
    score_multi = bm25f_score_terms(["test", "query"], Counter({"test": 1.0, "query": 1.0}), 10.0, {"test": 1.5, "query": 1.5}, 10.0)
    assert score_multi >= score_single

def test_compute_weighted_tf():
    result = compute_weighted_tf({"statement": ["word", "word", "test"]})
    assert isinstance(result, tuple)
    assert len(result) == 2

def test_bm25f_empty_query():
    score = bm25f_score_terms([], Counter({"test": 1.0}), 5.0, {}, 10.0)
    assert score == 0
