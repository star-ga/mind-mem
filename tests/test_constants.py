"""Tests for recall constants module."""
from __future__ import annotations

from scripts._recall_constants import (
    _STOPWORDS,
    _VALID_RECALL_KEYS,
    BM25_B,
    BM25_K1,
    FIELD_WEIGHTS,
    MAX_BLOCKS_PER_QUERY,
)


def test_bm25_k1_positive():
    """BM25 k1 parameter is positive."""
    assert BM25_K1 > 0


def test_bm25_b_range():
    """BM25 b parameter is between 0 and 1."""
    assert 0 <= BM25_B <= 1


def test_field_weights_not_empty():
    """Field weights dictionary is not empty."""
    assert len(FIELD_WEIGHTS) > 0


def test_field_weights_positive():
    """All field weights are positive."""
    for field, weight in FIELD_WEIGHTS.items():
        assert weight > 0, f"Field {field} has non-positive weight {weight}"


def test_max_blocks_positive():
    """Max blocks per query is positive."""
    assert MAX_BLOCKS_PER_QUERY > 0


def test_stopwords_not_empty():
    """Stopwords set is not empty."""
    assert len(_STOPWORDS) > 0


def test_stopwords_lowercase():
    """All stopwords are lowercase."""
    for word in _STOPWORDS:
        assert word == word.lower(), f"Stopword '{word}' is not lowercase"


def test_valid_recall_keys_not_empty():
    """Valid recall keys set is not empty."""
    assert len(_VALID_RECALL_KEYS) > 0


def test_common_stopwords_present():
    """Common English stopwords are in the set."""
    for word in ["the", "a", "is", "in", "of"]:
        assert word in _STOPWORDS, f"Common stopword '{word}' missing"
