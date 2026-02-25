"""Tests for bigram extraction."""
from __future__ import annotations

from scripts._recall_detection import get_bigrams


def test_bigrams_basic():
    """Basic bigram extraction."""
    bigrams = get_bigrams(["hello", "world", "test"])
    assert isinstance(bigrams, list)
    assert len(bigrams) == 2


def test_bigrams_single_token():
    """Single token produces no bigrams."""
    bigrams = get_bigrams(["hello"])
    assert bigrams == []


def test_bigrams_empty():
    """Empty list produces no bigrams."""
    bigrams = get_bigrams([])
    assert bigrams == []


def test_bigrams_two_tokens():
    """Two tokens produce one bigram."""
    bigrams = get_bigrams(["hello", "world"])
    assert len(bigrams) == 1


def test_bigrams_content():
    """Bigrams contain consecutive token pairs."""
    bigrams = get_bigrams(["a", "b", "c"])
    assert ("a", "b") in bigrams or "a_b" in bigrams or ["a", "b"] in bigrams
