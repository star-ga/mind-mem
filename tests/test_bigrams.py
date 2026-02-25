"""Tests for bigram extraction."""

from __future__ import annotations

from scripts._recall_detection import get_bigrams


def test_bigrams_basic():
    bigrams = get_bigrams(["hello", "world", "test"])
    assert isinstance(bigrams, set)
    assert len(bigrams) == 2


def test_bigrams_single_token():
    bigrams = get_bigrams(["hello"])
    assert len(bigrams) == 0


def test_bigrams_empty():
    bigrams = get_bigrams([])
    assert len(bigrams) == 0


def test_bigrams_two_tokens():
    bigrams = get_bigrams(["hello", "world"])
    assert len(bigrams) == 1


def test_bigrams_content():
    bigrams = get_bigrams(["a", "b", "c"])
    assert ("a", "b") in bigrams
