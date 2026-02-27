"""Tests for stopword handling."""

from __future__ import annotations

from mind_mem._recall_constants import _STOPWORDS
from mind_mem._recall_tokenization import tokenize


def test_stopwords_filtered():
    """Stopwords are removed from tokenization."""
    tokens = tokenize("the quick brown fox")
    assert "the" not in tokens


def test_stopwords_case_insensitive():
    """Stopwords filtering is case-insensitive."""
    tokens = tokenize("THE QUICK BROWN FOX")
    assert "the" not in tokens


def test_content_words_preserved():
    """Content words are preserved after stopword removal."""
    tokens = tokenize("important decision about deployment")
    assert len(tokens) >= 2


def test_all_stopwords_lowercase():
    """Stopword set contains only lowercase entries."""
    for sw in _STOPWORDS:
        assert sw == sw.lower()


def test_stopword_set_size():
    """Stopword set has reasonable size."""
    assert len(_STOPWORDS) >= 20
    assert len(_STOPWORDS) <= 1000
