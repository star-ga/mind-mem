"""Tests for tokenization module."""
from __future__ import annotations

from scripts._recall_tokenization import tokenize


def test_basic_tokenize():
    """Basic word tokenization."""
    tokens = tokenize("hello world")
    assert len(tokens) >= 2


def test_empty_string():
    """Empty string returns no tokens."""
    assert tokenize("") == []


def test_stopword_removal():
    """Common stopwords are removed."""
    tokens = tokenize("the a an is was were")
    # Most or all should be filtered as stopwords
    assert len(tokens) <= 2


def test_case_insensitive():
    """Tokenization is case-insensitive."""
    lower = tokenize("hello world")
    upper = tokenize("HELLO WORLD")
    assert lower == upper


def test_punctuation_handling():
    """Punctuation is stripped from tokens."""
    tokens = tokenize("hello, world! how are you?")
    for t in tokens:
        assert "," not in t
        assert "!" not in t
        assert "?" not in t


def test_numbers():
    """Numbers are tokenized."""
    tokens = tokenize("version 123 released")
    assert any("123" in t for t in tokens)


def test_underscore_words():
    """Underscore-connected words are kept."""
    tokens = tokenize("my_variable_name")
    assert len(tokens) >= 1


def test_unicode_text():
    """Unicode doesn't crash tokenizer."""
    tokens = tokenize("café résumé naïve")
    assert isinstance(tokens, list)


def test_very_long_text():
    """Very long text tokenizes without error."""
    text = "word " * 10000
    tokens = tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_mixed_content():
    """Mixed alphanumeric content."""
    tokens = tokenize("abc123 def456 ghi789")
    assert len(tokens) >= 3
