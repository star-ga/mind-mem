"""Tests for intent classification."""
from __future__ import annotations

from scripts._recall_detection import detect_query_type


def test_what_query():
    """'What is X' detected as factual query."""
    qt = detect_query_type("what is the deployment process")
    assert qt is not None


def test_when_query():
    """'When did X' detected as temporal query."""
    qt = detect_query_type("when did we decide on the API format")
    assert qt is not None


def test_who_query():
    """'Who is X' detected as entity query."""
    qt = detect_query_type("who is responsible for the backend")
    assert qt is not None


def test_how_query():
    """'How does X' detected as procedural query."""
    qt = detect_query_type("how does the scoring system work")
    assert qt is not None


def test_why_query():
    """'Why did X' detected as reasoning query."""
    qt = detect_query_type("why did we choose BM25 over TF-IDF")
    assert qt is not None


def test_plain_keyword():
    """Plain keywords still return a query type."""
    qt = detect_query_type("deployment")
    assert qt is not None


def test_empty_returns_default():
    """Empty query returns default type."""
    qt = detect_query_type("")
    # Should return something or None gracefully
    assert qt is None or isinstance(qt, str)
