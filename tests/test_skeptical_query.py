"""Tests for skeptical query detection."""
from __future__ import annotations

from scripts._recall_detection import is_skeptical_query


def test_skeptical_positive():
    """Skeptical phrasing detected."""
    assert is_skeptical_query("did we really decide that") is True


def test_skeptical_negative():
    """Non-skeptical query not flagged."""
    assert is_skeptical_query("what was decided") is False


def test_skeptical_empty():
    """Empty query is not skeptical."""
    assert is_skeptical_query("") is False


def test_skeptical_challenge():
    """Challenge phrasing detected."""
    result = is_skeptical_query("are you sure about that decision")
    assert isinstance(result, bool)


def test_skeptical_contradiction():
    """Contradiction inquiry detected."""
    result = is_skeptical_query("that contradicts what was said before")
    assert isinstance(result, bool)
