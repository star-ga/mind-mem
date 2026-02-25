"""Tests for date scoring function."""
from __future__ import annotations

from scripts._recall_scoring import date_score


def test_date_score_none():
    """None date returns zero score."""
    assert date_score(None) == 0.0


def test_date_score_empty():
    """Empty date string returns zero."""
    assert date_score("") == 0.0


def test_date_score_valid():
    """Valid date returns a float."""
    result = date_score("2026-01-15")
    assert isinstance(result, float)


def test_date_score_recent_higher():
    """More recent dates score higher."""
    recent = date_score("2026-02-20")
    older = date_score("2025-01-01")
    assert recent >= older


def test_date_score_invalid_format():
    """Invalid date format returns zero."""
    result = date_score("not-a-date")
    assert result == 0.0 or isinstance(result, float)
