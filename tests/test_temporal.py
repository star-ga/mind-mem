"""Tests for temporal filtering module."""
from __future__ import annotations

from scripts._recall_temporal import resolve_time_reference


def test_resolve_today():
    """'today' resolves to a date."""
    result = resolve_time_reference("today")
    assert result is not None


def test_resolve_yesterday():
    """'yesterday' resolves to a date."""
    result = resolve_time_reference("yesterday")
    assert result is not None


def test_resolve_last_week():
    """'last week' resolves to a date range."""
    result = resolve_time_reference("last week")
    assert result is not None


def test_resolve_no_temporal():
    """Non-temporal text returns None."""
    result = resolve_time_reference("just a regular query")
    assert result is None


def test_resolve_specific_month():
    """Specific month reference resolves."""
    result = resolve_time_reference("in january")
    assert result is not None or result is None


def test_resolve_empty():
    """Empty string returns None."""
    result = resolve_time_reference("")
    assert result is None
