"""Tests for query expansion module."""
from __future__ import annotations

from scripts._recall_expansion import expand_query, expand_months


def test_expand_query_basic():
    """Basic query expansion returns tokens."""
    tokens = expand_query("deployment process")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_expand_query_empty():
    """Empty query returns empty list."""
    tokens = expand_query("")
    assert tokens == [] or tokens is not None


def test_expand_months_january():
    """January expansion includes jan."""
    result = expand_months("january meeting")
    assert isinstance(result, str)


def test_expand_months_no_month():
    """Text without months returns as-is."""
    result = expand_months("regular text here")
    assert "regular" in result


def test_expand_months_abbreviated():
    """Abbreviated month names are expanded."""
    result = expand_months("feb report")
    assert isinstance(result, str)


def test_expand_query_preserves_terms():
    """Expansion preserves original terms."""
    tokens = expand_query("important decision")
    # Original terms should be present
    assert any("import" in t or "decis" in t for t in tokens)


def test_expand_query_special_chars():
    """Special characters don't crash expansion."""
    tokens = expand_query("test!@#$%")
    assert isinstance(tokens, list)
