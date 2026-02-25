"""Tests for query expansion module."""
from __future__ import annotations

from scripts._recall_expansion import expand_months, expand_query


def test_expand_query_basic():
    tokens = expand_query("deployment process")
    assert isinstance(tokens, list)
    assert len(tokens) > 0

def test_expand_query_empty():
    tokens = expand_query("")
    assert isinstance(tokens, list)

def test_expand_months_january():
    result = expand_months("january meeting", ["january", "meeting"])
    assert isinstance(result, list)

def test_expand_months_no_month():
    result = expand_months("regular text here", ["regular", "text", "here"])
    assert isinstance(result, list)

def test_expand_months_abbreviated():
    result = expand_months("feb report", ["feb", "report"])
    assert isinstance(result, list)

def test_expand_query_preserves_terms():
    tokens = expand_query("important decision")
    assert isinstance(tokens, list)

def test_expand_query_special_chars():
    tokens = expand_query("test!@#$%")
    assert isinstance(tokens, list)
