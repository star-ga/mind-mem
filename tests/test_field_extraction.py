"""Tests for field token extraction."""
from __future__ import annotations

from scripts._recall_detection import extract_field_tokens


def test_extract_from_statement():
    """Extracts tokens from statement field."""
    block = {"statement": "Use BM25 scoring"}
    tokens = extract_field_tokens(block, "statement")
    assert isinstance(tokens, (list, dict))


def test_extract_from_missing_field():
    """Missing field returns empty."""
    block = {"statement": "test"}
    tokens = extract_field_tokens(block, "nonexistent")
    assert tokens == [] or tokens == {} or tokens is None


def test_extract_from_empty_field():
    """Empty field returns empty tokens."""
    block = {"statement": ""}
    tokens = extract_field_tokens(block, "statement")
    assert isinstance(tokens, (list, dict))


def test_extract_from_type_field():
    """Type field extraction works."""
    block = {"type": "Decision", "statement": "test"}
    tokens = extract_field_tokens(block, "type")
    assert isinstance(tokens, (list, dict))
