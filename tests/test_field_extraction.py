"""Tests for field token extraction."""
from __future__ import annotations
from scripts._recall_detection import extract_field_tokens

def test_extract_from_block():
    block = {"Statement": "Use BM25 scoring", "Title": "Decision"}
    tokens = extract_field_tokens(block)
    assert isinstance(tokens, dict)

def test_extract_from_empty_block():
    block = {}
    tokens = extract_field_tokens(block)
    assert isinstance(tokens, dict)

def test_extract_from_statement_only():
    block = {"Statement": "Just a statement"}
    tokens = extract_field_tokens(block)
    assert isinstance(tokens, dict)

def test_extract_from_full_block():
    block = {"Statement": "Full block", "Title": "Decision", "Tags": "a, b"}
    tokens = extract_field_tokens(block)
    assert isinstance(tokens, dict)
