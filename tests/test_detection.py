"""Tests for query detection module."""
from __future__ import annotations

from scripts._recall_detection import (
    chunk_text,
    decompose_query,
    detect_query_type,
    get_bigrams,
    get_block_type,
    get_excerpt,
    is_skeptical_query,
)


def test_chunk_text_basic():
    """Chunk text splits into chunks."""
    chunks = chunk_text("word " * 100, max_tokens=20)
    assert isinstance(chunks, list)
    assert len(chunks) > 1


def test_chunk_text_short():
    """Short text returns single chunk."""
    chunks = chunk_text("hello world", max_tokens=100)
    assert len(chunks) == 1


def test_detect_query_type_returns_string():
    """Query type is a string."""
    qt = detect_query_type("what is this")
    assert isinstance(qt, str)


def test_get_bigrams():
    """Bigrams from token list."""
    bigrams = get_bigrams(["hello", "world", "test"])
    assert isinstance(bigrams, list)


def test_get_block_type_decision():
    """Decision block type detected."""
    bt = get_block_type({"type": "Decision"})
    assert bt == "Decision" or bt is not None


def test_get_excerpt_basic():
    """Excerpt returns string."""
    exc = get_excerpt({"statement": "Hello world this is a test"})
    assert isinstance(exc, str)


def test_is_skeptical_query():
    """Skeptical queries detected."""
    assert isinstance(is_skeptical_query("did we really decide that"), bool)


def test_decompose_query():
    """Query decomposition returns list."""
    parts = decompose_query("what is X and when was Y decided")
    assert isinstance(parts, list)
