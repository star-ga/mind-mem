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
    text = ". ".join(f"Sentence {i}" for i in range(30)) + "."
    chunks = chunk_text(text, chunk_size=3)
    assert isinstance(chunks, list)
    assert len(chunks) > 1


def test_chunk_text_short():
    chunks = chunk_text("hello world", chunk_size=100)
    assert len(chunks) >= 1


def test_detect_query_type_returns_string():
    qt = detect_query_type("what is this")
    assert isinstance(qt, str)


def test_get_bigrams():
    bigrams = get_bigrams(["hello", "world", "test"])
    assert isinstance(bigrams, set)


def test_get_block_type_decision():
    bt = get_block_type("DEC-001")
    assert isinstance(bt, str)


def test_get_excerpt_basic():
    exc = get_excerpt({"Statement": "Hello world this is a test"})
    assert isinstance(exc, str)


def test_is_skeptical_query():
    result = is_skeptical_query("did we really decide that")
    assert isinstance(result, bool)


def test_decompose_query():
    parts = decompose_query("what is X and when was Y decided")
    assert isinstance(parts, list)
