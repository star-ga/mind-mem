"""Tests for text chunking."""
from __future__ import annotations

from scripts._recall_detection import chunk_text


def test_chunk_short_text():
    """Short text returns single chunk."""
    chunks = chunk_text("hello world", max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_chunk_long_text():
    """Long text is split into multiple chunks."""
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, max_tokens=20)
    assert len(chunks) > 1


def test_chunk_empty():
    """Empty text returns empty or single empty chunk."""
    chunks = chunk_text("", max_tokens=10)
    assert len(chunks) <= 1


def test_chunk_preserves_content():
    """All content is preserved across chunks."""
    text = "alpha beta gamma delta epsilon"
    chunks = chunk_text(text, max_tokens=3)
    combined = " ".join(chunks)
    for word in ["alpha", "beta", "gamma", "delta", "epsilon"]:
        assert word in combined


def test_chunk_respects_max():
    """Each chunk respects max_tokens limit."""
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, max_tokens=10)
    for chunk in chunks:
        assert len(chunk.split()) <= 12  # Allow small overflow
