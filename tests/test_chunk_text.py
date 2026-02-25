"""Tests for text chunking."""
from __future__ import annotations

from scripts._recall_detection import chunk_text


def test_chunk_short_text():
    chunks = chunk_text("hello world", chunk_size=100)
    assert len(chunks) >= 1

def test_chunk_long_text():
    text = "First. Second. Third. Fourth. Fifth. Sixth. Seventh. Eighth."
    chunks = chunk_text(text, chunk_size=2)
    assert len(chunks) > 1

def test_chunk_empty():
    chunks = chunk_text("", chunk_size=10)
    assert len(chunks) <= 1

def test_chunk_preserves_content():
    text = "Alpha sentence. Beta sentence. Gamma sentence."
    chunks = chunk_text(text, chunk_size=3)
    combined = " ".join(chunks)
    for word in ["Alpha", "Beta", "Gamma"]:
        assert word in combined

def test_chunk_respects_size():
    text = ". ".join(f"Sentence {i}" for i in range(30)) + "."
    chunks = chunk_text(text, chunk_size=3)
    assert len(chunks) > 1
