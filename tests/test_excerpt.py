"""Tests for excerpt generation."""

from __future__ import annotations

from mind_mem._recall_detection import get_excerpt


def test_excerpt_basic():
    block = {"Statement": "This is a test statement about decisions"}
    excerpt = get_excerpt(block)
    assert isinstance(excerpt, str)
    assert len(excerpt) > 0


def test_excerpt_long_statement():
    block = {"Statement": "word " * 1000}
    excerpt = get_excerpt(block, max_len=100)
    assert len(excerpt) <= 103  # allow small overflow


def test_excerpt_empty():
    block = {"Statement": ""}
    excerpt = get_excerpt(block)
    assert isinstance(excerpt, str)


def test_excerpt_missing_statement():
    block = {"Type": "Decision"}
    excerpt = get_excerpt(block)
    assert isinstance(excerpt, str)


def test_excerpt_preserves_beginning():
    block = {"Statement": "Important first words then more content follows"}
    excerpt = get_excerpt(block)
    assert excerpt.startswith("Important") or "Important" in excerpt
