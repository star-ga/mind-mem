"""Tests for excerpt generation."""
from __future__ import annotations

from scripts._recall_detection import get_excerpt


def test_excerpt_basic():
    """Basic excerpt generation."""
    block = {"statement": "This is a test statement about decisions"}
    excerpt = get_excerpt(block)
    assert isinstance(excerpt, str)
    assert len(excerpt) > 0


def test_excerpt_long_statement():
    """Long statements are truncated."""
    block = {"statement": "word " * 1000}
    excerpt = get_excerpt(block)
    assert len(excerpt) < len(block["statement"])


def test_excerpt_empty():
    """Empty statement produces empty excerpt."""
    block = {"statement": ""}
    excerpt = get_excerpt(block)
    assert isinstance(excerpt, str)


def test_excerpt_missing_statement():
    """Missing statement field handled."""
    block = {"type": "Decision"}
    excerpt = get_excerpt(block)
    assert isinstance(excerpt, str)


def test_excerpt_preserves_beginning():
    """Excerpt preserves beginning of statement."""
    block = {"statement": "Important first words then more content follows"}
    excerpt = get_excerpt(block)
    assert "Important" in excerpt
