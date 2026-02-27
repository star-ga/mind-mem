"""Tests for RM3 query expansion."""

from __future__ import annotations

from collections import Counter

from mind_mem._recall_expansion import rm3_expand


def test_rm3_expand_basic():
    result = rm3_expand(
        ["test", "query"],
        [(["test", "document", "about", "queries"], 1.0)],
        Counter({"test": 5, "query": 3, "document": 2, "about": 1, "queries": 1}),
        100,
    )
    assert isinstance(result, dict)


def test_rm3_expand_empty_tokens():
    result = rm3_expand([], [(["test"], 1.0)], Counter({"test": 1}), 100)
    assert isinstance(result, dict)


def test_rm3_expand_empty_docs():
    result = rm3_expand(["test"], [], Counter({"test": 1}), 100)
    assert isinstance(result, dict)


def test_rm3_expand_preserves_original():
    tokens = ["important", "query"]
    result = rm3_expand(
        tokens,
        [(["related", "content", "important"], 1.0)],
        Counter({"important": 2, "query": 3, "related": 1, "content": 1}),
        100,
    )
    assert isinstance(result, dict)
    # Original terms should still have weight
    for t in tokens:
        assert t in result


def test_rm3_expand_no_crash_on_short_docs():
    result = rm3_expand(["x"], [(["y"], 1.0)], Counter({"x": 1, "y": 1}), 10)
    assert isinstance(result, dict)
