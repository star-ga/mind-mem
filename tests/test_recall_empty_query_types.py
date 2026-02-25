"""Tests for various empty/minimal query types."""

from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _ws():
    ws = tempfile.mkdtemp()
    init(ws)
    p = os.path.join(ws, "decisions", "eq.md")
    with open(p, "w") as f:
        f.write("[EQ-001]\nType: Decision\nStatement: Test block\n\n")
    return ws


def test_none_like_query():
    results = recall(_ws(), "   \t\n  ", limit=5)
    assert results == []


def test_single_stopword():
    results = recall(_ws(), "the", limit=5)
    assert isinstance(results, list)


def test_all_stopwords():
    results = recall(_ws(), "the a an is was", limit=5)
    assert isinstance(results, list)
    assert len(results) == 0


def test_punctuation_only():
    results = recall(_ws(), "... !!! ???", limit=5)
    assert isinstance(results, list)
