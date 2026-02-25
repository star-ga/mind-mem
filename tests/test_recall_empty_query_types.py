"""Tests for various empty/minimal query types."""

from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _ws():
    td = tempfile.TemporaryDirectory()
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    p = os.path.join(ws, "decisions", "eq.md")
    with open(p, "w") as f:
        f.write("[EQ-001]\nType: Decision\nStatement: Test block\n\n")
    return ws, td


def test_none_like_query():
    ws, td = _ws()
    try:
        results = recall(ws, "   \t\n  ", limit=5)
        assert results == []
    finally:
        td.cleanup()


def test_single_stopword():
    ws, td = _ws()
    try:
        results = recall(ws, "the", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_all_stopwords():
    ws, td = _ws()
    try:
        results = recall(ws, "the a an is was", limit=5)
        assert isinstance(results, list)
        assert len(results) == 0
    finally:
        td.cleanup()


def test_punctuation_only():
    ws, td = _ws()
    try:
        results = recall(ws, "... !!! ???", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()
