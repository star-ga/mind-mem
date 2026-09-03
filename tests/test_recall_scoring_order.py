"""Tests for recall result scoring order."""

from __future__ import annotations

import os

import pytest

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


@pytest.fixture
def ws(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "order_test.md")
    with open(blocks_md, "w", encoding="utf-8") as f:
        # Block with high relevance (multiple keyword matches)
        f.write("[ORD-001]\nType: Decision\nStatement: BM25 scoring algorithm for text retrieval search\n\n")
        # Block with medium relevance
        f.write("[ORD-002]\nType: Decision\nStatement: Something about scoring\n\n")
        # Block with low relevance
        f.write("[ORD-003]\nType: Decision\nStatement: Unrelated topic entirely\n\n")
    return ws


def test_results_sorted_by_score(ws):
    """Results are returned in descending score order."""
    results = recall(ws, "BM25 scoring algorithm", limit=10)
    # POSITIVE CONTROL: an ordering claim over fewer than two hits is not a
    # claim. This used to be a bare ``if``, which meant an unretrievable
    # fixture reported PASS instead of reporting that it had nothing to order.
    assert len(results) >= 2, f"fixture must yield two orderable hits, got {[r.get('_id') for r in results]}"
    scores = [r.get("score", r.get("_score", 0)) for r in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Score {scores[i]} < {scores[i + 1]}"


def test_more_relevant_ranked_higher(ws):
    """More relevant blocks rank higher."""
    results = recall(ws, "BM25 scoring algorithm text retrieval", limit=10)
    # A recall hit is keyed by ``_id``; it has never carried ``id`` or
    # ``block_id``. Reading those returned ``""`` and the claim rested entirely
    # on the ``len(results) == 1`` escape hatch beside it — so while the corpus
    # table hid this fixture from recall, the test passed on a single hit (or
    # on none) without ever checking which block ranked first.
    assert results, "fixture blocks are unretrievable — there is nothing to rank"
    ranked = [r.get("_id") for r in results]
    assert ranked[0] == "ORD-001", f"expected ORD-001 ranked first, got {ranked}"


def test_scores_are_positive(ws):
    """All scores are non-negative."""
    results = recall(ws, "scoring", limit=10)
    assert results, "fixture blocks are unretrievable — the check below would be vacuous"
    for r in results:
        score = r.get("score", r.get("_score", 0))
        assert score >= 0
