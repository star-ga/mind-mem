"""Tests for recall result scoring order."""
from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "order_test.md")
    with open(blocks_md, "w") as f:
        # Block with high relevance (multiple keyword matches)
        f.write("[ORD-001]\nType: Decision\nStatement: BM25 scoring algorithm for text retrieval search\n\n")
        # Block with medium relevance
        f.write("[ORD-002]\nType: Decision\nStatement: Something about scoring\n\n")
        # Block with low relevance
        f.write("[ORD-003]\nType: Decision\nStatement: Unrelated topic entirely\n\n")
    return ws


def test_results_sorted_by_score():
    """Results are returned in descending score order."""
    ws = _make_workspace()
    results = recall(ws, "BM25 scoring algorithm", limit=10)
    if len(results) >= 2:
        scores = [r.get("score", r.get("_score", 0)) for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Score {scores[i]} < {scores[i+1]}"


def test_more_relevant_ranked_higher():
    """More relevant blocks rank higher."""
    ws = _make_workspace()
    results = recall(ws, "BM25 scoring algorithm text retrieval", limit=10)
    if results:
        first_id = results[0].get("id") or results[0].get("block_id", "")
        assert "ORD-001" in str(first_id) or len(results) == 1


def test_scores_are_positive():
    """All scores are non-negative."""
    ws = _make_workspace()
    results = recall(ws, "scoring", limit=10)
    for r in results:
        score = r.get("score", r.get("_score", 0))
        assert score >= 0
