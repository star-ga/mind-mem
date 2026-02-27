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
    with open(blocks_md, "w") as f:
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
    if len(results) >= 2:
        scores = [r.get("score", r.get("_score", 0)) for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Score {scores[i]} < {scores[i + 1]}"


def test_more_relevant_ranked_higher(ws):
    """More relevant blocks rank higher."""
    results = recall(ws, "BM25 scoring algorithm text retrieval", limit=10)
    if results:
        first_id = results[0].get("id") or results[0].get("block_id", "")
        assert "ORD-001" in str(first_id) or len(results) == 1


def test_scores_are_positive(ws):
    """All scores are non-negative."""
    results = recall(ws, "scoring", limit=10)
    for r in results:
        score = r.get("score", r.get("_score", 0))
        assert score >= 0
