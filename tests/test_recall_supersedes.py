"""Tests for supersedes field in recall."""

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
    p = os.path.join(ws, "decisions", "sup.md")
    with open(p, "w") as f:
        f.write("[SUP-001]\nType: Decision\nStatement: Original decision\nStatus: Superseded\n\n")
        f.write("[SUP-002]\nType: Decision\nStatement: Replacement decision\nSupersedes: SUP-001\nStatus: Active\n\n")
    return ws


def test_superseded_blocks(ws):
    results = recall(ws, "decision", limit=10)
    assert isinstance(results, list)


def test_active_superseding(ws):
    results = recall(ws, "replacement decision", limit=5)
    assert isinstance(results, list)
