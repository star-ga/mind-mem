"""Tests for _allow_decompose recall parameter."""

from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "decompose_test.md")
    with open(blocks_md, "w") as f:
        f.write("[AD-001]\nType: Decision\nStatement: Decomposition test alpha\n\n")
        f.write("[AD-002]\nType: Decision\nStatement: Decomposition test beta\n\n")
    return ws


def test_allow_decompose_true():
    """_allow_decompose=True works."""
    ws = _make_workspace()
    results = recall(ws, "decomposition test alpha and beta", limit=5, _allow_decompose=True)
    assert isinstance(results, list)


def test_allow_decompose_false():
    """_allow_decompose=False works."""
    ws = _make_workspace()
    results = recall(ws, "decomposition test alpha and beta", limit=5, _allow_decompose=False)
    assert isinstance(results, list)
