"""Tests for recall on empty workspaces."""
from __future__ import annotations

import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def test_empty_decisions():
    ws = tempfile.mkdtemp()
    init(ws)
    results = recall(ws, "test query", limit=5)
    assert results == []

def test_empty_all_dirs():
    ws = tempfile.mkdtemp()
    init(ws)
    results = recall(ws, "anything", limit=10)
    assert isinstance(results, list)
    assert len(results) == 0

def test_workspace_only_memory_md():
    ws = tempfile.mkdtemp()
    init(ws)
    results = recall(ws, "memory", limit=5)
    assert isinstance(results, list)
