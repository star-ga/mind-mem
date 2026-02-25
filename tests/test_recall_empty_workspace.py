"""Tests for recall on empty workspaces."""

from __future__ import annotations

from scripts._recall_core import recall


def test_empty_decisions(workspace):
    results = recall(workspace, "test query", limit=5)
    assert results == []


def test_empty_all_dirs(workspace):
    results = recall(workspace, "anything", limit=10)
    assert isinstance(results, list)
    assert len(results) == 0


def test_workspace_only_memory_md(workspace):
    results = recall(workspace, "memory", limit=5)
    assert isinstance(results, list)
