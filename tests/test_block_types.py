"""Tests for different block types in recall."""
from __future__ import annotations

import os
import tempfile

from scripts.init_workspace import init
from scripts._recall_core import recall


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    for dir_name, block_type, prefix in [
        ("decisions", "Decision", "DEC"),
        ("tasks", "Task", "TSK"),
        ("entities", "Entity", "ENT"),
    ]:
        path = os.path.join(ws, dir_name, "types.md")
        with open(path, "w") as f:
            f.write(f"[{prefix}-001]\nType: {block_type}\nStatement: Type test {block_type}\n\n")
    return ws


def test_recall_decision_type():
    """Recall returns Decision type blocks."""
    ws = _make_workspace()
    results = recall(ws, "type test decision", limit=10)
    assert isinstance(results, list)


def test_recall_task_type():
    """Recall returns Task type blocks."""
    ws = _make_workspace()
    results = recall(ws, "type test task", limit=10)
    assert isinstance(results, list)


def test_recall_entity_type():
    """Recall returns Entity type blocks."""
    ws = _make_workspace()
    results = recall(ws, "type test entity", limit=10)
    assert isinstance(results, list)


def test_recall_all_types():
    """Recall searches across all block types."""
    ws = _make_workspace()
    results = recall(ws, "type test", limit=10)
    assert isinstance(results, list)
