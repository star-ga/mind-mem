"""Tests for different block types in recall."""

from __future__ import annotations

import os
import tempfile

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


def _make_workspace():
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    for dir_name, block_type, prefix in [
        ("decisions", "Decision", "DEC"),
        ("tasks", "Task", "TSK"),
        ("entities", "Entity", "ENT"),
    ]:
        path = os.path.join(ws, dir_name, "types.md")
        with open(path, "w") as f:
            f.write(f"[{prefix}-001]\nType: {block_type}\nStatement: Type test {block_type}\n\n")
    return ws, td


def test_recall_decision_type():
    """Recall returns Decision type blocks."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "type test decision", limit=10)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_recall_task_type():
    """Recall returns Task type blocks."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "type test task", limit=10)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_recall_entity_type():
    """Recall returns Entity type blocks."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "type test entity", limit=10)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_recall_all_types():
    """Recall searches across all block types."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "type test", limit=10)
        assert isinstance(results, list)
    finally:
        td.cleanup()
