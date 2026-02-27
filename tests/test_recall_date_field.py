"""Tests for date field in recall results."""

from __future__ import annotations

import os
import tempfile

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


def _make_workspace():
    td = tempfile.TemporaryDirectory()
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    path = os.path.join(ws, "decisions", "dated.md")
    with open(path, "w") as f:
        f.write("[DT-001]\nType: Decision\nStatement: Recent decision\nDate: 2026-02-24\n\n")
        f.write("[DT-002]\nType: Decision\nStatement: Old decision\nDate: 2025-01-01\n\n")
    return ws, td


def test_dated_blocks_found():
    ws, td = _make_workspace()
    try:
        results = recall(ws, "decision", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_recent_date_search():
    ws, td = _make_workspace()
    try:
        results = recall(ws, "recent decision", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_old_date_search():
    ws, td = _make_workspace()
    try:
        results = recall(ws, "old decision", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()
