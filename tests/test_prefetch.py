"""Tests for prefetch functionality."""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("mind_mem.prefetch")

from mind_mem.prefetch import prefetch  # noqa: E402

from mind_mem.init_workspace import init  # noqa: E402


def _make_workspace():
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "prefetch_test.md")
    with open(blocks_md, "w", encoding="utf-8") as f:
        f.write("[PF-001]\nType: Decision\nStatement: Prefetch test block\n\n")
    return ws, td


def test_prefetch_importable():
    """Prefetch module is importable."""
    assert callable(prefetch)


def test_prefetch_with_signals():
    """Prefetch with context signals returns results."""
    ws, td = _make_workspace()
    try:
        result = prefetch(ws, signals={"recent_queries": ["test"]})
        assert isinstance(result, (list, dict))
    finally:
        td.cleanup()


def test_prefetch_empty_signals():
    """Prefetch with no signals returns something."""
    ws, td = _make_workspace()
    try:
        result = prefetch(ws, signals={})
        assert isinstance(result, (list, dict, type(None)))
    finally:
        td.cleanup()
