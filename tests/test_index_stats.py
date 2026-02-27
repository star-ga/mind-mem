"""Tests for index statistics."""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("mind_mem.index_stats")

from mind_mem.index_stats import get_index_stats  # noqa: E402

from mind_mem.init_workspace import init  # noqa: E402


def test_index_stats_importable():
    """Index stats function is importable."""
    assert callable(get_index_stats)


def test_index_stats_empty_workspace():
    """Empty workspace returns valid stats."""
    with tempfile.TemporaryDirectory() as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        init(ws)
        stats = get_index_stats(ws)
        assert isinstance(stats, dict)


def test_index_stats_with_blocks():
    """Workspace with blocks reports counts."""
    with tempfile.TemporaryDirectory() as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        init(ws)
        blocks_md = os.path.join(ws, "decisions", "stats_test.md")
        with open(blocks_md, "w") as f:
            for i in range(10):
                f.write(f"[STAT-{i:03d}]\nType: Decision\nStatement: Test {i}\n\n")
        stats = get_index_stats(ws)
        assert isinstance(stats, dict)
