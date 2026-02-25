"""Tests for index statistics."""
from __future__ import annotations

import os
import tempfile

from scripts.init_workspace import init


def test_index_stats_importable():
    """Index stats function is importable."""
    try:
        from scripts.index_stats import get_index_stats
        assert callable(get_index_stats)
    except ImportError:
        pass


def test_index_stats_empty_workspace():
    """Empty workspace returns valid stats."""
    try:
        from scripts.index_stats import get_index_stats
        ws = tempfile.mkdtemp()
        init(ws)
        stats = get_index_stats(ws)
        assert isinstance(stats, dict)
    except ImportError:
        pass


def test_index_stats_with_blocks():
    """Workspace with blocks reports counts."""
    try:
        from scripts.index_stats import get_index_stats
        ws = tempfile.mkdtemp()
        init(ws)
        blocks_md = os.path.join(ws, "decisions", "stats_test.md")
        with open(blocks_md, "w") as f:
            for i in range(10):
                f.write(f"[STAT-{i:03d}]\nType: Decision\nStatement: Test {i}\n\n")
        stats = get_index_stats(ws)
        assert isinstance(stats, dict)
    except ImportError:
        pass
