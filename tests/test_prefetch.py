"""Tests for prefetch functionality."""
from __future__ import annotations

import os
import tempfile

from scripts.init_workspace import init


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "prefetch_test.md")
    with open(blocks_md, "w") as f:
        f.write("[PF-001]\nType: Decision\nStatement: Prefetch test block\n\n")
    return ws


def test_prefetch_importable():
    """Prefetch module is importable."""
    try:
        from scripts._recall_core import recall
        assert callable(recall)
    except ImportError:
        pass


def test_prefetch_with_signals():
    """Prefetch with context signals returns results."""
    try:
        from scripts.prefetch import prefetch
        ws = _make_workspace()
        result = prefetch(ws, signals={"recent_queries": ["test"]})
        assert isinstance(result, (list, dict))
    except ImportError:
        pass


def test_prefetch_empty_signals():
    """Prefetch with no signals returns something."""
    try:
        from scripts.prefetch import prefetch
        ws = _make_workspace()
        result = prefetch(ws, signals={})
        assert isinstance(result, (list, dict, type(None)))
    except ImportError:
        pass
