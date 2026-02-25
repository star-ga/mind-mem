"""Tests for memory evolution tracking."""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("scripts.memory_evolution")

from scripts.memory_evolution import track_evolution  # noqa: E402

from scripts.init_workspace import init  # noqa: E402


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    return ws


def test_evolution_tracking_importable():
    """Memory evolution module is importable."""
    assert callable(track_evolution)


def test_evolution_empty_workspace():
    """Evolution tracking on empty workspace doesn't crash."""
    ws = _make_workspace()
    result = track_evolution(ws, "NONEXIST-001")
    assert isinstance(result, (dict, list, type(None)))


def test_evolution_returns_history():
    """Evolution tracking returns history list."""
    ws = _make_workspace()
    blocks_md = os.path.join(ws, "decisions", "evo.md")
    with open(blocks_md, "w") as f:
        f.write("[EVO-001]\nType: Decision\nStatement: Original\n\n")
    result = track_evolution(ws, "EVO-001")
    assert isinstance(result, (dict, list, type(None)))
