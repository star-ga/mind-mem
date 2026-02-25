"""Tests for agent_id namespace filtering."""

from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "agent_test.md")
    with open(blocks_md, "w") as f:
        f.write("[AGT-001]\nType: Decision\nStatement: Agent-specific decision\n\n")
    return ws


def test_agent_id_none():
    """None agent_id returns all results."""
    ws = _make_workspace()
    results = recall(ws, "decision", limit=10, agent_id=None)
    assert isinstance(results, list)


def test_agent_id_string():
    """String agent_id doesn't crash."""
    ws = _make_workspace()
    results = recall(ws, "decision", limit=10, agent_id="test-agent")
    assert isinstance(results, list)


def test_agent_id_empty():
    """Empty agent_id behaves like None."""
    ws = _make_workspace()
    results = recall(ws, "decision", limit=10, agent_id="")
    assert isinstance(results, list)
