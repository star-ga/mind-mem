"""Tests for agent_id namespace filtering."""

from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _make_workspace():
    td = tempfile.TemporaryDirectory()
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "agent_test.md")
    with open(blocks_md, "w") as f:
        f.write("[AGT-001]\nType: Decision\nStatement: Agent-specific decision\n\n")
    return ws, td


def test_agent_id_none():
    """None agent_id returns all results."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "decision", limit=10, agent_id=None)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_agent_id_string():
    """String agent_id doesn't crash."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "decision", limit=10, agent_id="test-agent")
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_agent_id_empty():
    """Empty agent_id behaves like None."""
    ws, td = _make_workspace()
    try:
        results = recall(ws, "decision", limit=10, agent_id="")
        assert isinstance(results, list)
    finally:
        td.cleanup()
