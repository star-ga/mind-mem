"""Tests for propose_update functionality."""
from __future__ import annotations

import os
import tempfile

from scripts.init_workspace import init
from scripts.propose_update import propose_update


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    return ws


def test_propose_update_creates_proposal():
    """propose_update creates a proposal file."""
    ws = _make_workspace()
    result = propose_update(
        ws,
        block_id="NEW-001",
        content="[NEW-001]\nType: Decision\nStatement: Test proposal\n",
        reason="Testing proposal system",
    )
    assert result is not None


def test_propose_update_returns_id():
    """propose_update returns a proposal ID."""
    ws = _make_workspace()
    result = propose_update(
        ws,
        block_id="NEW-002",
        content="[NEW-002]\nType: Decision\nStatement: Another proposal\n",
        reason="Test",
    )
    assert isinstance(result, (str, dict))


def test_propose_update_with_empty_reason():
    """Empty reason is handled gracefully."""
    ws = _make_workspace()
    result = propose_update(
        ws,
        block_id="NEW-003",
        content="[NEW-003]\nType: Decision\nStatement: Minimal\n",
        reason="",
    )
    assert result is not None
