"""Tests for integrity scan engine."""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("mind_mem.scan_engine")

from mind_mem.scan_engine import scan  # noqa: E402

from mind_mem.init_workspace import init  # noqa: E402


def _make_workspace_with_blocks():
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "scan_test.md")
    with open(blocks_md, "w", encoding="utf-8") as f:
        f.write("[SCAN-001]\nType: Decision\nStatement: First decision\nStatus: Active\n\n")
        f.write("[SCAN-002]\nType: Decision\nStatement: Contradicts first\nStatus: Active\n\n")
    return ws, td


def test_scan_workspace_no_crash():
    """Scanning a valid workspace doesn't crash."""
    ws, td = _make_workspace_with_blocks()
    try:
        result = scan(ws)
        assert isinstance(result, (dict, list))
    finally:
        td.cleanup()


def test_scan_empty_workspace():
    """Scanning empty workspace returns clean result."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        init(ws)
        result = scan(ws)
        assert isinstance(result, (dict, list))
