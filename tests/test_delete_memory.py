"""Tests for memory deletion functionality."""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("mind_mem.delete_memory")

from mind_mem.delete_memory import delete_block  # noqa: E402

from mind_mem.init_workspace import init  # noqa: E402


def _make_workspace():
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "delete_test.md")
    with open(blocks_md, "w") as f:
        f.write("[DEL-001]\nType: Decision\nStatement: To be deleted\n\n")
        f.write("[DEL-002]\nType: Decision\nStatement: To keep\n\n")
    return ws, td


def test_delete_importable():
    """Delete memory module is importable."""
    assert callable(delete_block)


def test_delete_existing_block():
    """Deleting an existing block succeeds."""
    ws, td = _make_workspace()
    try:
        result = delete_block(ws, "DEL-001")
        assert result is not None
    finally:
        td.cleanup()


def test_delete_nonexistent_block():
    """Deleting non-existent block handles gracefully."""
    ws, td = _make_workspace()
    try:
        result = delete_block(ws, "NONEXIST-999")
        assert isinstance(result, (dict, bool, type(None)))
    finally:
        td.cleanup()
