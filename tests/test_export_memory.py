"""Tests for memory export functionality."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

pytest.importorskip("mind_mem.export_memory")

from mind_mem.export_memory import export_memory  # noqa: E402

from mind_mem.init_workspace import init  # noqa: E402


def _make_workspace():
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "export_test.md")
    with open(blocks_md, "w", encoding="utf-8") as f:
        f.write("[EXP-001]\nType: Decision\nStatement: Exportable decision\n\n")
        f.write("[EXP-002]\nType: Decision\nStatement: Another exportable\n\n")
    return ws, td


def test_export_produces_output():
    """Export produces non-empty output."""
    ws, td = _make_workspace()
    try:
        result = export_memory(ws)
        assert result is not None
        assert len(result) > 0
    finally:
        td.cleanup()


def test_export_format_jsonl():
    """Export produces valid JSONL."""
    ws, td = _make_workspace()
    try:
        result = export_memory(ws, format="jsonl")
        if isinstance(result, str):
            for line in result.strip().split("\n"):
                if line:
                    json.loads(line)  # Should not raise
    finally:
        td.cleanup()
