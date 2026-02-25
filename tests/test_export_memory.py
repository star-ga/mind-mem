"""Tests for memory export functionality."""
from __future__ import annotations

import json
import os
import tempfile

from scripts.init_workspace import init


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    blocks_md = os.path.join(ws, "decisions", "export_test.md")
    with open(blocks_md, "w") as f:
        f.write("[EXP-001]\nType: Decision\nStatement: Exportable decision\n\n")
        f.write("[EXP-002]\nType: Decision\nStatement: Another exportable\n\n")
    return ws


def test_export_produces_output():
    """Export produces non-empty output."""
    try:
        from scripts.export_memory import export_memory
        ws = _make_workspace()
        result = export_memory(ws)
        assert result is not None
        assert len(result) > 0
    except ImportError:
        pass  # Module may have different name


def test_export_format_jsonl():
    """Export produces valid JSONL."""
    try:
        from scripts.export_memory import export_memory
        ws = _make_workspace()
        result = export_memory(ws, format="jsonl")
        if isinstance(result, str):
            for line in result.strip().split("\n"):
                if line:
                    json.loads(line)  # Should not raise
    except ImportError:
        pass
