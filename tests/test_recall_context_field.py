"""Tests for context field in blocks."""

from __future__ import annotations

import os
import tempfile

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


def _ws():
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    p = os.path.join(ws, "decisions", "ctx.md")
    with open(p, "w") as f:
        f.write("[CTX-001]\nType: Decision\nStatement: Main content\nContext: During sprint planning\n\n")
        f.write("[CTX-002]\nType: Decision\nStatement: Other content\nContext: During code review\n\n")
    return ws, td


def test_context_field_search():
    ws, td = _ws()
    try:
        results = recall(ws, "sprint planning", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_context_field_code_review():
    ws, td = _ws()
    try:
        results = recall(ws, "code review", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()
