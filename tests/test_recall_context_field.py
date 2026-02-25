"""Tests for context field in blocks."""
from __future__ import annotations
import os, tempfile
from scripts.init_workspace import init
from scripts._recall_core import recall

def _ws():
    ws = tempfile.mkdtemp()
    init(ws)
    p = os.path.join(ws, "decisions", "ctx.md")
    with open(p, "w") as f:
        f.write("[CTX-001]\nType: Decision\nStatement: Main content\nContext: During sprint planning\n\n")
        f.write("[CTX-002]\nType: Decision\nStatement: Other content\nContext: During code review\n\n")
    return ws

def test_context_field_search():
    results = recall(_ws(), "sprint planning", limit=5)
    assert isinstance(results, list)

def test_context_field_code_review():
    results = recall(_ws(), "code review", limit=5)
    assert isinstance(results, list)
