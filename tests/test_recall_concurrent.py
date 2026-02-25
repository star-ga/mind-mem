"""Tests for concurrent recall queries."""
from __future__ import annotations
import os, tempfile, threading
from scripts.init_workspace import init
from scripts._recall_core import recall

def _ws():
    ws = tempfile.mkdtemp()
    init(ws)
    p = os.path.join(ws, "decisions", "conc.md")
    with open(p, "w") as f:
        for i in range(10):
            f.write(f"[CC-{i:03d}]\nType: Decision\nStatement: Concurrent test {i}\n\n")
    return ws

def test_sequential_recalls():
    ws = _ws()
    for _ in range(5):
        r = recall(ws, "concurrent test", limit=5)
        assert isinstance(r, list)

def test_threaded_recalls():
    ws = _ws()
    results = []
    errors = []
    def worker():
        try:
            r = recall(ws, "concurrent test", limit=3)
            results.append(r)
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(errors) == 0
    assert len(results) == 4
