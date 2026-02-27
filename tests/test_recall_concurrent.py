"""Tests for concurrent recall queries."""

from __future__ import annotations

import os
import tempfile
import threading

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


def _ws():
    td = tempfile.TemporaryDirectory()
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    p = os.path.join(ws, "decisions", "conc.md")
    with open(p, "w") as f:
        for i in range(10):
            f.write(f"[CC-{i:03d}]\nType: Decision\nStatement: Concurrent test {i}\n\n")
    return ws, td


def test_sequential_recalls():
    ws, td = _ws()
    try:
        for _ in range(5):
            r = recall(ws, "concurrent test", limit=5)
            assert isinstance(r, list)
    finally:
        td.cleanup()


def test_threaded_recalls():
    ws, td = _ws()
    try:
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
    finally:
        td.cleanup()
