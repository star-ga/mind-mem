"""Tests for recall with large workspaces."""

from __future__ import annotations

import os
import tempfile
import time

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


def _ws(n=100):
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    p = os.path.join(ws, "decisions", "large.md")
    with open(p, "w") as f:
        for i in range(n):
            f.write(f"[LG-{i:04d}]\nType: Decision\n")
            f.write(f"Statement: Large workspace block {i} about topic {chr(65 + i % 26)}\n\n")
    return ws, td


def test_100_blocks():
    ws, td = _ws(100)
    try:
        results = recall(ws, "large workspace block", limit=10)
        assert isinstance(results, list)
        assert len(results) <= 10
    finally:
        td.cleanup()


def test_recall_completes_in_time():
    ws, td = _ws(100)
    try:
        start = time.perf_counter()
        recall(ws, "workspace block topic", limit=10)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 5000, f"Recall took {elapsed:.0f}ms"
    finally:
        td.cleanup()


def test_limit_respected_large():
    ws, td = _ws(100)
    try:
        results = recall(ws, "block", limit=5)
        assert len(results) <= 5
    finally:
        td.cleanup()
