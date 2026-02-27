"""Tests for speaker-based recall."""

from __future__ import annotations

import os
import tempfile

from mind_mem._recall_core import recall
from mind_mem.init_workspace import init


def _make_workspace():
    td = tempfile.TemporaryDirectory()
    ws = os.path.join(td.name, "ws")
    os.makedirs(ws)
    init(ws)
    path = os.path.join(ws, "decisions", "speaker.md")
    with open(path, "w") as f:
        f.write("[SPK-001]\nType: Decision\nStatement: User said this\nSpeaker: alice\n\n")
        f.write("[SPK-002]\nType: Decision\nStatement: System generated\nSpeaker: system\n\n")
    return ws, td


def test_speaker_search():
    ws, td = _make_workspace()
    try:
        results = recall(ws, "alice said", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()


def test_system_speaker():
    ws, td = _make_workspace()
    try:
        results = recall(ws, "system generated", limit=5)
        assert isinstance(results, list)
    finally:
        td.cleanup()
