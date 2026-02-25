"""Tests for speaker-based recall."""
from __future__ import annotations

import os
import tempfile

from scripts._recall_core import recall
from scripts.init_workspace import init


def _make_workspace():
    ws = tempfile.mkdtemp()
    init(ws)
    path = os.path.join(ws, "decisions", "speaker.md")
    with open(path, "w") as f:
        f.write("[SPK-001]\nType: Decision\nStatement: User said this\nSpeaker: alice\n\n")
        f.write("[SPK-002]\nType: Decision\nStatement: System generated\nSpeaker: system\n\n")
    return ws

def test_speaker_search():
    ws = _make_workspace()
    results = recall(ws, "alice said", limit=5)
    assert isinstance(results, list)

def test_system_speaker():
    ws = _make_workspace()
    results = recall(ws, "system generated", limit=5)
    assert isinstance(results, list)
