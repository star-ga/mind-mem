"""Shared pytest fixtures for mind-mem test suite."""

import os
import sys

import pytest

ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mind_mem.init_workspace import init as init_workspace  # noqa: E402


@pytest.fixture
def workspace(tmp_path):
    """Create and return an initialized mind-mem workspace."""
    ws = str(tmp_path / "workspace")
    os.makedirs(ws)
    init_workspace(ws)
    return ws
