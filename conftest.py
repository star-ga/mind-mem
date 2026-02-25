"""Shared pytest fixtures for mind-mem test suite."""
import os

import pytest

from scripts.init_workspace import init as init_workspace


@pytest.fixture
def workspace(tmp_path):
    """Create and return an initialized mind-mem workspace."""
    ws = str(tmp_path / "workspace")
    os.makedirs(ws)
    init_workspace(ws)
    return ws
