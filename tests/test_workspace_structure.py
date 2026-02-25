"""Tests for workspace directory structure."""

from __future__ import annotations

import os
import tempfile

from scripts.init_workspace import init

EXPECTED_DIRS = ["decisions", "tasks", "entities", "memory", "intelligence", "summaries"]


def test_all_directories_created():
    """All expected directories are created."""
    with tempfile.TemporaryDirectory() as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        init(ws)
        for d in EXPECTED_DIRS:
            path = os.path.join(ws, d)
            assert os.path.isdir(path), f"Missing: {d}"


def test_directories_are_writable():
    """All directories are writable."""
    with tempfile.TemporaryDirectory() as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        init(ws)
        for d in EXPECTED_DIRS:
            test_file = os.path.join(ws, d, "write_test.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.unlink(test_file)


def test_memory_md_exists():
    """MEMORY.md is created at workspace root."""
    with tempfile.TemporaryDirectory() as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        init(ws)
        assert os.path.isfile(os.path.join(ws, "MEMORY.md"))


def test_memory_md_has_content():
    """MEMORY.md has initial content."""
    with tempfile.TemporaryDirectory() as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        init(ws)
        with open(os.path.join(ws, "MEMORY.md")) as f:
            content = f.read()
        assert len(content) > 0


def test_workspace_is_self_contained():
    """No files created outside workspace directory."""
    with tempfile.TemporaryDirectory() as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        parent = os.path.dirname(ws)
        before = set(os.listdir(parent))
        init(ws)
        after = set(os.listdir(parent))
        new_files = after - before
        # Only the workspace dir itself should be present
        assert len(new_files) <= 1
