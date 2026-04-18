"""Tests for workspace initialization."""

from __future__ import annotations

import os
import tempfile

from mind_mem.init_workspace import init


def test_init_creates_directories():
    """init() creates the expected directory structure."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        init(ws)
        for d in ["decisions", "tasks", "entities", "memory", "intelligence", "summaries"]:
            assert os.path.isdir(os.path.join(ws, d)), f"Missing directory: {d}"


def test_init_creates_memory_md():
    """init() creates MEMORY.md."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        init(ws)
        assert os.path.isfile(os.path.join(ws, "MEMORY.md"))


def test_init_idempotent():
    """Calling init() twice doesn't error."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        init(ws)
        init(ws)  # Should not raise
        assert os.path.isdir(os.path.join(ws, "decisions"))


def test_init_preserves_existing_files():
    """init() doesn't overwrite existing files."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        ws = os.path.join(td, "ws")
        os.makedirs(ws)
        init(ws)
        marker = os.path.join(ws, "decisions", "existing.md")
        with open(marker, "w") as f:
            f.write("keep me")
        init(ws)
        assert os.path.isfile(marker)
        with open(marker) as f:
            assert f.read() == "keep me"


def test_init_nested_path():
    """init() works with deeply nested paths."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        ws = os.path.join(td, "a", "b", "c")
        os.makedirs(ws)
        init(ws)
        assert os.path.isdir(os.path.join(ws, "decisions"))
