"""Tests for MIND FFI module."""
from __future__ import annotations

import os
import tempfile

import pytest

from scripts.mind_ffi import get_mind_dir, load_kernel_config, get_kernel_param


def test_get_mind_dir():
    """get_mind_dir returns .mind subdirectory."""
    ws = tempfile.mkdtemp()
    mind_dir = get_mind_dir(ws)
    assert mind_dir.endswith(".mind") or ".mind" in mind_dir


def test_load_kernel_config_missing():
    """Loading non-existent kernel returns None."""
    result = load_kernel_config("/nonexistent/path/kernel.mind")
    assert result is None


def test_load_kernel_config_empty():
    """Loading empty file returns None or empty config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
        f.write("")
        path = f.name
    try:
        result = load_kernel_config(path)
        assert result is None or isinstance(result, dict)
    finally:
        os.unlink(path)


def test_get_kernel_param_default():
    """Missing param returns default value."""
    result = get_kernel_param(None, "bm25", "k1", 1.2)
    assert result == 1.2


def test_get_kernel_param_from_config():
    """Param from valid config is returned."""
    config = {"bm25": {"k1": 2.0}}
    result = get_kernel_param(config, "bm25", "k1", 1.2)
    assert result == 2.0


def test_get_kernel_param_missing_section():
    """Missing section returns default."""
    config = {"other": {"key": "val"}}
    result = get_kernel_param(config, "bm25", "k1", 1.2)
    assert result == 1.2
