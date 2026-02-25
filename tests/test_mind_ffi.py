"""Tests for MIND FFI module."""
from __future__ import annotations
import os
import tempfile
from scripts.mind_ffi import get_mind_dir, load_kernel_config, get_kernel_param

def test_get_mind_dir():
    mind_dir = get_mind_dir("/tmp/test")
    assert isinstance(mind_dir, str)

def test_load_kernel_config_missing():
    result = load_kernel_config("/nonexistent/path/kernel.mind")
    assert isinstance(result, dict)

def test_load_kernel_config_empty():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mind", delete=False) as f:
        f.write("")
        path = f.name
    try:
        result = load_kernel_config(path)
        assert isinstance(result, dict)
    finally:
        os.unlink(path)

def test_get_kernel_param_default():
    result = get_kernel_param({}, "bm25", "k1", 1.2)
    assert result == 1.2

def test_get_kernel_param_from_config():
    config = {"bm25": {"k1": 2.0}}
    result = get_kernel_param(config, "bm25", "k1", 1.2)
    assert result == 2.0

def test_get_kernel_param_missing_section():
    config = {"other": {"key": "val"}}
    result = get_kernel_param(config, "bm25", "k1", 1.2)
    assert result == 1.2
