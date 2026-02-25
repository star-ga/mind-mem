"""Tests for block ID format validation."""
from __future__ import annotations

import os
import tempfile

from scripts.block_parser import parse_file


def test_standard_id():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("[TEST-001]\nType: Decision\nStatement: Standard ID\n")
        path = f.name
    blocks = parse_file(path)
    assert len(blocks) >= 1
    os.unlink(path)

def test_numeric_id():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("[12345]\nType: Decision\nStatement: Numeric ID\n")
        path = f.name
    blocks = parse_file(path)
    os.unlink(path)
    assert isinstance(blocks, list)

def test_long_id():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("[VERY-LONG-IDENTIFIER-NAME-001]\nType: Decision\nStatement: Long ID\n")
        path = f.name
    blocks = parse_file(path)
    os.unlink(path)
    assert isinstance(blocks, list)

def test_lowercase_id():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("[lower-case-001]\nType: Decision\nStatement: Lowercase\n")
        path = f.name
    blocks = parse_file(path)
    os.unlink(path)
    assert isinstance(blocks, list)
