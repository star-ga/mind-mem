"""Extended block parser tests."""

from __future__ import annotations

import os
import tempfile

import pytest

from scripts.block_parser import parse_file


def test_empty_file():
    """Parse an empty file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("")
        path = f.name
    try:
        blocks = parse_file(path)
        assert blocks == []
    finally:
        os.unlink(path)


def test_no_blocks():
    """Parse file with no block markers."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("Just some text\nNo blocks here\n")
        path = f.name
    try:
        blocks = parse_file(path)
        assert blocks == []
    finally:
        os.unlink(path)


def test_single_block():
    """Parse file with one block."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("[TEST-001]\nType: Decision\nStatement: Something\n")
        path = f.name
    try:
        blocks = parse_file(path)
        assert len(blocks) >= 1
    finally:
        os.unlink(path)


def test_multiple_blocks():
    """Parse file with multiple blocks."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        for i in range(5):
            f.write(f"[MULTI-{i:03d}]\nType: Decision\nStatement: Block {i}\n\n")
        path = f.name
    try:
        blocks = parse_file(path)
        assert len(blocks) == 5
    finally:
        os.unlink(path)


def test_block_with_multiline_statement():
    """Parse block with multi-line statement."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("[ML-001]\nType: Decision\nStatement: Line one\n  continued here\n")
        path = f.name
    try:
        blocks = parse_file(path)
        assert len(blocks) >= 1
    finally:
        os.unlink(path)


def test_nonexistent_file():
    """Parsing non-existent file raises error."""
    with pytest.raises((FileNotFoundError, OSError)):
        parse_file("/nonexistent/path.md")


def test_unicode_content():
    """Parse file with unicode content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("[UNI-001]\nType: Decision\nStatement: 日本語テスト\n")
        path = f.name
    try:
        blocks = parse_file(path)
        assert len(blocks) >= 1
    finally:
        os.unlink(path)
