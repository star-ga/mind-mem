"""Tests for parsing multiple files."""
from __future__ import annotations
import os, tempfile
from scripts.block_parser import parse_file

def test_parse_multiple_files():
    d = tempfile.mkdtemp()
    for i in range(3):
        p = os.path.join(d, f"file{i}.md")
        with open(p, "w") as f:
            f.write(f"[MF-{i:03d}]\nType: Decision\nStatement: File {i}\n\n")
    for i in range(3):
        blocks = parse_file(os.path.join(d, f"file{i}.md"))
        assert len(blocks) >= 1

def test_large_file_parse():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        for i in range(50):
            f.write(f"[LF-{i:03d}]\nType: Decision\nStatement: Block {i}\n\n")
        path = f.name
    blocks = parse_file(path)
    assert len(blocks) == 50
    os.unlink(path)

def test_mixed_content():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Header\nSome text\n\n[MX-001]\nType: Decision\nStatement: In mixed file\n\nMore text\n")
        path = f.name
    blocks = parse_file(path)
    assert len(blocks) >= 1
    os.unlink(path)
