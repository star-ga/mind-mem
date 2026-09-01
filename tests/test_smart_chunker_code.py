#!/usr/bin/env python3
"""Tests for code-aware chunking in smart_chunker.py."""

from __future__ import annotations

import unittest

from mind_mem.smart_chunker import (
    _detect_code_boundaries,
    _segment_document,
)


class TestDetectCodeBoundaries(unittest.TestCase):
    """Test the _detect_code_boundaries function."""

    def test_python_functions(self):
        """Should detect Python def and class boundaries."""
        code = "import os\n\ndef foo():\n    pass\n\ndef bar(x):\n    return x\n\nclass MyClass:\n    pass\n"
        bounds = _detect_code_boundaries(code)
        # Should find boundaries at 'def foo', 'def bar', 'class MyClass'
        # The first def starts at offset > 0 (after import line)
        self.assertGreaterEqual(len(bounds), 2)
        # Verify each boundary offset points to a def/class line
        for offset in bounds:
            remainder = code[offset:]
            self.assertTrue(
                remainder.startswith("def ") or remainder.startswith("class "),
                f"Boundary at {offset} does not start a def/class: {remainder[:30]!r}",
            )

    def test_javascript_boundaries(self):
        """Should detect JS function, class, and arrow function boundaries."""
        code = (
            "const x = 1;\n"
            "\n"
            "function hello() {\n"
            "  return 'hi';\n"
            "}\n"
            "\n"
            "class Widget {\n"
            "  render() {}\n"
            "}\n"
            "\n"
            "const greet = (name) => {\n"
            "  return name;\n"
            "};\n"
        )
        bounds = _detect_code_boundaries(code)
        self.assertGreaterEqual(len(bounds), 2)

    def test_rust_boundaries(self):
        """Should detect Rust fn, impl, and struct boundaries."""
        code = (
            "use std::io;\n"
            "\n"
            "fn main() {\n"
            '    println!("hello");\n'
            "}\n"
            "\n"
            "pub fn helper(x: i32) -> i32 {\n"
            "    x + 1\n"
            "}\n"
            "\n"
            "pub struct Config {\n"
            "    name: String,\n"
            "}\n"
        )
        bounds = _detect_code_boundaries(code)
        self.assertGreaterEqual(len(bounds), 2)

    def test_go_func_boundaries(self):
        """Should detect Go func boundaries including method receivers."""
        code = 'package main\n\nfunc main() {\n    fmt.Println("hello")\n}\n\nfunc (s *Server) Start() error {\n    return nil\n}\n'
        bounds = _detect_code_boundaries(code)
        self.assertGreaterEqual(len(bounds), 1)

    def test_no_boundaries_in_plain_text(self):
        """Plain text with no code patterns should return empty list."""
        text = "This is just some regular text.\nNothing special here.\n"
        bounds = _detect_code_boundaries(text)
        self.assertEqual(bounds, [])


class TestCodeAwareSegmentation(unittest.TestCase):
    """Test that code blocks with multiple functions get split at boundaries."""

    def test_code_block_split_at_function_boundaries(self):
        """A fenced code block with multiple functions should produce multiple segments."""
        doc = (
            "# Module docs\n"
            "\n"
            "Here is some code:\n"
            "\n"
            "```python\n"
            "def alpha():\n"
            "    return 1\n"
            "\n"
            "def beta():\n"
            "    return 2\n"
            "\n"
            "def gamma():\n"
            "    return 3\n"
            "```\n"
            "\n"
            "That was the code.\n"
        )
        segments = _segment_document(doc)
        code_segments = [s for s in segments if s.kind == "code"]
        # The code block should be split into multiple code segments
        # at function boundaries (at least 2 segments from 3 functions)
        self.assertGreaterEqual(
            len(code_segments),
            2,
            f"Expected code block to be split at function boundaries, got {len(code_segments)} code segment(s)",
        )

    def test_single_function_code_block_not_split(self):
        """A code block with only one function should remain as a single segment."""
        doc = "```python\ndef only_one():\n    return 42\n```\n"
        segments = _segment_document(doc)
        code_segments = [s for s in segments if s.kind == "code"]
        # Single function = no internal boundaries = one segment
        self.assertEqual(len(code_segments), 1)


if __name__ == "__main__":
    unittest.main()
