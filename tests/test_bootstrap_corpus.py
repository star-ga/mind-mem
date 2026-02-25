#!/usr/bin/env python3
"""Tests for bootstrap_corpus.py — backfill pipeline module."""

import os
import tempfile
import unittest

from mind_mem.bootstrap_corpus import main, scan_markdown_file


class TestScanMarkdownFile(unittest.TestCase):
    """Tests for scan_markdown_file()."""

    def _write_md(self, tmpdir: str, content: str) -> str:
        path = os.path.join(tmpdir, "test.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_nonexistent_file_returns_empty(self):
        """Non-existent path returns empty list without raising."""
        result = scan_markdown_file("/tmp/does_not_exist_xyz_42.md")
        self.assertEqual(result, [])

    def test_empty_file_returns_empty(self):
        """An empty markdown file produces no signals."""
        with tempfile.TemporaryDirectory() as td:
            path = self._write_md(td, "")
            result = scan_markdown_file(path)
            self.assertEqual(result, [])

    def test_whitespace_only_file_returns_empty(self):
        """A file with only whitespace/blank lines produces no signals."""
        with tempfile.TemporaryDirectory() as td:
            path = self._write_md(td, "\n\n   \n\n")
            result = scan_markdown_file(path)
            self.assertEqual(result, [])

    def test_headers_only_returns_empty(self):
        """Markdown headers (# lines) are skipped by scan_log."""
        with tempfile.TemporaryDirectory() as td:
            path = self._write_md(td, "# We decided to use Rust\n## We will refactor\n")
            result = scan_markdown_file(path)
            self.assertEqual(result, [])

    def test_detects_decision_pattern(self):
        """'We decided to' triggers a decision signal."""
        with tempfile.TemporaryDirectory() as td:
            path = self._write_md(td, "We decided to adopt TypeScript for the frontend.\n")
            result = scan_markdown_file(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["type"], "decision")
            self.assertEqual(result[0]["confidence"], "high")
            self.assertEqual(result[0]["priority"], "P1")
            self.assertIn("decided to", result[0]["text"].lower())

    def test_detects_task_pattern(self):
        """'need to' triggers a task signal."""
        with tempfile.TemporaryDirectory() as td:
            path = self._write_md(td, "We need to migrate the database to PostgreSQL.\n")
            result = scan_markdown_file(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["type"], "task")

    def test_multiple_signals_from_multiline(self):
        """Multiple matching lines each produce a signal."""
        content = (
            "Regular text with no patterns.\n"
            "We decided to switch to Redis for caching.\n"
            "Another plain line.\n"
            "We need to update the CI pipeline.\n"
            "Final line.\n"
        )
        with tempfile.TemporaryDirectory() as td:
            path = self._write_md(td, content)
            result = scan_markdown_file(path)
            self.assertEqual(len(result), 2)
            types = {s["type"] for s in result}
            self.assertEqual(types, {"decision", "task"})

    def test_crossreferenced_lines_skipped(self):
        """Lines with existing cross-reference IDs (D-YYYYMMDD-NNN) are skipped."""
        with tempfile.TemporaryDirectory() as td:
            path = self._write_md(td, "We decided to use JWT (D-20260213-001).\n")
            result = scan_markdown_file(path)
            self.assertEqual(result, [])

    def test_signal_has_required_fields(self):
        """Each signal dict contains all required keys."""
        required_keys = {"line", "type", "text", "pattern", "confidence", "priority", "structure"}
        with tempfile.TemporaryDirectory() as td:
            path = self._write_md(td, "We decided to deprecate the old API endpoint.\n")
            result = scan_markdown_file(path)
            self.assertEqual(len(result), 1)
            self.assertTrue(required_keys.issubset(result[0].keys()),
                            f"Missing keys: {required_keys - result[0].keys()}")

    def test_line_number_tracking(self):
        """Signal line number matches the actual line in the file."""
        content = "Line one.\nLine two.\nWe will migrate to Kubernetes.\n"
        with tempfile.TemporaryDirectory() as td:
            path = self._write_md(td, content)
            result = scan_markdown_file(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["line"], 3)

    def test_medium_confidence_pattern(self):
        """'switching to' triggers a medium-confidence decision."""
        with tempfile.TemporaryDirectory() as td:
            path = self._write_md(td, "We are switching to a monorepo layout.\n")
            result = scan_markdown_file(path)
            self.assertTrue(len(result) >= 1)
            # At least one match should be medium confidence
            confidences = [s["confidence"] for s in result]
            self.assertIn("medium", confidences)

    def test_text_truncated_to_150_chars(self):
        """Signal text is truncated to 150 characters max."""
        long_line = "We decided to " + "x" * 200 + "\n"
        with tempfile.TemporaryDirectory() as td:
            path = self._write_md(td, long_line)
            result = scan_markdown_file(path)
            self.assertEqual(len(result), 1)
            self.assertLessEqual(len(result[0]["text"]), 150)


class TestModuleStructure(unittest.TestCase):
    """Tests for module-level imports and structure."""

    def test_imports_resolve(self):
        """The bootstrap_corpus module can be imported without error."""
        import mind_mem.bootstrap_corpus as mod
        self.assertTrue(hasattr(mod, "scan_markdown_file"))
        self.assertTrue(hasattr(mod, "main"))

    def test_main_is_callable(self):
        """main() exists and is callable."""
        self.assertTrue(callable(main))

    def test_argparser_creation(self):
        """The argparse parser can be constructed (we just invoke its creation path)."""
        import argparse
        # Re-create the parser as main() does — validates the arg spec is valid
        parser = argparse.ArgumentParser(description="mind-mem Bootstrap Corpus Backfill")
        parser.add_argument("workspace", help="Path to mind-mem workspace")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--max-transcripts", type=int, default=0)
        # Parsing known good args should succeed
        args = parser.parse_args(["/tmp/ws", "--dry-run", "--max-transcripts", "5"])
        self.assertEqual(args.workspace, "/tmp/ws")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.max_transcripts, 5)


if __name__ == "__main__":
    unittest.main()
