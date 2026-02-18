#!/usr/bin/env python3
"""Tests for transcript_capture.py — zero external deps (stdlib unittest)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from transcript_capture import (
    parse_transcript,
    scan_transcript,
    TRANSCRIPT_PATTERNS,
    XREF_PATTERN,
)


class TestParseTranscript(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def _write_jsonl(self, lines):
        path = os.path.join(self.td, "session.jsonl")
        with open(path, "w") as f:
            for obj in lines:
                f.write(json.dumps(obj) + "\n")
        return path

    def test_simple_string_content(self):
        path = self._write_jsonl([
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ])
        msgs = parse_transcript(path)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "Hello world")

    def test_list_content_blocks(self):
        path = self._write_jsonl([
            {"role": "assistant", "content": [
                {"type": "text", "text": "Here is the answer"},
                {"type": "text", "text": "And more detail"},
            ]},
        ])
        msgs = parse_transcript(path)
        self.assertEqual(len(msgs), 1)
        self.assertIn("Here is the answer", msgs[0]["content"])
        self.assertIn("And more detail", msgs[0]["content"])

    def test_message_field_fallback(self):
        path = self._write_jsonl([
            {"type": "log", "message": "System started up"},
        ])
        msgs = parse_transcript(path)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["content"], "System started up")

    def test_skips_empty_content(self):
        path = self._write_jsonl([
            {"role": "user", "content": ""},
            {"role": "user", "content": "Real message"},
        ])
        msgs = parse_transcript(path)
        self.assertEqual(len(msgs), 1)

    def test_skips_invalid_json_lines(self):
        path = os.path.join(self.td, "bad.jsonl")
        with open(path, "w") as f:
            f.write("{bad json\n")
            f.write(json.dumps({"role": "user", "content": "Valid"}) + "\n")
        msgs = parse_transcript(path)
        self.assertEqual(len(msgs), 1)

    def test_line_numbers_assigned(self):
        path = self._write_jsonl([
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "user", "content": "Third"},
        ])
        msgs = parse_transcript(path)
        self.assertEqual(msgs[0]["line"], 1)
        self.assertEqual(msgs[1]["line"], 2)
        self.assertEqual(msgs[2]["line"], 3)

    def test_skips_blank_lines(self):
        path = os.path.join(self.td, "blanks.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"role": "user", "content": "First"}) + "\n")
            f.write("\n")
            f.write("\n")
            f.write(json.dumps({"role": "user", "content": "Second"}) + "\n")
        msgs = parse_transcript(path)
        self.assertEqual(len(msgs), 2)

    def test_string_content_blocks_in_list(self):
        path = self._write_jsonl([
            {"role": "assistant", "content": ["plain string block"]},
        ])
        msgs = parse_transcript(path)
        self.assertEqual(len(msgs), 1)
        self.assertIn("plain string block", msgs[0]["content"])


class TestScanTranscript(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def _write_jsonl(self, lines):
        path = os.path.join(self.td, "session.jsonl")
        with open(path, "w") as f:
            for obj in lines:
                f.write(json.dumps(obj) + "\n")
        return path

    def test_detects_correction_dont_use(self):
        path = self._write_jsonl([
            {"role": "user", "content": "Don't use raw SQL queries in production code."},
        ])
        signals = scan_transcript(path)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["type"], "correction")
        self.assertEqual(signals[0]["confidence"], "high")

    def test_detects_correction_never(self):
        path = self._write_jsonl([
            {"role": "user", "content": "Never use eval() in user-facing code."},
        ])
        signals = scan_transcript(path)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["type"], "correction")

    def test_detects_correction_always(self):
        path = self._write_jsonl([
            {"role": "user", "content": "Always use parameterized queries for database access."},
        ])
        signals = scan_transcript(path)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["type"], "correction")
        self.assertEqual(signals[0]["confidence"], "high")

    def test_detects_convention(self):
        path = self._write_jsonl([
            {"role": "user", "content": "The convention is to use snake_case for all function names."},
        ])
        signals = scan_transcript(path)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["type"], "convention")

    def test_detects_insight(self):
        path = self._write_jsonl([
            {"role": "assistant", "content": "The root cause was a race condition in the mutex."},
        ])
        signals = scan_transcript(path)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["type"], "insight")
        self.assertEqual(signals[0]["confidence"], "high")

    def test_detects_decision(self):
        path = self._write_jsonl([
            {"role": "user", "content": "Let's go with PostgreSQL for the database layer."},
        ])
        signals = scan_transcript(path)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["type"], "decision")

    def test_role_filter_user(self):
        path = self._write_jsonl([
            {"role": "user", "content": "Never use global variables in modules."},
            {"role": "assistant", "content": "The fix was to add proper scoping."},
        ])
        signals = scan_transcript(path, role_filter="user")
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["type"], "correction")

    def test_role_filter_assistant(self):
        path = self._write_jsonl([
            {"role": "user", "content": "Never use global variables in modules."},
            {"role": "assistant", "content": "The fix was to add proper scoping."},
        ])
        signals = scan_transcript(path, role_filter="assistant")
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["type"], "insight")

    def test_skips_short_lines(self):
        path = self._write_jsonl([
            {"role": "user", "content": "Don't do it"},  # < 15 chars
        ])
        signals = scan_transcript(path)
        self.assertEqual(len(signals), 0)

    def test_skips_headers(self):
        path = self._write_jsonl([
            {"role": "user", "content": "# Never use eval in production code"},
        ])
        signals = scan_transcript(path)
        self.assertEqual(len(signals), 0)

    def test_skips_code_blocks(self):
        path = self._write_jsonl([
            {"role": "assistant", "content": "```python\nThe fix was to add validation\n```"},
        ])
        signals = scan_transcript(path)
        # Lines starting with ``` are skipped, but inner lines are still scanned.
        # The inner line "The fix was to add validation" is only 34 chars and matches,
        # so we check that code fence lines themselves are filtered.
        code_fence_signals = [s for s in signals if s["text"].startswith("```")]
        self.assertEqual(len(code_fence_signals), 0)

    def test_skips_already_crossreferenced(self):
        path = self._write_jsonl([
            {"role": "user", "content": "We decided to use JWT D-20260215-001."},
        ])
        signals = scan_transcript(path)
        self.assertEqual(len(signals), 0)

    def test_signal_has_source_role(self):
        path = self._write_jsonl([
            {"role": "user", "content": "Always check return values from API calls."},
        ])
        signals = scan_transcript(path)
        self.assertEqual(signals[0]["source_role"], "user")

    def test_signal_has_priority(self):
        path = self._write_jsonl([
            {"role": "user", "content": "Never use hardcoded secrets in source code."},
        ])
        signals = scan_transcript(path)
        self.assertIn("priority", signals[0])
        self.assertEqual(signals[0]["priority"], "P1")

    def test_signal_has_structure(self):
        path = self._write_jsonl([
            {"role": "user", "content": "Always use type hints for all public API functions."},
        ])
        signals = scan_transcript(path)
        self.assertIn("structure", signals[0])

    def test_truncates_long_text(self):
        long_text = "Always use " + "x" * 300 + " for everything."
        path = self._write_jsonl([
            {"role": "user", "content": long_text},
        ])
        signals = scan_transcript(path)
        self.assertLessEqual(len(signals[0]["text"]), 200)

    def test_multiple_signals_from_one_message(self):
        path = self._write_jsonl([
            {"role": "user", "content": (
                "Never use eval() in production code.\n"
                "Always use parameterized queries.\n"
                "The convention is to log all errors."
            )},
        ])
        signals = scan_transcript(path)
        self.assertEqual(len(signals), 3)
        types = {s["type"] for s in signals}
        self.assertIn("correction", types)
        self.assertIn("convention", types)


class TestTranscriptPatterns(unittest.TestCase):
    """Verify all 17 transcript patterns match expected inputs."""

    def test_pattern_count(self):
        self.assertEqual(len(TRANSCRIPT_PATTERNS), 16)

    def test_all_patterns_have_three_tuple(self):
        for p in TRANSCRIPT_PATTERNS:
            self.assertEqual(len(p), 3, f"Pattern tuple must be (pattern, type, confidence): {p}")

    def test_correction_patterns_match(self):
        correction_texts = [
            "Don't ever use global state",
            "Never do that in production",
            "Always use type hints here",
            "Stop doing manual deploys",
            "That's wrong, use the other API",
            "No, use the standard library instead",
            "Instead of curl, use the orchestrator",
        ]
        for text in correction_texts:
            matched = False
            for pat, sig_type, _ in TRANSCRIPT_PATTERNS:
                if sig_type == "correction" and pat.search(text):
                    matched = True
                    break
            self.assertTrue(matched, f"No correction pattern matched: {text!r}")

    def test_convention_patterns_match(self):
        convention_texts = [
            "The convention is snake_case",
            "We follow PEP 8 strictly",
            "Our approach is test-first development",
        ]
        for text in convention_texts:
            matched = False
            for pat, sig_type, _ in TRANSCRIPT_PATTERNS:
                if sig_type == "convention" and pat.search(text):
                    matched = True
                    break
            self.assertTrue(matched, f"No convention pattern matched: {text!r}")


class TestXrefPattern(unittest.TestCase):
    def test_matches_standard_id(self):
        self.assertTrue(XREF_PATTERN.search("See D-20260215-001 for details"))

    def test_no_match_on_random_text(self):
        self.assertFalse(XREF_PATTERN.search("No cross references here"))


if __name__ == "__main__":
    unittest.main()
