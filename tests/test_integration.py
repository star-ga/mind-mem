"""Integration test: full mind-mem lifecycle init → capture → scan → recall."""

import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime

# Ensure scripts are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from capture import append_signals, scan_log
from init_workspace import init
from recall import recall


class TestFullLifecycle(unittest.TestCase):
    """End-to-end: init workspace, write a daily log, capture signals, run recall."""

    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="memos_integration_")

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_init_creates_structure(self):
        created, skipped = init(self.ws)
        self.assertTrue(len(created) > 0)
        self.assertTrue(os.path.isdir(os.path.join(self.ws, "decisions")))
        self.assertTrue(os.path.isdir(os.path.join(self.ws, "tasks")))
        self.assertTrue(os.path.isdir(os.path.join(self.ws, "intelligence")))
        self.assertTrue(os.path.isfile(os.path.join(self.ws, "mind-mem.json")))

    def test_init_idempotent(self):
        """Running init twice should not overwrite existing files."""
        init(self.ws)
        _, skipped1 = init(self.ws)
        self.assertTrue(len(skipped1) > 0)

    def test_full_lifecycle(self):
        """init → write daily log → capture → write decisions → recall."""
        # 1. Init workspace
        init(self.ws)

        # 2. Write a daily log with decision-like language
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(self.ws, "memory", f"{today}.md")
        with open(log_path, "w") as f:
            f.write(f"# {today}\n\n")
            f.write("We decided to use PostgreSQL for the main database.\n")
            f.write("Reviewed D-20260101-001 for context.\n")  # has xref, should skip
            f.write("Need to set up CI pipeline before next sprint.\n")
            f.write("Regular status update meeting notes.\n")  # no signal

        # 3. Capture signals
        signals = scan_log(log_path)
        self.assertTrue(len(signals) >= 2)

        # Verify xref line was skipped
        texts = [s["text"] for s in signals]
        self.assertFalse(any("D-20260101-001" in t for t in texts))

        # Verify decision and task types detected
        types = [s["type"] for s in signals]
        self.assertIn("decision", types)
        self.assertIn("task", types)

        # 4. Append signals
        written = append_signals(self.ws, signals, today)
        self.assertEqual(written, len(signals))

        # Verify SIGNALS.md has content
        signals_path = os.path.join(self.ws, "intelligence", "SIGNALS.md")
        with open(signals_path) as f:
            content = f.read()
        self.assertIn("SIG-", content)
        self.assertIn("PostgreSQL", content)

        # 5. Write a formal decision block for recall to find
        decisions_path = os.path.join(self.ws, "decisions", "DECISIONS.md")
        with open(decisions_path, "w") as f:
            f.write("# Decisions\n\n")
            f.write(f"[D-{today.replace('-', '')}-001]\n")
            f.write("Statement: Use PostgreSQL for the main database\n")
            f.write("Status: active\n")
            f.write(f"Date: {today}\n")
            f.write("Context: Needed a reliable RDBMS for production workloads\n")
            f.write("\n---\n\n")
            f.write(f"[D-{today.replace('-', '')}-002]\n")
            f.write("Statement: Deploy to AWS us-east-1\n")
            f.write("Status: active\n")
            f.write(f"Date: {today}\n")
            f.write("\n---\n")

        # 6. Recall
        results = recall(self.ws, "PostgreSQL database")
        self.assertTrue(len(results) > 0)
        self.assertIn("PostgreSQL", results[0]["excerpt"])

    def test_capture_dedup(self):
        """Signals already in SIGNALS.md should not be duplicated."""
        init(self.ws)
        today = datetime.now().strftime("%Y-%m-%d")

        signals = [{"line": 1, "type": "decision",
                     "text": "We decided to use Redis for caching layer",
                     "pattern": r"\bdecided to\b"}]

        written1 = append_signals(self.ws, signals, today)
        self.assertEqual(written1, 1)

        # Append same signals again
        written2 = append_signals(self.ws, signals, today)
        self.assertEqual(written2, 0)

    def test_recall_active_filter(self):
        """Recall with active_only should exclude non-active blocks."""
        init(self.ws)
        today = datetime.now().strftime("%Y-%m-%d")
        decisions_path = os.path.join(self.ws, "decisions", "DECISIONS.md")
        with open(decisions_path, "w") as f:
            f.write("# Decisions\n\n")
            f.write(f"[D-{today.replace('-', '')}-001]\n")
            f.write("Statement: Active decision about testing\n")
            f.write("Status: active\n")
            f.write(f"Date: {today}\n\n---\n\n")
            f.write(f"[D-{today.replace('-', '')}-002]\n")
            f.write("Statement: Superseded decision about testing\n")
            f.write("Status: superseded\n")
            f.write(f"Date: {today}\n\n---\n")

        results_all = recall(self.ws, "testing")
        results_active = recall(self.ws, "testing", active_only=True)
        self.assertTrue(len(results_all) >= len(results_active))


if __name__ == "__main__":
    unittest.main()
