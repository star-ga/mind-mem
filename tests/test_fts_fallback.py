#!/usr/bin/env python3
"""Tests for FTS fallback behavior, recall envelope structure, block size cap,
and config key validation in mind-mem.

Covers:
  1. FTS fallback — recall returns envelope with warnings when no FTS5 index exists
  2. Empty results message — envelope contains helpful message on zero matches
  3. Recall envelope structure — consistent shape across queries
  4. Block size cap in parser — parse_file truncates files >100KB without crash
  5. Config key validation — _load_backend warns on unknown keys and returns None
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

# Add scripts/ to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from block_parser import MAX_PARSE_SIZE, parse_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SERVER_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp_server.py")
_HAS_FASTMCP = importlib.util.find_spec("fastmcp") is not None


def _load_server(workspace: str):
    """Load the mcp_server module with a given workspace."""
    os.environ["MIND_MEM_WORKSPACE"] = workspace
    spec = importlib.util.spec_from_file_location("mcp_server", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_workspace(tmpdir, decisions_content="", config=None):
    """Create a minimal workspace with standard directories and files.

    Args:
        tmpdir: Root directory for the workspace.
        decisions_content: Content for decisions/DECISIONS.md.
        config: Optional dict to write as mind-mem.json.

    Returns:
        The workspace path (same as tmpdir).
    """
    for d in ["decisions", "tasks", "entities", "intelligence", "memory"]:
        os.makedirs(os.path.join(tmpdir, d), exist_ok=True)

    with open(os.path.join(tmpdir, "decisions", "DECISIONS.md"), "w") as f:
        f.write(decisions_content)
    with open(os.path.join(tmpdir, "tasks", "TASKS.md"), "w") as f:
        f.write("[T-20260213-099]\nTitle: Placeholder task\nStatus: active\n")

    for fname in [
        "entities/projects.md",
        "entities/people.md",
        "entities/tools.md",
        "entities/incidents.md",
        "intelligence/CONTRADICTIONS.md",
        "intelligence/DRIFT.md",
        "intelligence/SIGNALS.md",
    ]:
        path = os.path.join(tmpdir, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"# {os.path.basename(fname)}\n")

    if config is not None:
        with open(os.path.join(tmpdir, "mind-mem.json"), "w") as f:
            json.dump(config, f)

    return tmpdir


def _call_tool(fn, *args, **kwargs):
    """Call an MCP tool function, unwrapping FastMCP wrapper if needed."""
    if hasattr(fn, "fn"):
        return fn.fn(*args, **kwargs)
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# 1. FTS fallback behavior
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestFTSFallback(unittest.TestCase):
    """When no FTS5 index exists, recall should return an envelope with a
    warnings field mentioning the fallback."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        decisions = (
            "[D-20260101-001]\n"
            "Statement: Use PostgreSQL for the primary database\n"
            "Status: active\n"
            "Tags: database, infrastructure\n"
        )
        _make_workspace(self.td, decisions_content=decisions)
        # Ensure no FTS5 database exists
        for candidate in ["memory/fts.db", "memory/recall.db", ".mind-mem/fts.db"]:
            path = os.path.join(self.td, candidate)
            if os.path.exists(path):
                os.remove(path)
        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_fallback_envelope_has_warnings(self):
        """recall returns warnings list mentioning FTS5 when no index exists."""
        result = _call_tool(self.mod.recall, query="PostgreSQL", limit=5)
        envelope = json.loads(result)
        self.assertIn("warnings", envelope)
        self.assertIsInstance(envelope["warnings"], list)
        self.assertGreater(len(envelope["warnings"]), 0)
        # At least one warning should mention FTS5
        joined = " ".join(envelope["warnings"])
        self.assertIn("FTS5", joined)

    def test_fallback_envelope_schema_version(self):
        """Envelope _schema_version is '1.0'."""
        result = _call_tool(self.mod.recall, query="PostgreSQL", limit=5)
        envelope = json.loads(result)
        self.assertEqual(envelope["_schema_version"], "1.0")

    def test_fallback_backend_is_scan(self):
        """When no FTS5 index exists, backend should be 'scan'."""
        result = _call_tool(self.mod.recall, query="PostgreSQL", limit=5)
        envelope = json.loads(result)
        self.assertEqual(envelope["backend"], "scan")

    def test_fallback_results_is_list(self):
        """Results field is always a list, even in fallback mode."""
        result = _call_tool(self.mod.recall, query="PostgreSQL", limit=5)
        envelope = json.loads(result)
        self.assertIsInstance(envelope["results"], list)


# ---------------------------------------------------------------------------
# 2. Empty results message
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestEmptyResultsMessage(unittest.TestCase):
    """When recall returns zero matches, the envelope should contain a helpful
    message field."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        decisions = (
            "[D-20260101-001]\n"
            "Statement: Use PostgreSQL for primary database\n"
            "Status: active\n"
        )
        _make_workspace(self.td, decisions_content=decisions)
        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_no_results_contains_message(self):
        """Searching for a term that does not exist produces a message field."""
        result = _call_tool(self.mod.recall, query="xyznonexistentterm12345")
        envelope = json.loads(result)
        self.assertEqual(envelope["count"], 0)
        self.assertEqual(len(envelope["results"]), 0)
        self.assertIn("message", envelope)
        self.assertIsInstance(envelope["message"], str)
        self.assertGreater(len(envelope["message"]), 0)

    def test_empty_query_contains_message(self):
        """Empty query returns zero results and a message."""
        result = _call_tool(self.mod.recall, query="")
        envelope = json.loads(result)
        self.assertEqual(envelope["count"], 0)
        self.assertIn("message", envelope)

    def test_stopword_only_query_contains_message(self):
        """Query with only stopwords returns zero results and a message."""
        result = _call_tool(self.mod.recall, query="the a an is")
        envelope = json.loads(result)
        self.assertEqual(envelope["count"], 0)
        self.assertIn("message", envelope)


# ---------------------------------------------------------------------------
# 3. Recall envelope structure
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestRecallEnvelopeStructure(unittest.TestCase):
    """Various queries should always return envelopes with a consistent set
    of top-level keys."""

    REQUIRED_KEYS = {"_schema_version", "backend", "count", "results"}

    def setUp(self):
        self.td = tempfile.mkdtemp()
        decisions = (
            "[D-20260101-001]\n"
            "Statement: Use PostgreSQL for authentication storage\n"
            "Status: active\n"
            "Tags: database, auth\n"
            "\n---\n\n"
            "[D-20260102-001]\n"
            "Statement: Deploy Redis for session caching\n"
            "Status: active\n"
            "Tags: caching, performance\n"
        )
        _make_workspace(self.td, decisions_content=decisions)
        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _assert_envelope_shape(self, query):
        """Helper: call recall and verify required keys."""
        result = _call_tool(self.mod.recall, query=query, limit=10)
        envelope = json.loads(result)
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, envelope, f"Missing key '{key}' for query '{query}'")
        self.assertIsInstance(envelope["results"], list)
        self.assertIsInstance(envelope["count"], int)
        self.assertEqual(envelope["count"], len(envelope["results"]))
        return envelope

    def test_matching_query_envelope(self):
        """Query that matches blocks has correct envelope shape."""
        envelope = self._assert_envelope_shape("PostgreSQL")
        self.assertGreater(envelope["count"], 0)

    def test_nonmatching_query_envelope(self):
        """Query with no matches still has correct envelope shape."""
        envelope = self._assert_envelope_shape("kubernetes_nonexistent")
        self.assertEqual(envelope["count"], 0)

    def test_broad_query_envelope(self):
        """Broad single-word query has correct envelope shape."""
        self._assert_envelope_shape("database")

    def test_multi_word_query_envelope(self):
        """Multi-word query has correct envelope shape."""
        self._assert_envelope_shape("session caching Redis")

    def test_special_characters_query_envelope(self):
        """Query with regex-special characters has correct envelope shape."""
        self._assert_envelope_shape("[test](.*)")

    def test_schema_version_value(self):
        """_schema_version is always '1.0'."""
        envelope = self._assert_envelope_shape("auth")
        self.assertEqual(envelope["_schema_version"], "1.0")


# ---------------------------------------------------------------------------
# 4. Block size cap in parser
# ---------------------------------------------------------------------------

class TestBlockSizeCap(unittest.TestCase):
    """parse_file should handle files larger than MAX_PARSE_SIZE (100KB)
    by truncating without crashing."""

    def test_oversized_file_truncated_without_crash(self):
        """A file >100KB is truncated and still returns blocks where possible."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            # Write a valid block at the start
            f.write(
                "[D-20260101-001]\n"
                "Statement: First block before the overflow\n"
                "Status: active\n"
                "\n---\n\n"
            )
            # Pad with enough content to exceed 100KB
            # Each line is ~80 chars; need >100_000 bytes total
            padding_lines = (MAX_PARSE_SIZE // 40) + 100
            for i in range(padding_lines):
                f.write(f"# padding line {i:06d} " + "x" * 20 + "\n")
            tmp_path = f.name

        try:
            self.assertGreater(os.path.getsize(tmp_path), MAX_PARSE_SIZE)
            # parse_file should not raise
            blocks = parse_file(tmp_path)
            self.assertIsInstance(blocks, list)
            # The first block should have been parsed before truncation
            self.assertGreater(len(blocks), 0)
            self.assertEqual(blocks[0]["_id"], "D-20260101-001")
        finally:
            os.unlink(tmp_path)

    def test_exactly_at_limit_file(self):
        """A file of exactly MAX_PARSE_SIZE bytes is parsed without truncation."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            header = "[D-20260101-001]\nStatement: Exact limit test\nStatus: active\n"
            f.write(header)
            remaining = MAX_PARSE_SIZE - len(header.encode("utf-8"))
            if remaining > 0:
                f.write("x" * remaining)
            tmp_path = f.name

        try:
            blocks = parse_file(tmp_path)
            self.assertIsInstance(blocks, list)
            self.assertGreater(len(blocks), 0)
        finally:
            os.unlink(tmp_path)

    def test_small_file_unaffected(self):
        """Files well under the limit are parsed normally."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                "[D-20260101-001]\n"
                "Statement: Small file test\n"
                "Status: active\n"
                "\n---\n\n"
                "[D-20260101-002]\n"
                "Statement: Second block\n"
                "Status: active\n"
            )
            tmp_path = f.name

        try:
            blocks = parse_file(tmp_path)
            self.assertEqual(len(blocks), 2)
        finally:
            os.unlink(tmp_path)

    def test_oversized_file_with_block_at_boundary(self):
        """A block that starts just before the 100KB boundary may be
        partially parsed, but the parser must not crash."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            # Fill past the boundary with filler, then add a block
            filler_line = "# Filler line padding content here\n"  # 36 bytes
            repeats = (MAX_PARSE_SIZE // len(filler_line)) + 10
            f.write(filler_line * repeats)
            # Start a block after the boundary
            f.write(
                "\n---\n\n"
                "[D-20260101-099]\n"
                "Statement: Block near boundary that may be truncated\n"
                "Status: active\n"
                "Tags: boundary, test\n"
            )
            tmp_path = f.name

        try:
            self.assertGreater(os.path.getsize(tmp_path), MAX_PARSE_SIZE)
            blocks = parse_file(tmp_path)
            self.assertIsInstance(blocks, list)
            # We do not assert the boundary block is fully parsed;
            # the important thing is no crash.
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 5. Config key validation warning
# ---------------------------------------------------------------------------

class TestConfigKeyValidation(unittest.TestCase):
    """_load_backend should return None and not crash when mind-mem.json
    contains unknown keys in the recall section."""

    def test_unknown_keys_returns_none(self):
        """Unknown recall config keys cause _load_backend to return None
        (default BM25 scan) without raising."""
        from recall import _load_backend

        with tempfile.TemporaryDirectory() as td:
            config = {
                "recall": {
                    "backend": "scan",
                    "bogus_unknown_key": True,
                    "another_invalid": 42,
                }
            }
            with open(os.path.join(td, "mind-mem.json"), "w") as f:
                json.dump(config, f)

            result = _load_backend(td)
            # "scan" backend returns None (use built-in BM25)
            self.assertIsNone(result)

    def test_valid_keys_only_returns_none_for_scan(self):
        """All valid keys with backend=scan still returns None (built-in BM25)."""
        from recall import _load_backend

        with tempfile.TemporaryDirectory() as td:
            config = {
                "recall": {
                    "backend": "scan",
                    "limit": 10,
                    "rerank": True,
                }
            }
            with open(os.path.join(td, "mind-mem.json"), "w") as f:
                json.dump(config, f)

            result = _load_backend(td)
            self.assertIsNone(result)

    def test_no_config_file_returns_none(self):
        """Missing mind-mem.json returns None without error."""
        from recall import _load_backend

        with tempfile.TemporaryDirectory() as td:
            result = _load_backend(td)
            self.assertIsNone(result)

    def test_empty_recall_section_returns_none(self):
        """Empty recall section defaults to scan (None)."""
        from recall import _load_backend

        with tempfile.TemporaryDirectory() as td:
            config = {"recall": {}}
            with open(os.path.join(td, "mind-mem.json"), "w") as f:
                json.dump(config, f)

            result = _load_backend(td)
            self.assertIsNone(result)

    def test_malformed_json_returns_none(self):
        """Malformed JSON in mind-mem.json returns None without crash."""
        from recall import _load_backend

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "mind-mem.json"), "w") as f:
                f.write("{invalid json content!!!")

            result = _load_backend(td)
            self.assertIsNone(result)

    def test_sqlite_backend_returns_sqlite(self):
        """backend=sqlite returns the string 'sqlite'."""
        from recall import _load_backend

        with tempfile.TemporaryDirectory() as td:
            config = {"recall": {"backend": "sqlite"}}
            with open(os.path.join(td, "mind-mem.json"), "w") as f:
                json.dump(config, f)

            result = _load_backend(td)
            self.assertEqual(result, "sqlite")


if __name__ == "__main__":
    unittest.main()
