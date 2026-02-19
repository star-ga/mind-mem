#!/usr/bin/env python3
"""Edge-case and stress tests for mind-mem — block_parser, recall, and MCP server."""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from block_parser import parse_blocks
from recall import extract_text, recall, tokenize

# ---------------------------------------------------------------------------
# Block parser edge cases
# ---------------------------------------------------------------------------

class TestBlockParserEdgeCases(unittest.TestCase):

    def test_empty_file(self):
        """Empty string produces zero blocks."""
        self.assertEqual(parse_blocks(""), [])

    def test_whitespace_only(self):
        """File containing only whitespace produces zero blocks."""
        self.assertEqual(parse_blocks("   \n\n  \t\n  \n"), [])

    def test_block_with_no_fields(self):
        """A block with only an ID line and no fields should still parse."""
        blocks = parse_blocks("[D-20260215-001]\n")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["_id"], "D-20260215-001")
        # Should have _id and _line only
        non_internal = {k: v for k, v in blocks[0].items() if not k.startswith("_")}
        self.assertEqual(len(non_internal), 0)

    def test_unicode_statements(self):
        """Blocks with unicode characters in statements should parse correctly."""
        text = (
            "[D-20260215-001]\n"
            "Statement: Utiliser l'encodage UTF-8 pour tous les fichiers\n"
            "Status: active\n"
            "\n---\n\n"
            "[D-20260215-002]\n"
            "Statement: \u4f7f\u7528PostgreSQL\u4f5c\u4e3a\u4e3b\u6570\u636e\u5e93\n"
            "Status: active\n"
            "\n---\n\n"
            "[D-20260215-003]\n"
            "Statement: \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u044c JWT \u0434\u043b\u044f \u0430\u0443\u0442\u0435\u043d\u0442\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u0438\n"
            "Status: active\n"
        )
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 3)
        self.assertIn("UTF-8", blocks[0]["Statement"])
        self.assertIn("PostgreSQL", blocks[1]["Statement"])
        self.assertIn("JWT", blocks[2]["Statement"])

    def test_extremely_long_field_values(self):
        """Block with a 10000-char field value should parse without error."""
        long_val = "x" * 10000
        text = f"[D-20260215-001]\nStatement: {long_val}\nStatus: active\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]["Statement"]), 10000)

    def test_malformed_id_missing_brackets(self):
        """ID without brackets should not be parsed as a block."""
        text = "D-20260215-001\nStatement: Missing brackets\nStatus: active\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 0)

    def test_malformed_id_lowercase(self):
        """Lowercase ID prefix should not match the block header pattern."""
        text = "[d-20260215-001]\nStatement: Lowercase prefix\nStatus: active\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 0)

    def test_multiple_consecutive_separators(self):
        """Multiple --- lines in a row should not produce phantom blocks."""
        text = (
            "[D-20260215-001]\nStatement: First\nStatus: active\n"
            "\n---\n---\n---\n---\n\n"
            "[D-20260215-002]\nStatement: Second\nStatus: active\n"
        )
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["_id"], "D-20260215-001")
        self.assertEqual(blocks[1]["_id"], "D-20260215-002")

    def test_duplicate_field_names(self):
        """When a field name appears twice, the last value wins."""
        text = (
            "[D-20260215-001]\n"
            "Statement: First statement\n"
            "Status: active\n"
            "Statement: Second statement\n"
        )
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        # The parser overwrites, so we should get the second value
        self.assertEqual(blocks[0]["Statement"], "Second statement")

    def test_block_id_with_extra_text_after_bracket(self):
        """Text after closing bracket should prevent block match."""
        text = "[D-20260215-001] extra text\nStatement: Should not parse\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 0)

    def test_hash_prefixed_id_ignored(self):
        """## [ID] header should not be parsed as a block."""
        text = "## [D-20260215-001]\nStatement: Should not parse\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 0)

    def test_separator_at_start_of_file(self):
        """Separator before any block should not cause errors."""
        text = "---\n\n[D-20260215-001]\nStatement: After separator\nStatus: active\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["_id"], "D-20260215-001")


# ---------------------------------------------------------------------------
# Recall edge cases
# ---------------------------------------------------------------------------

class TestRecallEdgeCases(unittest.TestCase):

    def _setup_workspace(self, tmpdir, decisions_content=""):
        """Create a minimal workspace for recall testing."""
        for d in ["decisions", "tasks", "entities", "intelligence"]:
            os.makedirs(os.path.join(tmpdir, d), exist_ok=True)
        with open(os.path.join(tmpdir, "decisions", "DECISIONS.md"), "w") as f:
            f.write(decisions_content)
        with open(os.path.join(tmpdir, "tasks", "TASKS.md"), "w") as f:
            f.write("[T-20260213-099]\nTitle: Unrelated placeholder task\nStatus: active\n")
        for fname in ["entities/projects.md", "entities/people.md",
                       "entities/tools.md", "entities/incidents.md",
                       "intelligence/CONTRADICTIONS.md", "intelligence/DRIFT.md",
                       "intelligence/SIGNALS.md"]:
            path = os.path.join(tmpdir, fname)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(f"# {os.path.basename(fname)}\n")
        return tmpdir

    def test_empty_workspace_no_md_files(self):
        """Workspace with no .md files should return empty results."""
        with tempfile.TemporaryDirectory() as td:
            # Create empty dirs but no .md files
            for d in ["decisions", "tasks", "entities", "intelligence"]:
                os.makedirs(os.path.join(td, d), exist_ok=True)
            results = recall(td, "anything")
            self.assertEqual(results, [])

    def test_query_with_only_stopwords(self):
        """Query consisting entirely of stopwords should return empty."""
        with tempfile.TemporaryDirectory() as td:
            ws = self._setup_workspace(td, "[D-20260215-001]\nStatement: Use JWT\nStatus: active\n")
            results = recall(ws, "the a an is")
            self.assertEqual(results, [])

    def test_query_with_special_regex_characters(self):
        """Regex special characters in query should not cause errors."""
        with tempfile.TemporaryDirectory() as td:
            ws = self._setup_workspace(td, "[D-20260215-001]\nStatement: Use foo.bar pattern\nStatus: active\n")
            # Should not raise, even though query has regex metacharacters
            results = recall(ws, "foo.*bar")
            self.assertIsInstance(results, list)

    def test_very_long_query(self):
        """Very long query (500+ words) should not crash."""
        with tempfile.TemporaryDirectory() as td:
            ws = self._setup_workspace(td, "[D-20260215-001]\nStatement: Use JWT\nStatus: active\n")
            long_query = " ".join(f"word{i}" for i in range(500))
            results = recall(ws, long_query)
            self.assertIsInstance(results, list)

    def test_workspace_with_empty_md_files(self):
        """Workspace where all .md files are empty should return empty results."""
        with tempfile.TemporaryDirectory() as td:
            for d in ["decisions", "tasks", "entities", "intelligence"]:
                os.makedirs(os.path.join(td, d), exist_ok=True)
            # Create empty files
            for fname in ["decisions/DECISIONS.md", "tasks/TASKS.md",
                           "entities/projects.md", "entities/people.md",
                           "entities/tools.md", "entities/incidents.md",
                           "intelligence/CONTRADICTIONS.md", "intelligence/DRIFT.md",
                           "intelligence/SIGNALS.md"]:
                with open(os.path.join(td, fname), "w") as f:
                    f.write("")
            results = recall(td, "database")
            self.assertEqual(results, [])

    def test_search_with_limit_zero(self):
        """limit=0 should return no results."""
        with tempfile.TemporaryDirectory() as td:
            content = "[D-20260215-001]\nStatement: Use JWT for authentication\nStatus: active\n"
            ws = self._setup_workspace(td, content)
            results = recall(ws, "JWT", limit=0)
            self.assertEqual(results, [])

    def test_search_with_very_large_limit(self):
        """limit=1000 should not crash even when fewer blocks exist."""
        with tempfile.TemporaryDirectory() as td:
            content = "[D-20260215-001]\nStatement: Use JWT\nStatus: active\n"
            ws = self._setup_workspace(td, content)
            results = recall(ws, "JWT", limit=1000)
            self.assertIsInstance(results, list)
            # Should return whatever matches exist, not error
            self.assertLessEqual(len(results), 1000)

    def test_tokenize_empty_string(self):
        """Tokenizing empty string returns empty list."""
        self.assertEqual(tokenize(""), [])

    def test_tokenize_only_stopwords(self):
        """Tokenizing only stopwords returns empty list."""
        self.assertEqual(tokenize("the a an is are was"), [])

    def test_extract_text_empty_block(self):
        """extract_text on empty dict returns empty-ish string."""
        result = extract_text({})
        self.assertEqual(result.strip(), "")


# ---------------------------------------------------------------------------
# MCP server edge cases
# ---------------------------------------------------------------------------

_SERVER_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp_server.py")
_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load_server(workspace: str):
    """Load the mcp_server module with a given workspace."""
    os.environ["MIND_MEM_WORKSPACE"] = workspace
    spec = importlib.util.spec_from_file_location("mcp_server", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(
    importlib.util.find_spec("fastmcp") is not None,
    "fastmcp not installed"
)
class TestMCPEdgeCases(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        os.makedirs(os.path.join(self.td, "tasks"))
        os.makedirs(os.path.join(self.td, "entities"))
        os.makedirs(os.path.join(self.td, "intelligence"))
        os.makedirs(os.path.join(self.td, "memory"))

        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write("[D-20260101-001]\nStatement: Use PostgreSQL\nStatus: active\n")
        with open(os.path.join(self.td, "tasks", "TASKS.md"), "w") as f:
            f.write("[T-20260101-001]\nDescription: Setup DB\nStatus: open\n")
        for fname in ["entities/projects.md", "entities/people.md",
                       "entities/tools.md", "entities/incidents.md",
                       "intelligence/CONTRADICTIONS.md", "intelligence/DRIFT.md",
                       "intelligence/SIGNALS.md"]:
            path = os.path.join(self.td, fname)
            with open(path, "w") as f:
                f.write(f"# {os.path.basename(fname)}\n")

        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _call_tool(self, fn, *args, **kwargs):
        """Call an MCP tool function, unwrapping FastMCP wrapper if needed."""
        if hasattr(fn, "fn"):
            return fn.fn(*args, **kwargs)
        return fn(*args, **kwargs)

    def test_propose_update_empty_statement(self):
        """propose_update with empty statement should not crash."""
        result = self._call_tool(
            self.mod.propose_update,
            block_type="decision",
            statement="",
        )
        parsed = json.loads(result)
        # Should either succeed (writing an empty signal) or return an error,
        # but must not raise an exception
        self.assertIsInstance(parsed, dict)

    def test_propose_update_very_long_statement(self):
        """propose_update with a 5000-char statement should not crash."""
        long_stmt = "x" * 5000
        result = self._call_tool(
            self.mod.propose_update,
            block_type="decision",
            statement=long_stmt,
        )
        parsed = json.loads(result)
        self.assertIsInstance(parsed, dict)
        # Should succeed in proposing
        self.assertIn("status", parsed)

    def test_propose_update_invalid_block_type(self):
        """propose_update with invalid block_type should return an error."""
        result = self._call_tool(
            self.mod.propose_update,
            block_type="invalid_type",
            statement="Some statement",
        )
        parsed = json.loads(result)
        self.assertIn("error", parsed)

    def test_propose_update_invalid_confidence(self):
        """propose_update with invalid confidence should still work (defaults to P2)."""
        result = self._call_tool(
            self.mod.propose_update,
            block_type="decision",
            statement="Some statement",
            confidence="invalid_level",
        )
        parsed = json.loads(result)
        # Should succeed because CONFIDENCE_TO_PRIORITY.get() returns "P2" as default
        self.assertIsInstance(parsed, dict)

    def test_recall_with_empty_query(self):
        """MCP recall tool with empty query should return empty results."""
        result = self._call_tool(self.mod.recall, query="")
        parsed = json.loads(result)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 0)

    def test_recall_with_special_characters(self):
        """MCP recall with regex special chars should not crash."""
        result = self._call_tool(self.mod.recall, query="[test](.*)")
        parsed = json.loads(result)
        self.assertIsInstance(parsed, list)

    def test_scan_on_minimal_workspace(self):
        """scan tool should work on a minimal workspace."""
        result = self._call_tool(self.mod.scan)
        parsed = json.loads(result)
        self.assertIn("checks", parsed)
        self.assertIn("decisions", parsed["checks"])


if __name__ == "__main__":
    unittest.main()
