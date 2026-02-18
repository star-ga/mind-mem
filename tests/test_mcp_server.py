#!/usr/bin/env python3
"""Tests for mcp_server.py — tests the MCP server resources and tool logic.

Uses importlib to load the server module and tests the underlying functions.
FastMCP wraps these into MCP protocol handlers; we test the logic layer.
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

# Load the mcp_server module
_SERVER_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp_server.py")
_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS_DIR)

_HAS_FASTMCP = importlib.util.find_spec("fastmcp") is not None


def _load_server(workspace: str):
    """Load the mcp_server module with a given workspace."""
    os.environ["MIND_MEM_WORKSPACE"] = workspace
    spec = importlib.util.spec_from_file_location("mcp_server", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestMCPServerHelpers(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        os.makedirs(os.path.join(self.td, "tasks"))
        os.makedirs(os.path.join(self.td, "entities"))
        os.makedirs(os.path.join(self.td, "intelligence"))
        os.makedirs(os.path.join(self.td, "memory"))

        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL for user database\n"
                "Status: active\n"
                "Date: 2026-01-01\n"
                "\n---\n\n"
                "[D-20260102-001]\n"
                "Statement: Use Redis for caching\n"
                "Status: superseded\n"
                "Date: 2026-01-02\n"
            )

        with open(os.path.join(self.td, "tasks", "TASKS.md"), "w") as f:
            f.write(
                "[T-20260101-001]\n"
                "Description: Set up PostgreSQL in staging\n"
                "Status: open\n"
            )

        with open(os.path.join(self.td, "entities", "projects.md"), "w") as f:
            f.write("[PRJ-001]\nName: mind-mem\nStatus: active\n")

        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_workspace_resolution(self):
        self.assertEqual(self.mod._workspace(), os.path.abspath(self.td))

    def test_read_file_exists(self):
        content = self.mod._read_file("decisions/DECISIONS.md")
        self.assertIn("PostgreSQL", content)

    def test_read_file_missing(self):
        content = self.mod._read_file("nonexistent.md")
        self.assertIn("not found", content.lower())

    def test_blocks_to_json(self):
        blocks = [{"_id": "D-001", "Statement": "Test"}]
        result = self.mod._blocks_to_json(blocks)
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["_id"], "D-001")


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestMCPResources(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        os.makedirs(os.path.join(self.td, "tasks"))
        os.makedirs(os.path.join(self.td, "entities"))
        os.makedirs(os.path.join(self.td, "intelligence"))
        os.makedirs(os.path.join(self.td, "memory"))

        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL\n"
                "Status: active\n"
                "Date: 2026-01-01\n"
                "\n---\n\n"
                "[D-20260102-001]\n"
                "Statement: Use Redis\n"
                "Status: superseded\n"
            )

        with open(os.path.join(self.td, "tasks", "TASKS.md"), "w") as f:
            f.write("[T-20260101-001]\nDescription: Setup DB\nStatus: open\n")

        with open(os.path.join(self.td, "entities", "projects.md"), "w") as f:
            f.write("[PRJ-001]\nName: mind-mem\n")

        with open(os.path.join(self.td, "intelligence", "SIGNALS.md"), "w") as f:
            f.write("# Signals\n\nNo signals yet.\n")

        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_get_decisions_returns_active_only(self):
        # The resource function is wrapped by FastMCP, access the original
        fn = self.mod.get_decisions
        if hasattr(fn, "fn"):
            result = fn.fn()
        else:
            result = fn()
        blocks = json.loads(result)
        # Should only return active decisions
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["Statement"], "Use PostgreSQL")

    def test_get_tasks(self):
        fn = self.mod.get_tasks
        if hasattr(fn, "fn"):
            result = fn.fn()
        else:
            result = fn()
        blocks = json.loads(result)
        self.assertEqual(len(blocks), 1)

    def test_get_entities_valid(self):
        fn = self.mod.get_entities
        if hasattr(fn, "fn"):
            result = fn.fn("projects")
        else:
            result = fn("projects")
        self.assertIn("mind-mem", result)

    def test_get_entities_invalid(self):
        fn = self.mod.get_entities
        if hasattr(fn, "fn"):
            result = fn.fn("invalid_type")
        else:
            result = fn("invalid_type")
        parsed = json.loads(result)
        self.assertIn("error", parsed)

    def test_get_health(self):
        fn = self.mod.get_health
        if hasattr(fn, "fn"):
            result = fn.fn()
        else:
            result = fn()
        health = json.loads(result)
        self.assertIn("workspace", health)
        self.assertIn("files", health)
        self.assertIn("decisions", health["files"])
        self.assertEqual(health["files"]["decisions"]["total"], 2)

    def test_get_signals(self):
        fn = self.mod.get_signals
        if hasattr(fn, "fn"):
            result = fn.fn()
        else:
            result = fn()
        self.assertIn("Signals", result)


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestMCPRecallEngine(unittest.TestCase):
    """Test recall through the module's internal recall engine."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        os.makedirs(os.path.join(self.td, "tasks"))
        os.makedirs(os.path.join(self.td, "entities"))
        os.makedirs(os.path.join(self.td, "intelligence"))

        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL for the primary database\n"
                "Status: active\n"
                "Date: 2026-01-01\n"
                "Tags: database, infrastructure\n"
            )

        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_recall_via_engine(self):
        from recall import recall as r
        results = r(self.td, "PostgreSQL", limit=5)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["_id"], "D-20260101-001")

    def test_recall_no_results(self):
        from recall import recall as r
        results = r(self.td, "kubernetes", limit=5)
        self.assertEqual(len(results), 0)


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestMCPApproveApply(unittest.TestCase):
    """Test the approve_apply MCP tool."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        os.makedirs(os.path.join(self.td, "tasks"))
        os.makedirs(os.path.join(self.td, "entities"))
        os.makedirs(os.path.join(self.td, "intelligence", "proposed"), exist_ok=True)
        os.makedirs(os.path.join(self.td, "intelligence", "applied"), exist_ok=True)
        os.makedirs(os.path.join(self.td, "memory"))

        # Create empty decisions file
        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write("# Decisions\n")

        # Create mind-mem.json with detect_only mode (blocks apply)
        with open(os.path.join(self.td, "mind-mem.json"), "w") as f:
            json.dump({"governance_mode": "detect_only"}, f)

        # Create empty proposed files
        for name in ("DECISIONS_PROPOSED.md", "TASKS_PROPOSED.md", "EDITS_PROPOSED.md"):
            with open(os.path.join(self.td, "intelligence", "proposed", name), "w") as f:
                f.write(f"# {name}\n")

        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_invalid_proposal_id_format(self):
        fn = self.mod.approve_apply
        if hasattr(fn, "fn"):
            result = fn.fn("bad-id", dry_run=True)
        else:
            result = fn("bad-id", dry_run=True)
        parsed = json.loads(result)
        self.assertIn("error", parsed)
        self.assertIn("Invalid proposal ID", parsed["error"])

    def test_nonexistent_proposal(self):
        fn = self.mod.approve_apply
        if hasattr(fn, "fn"):
            result = fn.fn("P-20260101-999", dry_run=True)
        else:
            result = fn("P-20260101-999", dry_run=True)
        parsed = json.loads(result)
        self.assertFalse(parsed["success"])

    def test_defaults_to_dry_run(self):
        fn = self.mod.approve_apply
        if hasattr(fn, "fn"):
            result = fn.fn("P-20260101-001")
        else:
            result = fn("P-20260101-001")
        parsed = json.loads(result)
        self.assertTrue(parsed["dry_run"])


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestMCPRollback(unittest.TestCase):
    """Test the rollback_proposal MCP tool."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        os.makedirs(os.path.join(self.td, "intelligence", "applied"), exist_ok=True)
        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_invalid_timestamp_format(self):
        fn = self.mod.rollback_proposal
        if hasattr(fn, "fn"):
            result = fn.fn("bad-timestamp")
        else:
            result = fn("bad-timestamp")
        parsed = json.loads(result)
        self.assertIn("error", parsed)
        self.assertIn("Invalid receipt timestamp", parsed["error"])

    def test_nonexistent_snapshot(self):
        fn = self.mod.rollback_proposal
        if hasattr(fn, "fn"):
            result = fn.fn("20260101-120000")
        else:
            result = fn("20260101-120000")
        parsed = json.loads(result)
        self.assertFalse(parsed["success"])
        self.assertEqual(parsed["status"], "rollback_failed")

    def test_valid_format_accepted(self):
        fn = self.mod.rollback_proposal
        if hasattr(fn, "fn"):
            result = fn.fn("20260215-143000")
        else:
            result = fn("20260215-143000")
        parsed = json.loads(result)
        # Should fail (no snapshot exists) but not due to format validation
        self.assertNotIn("error", parsed)
        self.assertEqual(parsed["receipt_ts"], "20260215-143000")


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestMCPTokenAuth(unittest.TestCase):
    """Test token auth helper."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        self._orig_token = os.environ.get("MIND_MEM_TOKEN")
        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)
        if self._orig_token is not None:
            os.environ["MIND_MEM_TOKEN"] = self._orig_token
        else:
            os.environ.pop("MIND_MEM_TOKEN", None)

    def test_no_token_returns_none(self):
        os.environ.pop("MIND_MEM_TOKEN", None)
        self.assertIsNone(self.mod._check_token())

    def test_token_from_env(self):
        os.environ["MIND_MEM_TOKEN"] = "test-secret-123"
        self.assertEqual(self.mod._check_token(), "test-secret-123")


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestMCPServerMeta(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_server_name(self):
        self.assertEqual(self.mod.mcp.name, "mind-mem")

    def test_server_has_instructions(self):
        self.assertIn("contradiction", self.mod.mcp.instructions.lower())


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestTokenVerification(unittest.TestCase):
    """Security tests for MCP HTTP token enforcement."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        self.mod = _load_server(self.td)
        self._orig_token = os.environ.get("MIND_MEM_TOKEN")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)
        if self._orig_token is not None:
            os.environ["MIND_MEM_TOKEN"] = self._orig_token
        else:
            os.environ.pop("MIND_MEM_TOKEN", None)

    def test_no_token_configured_allows_all(self):
        os.environ.pop("MIND_MEM_TOKEN", None)
        self.assertTrue(self.mod.verify_token({}))
        self.assertTrue(self.mod.verify_token({"Authorization": "Bearer anything"}))

    def test_valid_bearer_token(self):
        os.environ["MIND_MEM_TOKEN"] = "secret123"
        self.assertTrue(self.mod.verify_token({"Authorization": "Bearer secret123"}))

    def test_invalid_bearer_token(self):
        os.environ["MIND_MEM_TOKEN"] = "secret123"
        self.assertFalse(self.mod.verify_token({"Authorization": "Bearer wrong"}))

    def test_missing_token_rejected(self):
        os.environ["MIND_MEM_TOKEN"] = "secret123"
        self.assertFalse(self.mod.verify_token({}))

    def test_alt_header_accepted(self):
        os.environ["MIND_MEM_TOKEN"] = "secret123"
        self.assertTrue(self.mod.verify_token({"X-MemOS-Token": "secret123"}))

    def test_alt_header_wrong_rejected(self):
        os.environ["MIND_MEM_TOKEN"] = "secret123"
        self.assertFalse(self.mod.verify_token({"X-MemOS-Token": "wrong"}))


if __name__ == "__main__":
    unittest.main()
