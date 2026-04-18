#!/usr/bin/env python3
"""Tests for MCP v1.4.0 features — issues #29, #31, #35, #36.

- #29: SQLite locked database returns structured "busy" error
- #31: Query-level observability decorator logs tool calls
- #35: New tools: delete_memory_item, export_memory
- #36: _schema_version field in ALL MCP JSON responses
"""

import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Load the mcp_server module
_SERVER_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp_server.py")
_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "mind_mem")

_HAS_FASTMCP = importlib.util.find_spec("fastmcp") is not None


def _load_server(workspace: str):
    """Load the mcp_server module with a given workspace."""
    os.environ["MIND_MEM_WORKSPACE"] = workspace
    spec = importlib.util.spec_from_file_location("mcp_server_v140", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call_tool(mod_attr, *args, **kwargs):
    """Call an MCP tool, unwrapping FunctionTool if needed.

    FastMCP's @mcp.tool decorator wraps functions into FunctionTool objects.
    This helper extracts the underlying callable via .fn and invokes it.
    """
    fn = mod_attr
    if hasattr(fn, "fn"):
        fn = fn.fn
    return fn(*args, **kwargs)


def _setup_workspace(td):
    """Create a standard workspace layout for tests."""
    os.makedirs(os.path.join(td, "decisions"), exist_ok=True)
    os.makedirs(os.path.join(td, "tasks"), exist_ok=True)
    os.makedirs(os.path.join(td, "entities"), exist_ok=True)
    os.makedirs(os.path.join(td, "intelligence"), exist_ok=True)
    os.makedirs(os.path.join(td, "memory"), exist_ok=True)

    with open(os.path.join(td, "decisions", "DECISIONS.md"), "w") as f:
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

    with open(os.path.join(td, "tasks", "TASKS.md"), "w") as f:
        f.write("[T-20260101-001]\nDescription: Set up PostgreSQL in staging\nStatus: open\n")

    with open(os.path.join(td, "entities", "projects.md"), "w") as f:
        f.write("[PRJ-mind-mem]\nName: mind-mem\nStatus: active\n")

    with open(os.path.join(td, "entities", "tools.md"), "w") as f:
        f.write("[TOOL-pytest]\nName: pytest\nStatus: active\n")


# ---------------------------------------------------------------------------
# #29: SQLite locked database returns structured "busy" error
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestSQLiteBusyError(unittest.TestCase):
    """Issue #29: SQLite locked database returns structured busy error."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        _setup_workspace(self.td)
        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_sqlite_busy_error_structure(self):
        result = json.loads(self.mod._sqlite_busy_error())
        self.assertEqual(result["error"], "database_busy")
        self.assertEqual(result["retry_after_seconds"], 1)
        self.assertIn("_schema_version", result)
        self.assertEqual(result["_schema_version"], "1.0")

    def test_is_db_locked_positive(self):
        exc = sqlite3.OperationalError("database is locked")
        self.assertTrue(self.mod._is_db_locked(exc))

    def test_is_db_locked_negative(self):
        exc = sqlite3.OperationalError("no such table: foo")
        self.assertFalse(self.mod._is_db_locked(exc))

    def test_recall_catches_locked_db(self):
        """Recall tool should catch sqlite3.OperationalError for locked DB."""
        db_path = self.mod.fts_db_path(self.td)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with open(db_path, "w", encoding="utf-8") as f:
            f.write("")  # dummy file

        with patch.object(self.mod, "fts_query", side_effect=sqlite3.OperationalError("database is locked")):
            result_str = _call_tool(self.mod.recall, "test query", backend="bm25")
            result = json.loads(result_str)
            self.assertEqual(result["error"], "database_busy")
            self.assertEqual(result["retry_after_seconds"], 1)

    def test_hybrid_search_catches_locked_db(self):
        """hybrid_search should catch sqlite3.OperationalError for locked DB."""
        with patch.dict("sys.modules", {"mind_mem.hybrid_recall": MagicMock()}):
            mock_hybrid = sys.modules["mind_mem.hybrid_recall"]
            mock_hybrid.validate_recall_config.return_value = []
            mock_backend = MagicMock()
            mock_backend.search.side_effect = sqlite3.OperationalError("database is locked")
            mock_hybrid.HybridBackend.from_config.return_value = mock_backend

            result_str = _call_tool(self.mod.hybrid_search, "test query")
            result = json.loads(result_str)
            self.assertEqual(result["error"], "database_busy")


# ---------------------------------------------------------------------------
# #31: Query-level observability decorator
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestObservabilityDecorator(unittest.TestCase):
    """Issue #31: mcp_tool_observe decorator logs structured data."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        _setup_workspace(self.td)
        self.mod = _load_server(self.td)
        # Reset metrics for clean test
        self.mod.metrics.reset()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_decorator_preserves_function_name(self):
        """Decorated functions should keep their original name via .fn."""
        fn = self.mod.recall
        if hasattr(fn, "fn"):
            self.assertEqual(fn.fn.__name__, "recall")
        else:
            self.assertEqual(fn.__name__, "recall")

    def test_decorator_increments_success_counter(self):
        """Successful tool call should increment mcp_tool_success."""
        _call_tool(self.mod.scan)
        self.assertGreater(self.mod.metrics.get("mcp_tool_success"), 0)

    def test_decorator_records_duration(self):
        """Tool calls should record duration_ms observation."""
        _call_tool(self.mod.scan)
        summary = self.mod.metrics.summary()
        self.assertIn("mcp_tool_duration_ms", summary.get("observations", {}))

    def test_decorator_on_recall(self):
        """recall tool should be wrapped with observability."""
        _call_tool(self.mod.recall, "PostgreSQL")
        self.assertGreater(self.mod.metrics.get("mcp_tool_success"), 0)

    def test_decorator_on_list_contradictions(self):
        """list_contradictions should be wrapped with observability."""
        with open(os.path.join(self.td, "intelligence", "CONTRADICTIONS.md"), "w") as f:
            f.write("# Contradictions\n")
        _call_tool(self.mod.list_contradictions)
        self.assertGreater(self.mod.metrics.get("mcp_tool_success"), 0)

    def test_failure_increments_failure_counter(self):
        """Failed tool call should increment mcp_tool_failure."""

        @self.mod.mcp_tool_observe
        def failing_tool():
            raise ValueError("test error")

        with self.assertRaises(ValueError):
            failing_tool()
        self.assertGreater(self.mod.metrics.get("mcp_tool_failure"), 0)


# ---------------------------------------------------------------------------
# #35: New tools — delete_memory_item, export_memory
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestDeleteMemoryItem(unittest.TestCase):
    """Issue #35: delete_memory_item tool."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        _setup_workspace(self.td)
        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_delete_existing_block(self):
        """Should delete a block and return confirmation."""
        result_str = _call_tool(self.mod.delete_memory_item, "D-20260101-001")
        result = json.loads(result_str)
        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["block_id"], "D-20260101-001")
        self.assertEqual(result["_schema_version"], "1.0")

        # Verify block is actually gone from file
        with open(os.path.join(self.td, "decisions", "DECISIONS.md")) as f:
            content = f.read()
        self.assertNotIn("[D-20260101-001]", content)
        # Other blocks should still exist
        self.assertIn("[D-20260102-001]", content)

    def test_delete_task_block(self):
        """Should delete a task block."""
        result_str = _call_tool(self.mod.delete_memory_item, "T-20260101-001")
        result = json.loads(result_str)
        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["block_id"], "T-20260101-001")

        # Verify block removed
        with open(os.path.join(self.td, "tasks", "TASKS.md")) as f:
            content = f.read()
        self.assertNotIn("[T-20260101-001]", content)

    def test_delete_nonexistent_block(self):
        """Should return error for non-existent block ID."""
        result_str = _call_tool(self.mod.delete_memory_item, "D-20260199-999")
        result = json.loads(result_str)
        self.assertIn("error", result)
        self.assertIn("not found", result["error"].lower())

    def test_delete_invalid_id_format(self):
        """Should reject invalid block ID format."""
        result_str = _call_tool(self.mod.delete_memory_item, "invalid!!!")
        result = json.loads(result_str)
        self.assertIn("error", result)
        self.assertIn("Invalid block ID", result["error"])

    def test_delete_unrecognized_prefix(self):
        """Should return error for unknown prefix."""
        result_str = _call_tool(self.mod.delete_memory_item, "UNKNOWN-123")
        result = json.loads(result_str)
        self.assertIn("error", result)
        self.assertIn("Unrecognized", result["error"])

    def test_delete_has_schema_version(self):
        """All responses should include _schema_version."""
        # Success case
        result = json.loads(_call_tool(self.mod.delete_memory_item, "D-20260101-001"))
        self.assertEqual(result["_schema_version"], "1.0")
        # Error case
        result = json.loads(_call_tool(self.mod.delete_memory_item, "D-20260199-999"))
        self.assertEqual(result["_schema_version"], "1.0")

    def test_delete_is_admin_tool(self):
        """delete_memory_item should be in ADMIN_TOOLS."""
        self.assertIn("delete_memory_item", self.mod.ADMIN_TOOLS)


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestExportMemory(unittest.TestCase):
    """Issue #35: export_memory tool."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        _setup_workspace(self.td)
        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_export_jsonl_basic(self):
        """Should export all blocks as JSONL."""
        result_str = _call_tool(self.mod.export_memory)
        result = json.loads(result_str)
        self.assertEqual(result["_schema_version"], "1.0")
        self.assertEqual(result["format"], "jsonl")
        self.assertGreater(result["block_count"], 0)
        self.assertIn("data", result)

        # Parse JSONL data
        lines = result["data"].strip().split("\n")
        self.assertGreater(len(lines), 0)
        for line in lines:
            block = json.loads(line)
            self.assertIn("_id", block)
            self.assertIn("_source_file", block)

    def test_export_strips_metadata_by_default(self):
        """Without include_metadata, internal _ fields should be stripped."""
        result_str = _call_tool(self.mod.export_memory, format="jsonl", include_metadata=False)
        result = json.loads(result_str)
        lines = result["data"].strip().split("\n")
        for line in lines:
            block = json.loads(line)
            internal_keys = [k for k in block if k.startswith("_") and k not in ("_id", "_source_file")]
            self.assertEqual(internal_keys, [], f"Unexpected metadata keys: {internal_keys}")

    def test_export_includes_metadata_when_requested(self):
        """With include_metadata=True, internal _ fields should be preserved."""
        result_str = _call_tool(self.mod.export_memory, format="jsonl", include_metadata=True)
        result = json.loads(result_str)
        # This test just verifies the call succeeds and returns data
        self.assertIn("data", result)
        self.assertGreater(result["block_count"], 0)

    def test_export_unsupported_format(self):
        """Should reject unsupported format."""
        result_str = _call_tool(self.mod.export_memory, format="csv")
        result = json.loads(result_str)
        self.assertIn("error", result)
        self.assertIn("Unsupported format", result["error"])
        self.assertEqual(result["_schema_version"], "1.0")

    def test_export_includes_multiple_dirs(self):
        """Should include blocks from decisions, tasks, and entities."""
        result_str = _call_tool(self.mod.export_memory)
        result = json.loads(result_str)
        lines = result["data"].strip().split("\n")
        source_files = {json.loads(line)["_source_file"] for line in lines}
        # Should have blocks from decisions and tasks at minimum
        has_decisions = any("decisions/" in sf for sf in source_files)
        has_tasks = any("tasks/" in sf for sf in source_files)
        self.assertTrue(has_decisions, "Expected blocks from decisions/")
        self.assertTrue(has_tasks, "Expected blocks from tasks/")

    def test_export_is_admin_tool(self):
        """export_memory should be in ADMIN_TOOLS (moved from USER_TOOLS, #447)."""
        self.assertIn("export_memory", self.mod.ADMIN_TOOLS)
        self.assertNotIn("export_memory", self.mod.USER_TOOLS)


# ---------------------------------------------------------------------------
# #36: _schema_version field in ALL MCP JSON responses
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestSchemaVersionInAllResponses(unittest.TestCase):
    """Issue #36: Every tool response should have _schema_version: 1.0."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        _setup_workspace(self.td)
        # Create contradictions file for list_contradictions
        with open(os.path.join(self.td, "intelligence", "CONTRADICTIONS.md"), "w") as f:
            f.write("# Contradictions\n")
        self.mod = _load_server(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _assert_schema_version(self, result_str, tool_name):
        """Assert that a JSON result string contains _schema_version: 1.0."""
        try:
            result = json.loads(result_str)
        except json.JSONDecodeError:
            self.fail(f"{tool_name} did not return valid JSON: {result_str[:200]}")
        if isinstance(result, dict):
            self.assertIn("_schema_version", result, f"{tool_name} response missing _schema_version")
            self.assertEqual(result["_schema_version"], "1.0", f"{tool_name} _schema_version is not '1.0'")

    def test_recall_has_schema_version(self):
        result = _call_tool(self.mod.recall, "PostgreSQL")
        self._assert_schema_version(result, "recall")

    def test_scan_has_schema_version(self):
        result = _call_tool(self.mod.scan)
        self._assert_schema_version(result, "scan")

    def test_list_contradictions_has_schema_version(self):
        result = _call_tool(self.mod.list_contradictions)
        self._assert_schema_version(result, "list_contradictions")

    def test_approve_apply_has_schema_version(self):
        result = _call_tool(self.mod.approve_apply, "P-20260101-001")
        self._assert_schema_version(result, "approve_apply")

    def test_rollback_proposal_has_schema_version(self):
        result = _call_tool(self.mod.rollback_proposal, "20260101-120000")
        self._assert_schema_version(result, "rollback_proposal")

    def test_hybrid_search_has_schema_version(self):
        result = _call_tool(self.mod.hybrid_search, "test")
        self._assert_schema_version(result, "hybrid_search")

    def test_find_similar_has_schema_version(self):
        result = _call_tool(self.mod.find_similar, "D-20260101-001")
        self._assert_schema_version(result, "find_similar")

    def test_intent_classify_has_schema_version(self):
        result = _call_tool(self.mod.intent_classify, "Why use PostgreSQL?")
        self._assert_schema_version(result, "intent_classify")

    def test_index_stats_has_schema_version(self):
        result = _call_tool(self.mod.index_stats)
        self._assert_schema_version(result, "index_stats")

    def test_reindex_has_schema_version(self):
        result = _call_tool(self.mod.reindex)
        self._assert_schema_version(result, "reindex")

    def test_memory_evolution_has_schema_version(self):
        result = _call_tool(self.mod.memory_evolution, "D-20260101-001")
        self._assert_schema_version(result, "memory_evolution")

    def test_category_summary_has_schema_version(self):
        result = _call_tool(self.mod.category_summary, "database")
        self._assert_schema_version(result, "category_summary")

    def test_prefetch_has_schema_version(self):
        result = _call_tool(self.mod.prefetch, "PostgreSQL, database")
        self._assert_schema_version(result, "prefetch")

    def test_prefetch_empty_has_schema_version(self):
        result = _call_tool(self.mod.prefetch, "")
        self._assert_schema_version(result, "prefetch (empty)")

    def test_list_mind_kernels_has_schema_version(self):
        result = _call_tool(self.mod.list_mind_kernels)
        self._assert_schema_version(result, "list_mind_kernels")

    def test_get_mind_kernel_has_schema_version(self):
        result = _call_tool(self.mod.get_mind_kernel, "nonexistent")
        self._assert_schema_version(result, "get_mind_kernel")

    def test_get_mind_kernel_invalid_name_has_schema_version(self):
        result = _call_tool(self.mod.get_mind_kernel, "../../../etc/passwd")
        self._assert_schema_version(result, "get_mind_kernel (invalid)")

    def test_delete_memory_item_has_schema_version(self):
        result = _call_tool(self.mod.delete_memory_item, "D-20260101-001")
        self._assert_schema_version(result, "delete_memory_item")

    def test_export_memory_has_schema_version(self):
        result = _call_tool(self.mod.export_memory)
        self._assert_schema_version(result, "export_memory")

    def test_sqlite_busy_error_has_schema_version(self):
        result = self.mod._sqlite_busy_error()
        self._assert_schema_version(result, "_sqlite_busy_error")


# ---------------------------------------------------------------------------
# Integration: observability + schema version combined
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_FASTMCP, "fastmcp not installed")
class TestObservabilityIntegration(unittest.TestCase):
    """Integration tests: observability decorator works with all tools."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        _setup_workspace(self.td)
        with open(os.path.join(self.td, "intelligence", "CONTRADICTIONS.md"), "w") as f:
            f.write("# Contradictions\n")
        self.mod = _load_server(self.td)
        self.mod.metrics.reset()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_multiple_tools_accumulate_metrics(self):
        """Multiple tool calls should accumulate success counters."""
        _call_tool(self.mod.recall, "test")
        _call_tool(self.mod.scan)
        _call_tool(self.mod.list_contradictions)
        self.assertEqual(self.mod.metrics.get("mcp_tool_success"), 3)

    def test_duration_observations_grow(self):
        """Each tool call should add a duration observation."""
        _call_tool(self.mod.recall, "test")
        _call_tool(self.mod.scan)
        summary = self.mod.metrics.summary()
        obs = summary.get("observations", {}).get("mcp_tool_duration_ms", {})
        self.assertEqual(obs.get("count"), 2)


if __name__ == "__main__":
    unittest.main()
