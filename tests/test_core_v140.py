"""Tests for v1.4.0 core hardening: issues #28, #30, #32, #34."""

import os
import shutil
import sqlite3
import tempfile
import threading
import unittest

from mind_mem.block_metadata import BlockMetadataManager
from mind_mem.block_parser import BlockCorruptedError, parse_blocks, parse_file
from mind_mem.hybrid_recall import HybridBackend, validate_recall_config

# ---------------------------------------------------------------------------
# Issue #28 — Strict schema checks on hybrid fallback chain
# ---------------------------------------------------------------------------


class TestValidateRecallConfig(unittest.TestCase):
    """validate_recall_config should catch non-numeric and non-positive values."""

    def test_valid_config_no_errors(self):
        cfg = {"bm25_weight": 1.0, "vector_weight": 0.5, "rrf_k": 60}
        self.assertEqual(validate_recall_config(cfg), [])

    def test_empty_config_no_errors(self):
        self.assertEqual(validate_recall_config({}), [])

    def test_string_numeric_accepted(self):
        cfg = {"bm25_weight": "2.0", "rrf_k": "30"}
        self.assertEqual(validate_recall_config(cfg), [])

    def test_non_numeric_rejected(self):
        cfg = {"bm25_weight": "abc"}
        errors = validate_recall_config(cfg)
        self.assertEqual(len(errors), 1)
        self.assertIn("bm25_weight", errors[0])
        self.assertIn("must be numeric", errors[0])

    def test_negative_rejected(self):
        cfg = {"vector_weight": -1.0}
        errors = validate_recall_config(cfg)
        self.assertEqual(len(errors), 1)
        self.assertIn("must be positive", errors[0])

    def test_zero_rejected(self):
        cfg = {"rrf_k": 0}
        errors = validate_recall_config(cfg)
        self.assertEqual(len(errors), 1)
        self.assertIn("must be positive", errors[0])

    def test_multiple_errors(self):
        cfg = {"bm25_weight": "bad", "vector_weight": -1, "rrf_k": None}
        errors = validate_recall_config(cfg)
        self.assertEqual(len(errors), 3)

    def test_none_value_rejected(self):
        cfg = {"bm25_weight": None}
        errors = validate_recall_config(cfg)
        self.assertEqual(len(errors), 1)


class TestHybridBackendSchemaValidation(unittest.TestCase):
    """HybridBackend should fall back to defaults on bad config."""

    def test_bad_bm25_weight_uses_default(self):
        hb = HybridBackend(config={"bm25_weight": "invalid"})
        self.assertEqual(hb.bm25_weight, 1.0)  # default
        self.assertTrue(len(hb._config_errors) > 0)

    def test_negative_rrf_k_uses_default(self):
        hb = HybridBackend(config={"rrf_k": -10})
        self.assertEqual(hb.rrf_k, 60)  # default

    def test_valid_config_no_errors(self):
        hb = HybridBackend(config={"bm25_weight": 2.0, "rrf_k": 30})
        self.assertEqual(hb.bm25_weight, 2.0)
        self.assertEqual(hb.rrf_k, 30)
        self.assertEqual(hb._config_errors, [])


class TestFromConfigValidation(unittest.TestCase):
    """HybridBackend.from_config should handle missing/bad recall section."""

    def test_missing_recall_section(self):
        hb = HybridBackend.from_config({})
        self.assertEqual(hb.rrf_k, 60)

    def test_recall_not_dict(self):
        hb = HybridBackend.from_config({"recall": "invalid"})
        self.assertEqual(hb.rrf_k, 60)

    def test_recall_none(self):
        hb = HybridBackend.from_config({"recall": None})
        self.assertEqual(hb.rrf_k, 60)

    def test_valid_recall_section(self):
        hb = HybridBackend.from_config({"recall": {"rrf_k": 42}})
        self.assertEqual(hb.rrf_k, 42)


# ---------------------------------------------------------------------------
# Issue #30 — BlockCorruptedError and parse_file error handling
# ---------------------------------------------------------------------------


class TestBlockCorruptedError(unittest.TestCase):
    """BlockCorruptedError should carry metadata."""

    def test_error_attributes(self):
        err = BlockCorruptedError(
            "test error", block_line_number=10, file_path="/tmp/test.md", context="[BAD-BLOCK]\nStatus: ?"
        )
        self.assertEqual(err.block_line_number, 10)
        self.assertEqual(err.file_path, "/tmp/test.md")
        self.assertIn("BAD-BLOCK", err.context)
        self.assertIsInstance(err, ValueError)

    def test_error_defaults(self):
        err = BlockCorruptedError("msg")
        self.assertEqual(err.block_line_number, 0)
        self.assertEqual(err.file_path, "")
        self.assertEqual(err.context, "")


class TestParseFileCorruptionHandling(unittest.TestCase):
    """parse_file should skip bad blocks in non-strict mode, raise in strict."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_valid_blocks_parsed(self):
        path = self._write("good.md", "[D-20260101-001]\nStatus: active\nStatement: OK\n")
        blocks = parse_file(path)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["_id"], "D-20260101-001")

    def test_parse_blocks_basic(self):
        text = "[T-20260101-001]\nStatus: active\n---\n[T-20260101-002]\nStatus: done\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 2)

    def test_empty_file(self):
        path = self._write("empty.md", "")
        blocks = parse_file(path)
        self.assertEqual(blocks, [])


# ---------------------------------------------------------------------------
# Issue #32 — Threading locks on BlockMetadataManager
# ---------------------------------------------------------------------------


class TestBlockMetadataManagerLock(unittest.TestCase):
    """BlockMetadataManager should have RLock and be thread-safe."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "meta.db")
        self.mgr = BlockMetadataManager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_has_rlock(self):
        self.assertIsInstance(self.mgr._lock, type(threading.RLock()))

    def test_concurrent_record_access(self):
        """Multiple threads calling record_access should not corrupt state."""
        errors = []

        def worker(tid):
            try:
                for _ in range(20):
                    self.mgr.record_access([f"B-{tid}"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")

        # Verify each block was accessed 20 times
        conn = sqlite3.connect(self.db_path)
        for i in range(10):
            row = conn.execute("SELECT access_count FROM block_meta WHERE id = ?", (f"B-{i}",)).fetchone()
            self.assertIsNotNone(row, f"B-{i} not found")
            self.assertEqual(row[0], 20, f"B-{i} has {row[0]} accesses, expected 20")
        conn.close()

    def test_concurrent_importance_updates(self):
        """Concurrent importance updates should not raise."""
        self.mgr.record_access(["D-001"])
        errors = []

        def worker():
            try:
                for _ in range(10):
                    imp = self.mgr.update_importance("D-001")
                    assert 0.8 <= imp <= 1.5
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")

    def test_concurrent_evolve_keywords(self):
        """Concurrent keyword evolution should not corrupt."""
        self.mgr.record_access(["D-001"])
        errors = []

        def worker(tid):
            try:
                self.mgr.evolve_keywords(
                    "D-001",
                    [f"kw{tid}", "common"],
                    block_content=f"kw{tid} common text",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")

    def test_rlock_allows_reentrant_calls(self):
        """RLock should allow methods that call each other internally."""
        # update_importance uses _get_conn internally, test it doesn't deadlock
        self.mgr.record_access(["D-001"])
        imp = self.mgr.update_importance("D-001")
        boost = self.mgr.get_importance_boost("D-001")
        self.assertGreaterEqual(imp, 0.8)
        self.assertGreaterEqual(boost, 0.8)


# ---------------------------------------------------------------------------
# Issue #34 — FTS5 index staleness check
# ---------------------------------------------------------------------------


class TestFTS5Staleness(unittest.TestCase):
    """is_stale should detect changed files."""

    def test_no_index_is_stale(self):
        from mind_mem.sqlite_index import is_stale

        # Non-existent workspace has no DB
        self.assertTrue(is_stale("/tmp/nonexistent_workspace_xyz"))

    def test_is_stale_function_exists(self):
        from mind_mem.sqlite_index import is_stale

        self.assertTrue(callable(is_stale))

    def test_index_status_includes_stale_files(self):
        from mind_mem.sqlite_index import index_status

        status = index_status("/tmp/nonexistent_workspace_xyz")
        self.assertIn("stale_files", status)


# ---------------------------------------------------------------------------
# Integration: MCP hybrid_search config validation envelope
# ---------------------------------------------------------------------------


class TestHybridSearchEnvelope(unittest.TestCase):
    """hybrid_search MCP tool should return config_warnings on bad config."""

    def test_validate_recall_config_integration(self):
        """Bad config values produce warnings that would appear in envelope."""
        cfg = {"bm25_weight": "not_a_number", "rrf_k": -5}
        errors = validate_recall_config(cfg)
        self.assertEqual(len(errors), 2)
        # These warnings would be included in the JSON envelope
        for e in errors:
            self.assertIsInstance(e, str)


if __name__ == "__main__":
    unittest.main()
