#!/usr/bin/env python3
"""Tests for sqlite_index.py — SQLite FTS5 index for mind-mem recall."""

import os
import shutil
import tempfile
import unittest

from mind_mem.sqlite_index import (
    _compute_block_hash,
    _connect,
    _db_path,
    _file_hash,
    build_index,
    index_status,
    query_index,
)


class _WorkspaceMixin:
    """Helper to create a minimal workspace for testing."""

    def _setup_workspace(self, tmpdir, decisions="", tasks="", entities_projects=""):
        for d in ["decisions", "tasks", "entities", "intelligence", "memory"]:
            os.makedirs(os.path.join(tmpdir, d), exist_ok=True)

        if decisions:
            with open(os.path.join(tmpdir, "decisions", "DECISIONS.md"), "w") as f:
                f.write(decisions)
        else:
            with open(os.path.join(tmpdir, "decisions", "DECISIONS.md"), "w") as f:
                f.write("# Decisions\n")

        if tasks:
            with open(os.path.join(tmpdir, "tasks", "TASKS.md"), "w") as f:
                f.write(tasks)

        if entities_projects:
            with open(os.path.join(tmpdir, "entities", "projects.md"), "w") as f:
                f.write(entities_projects)

        # Create empty files for other corpus entries
        for fname in [
            "entities/people.md",
            "entities/tools.md",
            "entities/incidents.md",
            "intelligence/CONTRADICTIONS.md",
            "intelligence/DRIFT.md",
            "intelligence/SIGNALS.md",
        ]:
            path = os.path.join(tmpdir, fname)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {os.path.basename(fname)}\n")

        return tmpdir


class TestBuildIndex(_WorkspaceMixin, unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_full_build_creates_db(self):
        ws = self._setup_workspace(
            self.td,
            decisions=("[D-20260101-001]\nStatement: Use PostgreSQL for the database\nStatus: active\nDate: 2026-01-01\n"),
        )
        result = build_index(ws, incremental=False)
        self.assertTrue(os.path.isfile(_db_path(ws)))
        self.assertGreater(result["blocks_indexed"], 0)
        self.assertIn("elapsed_ms", result)

    def test_incremental_skips_unchanged(self):
        ws = self._setup_workspace(self.td, decisions=("[D-20260101-001]\nStatement: Use PostgreSQL\nStatus: active\n"))
        # First build
        build_index(ws, incremental=False)
        # Second build (no changes)
        r2 = build_index(ws, incremental=True)
        self.assertEqual(r2["files_indexed"], 0)
        self.assertEqual(r2["blocks_indexed"], 0)

    def test_incremental_detects_change(self):
        ws = self._setup_workspace(self.td, decisions=("[D-20260101-001]\nStatement: Use PostgreSQL\nStatus: active\n"))
        build_index(ws, incremental=False)

        # Modify the decisions file
        with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a") as f:
            f.write("\n---\n\n[D-20260101-002]\nStatement: Use Redis\nStatus: active\n")

        r2 = build_index(ws, incremental=True)
        self.assertGreater(r2["files_indexed"], 0)
        self.assertGreater(r2["blocks_indexed"], 0)

    def test_empty_workspace_builds(self):
        ws = self._setup_workspace(self.td)
        result = build_index(ws, incremental=False)
        self.assertTrue(os.path.isfile(_db_path(ws)))
        self.assertEqual(result["blocks_indexed"], 0)

    def test_xref_edges_populated(self):
        ws = self._setup_workspace(
            self.td,
            decisions=("[D-20260101-001]\nStatement: Use JWT\nContext: See T-20260101-001\nStatus: active\n"),
            tasks=("[T-20260101-001]\nTitle: Implement auth\nStatus: open\n"),
        )
        build_index(ws, incremental=False)
        conn = _connect(ws, readonly=True)
        edges = conn.execute("SELECT * FROM xref_edges").fetchall()
        conn.close()
        # Should have bidirectional edges
        src_dst = {(e["src"], e["dst"]) for e in edges}
        self.assertIn(("D-20260101-001", "T-20260101-001"), src_dst)
        self.assertIn(("T-20260101-001", "D-20260101-001"), src_dst)


class TestQueryIndex(_WorkspaceMixin, unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.ws = self._setup_workspace(
            self.td,
            decisions=(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL for the primary database\n"
                "Status: active\n"
                "Date: 2026-01-01\n"
                "Tags: database, infrastructure\n"
                "\n---\n\n"
                "[D-20260101-002]\n"
                "Statement: Use Redis for caching layer\n"
                "Status: active\n"
                "Date: 2026-01-02\n"
                "Tags: cache, infrastructure\n"
            ),
        )
        build_index(self.ws, incremental=False)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_basic_query(self):
        results = query_index(self.ws, "PostgreSQL")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["_id"], "D-20260101-001")

    def test_query_returns_correct_format(self):
        results = query_index(self.ws, "database")
        self.assertGreater(len(results), 0)
        r = results[0]
        self.assertIn("_id", r)
        self.assertIn("type", r)
        self.assertIn("score", r)
        self.assertIn("excerpt", r)
        self.assertIn("file", r)
        self.assertIn("line", r)
        self.assertIn("status", r)

    def test_no_results(self):
        results = query_index(self.ws, "kubernetes")
        self.assertEqual(len(results), 0)

    def test_limit_respected(self):
        results = query_index(self.ws, "infrastructure", limit=1)
        self.assertLessEqual(len(results), 1)

    def test_active_only_filter(self):
        # Add a superseded block
        with open(os.path.join(self.ws, "decisions", "DECISIONS.md"), "a") as f:
            f.write("\n---\n\n[D-20260101-003]\nStatement: Use MySQL for database\nStatus: superseded\n")
        build_index(self.ws, incremental=True)

        results = query_index(self.ws, "database", active_only=True)
        ids = [r["_id"] for r in results]
        self.assertNotIn("D-20260101-003", ids)

    def test_fallback_when_no_index(self):
        """When index doesn't exist, should fall back to filesystem recall."""
        empty = tempfile.mkdtemp()
        try:
            ws = self._setup_workspace(empty, decisions=("[D-20260101-001]\nStatement: Use PostgreSQL\nStatus: active\n"))
            # Don't build index — should fallback
            results = query_index(ws, "PostgreSQL")
            self.assertGreater(len(results), 0)
        finally:
            shutil.rmtree(empty, ignore_errors=True)


class TestIndexStatus(_WorkspaceMixin, unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_no_index(self):
        ws = self._setup_workspace(self.td)
        status = index_status(ws)
        self.assertFalse(status["exists"])

    def test_index_exists(self):
        ws = self._setup_workspace(self.td, decisions=("[D-20260101-001]\nStatement: Test\nStatus: active\n"))
        build_index(ws, incremental=False)
        status = index_status(ws)
        self.assertTrue(status["exists"])
        self.assertGreater(status["blocks"], 0)
        self.assertIsNotNone(status["last_build"])

    def test_stale_detection(self):
        ws = self._setup_workspace(self.td, decisions=("[D-20260101-001]\nStatement: Test\nStatus: active\n"))
        build_index(ws, incremental=False)

        # Modify file
        with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a") as f:
            f.write("\n---\n\n[D-20260101-002]\nStatement: New\nStatus: active\n")

        status = index_status(ws)
        self.assertGreater(status["stale_files"], 0)


class TestBlockLevelIncremental(_WorkspaceMixin, unittest.TestCase):
    """Tests for block-level incremental FTS indexing (issue #17)."""

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_initial_build_all_new(self):
        """First build should report all blocks as new."""
        ws = self._setup_workspace(
            self.td,
            decisions=(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL\n"
                "Status: active\n"
                "\n---\n\n"
                "[D-20260101-002]\n"
                "Statement: Use Redis\n"
                "Status: active\n"
            ),
        )
        result = build_index(ws, incremental=False)
        self.assertEqual(result["blocks_new"], 2)
        self.assertEqual(result["blocks_modified"], 0)
        self.assertEqual(result["blocks_deleted"], 0)
        self.assertEqual(result["blocks_unchanged"], 0)

    def test_unchanged_blocks_skipped(self):
        """Unchanged blocks should not be re-indexed."""
        ws = self._setup_workspace(
            self.td,
            decisions=(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL\n"
                "Status: active\n"
                "\n---\n\n"
                "[D-20260101-002]\n"
                "Statement: Use Redis\n"
                "Status: active\n"
            ),
        )
        build_index(ws, incremental=False)

        # Touch the file (change mtime) but keep same content
        path = os.path.join(ws, "decisions", "DECISIONS.md")
        with open(path, encoding="utf-8") as _f:
            content = _f.read()
        import time

        time.sleep(0.05)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        result = build_index(ws, incremental=True)
        # File changed (mtime), so it's in the changed list
        self.assertGreater(result["files_indexed"], 0)
        # But all blocks unchanged at block level
        self.assertEqual(result["blocks_new"], 0)
        self.assertEqual(result["blocks_modified"], 0)
        self.assertEqual(result["blocks_unchanged"], 2)

    def test_modified_block_detected(self):
        """Modifying a single block should only re-index that block."""
        ws = self._setup_workspace(
            self.td,
            decisions=(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL\n"
                "Status: active\n"
                "\n---\n\n"
                "[D-20260101-002]\n"
                "Statement: Use Redis\n"
                "Status: active\n"
            ),
        )
        build_index(ws, incremental=False)

        # Modify only the second block
        with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
            f.write(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL\n"
                "Status: active\n"
                "\n---\n\n"
                "[D-20260101-002]\n"
                "Statement: Use Memcached instead of Redis\n"
                "Status: active\n"
            )

        result = build_index(ws, incremental=True)
        self.assertEqual(result["blocks_modified"], 1)
        self.assertEqual(result["blocks_unchanged"], 1)
        self.assertEqual(result["blocks_new"], 0)
        self.assertEqual(result["blocks_deleted"], 0)

    def test_new_block_added(self):
        """Adding a block should be detected as new."""
        ws = self._setup_workspace(self.td, decisions=("[D-20260101-001]\nStatement: Use PostgreSQL\nStatus: active\n"))
        build_index(ws, incremental=False)

        # Add a second block
        with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a") as f:
            f.write("\n---\n\n[D-20260101-002]\nStatement: Use Redis\nStatus: active\n")

        result = build_index(ws, incremental=True)
        self.assertEqual(result["blocks_new"], 1)
        self.assertEqual(result["blocks_unchanged"], 1)

    def test_block_deleted(self):
        """Removing a block should be detected as deleted."""
        ws = self._setup_workspace(
            self.td,
            decisions=(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL\n"
                "Status: active\n"
                "\n---\n\n"
                "[D-20260101-002]\n"
                "Statement: Use Redis\n"
                "Status: active\n"
            ),
        )
        build_index(ws, incremental=False)

        # Remove the second block
        with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
            f.write("[D-20260101-001]\nStatement: Use PostgreSQL\nStatus: active\n")

        result = build_index(ws, incremental=True)
        self.assertEqual(result["blocks_deleted"], 1)
        self.assertEqual(result["blocks_unchanged"], 1)

        # Verify deleted block is gone from FTS
        results = query_index(ws, "Redis")
        ids = [r["_id"] for r in results]
        self.assertNotIn("D-20260101-002", ids)

    def test_mixed_add_modify_delete(self):
        """All three operations in a single reindex."""
        ws = self._setup_workspace(
            self.td,
            decisions=(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL\n"
                "Status: active\n"
                "\n---\n\n"
                "[D-20260101-002]\n"
                "Statement: Use Redis\n"
                "Status: active\n"
                "\n---\n\n"
                "[D-20260101-003]\n"
                "Statement: Use Docker\n"
                "Status: active\n"
            ),
        )
        build_index(ws, incremental=False)

        # 001: unchanged, 002: modify, 003: delete, 004: new
        with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
            f.write(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL\n"
                "Status: active\n"
                "\n---\n\n"
                "[D-20260101-002]\n"
                "Statement: Use Memcached\n"
                "Status: active\n"
                "\n---\n\n"
                "[D-20260101-004]\n"
                "Statement: Use Kubernetes\n"
                "Status: active\n"
            )

        result = build_index(ws, incremental=True)
        self.assertEqual(result["blocks_unchanged"], 1)  # 001
        self.assertEqual(result["blocks_modified"], 1)  # 002
        self.assertEqual(result["blocks_deleted"], 1)  # 003
        self.assertEqual(result["blocks_new"], 1)  # 004

    def test_force_rebuild_reindexes_all(self):
        """Full rebuild should re-index all blocks even if unchanged."""
        ws = self._setup_workspace(self.td, decisions=("[D-20260101-001]\nStatement: Use PostgreSQL\nStatus: active\n"))
        build_index(ws, incremental=False)

        # Full rebuild — no file changes, but force=True re-indexes
        result = build_index(ws, incremental=False)
        self.assertGreater(result["blocks_indexed"], 0)

    def test_index_meta_populated(self):
        """index_meta table should contain correct hashes after build."""
        ws = self._setup_workspace(self.td, decisions=("[D-20260101-001]\nStatement: Use PostgreSQL\nStatus: active\n"))
        build_index(ws, incremental=False)

        conn = _connect(ws, readonly=True)
        rows = conn.execute("SELECT * FROM index_meta").fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["block_id"], "D-20260101-001")
        self.assertTrue(len(rows[0]["content_hash"]) == 64)  # SHA-256 hex

    def test_modified_block_queryable(self):
        """After modifying a block, the new content should be searchable."""
        ws = self._setup_workspace(self.td, decisions=("[D-20260101-001]\nStatement: Use MySQL for database\nStatus: active\n"))
        build_index(ws, incremental=False)

        # Modify content
        with open(os.path.join(ws, "decisions", "DECISIONS.md"), "w") as f:
            f.write("[D-20260101-001]\nStatement: Use PostgreSQL for database\nStatus: active\n")
        build_index(ws, incremental=True)

        # New content should be searchable
        results = query_index(ws, "PostgreSQL")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["_id"], "D-20260101-001")

        # Old content should NOT match
        results = query_index(ws, "MySQL")
        mysql_ids = [r["_id"] for r in results]
        self.assertNotIn("D-20260101-001", mysql_ids)


class TestComputeBlockHash(unittest.TestCase):
    """Tests for _compute_block_hash determinism."""

    def test_same_content_same_hash(self):
        block = {"_id": "D-001", "Statement": "hello", "Status": "active"}
        self.assertEqual(_compute_block_hash(block), _compute_block_hash(block))

    def test_different_content_different_hash(self):
        b1 = {"_id": "D-001", "Statement": "hello"}
        b2 = {"_id": "D-001", "Statement": "world"}
        self.assertNotEqual(_compute_block_hash(b1), _compute_block_hash(b2))

    def test_line_number_ignored(self):
        """_line changes should not change the hash (blocks shift around)."""
        b1 = {"_id": "D-001", "Statement": "hello", "_line": 5}
        b2 = {"_id": "D-001", "Statement": "hello", "_line": 100}
        self.assertEqual(_compute_block_hash(b1), _compute_block_hash(b2))

    def test_hash_is_sha256_hex(self):
        block = {"_id": "D-001", "Statement": "test"}
        h = _compute_block_hash(block)
        self.assertEqual(len(h), 64)
        int(h, 16)  # Should parse as hex


class TestFileHash(unittest.TestCase):
    def test_same_content_same_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            h1 = _file_hash(path)
            h2 = _file_hash(path)
            self.assertEqual(h1, h2)
        finally:
            os.unlink(path)

    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("hello")
            path1 = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("world")
            path2 = f.name
        try:
            self.assertNotEqual(_file_hash(path1), _file_hash(path2))
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_missing_file(self):
        self.assertEqual(_file_hash("/nonexistent/file.md"), "")


if __name__ == "__main__":
    unittest.main()
