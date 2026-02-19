#!/usr/bin/env python3
"""Tests for sqlite_index.py — SQLite FTS5 index for mind-mem recall."""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from sqlite_index import (
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
            "entities/people.md", "entities/tools.md", "entities/incidents.md",
            "intelligence/CONTRADICTIONS.md", "intelligence/DRIFT.md",
            "intelligence/SIGNALS.md",
        ]:
            path = os.path.join(tmpdir, fname)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(f"# {os.path.basename(fname)}\n")

        return tmpdir


class TestBuildIndex(_WorkspaceMixin, unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_full_build_creates_db(self):
        ws = self._setup_workspace(self.td, decisions=(
            "[D-20260101-001]\n"
            "Statement: Use PostgreSQL for the database\n"
            "Status: active\n"
            "Date: 2026-01-01\n"
        ))
        result = build_index(ws, incremental=False)
        self.assertTrue(os.path.isfile(_db_path(ws)))
        self.assertGreater(result["blocks_indexed"], 0)
        self.assertIn("elapsed_ms", result)

    def test_incremental_skips_unchanged(self):
        ws = self._setup_workspace(self.td, decisions=(
            "[D-20260101-001]\n"
            "Statement: Use PostgreSQL\n"
            "Status: active\n"
        ))
        # First build
        build_index(ws, incremental=False)
        # Second build (no changes)
        r2 = build_index(ws, incremental=True)
        self.assertEqual(r2["files_indexed"], 0)
        self.assertEqual(r2["blocks_indexed"], 0)

    def test_incremental_detects_change(self):
        ws = self._setup_workspace(self.td, decisions=(
            "[D-20260101-001]\n"
            "Statement: Use PostgreSQL\n"
            "Status: active\n"
        ))
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
            decisions=(
                "[D-20260101-001]\nStatement: Use JWT\nContext: See T-20260101-001\nStatus: active\n"
            ),
            tasks=(
                "[T-20260101-001]\nTitle: Implement auth\nStatus: open\n"
            ),
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
        self.ws = self._setup_workspace(self.td, decisions=(
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
        ))
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
            f.write(
                "\n---\n\n[D-20260101-003]\n"
                "Statement: Use MySQL for database\n"
                "Status: superseded\n"
            )
        build_index(self.ws, incremental=True)

        results = query_index(self.ws, "database", active_only=True)
        ids = [r["_id"] for r in results]
        self.assertNotIn("D-20260101-003", ids)

    def test_fallback_when_no_index(self):
        """When index doesn't exist, should fall back to filesystem recall."""
        empty = tempfile.mkdtemp()
        try:
            ws = self._setup_workspace(empty, decisions=(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL\n"
                "Status: active\n"
            ))
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
        ws = self._setup_workspace(self.td, decisions=(
            "[D-20260101-001]\nStatement: Test\nStatus: active\n"
        ))
        build_index(ws, incremental=False)
        status = index_status(ws)
        self.assertTrue(status["exists"])
        self.assertGreater(status["blocks"], 0)
        self.assertIsNotNone(status["last_build"])

    def test_stale_detection(self):
        ws = self._setup_workspace(self.td, decisions=(
            "[D-20260101-001]\nStatement: Test\nStatus: active\n"
        ))
        build_index(ws, incremental=False)

        # Modify file
        with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a") as f:
            f.write("\n---\n\n[D-20260101-002]\nStatement: New\nStatus: active\n")

        status = index_status(ws)
        self.assertGreater(status["stale_files"], 0)


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
