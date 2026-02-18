"""Tests for A-MEM block metadata evolution."""

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

# Allow imports from scripts/
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from block_metadata import BlockMetadataManager


class TestBlockMetadataManager(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "index.db")
        self.mgr = BlockMetadataManager(self.db_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_access_creates_entry(self):
        self.mgr.record_access(["D-001"], "test query")
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT access_count FROM block_meta WHERE id='D-001'").fetchone()
        conn.close()
        self.assertEqual(row[0], 1)

    def test_record_access_increments(self):
        self.mgr.record_access(["D-001"])
        self.mgr.record_access(["D-001"])
        self.mgr.record_access(["D-001"])
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT access_count FROM block_meta WHERE id='D-001'").fetchone()
        conn.close()
        self.assertEqual(row[0], 3)

    def test_importance_range(self):
        self.mgr.record_access(["D-001"])
        imp = self.mgr.update_importance("D-001")
        self.assertGreaterEqual(imp, 0.8)
        self.assertLessEqual(imp, 1.5)

    def test_importance_missing_block(self):
        imp = self.mgr.update_importance("NONEXISTENT")
        self.assertEqual(imp, 1.0)

    def test_importance_boost_default(self):
        boost = self.mgr.get_importance_boost("NONEXISTENT")
        self.assertEqual(boost, 1.0)

    def test_importance_increases_with_access(self):
        self.mgr.record_access(["D-001"])
        imp1 = self.mgr.update_importance("D-001")
        for _ in range(10):
            self.mgr.record_access(["D-001"])
        imp2 = self.mgr.update_importance("D-001")
        self.assertGreaterEqual(imp2, imp1)

    def test_keyword_evolution(self):
        self.mgr.record_access(["D-001"])
        self.mgr.evolve_keywords("D-001", ["auth", "login"], "authentication login system")
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT keywords FROM block_meta WHERE id='D-001'").fetchone()
        conn.close()
        keywords = row[0].split(",") if row[0] else []
        self.assertIn("auth", keywords)
        self.assertIn("login", keywords)

    def test_keyword_respects_max(self):
        self.mgr.record_access(["D-001"])
        tokens = [f"word{i}" for i in range(30)]
        content = " ".join(tokens)
        self.mgr.evolve_keywords("D-001", tokens, content, max_keywords=5)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT keywords FROM block_meta WHERE id='D-001'").fetchone()
        conn.close()
        keywords = row[0].split(",") if row[0] else []
        self.assertLessEqual(len(keywords), 5)

    def test_co_occurrence_tracking(self):
        self.mgr.record_access(["D-001", "D-002", "D-003"])
        co = self.mgr.get_co_occurring_blocks("D-001")
        self.assertIn("D-002", co)
        self.assertIn("D-003", co)

    def test_co_occurrence_empty(self):
        co = self.mgr.get_co_occurring_blocks("NONEXISTENT")
        self.assertEqual(co, [])

    def test_graceful_degradation_bad_db(self):
        """Manager should not crash with bad DB path."""
        mgr = BlockMetadataManager("/nonexistent/path/db.sqlite")
        mgr.record_access(["D-001"])  # Should not raise
        self.assertEqual(mgr.get_importance_boost("D-001"), 1.0)

    def test_record_access_empty_list(self):
        self.mgr.record_access([])  # Should not raise


if __name__ == "__main__":
    unittest.main()
