"""Tests for A-MEM block metadata integration in recall pipeline."""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestBlockMetadataIntegration(unittest.TestCase):
    """Test that BlockMetadataManager is wired into recall."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        os.makedirs(os.path.join(self.td, ".mind-mem"))
        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL for the database\n"
                "Status: active\n"
                "Date: 2026-01-01\n"
                "Tags: database\n"
                "\n---\n\n"
                "[D-20260102-001]\n"
                "Statement: Use Redis for caching layer\n"
                "Status: active\n"
                "Date: 2026-01-02\n"
                "Tags: caching\n"
            )

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_importance_boost_applied(self):
        """Verify that A-MEM importance modifies recall scores."""
        from block_metadata import BlockMetadataManager
        db_path = os.path.join(self.td, ".mind-mem", "block_meta.db")
        mgr = BlockMetadataManager(db_path)

        # Pre-seed high importance for one block
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT OR REPLACE INTO block_meta (id, importance, access_count) "
            "VALUES (?, ?, ?)", ("D-20260101-001", 1.5, 100)
        )
        conn.commit()
        conn.close()

        boost = mgr.get_importance_boost("D-20260101-001")
        self.assertEqual(boost, 1.5)

        default = mgr.get_importance_boost("D-20260102-001")
        self.assertEqual(default, 1.0)

    def test_record_access_updates_count(self):
        """Verify record_access increments access counts."""
        from block_metadata import BlockMetadataManager
        db_path = os.path.join(self.td, ".mind-mem", "block_meta.db")
        mgr = BlockMetadataManager(db_path)

        mgr.record_access(["D-20260101-001", "D-20260102-001"], query="database")

        import sqlite3
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT access_count FROM block_meta WHERE id = ?",
            ("D-20260101-001",)
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], 1)

    def test_evolve_keywords(self):
        """Verify evolve_keywords adds query tokens to block keywords."""
        from block_metadata import BlockMetadataManager
        db_path = os.path.join(self.td, ".mind-mem", "block_meta.db")
        mgr = BlockMetadataManager(db_path)

        mgr.record_access(["D-20260101-001"])
        mgr.evolve_keywords("D-20260101-001", ["postgresql", "database"],
                            "Use PostgreSQL for the database")

        import sqlite3
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT keywords FROM block_meta WHERE id = ?",
            ("D-20260101-001",)
        ).fetchone()
        conn.close()
        self.assertIn("postgresql", row[0])
        self.assertIn("database", row[0])

    def test_graceful_degradation_missing_dir(self):
        """Verify recall works when .mind-mem directory is missing."""
        from recall import recall as r
        # Remove the .mind-mem dir
        shutil.rmtree(os.path.join(self.td, ".mind-mem"))
        results = r(self.td, "PostgreSQL", limit=5)
        self.assertGreater(len(results), 0)

    def test_importance_affects_ranking(self):
        """Block with higher importance should rank higher (all else equal)."""
        from block_metadata import BlockMetadataManager
        db_path = os.path.join(self.td, ".mind-mem", "block_meta.db")
        BlockMetadataManager(db_path)  # ensure table exists

        # Give D-20260102-001 very high importance
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT OR REPLACE INTO block_meta (id, importance) VALUES (?, ?)",
            ("D-20260102-001", 1.5)
        )
        conn.execute(
            "INSERT OR REPLACE INTO block_meta (id, importance) VALUES (?, ?)",
            ("D-20260101-001", 0.8)
        )
        conn.commit()
        conn.close()

        from recall import recall as r
        # Both blocks should match "active" — importance should reorder them
        results = r(self.td, "active status", limit=10)
        if len(results) >= 2:
            # The one with importance 1.5 should score higher
            ids = [r_["_id"] for r_ in results]
            if "D-20260102-001" in ids and "D-20260101-001" in ids:
                idx_high = ids.index("D-20260102-001")
                idx_low = ids.index("D-20260101-001")
                self.assertLess(idx_high, idx_low)

    def test_co_occurrence_tracking(self):
        """Verify co-occurrence is recorded between co-returned blocks."""
        from block_metadata import BlockMetadataManager
        db_path = os.path.join(self.td, ".mind-mem", "block_meta.db")
        mgr = BlockMetadataManager(db_path)

        mgr.record_access(["D-20260101-001", "D-20260102-001"])
        cooccurring = mgr.get_co_occurring_blocks("D-20260101-001")
        self.assertIn("D-20260102-001", cooccurring)


if __name__ == "__main__":
    unittest.main()
