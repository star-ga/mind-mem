#!/usr/bin/env python3
"""Tests for Feature 2 (fact card indexing) and Feature 4 (metadata-augmented embeddings)."""

import json
import os
import shutil
import tempfile
import unittest

from mind_mem.sqlite_index import (
    _aggregate_facts_to_parents,
    _connect,
    _init_schema,
    _insert_block,
)


class _WorkspaceMixin:
    """Helper to create a minimal workspace for testing."""

    def _setup_workspace(self, tmpdir, decisions=""):
        for d in ["decisions", "tasks", "entities", "intelligence", "memory"]:
            os.makedirs(os.path.join(tmpdir, d), exist_ok=True)

        with open(os.path.join(tmpdir, "decisions", "DECISIONS.md"), "w") as f:
            f.write(decisions or "# Decisions\n")

        for fname in [
            "tasks/TASKS.md",
            "entities/people.md",
            "entities/tools.md",
            "entities/incidents.md",
            "entities/projects.md",
            "intelligence/CONTRADICTIONS.md",
            "intelligence/DRIFT.md",
            "intelligence/SIGNALS.md",
        ]:
            path = os.path.join(tmpdir, fname)
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(f"# {os.path.basename(fname)}\n")


class TestFactSubBlockIndexing(_WorkspaceMixin, unittest.TestCase):
    """Test that _insert_block creates fact sub-blocks from Statement text."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._setup_workspace(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_fact_cards_created_for_conversation_block(self):
        conn = _connect(self.tmpdir)
        _init_schema(conn)

        block = {
            "_id": "DIA-D1-3",
            "_line": 10,
            "Statement": "I went to a LGBTQ support group yesterday and I started volunteering at the shelter",
            "Tags": "EVENT, Caroline",
            "Date": "2023-05-07",
            "Status": "active",
            "DiaID": "D1:3",
        }
        _insert_block(conn, block, "DIA-D1-3", "decisions/DECISIONS.md", set())
        conn.commit()

        # Check that fact sub-blocks were created
        fact_rows = conn.execute("SELECT * FROM blocks WHERE parent_id = 'DIA-D1-3'").fetchall()
        self.assertGreater(len(fact_rows), 0, "No fact sub-blocks created")

        # Verify fact IDs follow the ::F pattern
        for row in fact_rows:
            self.assertIn("::F", row["id"])
            self.assertTrue(row["id"].startswith("DIA-D1-3::F"))

        # Verify FTS entries exist for fact sub-blocks
        fact_ids = [r["id"] for r in fact_rows]
        for fid in fact_ids:
            fts_row = conn.execute("SELECT * FROM blocks_fts WHERE block_id = ?", (fid,)).fetchone()
            self.assertIsNotNone(fts_row, f"No FTS entry for {fid}")

        conn.close()

    def test_no_facts_for_short_statements(self):
        conn = _connect(self.tmpdir)
        _init_schema(conn)

        block = {
            "_id": "D-001",
            "_line": 1,
            "Statement": "Yes",
            "Tags": "",
            "Status": "active",
        }
        _insert_block(conn, block, "D-001", "decisions/DECISIONS.md", set())
        conn.commit()

        fact_rows = conn.execute("SELECT * FROM blocks WHERE parent_id = 'D-001'").fetchall()
        self.assertEqual(len(fact_rows), 0)
        conn.close()

    def test_parent_block_has_empty_parent_id(self):
        conn = _connect(self.tmpdir)
        _init_schema(conn)

        block = {
            "_id": "D-002",
            "_line": 1,
            "Statement": "I love painting landscapes",
            "Tags": "",
            "Status": "active",
        }
        _insert_block(conn, block, "D-002", "decisions/DECISIONS.md", set())
        conn.commit()

        parent = conn.execute("SELECT parent_id FROM blocks WHERE id = 'D-002'").fetchone()
        self.assertEqual(parent["parent_id"], "")
        conn.close()

    def test_fact_subblocks_inherit_parent_metadata(self):
        conn = _connect(self.tmpdir)
        _init_schema(conn)

        block = {
            "_id": "DIA-D1-5",
            "_line": 20,
            "Statement": "I started learning piano last month",
            "Tags": "EVENT, Alice",
            "Date": "2023-06-15",
            "Status": "active",
            "DiaID": "D1:5",
        }
        _insert_block(conn, block, "DIA-D1-5", "decisions/DECISIONS.md", set())
        conn.commit()

        facts = conn.execute("SELECT * FROM blocks WHERE parent_id = 'DIA-D1-5'").fetchall()
        for fact in facts:
            self.assertEqual(fact["parent_id"], "DIA-D1-5")
            self.assertEqual(fact["dia_id"], "D1:5")
            self.assertEqual(fact["file"], "decisions/DECISIONS.md")
        conn.close()


class TestFactAggregation(unittest.TestCase):
    """Test _aggregate_facts_to_parents() — small-to-big retrieval."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_fact_scores_boost_parent(self):
        conn = _connect(self.tmpdir)
        _init_schema(conn)

        # Insert a parent block
        conn.execute(
            """INSERT INTO blocks (id, type, file, line, status, parent_id, json_blob)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "P-001",
                "D",
                "decisions/DECISIONS.md",
                1,
                "active",
                "",
                json.dumps({"Statement": "Parent content", "_id": "P-001"}),
            ),
        )
        conn.commit()

        results = [
            {
                "_id": "P-001",
                "score": 0.5,
                "file": "f",
                "line": 1,
                "status": "active",
                "type": "D",
                "tags": "",
                "speaker": "",
                "excerpt": "Parent",
            },
            {
                "_id": "P-001::F1",
                "score": 0.9,
                "file": "f",
                "line": 1,
                "status": "active",
                "type": "FACT",
                "tags": "",
                "speaker": "",
                "excerpt": "Fact card",
            },
        ]
        aggregated = _aggregate_facts_to_parents(conn, results)
        conn.close()

        # Fact sub-block should be removed
        ids = [r["_id"] for r in aggregated]
        self.assertNotIn("P-001::F1", ids)
        self.assertIn("P-001", ids)

        # Parent should be boosted
        parent = next(r for r in aggregated if r["_id"] == "P-001")
        self.assertGreater(parent["score"], 0.5)
        self.assertTrue(parent.get("_fact_boost"))

    def test_no_facts_returns_unchanged(self):
        conn = _connect(self.tmpdir)
        _init_schema(conn)
        conn.commit()

        results = [
            {"_id": "A", "score": 0.9},
            {"_id": "B", "score": 0.5},
        ]
        aggregated = _aggregate_facts_to_parents(conn, results)
        conn.close()
        self.assertEqual(len(aggregated), 2)

    def test_missing_parent_injected(self):
        conn = _connect(self.tmpdir)
        _init_schema(conn)

        # Insert parent block in DB but not in results
        conn.execute(
            """INSERT INTO blocks (id, type, file, line, status, parent_id, json_blob)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "MISS-001",
                "D",
                "decisions/DECISIONS.md",
                1,
                "active",
                "",
                json.dumps({"Statement": "Missing parent", "_id": "MISS-001"}),
            ),
        )
        conn.commit()

        # Only the fact card is in results, parent is not
        results = [
            {
                "_id": "MISS-001::F1",
                "score": 0.8,
                "file": "f",
                "line": 1,
                "status": "active",
                "type": "FACT",
                "tags": "",
                "speaker": "",
            },
        ]
        aggregated = _aggregate_facts_to_parents(conn, results)
        conn.close()

        ids = [r["_id"] for r in aggregated]
        self.assertIn("MISS-001", ids)
        self.assertNotIn("MISS-001::F1", ids)

    def test_empty_results(self):
        conn = _connect(self.tmpdir)
        _init_schema(conn)
        conn.commit()
        result = _aggregate_facts_to_parents(conn, [])
        conn.close()
        self.assertEqual(result, [])


class TestMetadataAugmentedEmbeddings(unittest.TestCase):
    """Test Feature 4 — metadata-augmented embeddings in VectorBackend."""

    def test_augment_prepends_metadata(self):
        from mind_mem.recall_vector import VectorBackend

        block = {
            "Category": "FACT",
            "Speaker": "Caroline",
            "Date": "2023-05-07",
            "Tags": "identity, personal",
        }
        augmented = VectorBackend._augment_for_embedding(block, "She is a nurse")
        self.assertIn("[FACT]", augmented)
        self.assertIn("[Caroline]", augmented)
        self.assertIn("[2023-05-07]", augmented)
        self.assertIn("[identity, personal]", augmented)
        self.assertIn("She is a nurse", augmented)

    def test_augment_handles_missing_metadata(self):
        from mind_mem.recall_vector import VectorBackend

        block = {}
        augmented = VectorBackend._augment_for_embedding(block, "raw text")
        self.assertEqual(augmented, "raw text")

    def test_augment_uses_fallback_keys(self):
        from mind_mem.recall_vector import VectorBackend

        block = {"type": "EVENT", "speaker": "Alice", "date": "2023-01"}
        augmented = VectorBackend._augment_for_embedding(block, "went to gym")
        self.assertIn("[EVENT]", augmented)
        self.assertIn("[Alice]", augmented)
        self.assertIn("[2023-01]", augmented)

    def test_augment_truncates_long_tags(self):
        from mind_mem.recall_vector import VectorBackend

        block = {"Tags": "a" * 100}
        augmented = VectorBackend._augment_for_embedding(block, "text")
        # Tags should be truncated to 50 chars
        tag_start = augmented.index("[")
        tag_end = augmented.index("]")
        self.assertLessEqual(tag_end - tag_start - 1, 50)


class TestValidRecallKeys(unittest.TestCase):
    """Test that new config keys are registered."""

    def test_knee_cutoff_key_valid(self):
        from mind_mem._recall_constants import _VALID_RECALL_KEYS

        self.assertIn("knee_cutoff", _VALID_RECALL_KEYS)

    def test_min_score_key_valid(self):
        from mind_mem._recall_constants import _VALID_RECALL_KEYS

        self.assertIn("min_score", _VALID_RECALL_KEYS)


class TestFactDeletion(_WorkspaceMixin, unittest.TestCase):
    """Test that deleting parent blocks also deletes fact sub-blocks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._setup_workspace(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_delete_parent_removes_fact_children(self):
        from mind_mem.sqlite_index import _delete_blocks

        conn = _connect(self.tmpdir)
        _init_schema(conn)

        # Insert parent + fact child
        block = {
            "_id": "DEL-001",
            "_line": 1,
            "Statement": "I went to the doctor yesterday",
            "Tags": "EVENT, Bob",
            "Status": "active",
        }
        _insert_block(conn, block, "DEL-001", "decisions/DECISIONS.md", set())
        conn.commit()

        # Verify fact children exist
        children = conn.execute("SELECT id FROM blocks WHERE parent_id = 'DEL-001'").fetchall()
        self.assertGreater(len(children), 0)

        # Delete parent
        _delete_blocks(conn, ["DEL-001"], "decisions/DECISIONS.md")
        conn.commit()

        # Verify all children gone
        remaining = conn.execute("SELECT id FROM blocks WHERE parent_id = 'DEL-001'").fetchall()
        self.assertEqual(len(remaining), 0)

        # Verify parent gone
        parent = conn.execute("SELECT id FROM blocks WHERE id = 'DEL-001'").fetchone()
        self.assertIsNone(parent)
        conn.close()


if __name__ == "__main__":
    unittest.main()
