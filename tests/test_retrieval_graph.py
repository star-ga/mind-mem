#!/usr/bin/env python3
"""Tests for retrieval_graph.py — retrieval logging, co-retrieval graph, hard negatives."""

import json
import os
import tempfile
import unittest

from mind_mem.retrieval_graph import (
    _connect,
    ensure_graph_tables,
    get_hard_negative_ids,
    log_retrieval,
    propagate_scores,
    record_hard_negatives,
)


class TestEnsureGraphTables(unittest.TestCase):
    """Test schema creation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_creates_tables(self):
        ensure_graph_tables(self.tmpdir)
        conn = _connect(self.tmpdir)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self.assertIn("retrieval_log", tables)
        self.assertIn("co_retrieval", tables)
        self.assertIn("hard_negatives", tables)
        conn.close()

    def test_idempotent(self):
        ensure_graph_tables(self.tmpdir)
        ensure_graph_tables(self.tmpdir)  # Should not error


class TestLogRetrieval(unittest.TestCase):
    """Test retrieval logging and co-retrieval edge updates."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_log_empty_results_noop(self):
        log_retrieval(self.tmpdir, "test query", [])
        # No tables created since results are empty
        db_path = os.path.join(self.tmpdir, ".mind-mem-index", "recall.db")
        self.assertFalse(os.path.exists(db_path))

    def test_log_single_result(self):
        results = [{"_id": "D-001", "score": 0.9}]
        log_retrieval(self.tmpdir, "test query", results)
        conn = _connect(self.tmpdir)
        rows = conn.execute("SELECT * FROM retrieval_log").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query_text"], "test query")
        self.assertEqual(json.loads(rows[0]["mem_ids"]), ["D-001"])
        conn.close()

    def test_co_retrieval_edges(self):
        results = [
            {"_id": "A", "score": 0.9},
            {"_id": "B", "score": 0.8},
            {"_id": "C", "score": 0.7},
        ]
        log_retrieval(self.tmpdir, "query 1", results)
        conn = _connect(self.tmpdir)
        edges = conn.execute("SELECT * FROM co_retrieval ORDER BY mem1_id, mem2_id").fetchall()
        # 3 blocks → 3 edges: (A,B), (A,C), (B,C)
        self.assertEqual(len(edges), 3)
        pairs = {(e["mem1_id"], e["mem2_id"]) for e in edges}
        self.assertIn(("A", "B"), pairs)
        self.assertIn(("A", "C"), pairs)
        self.assertIn(("B", "C"), pairs)
        conn.close()

    def test_co_retrieval_weight_accumulates(self):
        results = [{"_id": "X", "score": 0.9}, {"_id": "Y", "score": 0.8}]
        log_retrieval(self.tmpdir, "q1", results)
        log_retrieval(self.tmpdir, "q2", results)
        conn = _connect(self.tmpdir)
        row = conn.execute(
            "SELECT weight, hit_count FROM co_retrieval WHERE mem1_id=? AND mem2_id=?", ("X", "Y")
        ).fetchone()
        self.assertEqual(row["hit_count"], 2)
        self.assertGreater(row["weight"], 0.5)  # 0.5 + 0.5 = 1.0
        conn.close()

    def test_no_crash_on_missing_ids(self):
        results = [{"score": 0.5}, {"_id": "", "score": 0.3}]
        log_retrieval(self.tmpdir, "test", results)  # Should not crash


class TestPropagateScores(unittest.TestCase):
    """Test PageRank-like score propagation across co-retrieval graph."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_no_graph_returns_initial(self):
        initial = {"A": 0.9, "B": 0.5}
        result = propagate_scores(self.tmpdir, initial)
        self.assertEqual(result["A"], 0.9)
        self.assertEqual(result["B"], 0.5)

    def test_propagation_boosts_neighbors(self):
        # Create some co-retrieval edges
        results = [
            {"_id": "A", "score": 0.9},
            {"_id": "B", "score": 0.8},
            {"_id": "C", "score": 0.7},
        ]
        # Log multiple times to build up edge weights above min_edge
        for _ in range(5):
            log_retrieval(self.tmpdir, "shared topic", results)

        # Now propagate from A only
        initial = {"A": 1.0}
        propagated = propagate_scores(self.tmpdir, initial, iterations=3, damping=0.3)
        # B and C should get non-zero scores via edges from A
        self.assertIn("B", propagated)
        self.assertIn("C", propagated)
        self.assertGreater(propagated.get("B", 0), 0)

    def test_empty_initial_scores(self):
        result = propagate_scores(self.tmpdir, {})
        self.assertEqual(result, {})


class TestRecordHardNegatives(unittest.TestCase):
    """Test hard negative recording."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_records_hard_negative(self):
        candidates = [
            {"_id": "HN-1", "score": 0.8, "ce_score": 0.1},  # Hard neg: high BM25, low CE
            {"_id": "OK-1", "score": 0.7, "ce_score": 0.9},  # Not a hard neg
        ]
        count = record_hard_negatives(self.tmpdir, "test query", candidates)
        self.assertEqual(count, 1)

    def test_get_hard_negative_ids(self):
        candidates = [
            {"_id": "HN-A", "score": 0.5, "ce_score": 0.1},
            {"_id": "HN-B", "score": 0.3, "ce_score": 0.2},
        ]
        record_hard_negatives(self.tmpdir, "q1", candidates)
        ids = get_hard_negative_ids(self.tmpdir)
        self.assertIn("HN-A", ids)
        self.assertIn("HN-B", ids)

    def test_no_hard_negatives_when_ce_high(self):
        candidates = [
            {"_id": "OK-1", "score": 0.9, "ce_score": 0.8},
        ]
        count = record_hard_negatives(self.tmpdir, "q1", candidates)
        self.assertEqual(count, 0)

    def test_no_hard_negatives_when_bm25_low(self):
        candidates = [
            {"_id": "LOW-1", "score": 0.05, "ce_score": 0.1},
        ]
        count = record_hard_negatives(self.tmpdir, "q1", candidates)
        self.assertEqual(count, 0)

    def test_empty_when_no_records(self):
        ids = get_hard_negative_ids(self.tmpdir)
        self.assertEqual(ids, set())


class TestKneeCutoff(unittest.TestCase):
    """Test knee score cutoff (adaptive top-K truncation)."""

    def test_basic_knee_detection(self):
        from mind_mem._recall_core import knee_cutoff

        # Clear knee at position 2: 0.9, 0.8, 0.1, 0.05
        results = [
            {"score": 0.9, "_id": "A"},
            {"score": 0.8, "_id": "B"},
            {"score": 0.1, "_id": "C"},
            {"score": 0.05, "_id": "D"},
        ]
        cut = knee_cutoff(results)
        self.assertEqual(len(cut), 2)  # Cuts at the 0.8→0.1 drop

    def test_no_knee_returns_all(self):
        from mind_mem._recall_core import knee_cutoff

        # Gradual decline — no sharp drop
        results = [
            {"score": 0.9, "_id": "A"},
            {"score": 0.85, "_id": "B"},
            {"score": 0.80, "_id": "C"},
            {"score": 0.75, "_id": "D"},
        ]
        cut = knee_cutoff(results)
        self.assertEqual(len(cut), 4)

    def test_min_results_respected(self):
        from mind_mem._recall_core import knee_cutoff

        results = [
            {"score": 0.9, "_id": "A"},
            {"score": 0.01, "_id": "B"},
        ]
        cut = knee_cutoff(results, min_results=2)
        self.assertEqual(len(cut), 2)

    def test_min_score_filter(self):
        from mind_mem._recall_core import knee_cutoff

        results = [
            {"score": 0.9, "_id": "A"},
            {"score": 0.8, "_id": "B"},
            {"score": 0.7, "_id": "C"},
        ]
        cut = knee_cutoff(results, min_score=0.75)
        self.assertTrue(all(r["score"] >= 0.75 for r in cut))

    def test_empty_input(self):
        from mind_mem._recall_core import knee_cutoff

        self.assertEqual(knee_cutoff([]), [])

    def test_single_result(self):
        from mind_mem._recall_core import knee_cutoff

        results = [{"score": 0.5, "_id": "A"}]
        self.assertEqual(knee_cutoff(results), results)

    def test_zero_scores(self):
        from mind_mem._recall_core import knee_cutoff

        results = [{"score": 0, "_id": "A"}, {"score": 0, "_id": "B"}]
        cut = knee_cutoff(results)
        self.assertEqual(len(cut), 2)


if __name__ == "__main__":
    unittest.main()
