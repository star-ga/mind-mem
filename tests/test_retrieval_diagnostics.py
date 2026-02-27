#!/usr/bin/env python3
"""Tests for retrieval diagnostics (#428), corpus isolation (#429), and intent instrumentation (#430)."""

import json
import os
import shutil
import tempfile
import unittest

from mind_mem.retrieval_graph import (
    _connect,
    log_retrieval,
    record_hard_negatives,
    retrieval_diagnostics,
)


class TestRetrievalDiagnostics(unittest.TestCase):
    """Test the retrieval_diagnostics aggregation function (#428)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_empty_db(self):
        result = retrieval_diagnostics(self.tmpdir)
        self.assertEqual(result["queries_analyzed"], 0)
        self.assertEqual(result["intent_distribution"], {})
        self.assertEqual(result["stage_stats"], {})

    def test_aggregates_stage_counts(self):
        # Log a query with stage counts
        log_retrieval(
            self.tmpdir,
            "test query",
            [{"_id": "D-001", "score": 0.9}],
            intent_type="WHY",
            stage_counts={
                "corpus_loaded": 100,
                "bm25_passed": 20,
                "deduped": 15,
                "final": 5,
            },
        )
        result = retrieval_diagnostics(self.tmpdir, last_n=10)
        self.assertEqual(result["queries_analyzed"], 1)
        self.assertEqual(result["intent_distribution"], {"WHY": 1})
        self.assertIn("corpus_loaded", result["stage_stats"])
        self.assertEqual(result["stage_stats"]["corpus_loaded"]["avg"], 100)
        self.assertEqual(result["stage_stats"]["bm25_passed"]["avg"], 20)
        self.assertEqual(result["stage_stats"]["final"]["avg"], 5)

    def test_rejection_rates(self):
        log_retrieval(
            self.tmpdir,
            "q1",
            [{"_id": "D-001", "score": 0.8}],
            intent_type="WHAT",
            stage_counts={
                "corpus_loaded": 200,
                "bm25_passed": 50,
                "deduped": 40,
                "reranked": 40,
                "final": 10,
            },
        )
        result = retrieval_diagnostics(self.tmpdir)
        rates = result["rejection_rates"]
        # corpus_loaded→bm25_passed should show 75% rejection
        self.assertAlmostEqual(rates["corpus_loaded_to_bm25_passed"], 0.75, places=2)

    def test_multiple_queries_aggregate(self):
        for i in range(5):
            log_retrieval(
                self.tmpdir,
                f"query {i}",
                [{"_id": f"D-{i:03d}", "score": 0.5 + i * 0.1}],
                intent_type="WHY" if i < 3 else "WHEN",
                stage_counts={"corpus_loaded": 100, "bm25_passed": 20 + i},
            )
        result = retrieval_diagnostics(self.tmpdir)
        self.assertEqual(result["queries_analyzed"], 5)
        self.assertEqual(result["intent_distribution"]["WHY"], 3)
        self.assertEqual(result["intent_distribution"]["WHEN"], 2)

    def test_hard_negatives_summary(self):
        record_hard_negatives(
            self.tmpdir,
            "test query",
            [
                {"_id": "D-001", "score": 0.5, "ce_score": 0.1},
                {"_id": "D-002", "score": 0.3, "ce_score": 0.2},
            ],
        )
        result = retrieval_diagnostics(self.tmpdir)
        self.assertEqual(result["hard_negatives"]["total"], 2)
        self.assertEqual(result["hard_negatives"]["unique_blocks"], 2)

    def test_score_distribution(self):
        for i in range(10):
            log_retrieval(
                self.tmpdir,
                f"q{i}",
                [{"_id": f"D-{i:03d}", "score": 0.1 * (i + 1)}],
                intent_type="WHAT",
            )
        result = retrieval_diagnostics(self.tmpdir)
        dist = result["score_distribution"]
        self.assertIn("p50", dist)
        self.assertIn("avg_final_count", dist)
        self.assertAlmostEqual(dist["avg_final_count"], 1.0, places=1)


class TestIntentQualityTracking(unittest.TestCase):
    """Test per-intent quality breakdown (#430)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_intent_quality_summary(self):
        # WHY queries with high scores
        for i in range(3):
            log_retrieval(
                self.tmpdir,
                f"why q{i}",
                [{"_id": f"D-{i:03d}", "score": 0.9}],
                intent_type="WHY",
                stage_counts={"intent_confidence": 0.8},
            )
        # ENTITY queries with low scores
        for i in range(2):
            log_retrieval(
                self.tmpdir,
                f"entity q{i}",
                [{"_id": f"E-{i:03d}", "score": 0.3}],
                intent_type="ENTITY",
                stage_counts={"intent_confidence": 0.5},
            )
        result = retrieval_diagnostics(self.tmpdir)
        iq = result["intent_quality"]
        self.assertIn("WHY", iq)
        self.assertIn("ENTITY", iq)
        self.assertGreater(iq["WHY"]["avg_top_score"], iq["ENTITY"]["avg_top_score"])
        self.assertEqual(iq["WHY"]["queries"], 3)
        self.assertAlmostEqual(iq["WHY"]["avg_confidence"], 0.8, places=2)

    def test_low_confidence_queries(self):
        log_retrieval(
            self.tmpdir,
            "ambiguous query here",
            [{"_id": "D-001", "score": 0.4}],
            intent_type="WHAT",
            stage_counts={"intent_confidence": 0.1},
        )
        log_retrieval(
            self.tmpdir,
            "clear why query",
            [{"_id": "D-002", "score": 0.9}],
            intent_type="WHY",
            stage_counts={"intent_confidence": 0.9},
        )
        result = retrieval_diagnostics(self.tmpdir)
        low = result["low_confidence_queries"]
        self.assertEqual(len(low), 1)
        self.assertIn("ambiguous", low[0]["query"])
        self.assertAlmostEqual(low[0]["confidence"], 0.1, places=2)


class TestLogRetrievalExtended(unittest.TestCase):
    """Test extended log_retrieval with intent_type and stage_counts."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_log_with_intent_and_stages(self):
        log_retrieval(
            self.tmpdir,
            "why did X happen",
            [{"_id": "D-001", "score": 0.9}],
            intent_type="WHY",
            stage_counts={"corpus_loaded": 50, "bm25_passed": 10, "final": 3},
        )
        conn = _connect(self.tmpdir)
        row = conn.execute("SELECT * FROM retrieval_log LIMIT 1").fetchone()
        self.assertEqual(row["intent_type"], "WHY")
        sc = json.loads(row["stage_counts"])
        self.assertEqual(sc["corpus_loaded"], 50)
        self.assertEqual(sc["final"], 3)
        conn.close()

    def test_log_backward_compatible(self):
        """Old callers without intent_type/stage_counts still work."""
        log_retrieval(self.tmpdir, "old query", [{"_id": "D-001", "score": 0.5}])
        conn = _connect(self.tmpdir)
        row = conn.execute("SELECT * FROM retrieval_log LIMIT 1").fetchone()
        self.assertEqual(row["intent_type"], "")
        self.assertEqual(json.loads(row["stage_counts"]), {})
        conn.close()


class TestCorpusIsolation(unittest.TestCase):
    """Test that pending signals are excluded from recall (#429)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create workspace structure for recall
        intel_dir = os.path.join(self.tmpdir, "intelligence")
        os.makedirs(intel_dir, exist_ok=True)
        decisions_dir = os.path.join(self.tmpdir, "decisions")
        os.makedirs(decisions_dir, exist_ok=True)
        # Write a DECISIONS.md with one active decision
        with open(os.path.join(decisions_dir, "DECISIONS.md"), "w") as f:
            f.write("# Decisions\n\n[D-20260226-001]\n"
                    "Date: 2026-02-26\nStatus: active\n"
                    "Statement: Use BM25 for scoring\n")
        # Write SIGNALS.md with one pending and one active signal
        with open(os.path.join(intel_dir, "SIGNALS.md"), "w") as f:
            f.write(
                "# Signals\n\n"
                "[SIG-20260226-001]\n"
                "Date: 2026-02-26\nStatus: pending\n"
                "Statement: Poisoned proposal should not appear\n\n"
                "[SIG-20260226-002]\n"
                "Date: 2026-02-26\nStatus: active\n"
                "Statement: Approved signal about scoring\n"
            )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_default_excludes_pending(self):
        from mind_mem._recall_core import recall

        results = recall(self.tmpdir, "scoring", limit=20)
        ids = [r["_id"] for r in results]
        # Pending signal should NOT appear
        self.assertNotIn("SIG-20260226-001", ids)
        # Active signal and decision may appear if they match
        # (depends on BM25 scoring, but at least pending is excluded)

    def test_include_pending_flag(self):
        from mind_mem._recall_core import recall

        results = recall(self.tmpdir, "poisoned proposal", limit=20, include_pending=True)
        ids = [r["_id"] for r in results]
        # With include_pending=True, the pending signal CAN appear
        # (it matches the query "poisoned proposal")
        self.assertIn("SIG-20260226-001", ids)


if __name__ == "__main__":
    unittest.main()
