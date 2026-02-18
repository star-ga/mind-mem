#!/usr/bin/env python3
"""Tests for hybrid_recall.py -- HybridBackend + RRF fusion."""

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from hybrid_recall import HybridBackend, rrf_fuse, _get_block_id


# ---------------------------------------------------------------------------
# RRF fusion unit tests
# ---------------------------------------------------------------------------


class TestRRFFuse(unittest.TestCase):
    """Test RRF score computation with known rankings."""

    def test_single_list_scores(self):
        """RRF with one list should produce weight/(k+rank)."""
        items = [
            {"_id": "A", "score": 10},
            {"_id": "B", "score": 5},
            {"_id": "C", "score": 1},
        ]
        fused = rrf_fuse([items], weights=[1.0], k=60)
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["_id"], "A")
        # rank=0 -> 1/(60+1) = 0.016393...
        self.assertAlmostEqual(fused[0]["rrf_score"], 1.0 / 61, places=5)
        self.assertAlmostEqual(fused[1]["rrf_score"], 1.0 / 62, places=5)
        self.assertAlmostEqual(fused[2]["rrf_score"], 1.0 / 63, places=5)

    def test_two_lists_fusion(self):
        """RRF with two lists should sum scores for shared IDs."""
        list1 = [{"_id": "A"}, {"_id": "B"}, {"_id": "C"}]
        list2 = [{"_id": "B"}, {"_id": "C"}, {"_id": "D"}]
        fused = rrf_fuse([list1, list2], weights=[1.0, 1.0], k=60)

        scores = {r["_id"]: r["rrf_score"] for r in fused}
        # B appears at rank 0 in list1 (index 1) and rank 0 in list2 (index 0)
        # B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        expected_b = 1.0 / 62 + 1.0 / 61
        self.assertAlmostEqual(scores["B"], expected_b, places=5)
        # D only in list2 at rank 2
        expected_d = 1.0 / 63
        self.assertAlmostEqual(scores["D"], expected_d, places=5)
        # B should rank higher than A (A is only in list1)
        b_idx = next(i for i, r in enumerate(fused) if r["_id"] == "B")
        a_idx = next(i for i, r in enumerate(fused) if r["_id"] == "A")
        self.assertLess(b_idx, a_idx)

    def test_weighted_fusion(self):
        """Weights should scale RRF contributions proportionally."""
        list1 = [{"_id": "A"}]
        list2 = [{"_id": "A"}]
        fused_equal = rrf_fuse([list1, list2], weights=[1.0, 1.0], k=60)
        fused_heavy = rrf_fuse([list1, list2], weights=[2.0, 1.0], k=60)

        score_equal = fused_equal[0]["rrf_score"]
        score_heavy = fused_heavy[0]["rrf_score"]
        # 2x weight on first list -> higher combined score
        self.assertGreater(score_heavy, score_equal)
        # Exact: equal = 2*(1/61), heavy = (2/61 + 1/61) = 3/61
        self.assertAlmostEqual(score_equal, 2.0 / 61, places=5)
        self.assertAlmostEqual(score_heavy, 3.0 / 61, places=5)

    def test_k_parameter_effect(self):
        """Higher k should compress score differences between ranks."""
        items = [{"_id": "A"}, {"_id": "B"}]
        fused_low_k = rrf_fuse([items], weights=[1.0], k=1)
        fused_high_k = rrf_fuse([items], weights=[1.0], k=1000)

        # With low k, difference between rank 0 and rank 1 is larger
        diff_low = fused_low_k[0]["rrf_score"] - fused_low_k[1]["rrf_score"]
        diff_high = fused_high_k[0]["rrf_score"] - fused_high_k[1]["rrf_score"]
        self.assertGreater(diff_low, diff_high)

    def test_empty_lists(self):
        """RRF with empty input should return empty."""
        self.assertEqual(rrf_fuse([], weights=[]), [])
        self.assertEqual(rrf_fuse([[]], weights=[1.0]), [])

    def test_dedup_preserves_first_seen(self):
        """When same ID appears in multiple lists, first-seen data is kept."""
        list1 = [{"_id": "X", "source": "bm25", "excerpt": "from bm25"}]
        list2 = [{"_id": "X", "source": "vector", "excerpt": "from vector"}]
        fused = rrf_fuse([list1, list2], weights=[1.0, 1.0], k=60)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["source"], "bm25")  # first seen wins

    def test_fusion_field_injected(self):
        """Fused results should have 'fusion' and 'rrf_score' fields."""
        items = [{"_id": "A"}]
        fused = rrf_fuse([items], weights=[1.0], k=60)
        self.assertEqual(fused[0]["fusion"], "rrf")
        self.assertIn("rrf_score", fused[0])

    def test_large_candidate_pool(self):
        """Fusion should handle 200+ items per list efficiently."""
        list1 = [{"_id": f"doc-{i}"} for i in range(200)]
        list2 = [{"_id": f"doc-{199 - i}"} for i in range(200)]  # reversed
        fused = rrf_fuse([list1, list2], weights=[1.0, 1.0], k=60)
        self.assertEqual(len(fused), 200)
        # Doc at rank 0 in one list and rank 199 in other should exist
        ids = {r["_id"] for r in fused}
        self.assertIn("doc-0", ids)
        self.assertIn("doc-199", ids)


class TestGetBlockId(unittest.TestCase):
    """Test _get_block_id helper."""

    def test_primary_key(self):
        self.assertEqual(_get_block_id({"_id": "DEC-1"}, "_id"), "DEC-1")

    def test_fallback_keys(self):
        self.assertEqual(_get_block_id({"id": "X"}, "_id"), "X")
        self.assertEqual(_get_block_id({"block_id": "Y"}, "_id"), "Y")

    def test_file_line_fallback(self):
        result = _get_block_id({"file": "a.md", "line": 5}, "_id")
        self.assertEqual(result, "a.md:5")


# ---------------------------------------------------------------------------
# HybridBackend tests
# ---------------------------------------------------------------------------


class TestHybridBackendInit(unittest.TestCase):
    """Test HybridBackend configuration and initialization."""

    def test_default_config(self):
        hb = HybridBackend()
        self.assertEqual(hb.rrf_k, 60)
        self.assertEqual(hb.bm25_weight, 1.0)
        self.assertEqual(hb.vector_weight, 1.0)
        self.assertFalse(hb.vector_enabled)
        self.assertFalse(hb.vector_available)

    def test_custom_config(self):
        cfg = {"rrf_k": 30, "bm25_weight": 2.0, "vector_weight": 0.5}
        hb = HybridBackend(config=cfg)
        self.assertEqual(hb.rrf_k, 30)
        self.assertEqual(hb.bm25_weight, 2.0)
        self.assertEqual(hb.vector_weight, 0.5)

    def test_from_config_factory(self):
        full_config = {"recall": {"rrf_k": 42, "bm25_weight": 1.5}}
        hb = HybridBackend.from_config(full_config)
        self.assertEqual(hb.rrf_k, 42)
        self.assertEqual(hb.bm25_weight, 1.5)

    def test_from_config_empty(self):
        hb = HybridBackend.from_config({})
        self.assertEqual(hb.rrf_k, 60)

    def test_vector_disabled_by_default(self):
        """vector_enabled=false means no vector probe at all."""
        hb = HybridBackend(config={"vector_enabled": False})
        self.assertFalse(hb._vector_available)


class TestHybridBackendFallback(unittest.TestCase):
    """Test that HybridBackend falls back to BM25 when vector is unavailable."""

    def test_empty_query_returns_empty(self):
        hb = HybridBackend()
        self.assertEqual(hb.search("", "/tmp"), [])
        self.assertEqual(hb.search("  ", "/tmp"), [])

    def test_bm25_fallback_no_vector(self):
        """When vector is unavailable, search delegates to BM25."""
        hb = HybridBackend(config={"vector_enabled": False})
        # Patch _bm25_search to verify it gets called
        called = threading.Event()
        def mock_bm25(*a, **kw):
            called.set()
            return [{"_id": "test", "score": 1.0}]

        hb._bm25_search = mock_bm25
        result = hb.search("test query", "/tmp/workspace")
        self.assertTrue(called.is_set())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["_id"], "test")


class TestHybridBackendThreadSafety(unittest.TestCase):
    """Test that concurrent search calls don't interfere."""

    def test_concurrent_rrf_fuse(self):
        """Multiple threads running rrf_fuse concurrently should be safe."""
        results = [None] * 10
        errors = []

        def worker(idx):
            try:
                list1 = [{"_id": f"t{idx}-{i}"} for i in range(50)]
                list2 = [{"_id": f"t{idx}-{49 - i}"} for i in range(50)]
                fused = rrf_fuse([list1, list2], weights=[1.0, 1.0], k=60)
                results[idx] = len(fused)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        for i, count in enumerate(results):
            self.assertEqual(count, 50, f"Thread {i} got {count} results")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation(unittest.TestCase):
    """Test config edge cases."""

    def test_none_config(self):
        hb = HybridBackend(config=None)
        self.assertEqual(hb.rrf_k, 60)

    def test_string_coercion(self):
        """Numeric config values passed as strings should be coerced."""
        cfg = {"rrf_k": "30", "bm25_weight": "2.0", "vector_weight": "0.5"}
        hb = HybridBackend(config=cfg)
        self.assertEqual(hb.rrf_k, 30)
        self.assertAlmostEqual(hb.bm25_weight, 2.0)

    def test_vector_enabled_truthy(self):
        hb = HybridBackend(config={"vector_enabled": True})
        self.assertTrue(hb.vector_enabled)
        # vector_available depends on whether recall_vector is importable
        # (it is in this repo since it's a local script), so just check
        # it returns a bool
        self.assertIsInstance(hb.vector_available, bool)


# ---------------------------------------------------------------------------
# RRF mathematical properties
# ---------------------------------------------------------------------------


class TestRRFMathProperties(unittest.TestCase):
    """Verify mathematical properties of RRF scoring."""

    def test_score_monotonically_decreasing(self):
        """Items ranked higher should always have higher RRF scores."""
        items = [{"_id": f"d{i}"} for i in range(20)]
        fused = rrf_fuse([items], weights=[1.0], k=60)
        for i in range(len(fused) - 1):
            self.assertGreaterEqual(fused[i]["rrf_score"], fused[i + 1]["rrf_score"])

    def test_sum_property(self):
        """Score of doc in both lists = sum of individual contributions."""
        list1 = [{"_id": "A"}, {"_id": "B"}]
        list2 = [{"_id": "A"}, {"_id": "C"}]
        k = 60
        fused = rrf_fuse([list1, list2], weights=[1.0, 1.0], k=k)
        a_score = next(r["rrf_score"] for r in fused if r["_id"] == "A")
        expected = 1.0 / (k + 1) + 1.0 / (k + 1)  # rank 0 in both
        self.assertAlmostEqual(a_score, expected, places=5)

    def test_disjoint_lists(self):
        """Disjoint lists should produce all items with single-list scores."""
        list1 = [{"_id": "A"}, {"_id": "B"}]
        list2 = [{"_id": "C"}, {"_id": "D"}]
        fused = rrf_fuse([list1, list2], weights=[1.0, 1.0], k=60)
        self.assertEqual(len(fused), 4)
        ids = {r["_id"] for r in fused}
        self.assertEqual(ids, {"A", "B", "C", "D"})


if __name__ == "__main__":
    unittest.main()
