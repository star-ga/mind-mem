#!/usr/bin/env python3
"""Tests for baseline snapshot and drift detection (#431)."""

import json
import os
import shutil
import tempfile
import unittest

from mind_mem.baseline_snapshot import (
    _chi_squared,
    _config_fingerprint,
    _list_versions,
    _next_version,
    compare_baselines,
    detect_drift,
    freeze_baseline,
    list_baselines,
)
from mind_mem.retrieval_graph import log_retrieval


def _populate_queries(workspace: str, intent_counts: dict[str, int]) -> None:
    """Helper: log retrieval queries to populate diagnostics data."""
    for intent, count in intent_counts.items():
        for i in range(count):
            conf = 0.8 if intent != "unknown" else 0.2
            log_retrieval(
                workspace,
                f"test query {intent} {i}",
                [{"_id": f"D-{intent}-{i}", "score": 0.85}],
                intent_type=intent,
                stage_counts={
                    "corpus_loaded": 100,
                    "bm25_passed": 50,
                    "final": 5,
                    "intent_confidence": conf,
                },
            )


class TestChiSquared(unittest.TestCase):
    """Test chi-squared goodness-of-fit implementation."""

    def test_identical_distributions(self):
        dist = {"WHAT": 10, "WHEN": 5, "HOW": 3}
        result = _chi_squared(dist, dist)
        self.assertAlmostEqual(result["chi2"], 0.0, places=2)
        self.assertEqual(result["p_value"], 1.0)

    def test_different_distributions(self):
        observed = {"WHAT": 20, "WHEN": 5, "HOW": 5}
        expected = {"WHAT": 10, "WHEN": 10, "HOW": 10}
        result = _chi_squared(observed, expected)
        self.assertGreater(result["chi2"], 0)
        self.assertLess(result["p_value"], 1.0)
        self.assertIn("WHAT", result["per_intent"])

    def test_empty_distributions(self):
        result = _chi_squared({}, {})
        self.assertEqual(result["chi2"], 0.0)
        self.assertEqual(result["p_value"], 1.0)

    def test_new_intent_in_observed(self):
        observed = {"WHAT": 10, "WHEN": 5, "TRACE": 3}
        expected = {"WHAT": 10, "WHEN": 5}
        result = _chi_squared(observed, expected)
        self.assertIn("TRACE", result["per_intent"])
        self.assertGreater(result["chi2"], 0)

    def test_missing_intent_in_observed(self):
        observed = {"WHAT": 10}
        expected = {"WHAT": 10, "WHEN": 5}
        result = _chi_squared(observed, expected)
        self.assertIn("WHEN", result["per_intent"])
        self.assertEqual(result["per_intent"]["WHEN"]["observed"], 0)

    def test_scaling(self):
        """Expected is scaled to match observed total — tests shape comparison."""
        observed = {"WHAT": 20, "WHEN": 10}  # total=30, ratio 2:1
        expected = {"WHAT": 10, "WHEN": 5}  # total=15, ratio 2:1
        result = _chi_squared(observed, expected)
        self.assertAlmostEqual(result["chi2"], 0.0, places=2)


class TestConfigFingerprint(unittest.TestCase):
    """Test config fingerprint capture."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_default_config(self):
        """Returns defaults when no config file exists."""
        fp = _config_fingerprint(self.tmpdir)
        self.assertEqual(fp["rrf_k"], 60)
        self.assertEqual(fp["vector_model"], "all-MiniLM-L6-v2")
        self.assertIn("_hash", fp)

    def test_reads_config_file(self):
        cfg = {"recall": {"rrf_k": 100, "vector_model": "custom-model"}}
        with open(os.path.join(self.tmpdir, "mind-mem.json"), "w") as f:
            json.dump(cfg, f)
        fp = _config_fingerprint(self.tmpdir)
        self.assertEqual(fp["rrf_k"], 100)
        self.assertEqual(fp["vector_model"], "custom-model")

    def test_hash_changes_with_config(self):
        fp1 = _config_fingerprint(self.tmpdir)
        cfg = {"recall": {"rrf_k": 999}}
        with open(os.path.join(self.tmpdir, "mind-mem.json"), "w") as f:
            json.dump(cfg, f)
        fp2 = _config_fingerprint(self.tmpdir)
        self.assertNotEqual(fp1["_hash"], fp2["_hash"])


class TestFreezeBaseline(unittest.TestCase):
    """Test baseline freeze (snapshot capture)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_freeze_empty_workspace(self):
        result = freeze_baseline(self.tmpdir)
        self.assertEqual(result["version_tag"], "v1")
        self.assertEqual(result["queries_analyzed"], 0)
        self.assertEqual(result["intent_distribution"], {})
        self.assertIn("config_fingerprint", result)
        # File written
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "intelligence/baselines/intent-baseline-v1.json")))

    def test_freeze_with_data(self):
        _populate_queries(self.tmpdir, {"WHAT": 10, "WHEN": 5, "HOW": 3})
        result = freeze_baseline(self.tmpdir)
        self.assertEqual(result["version_tag"], "v1")
        self.assertGreater(result["queries_analyzed"], 0)
        self.assertIn("WHAT", result["intent_distribution"])
        self.assertIn("WHEN", result["intent_distribution"])

    def test_auto_version_increment(self):
        freeze_baseline(self.tmpdir)
        result2 = freeze_baseline(self.tmpdir)
        self.assertEqual(result2["version_tag"], "v2")
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "intelligence/baselines/intent-baseline-v2.json")))

    def test_explicit_tag(self):
        result = freeze_baseline(self.tmpdir, tag=42)
        self.assertEqual(result["version_tag"], "v42")

    def test_low_confidence_fraction(self):
        _populate_queries(self.tmpdir, {"WHAT": 8, "unknown": 2})
        result = freeze_baseline(self.tmpdir)
        # "unknown" queries have confidence 0.2 (< 0.3 threshold)
        self.assertGreater(result["low_confidence_fraction"], 0)

    def test_baseline_is_valid_json(self):
        _populate_queries(self.tmpdir, {"WHAT": 5})
        freeze_baseline(self.tmpdir)
        path = os.path.join(self.tmpdir, "intelligence/baselines/intent-baseline-v1.json")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("frozen_at", data)
        self.assertIn("config_fingerprint", data)


class TestListBaselines(unittest.TestCase):
    """Test baseline listing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_no_baselines(self):
        result = list_baselines(self.tmpdir)
        self.assertEqual(result, [])

    def test_lists_frozen_baselines(self):
        freeze_baseline(self.tmpdir)
        freeze_baseline(self.tmpdir)
        result = list_baselines(self.tmpdir)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["version_tag"], "v1")
        self.assertEqual(result[1]["version_tag"], "v2")


class TestDetectDrift(unittest.TestCase):
    """Test drift detection against a baseline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_no_baselines_error(self):
        result = detect_drift(self.tmpdir)
        self.assertIn("error", result)

    def test_no_drift_same_data(self):
        _populate_queries(self.tmpdir, {"WHAT": 10, "WHEN": 5})
        freeze_baseline(self.tmpdir)
        result = detect_drift(self.tmpdir)
        self.assertFalse(result["significant_drift"])
        self.assertIn("baseline_version", result)
        self.assertIn("chi_squared_test", result)

    def test_detects_distribution_shift(self):
        # Freeze baseline with one distribution
        _populate_queries(self.tmpdir, {"WHAT": 20, "WHEN": 5})
        freeze_baseline(self.tmpdir)

        # Add queries with very different distribution
        _populate_queries(self.tmpdir, {"WHEN": 40, "TRACE": 20})
        result = detect_drift(self.tmpdir)
        # The chi2 should be non-trivial
        self.assertGreater(result["chi_squared_test"]["chi2"], 0)

    def test_config_change_detection(self):
        _populate_queries(self.tmpdir, {"WHAT": 10})
        freeze_baseline(self.tmpdir)

        # Change config
        cfg = {"recall": {"rrf_k": 999, "vector_model": "changed-model"}}
        with open(os.path.join(self.tmpdir, "mind-mem.json"), "w") as f:
            json.dump(cfg, f)

        result = detect_drift(self.tmpdir)
        self.assertTrue(result["config_changed"])
        self.assertIsNotNone(result["config_diff"])

    def test_low_confidence_direction(self):
        _populate_queries(self.tmpdir, {"WHAT": 10, "unknown": 5})
        freeze_baseline(self.tmpdir)

        # Add more low-confidence queries
        _populate_queries(self.tmpdir, {"unknown": 20})
        result = detect_drift(self.tmpdir)
        # Direction should be growing or stable (not shrinking)
        self.assertIn(result["low_confidence"]["direction"], ["growing", "stable"])

    def test_specific_baseline_version(self):
        _populate_queries(self.tmpdir, {"WHAT": 10})
        freeze_baseline(self.tmpdir)
        freeze_baseline(self.tmpdir, tag=5)
        result = detect_drift(self.tmpdir, baseline_tag=1)
        self.assertEqual(result["baseline_version"], "v1")


class TestCompareBaselines(unittest.TestCase):
    """Test baseline-to-baseline comparison."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, ".mind-mem-index"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_compare_identical(self):
        _populate_queries(self.tmpdir, {"WHAT": 10, "WHEN": 5})
        freeze_baseline(self.tmpdir)
        freeze_baseline(self.tmpdir)
        result = compare_baselines(self.tmpdir, 1, 2)
        self.assertEqual(result["v1"], "v1")
        self.assertEqual(result["v2"], "v2")
        self.assertAlmostEqual(result["chi_squared_test"]["chi2"], 0.0, places=2)

    def test_compare_different(self):
        _populate_queries(self.tmpdir, {"WHAT": 20})
        freeze_baseline(self.tmpdir)
        _populate_queries(self.tmpdir, {"WHEN": 30})
        freeze_baseline(self.tmpdir)
        result = compare_baselines(self.tmpdir, 1, 2)
        self.assertGreater(result["chi_squared_test"]["chi2"], 0)

    def test_missing_baseline_raises(self):
        with self.assertRaises(FileNotFoundError):
            compare_baselines(self.tmpdir, 1, 2)


class TestVersioning(unittest.TestCase):
    """Test version auto-increment and listing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_no_versions(self):
        self.assertEqual(_list_versions(self.tmpdir), [])
        self.assertEqual(_next_version(self.tmpdir), 1)

    def test_version_increment(self):
        d = os.path.join(self.tmpdir, "intelligence/baselines")
        os.makedirs(d)
        with open(os.path.join(d, "intent-baseline-v1.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(d, "intent-baseline-v3.json"), "w") as f:
            f.write("{}")
        self.assertEqual(_list_versions(self.tmpdir), [1, 3])
        self.assertEqual(_next_version(self.tmpdir), 4)

    def test_ignores_non_baseline_files(self):
        d = os.path.join(self.tmpdir, "intelligence/baselines")
        os.makedirs(d)
        with open(os.path.join(d, "intent-baseline-v1.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(d, "other-file.json"), "w") as f:
            f.write("{}")
        self.assertEqual(_list_versions(self.tmpdir), [1])


if __name__ == "__main__":
    unittest.main()
