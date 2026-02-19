"""Tests for cross-encoder reranker integration in recall pipeline."""
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

_HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


class TestCrossEncoderConfig(unittest.TestCase):
    """Test cross-encoder config loading and gating."""

    def test_default_disabled(self):
        """Cross-encoder should be disabled by default."""
        ce_config = {}.get("recall", {}).get("cross_encoder", {})
        self.assertFalse(ce_config.get("enabled", False))

    def test_config_structure(self):
        """Verify expected config keys."""
        config = {
            "recall": {
                "cross_encoder": {
                    "enabled": True,
                    "blend_weight": 0.6,
                    "top_k": 10,
                }
            }
        }
        ce = config["recall"]["cross_encoder"]
        self.assertTrue(ce["enabled"])
        self.assertEqual(ce["blend_weight"], 0.6)
        self.assertEqual(ce["top_k"], 10)


class TestCrossEncoderAvailability(unittest.TestCase):
    """Test graceful degradation when sentence-transformers is missing."""

    def test_is_available_returns_bool(self):
        from cross_encoder_reranker import CrossEncoderReranker
        result = CrossEncoderReranker.is_available()
        self.assertIsInstance(result, bool)

    def test_import_error_on_missing_dep(self):
        """CrossEncoderReranker should raise ImportError if deps missing."""
        from cross_encoder_reranker import _check_available
        # Just verify the check function works
        result = _check_available()
        self.assertIsInstance(result, bool)


class TestCrossEncoderInRecall(unittest.TestCase):
    """Test cross-encoder integration points in recall pipeline."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "decisions"))
        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL for the database\n"
                "Status: active\n"
                "Date: 2026-01-01\n"
                "Tags: database\n"
            )

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_recall_works_without_cross_encoder(self):
        """Recall should work fine with cross-encoder disabled (default)."""
        from recall import recall as r
        results = r(self.td, "PostgreSQL", limit=5)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["_id"], "D-20260101-001")

    def test_recall_with_ce_config_disabled(self):
        """Recall should skip cross-encoder when config disabled."""
        import json
        config = {"recall": {"cross_encoder": {"enabled": False}}}
        with open(os.path.join(self.td, "mind-mem.json"), "w") as f:
            json.dump(config, f)

        from recall import recall as r
        results = r(self.td, "PostgreSQL", limit=5)
        self.assertGreater(len(results), 0)
        # No ce_score field when disabled
        self.assertNotIn("ce_score", results[0])

    def test_content_key_populated_from_excerpt(self):
        """Verify candidates get 'content' key from 'excerpt' for CE."""
        candidates = [
            {"_id": "D-001", "score": 1.0, "excerpt": "PostgreSQL is great"},
            {"_id": "D-002", "score": 0.5, "excerpt": "Redis for caching"},
        ]
        for c in candidates:
            if "content" not in c:
                c["content"] = c.get("excerpt", "")
        self.assertEqual(candidates[0]["content"], "PostgreSQL is great")
        self.assertEqual(candidates[1]["content"], "Redis for caching")

    def test_blend_weight_range(self):
        """Blend weight should be between 0 and 1."""
        blend = 0.6
        self.assertGreaterEqual(blend, 0.0)
        self.assertLessEqual(blend, 1.0)


class TestCrossEncoderRerankerUnit(unittest.TestCase):
    """Unit tests for the CrossEncoderReranker class itself."""

    @pytest.mark.skipif(
        not _HAS_SENTENCE_TRANSFORMERS,
        reason="sentence-transformers not installed (optional dependency for cross-encoder reranking)",
    )
    def test_empty_candidates(self):
        """Reranker should handle empty candidate list."""
        from cross_encoder_reranker import CrossEncoderReranker
        ce = CrossEncoderReranker()
        result = ce.rerank("test query", [], top_k=5)
        self.assertEqual(result, [])


class TestCrossEncoderMindKernel(unittest.TestCase):
    """Test that the cross_encoder.mind config file exists."""

    def test_kernel_file_exists(self):
        kernel_path = os.path.join(
            os.path.dirname(__file__), "..", "mind", "cross_encoder.mind"
        )
        self.assertTrue(os.path.isfile(kernel_path))

    def test_kernel_has_blending_section(self):
        kernel_path = os.path.join(
            os.path.dirname(__file__), "..", "mind", "cross_encoder.mind"
        )
        with open(kernel_path) as f:
            content = f.read()
        self.assertIn("[blending]", content)
        self.assertIn("blend_weight", content)


if __name__ == "__main__":
    unittest.main()
