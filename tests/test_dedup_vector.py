#!/usr/bin/env python3
"""Tests for vector-enhanced cosine dedup (Layer 2b)."""

import unittest
from unittest.mock import MagicMock, patch

from mind_mem.dedup import (
    DedupConfig,
    deduplicate_results,
    layer_vector_cosine_dedup,
)


def _make_result(
    id_: str = "D-001",
    score: float = 1.0,
    excerpt: str = "test content",
    file: str = "memory.md",
    type_: str = "decision",
) -> dict:
    """Create a minimal result dict matching the recall output shape."""
    return {
        "_id": id_,
        "score": score,
        "excerpt": excerpt,
        "file": file,
        "type": type_,
        "tags": "",
        "line": 1,
        "status": "active",
    }


class TestLayerVectorCosineDedup(unittest.TestCase):
    """Test the vector cosine dedup layer."""

    def test_empty_results_passthrough(self):
        """Empty input should return empty output."""
        self.assertEqual(layer_vector_cosine_dedup([]), [])

    def test_fallback_without_workspace(self):
        """Without a workspace, should fall back to term-frequency dedup."""
        results = [
            _make_result("D-001", 1.0, "the quick brown fox jumps"),
            _make_result("D-002", 0.9, "the quick brown fox jumps"),
        ]
        kept = layer_vector_cosine_dedup(results, threshold=0.85, workspace=None)
        # Term-frequency fallback should deduplicate identical texts
        self.assertEqual(len(kept), 1)

    @patch("mind_mem.dedup.layer_cosine_dedup")
    def test_fallback_on_import_error(self, mock_tf_dedup):
        """Should fall back to term-frequency dedup when recall_vector is unavailable."""
        mock_tf_dedup.return_value = [_make_result("D-001", 1.0, "fallback")]
        results = [_make_result("D-001", 1.0, "test"), _make_result("D-002", 0.9, "test")]

        with patch.dict("sys.modules", {"mind_mem.recall_vector": None}):
            kept = layer_vector_cosine_dedup(
                results, threshold=0.85, workspace="/tmp/test-ws"
            )

        # Should have called the fallback
        mock_tf_dedup.assert_called_once()

    @patch("mind_mem.dedup.layer_cosine_dedup")
    def test_vector_dedup_removes_duplicates(self, mock_tf_dedup):
        """When embeddings are available, should use vector similarity for dedup."""
        results = [
            _make_result("D-001", 1.0, "authentication flow"),
            _make_result("D-002", 0.9, "login process"),
            _make_result("D-003", 0.8, "database schema design"),
        ]

        # Mock the VectorBackend
        mock_backend = MagicMock()
        # Embeddings: D-001 and D-002 are similar, D-003 is different
        mock_backend.embed.return_value = [
            [1.0, 0.0, 0.0],  # D-001
            [0.99, 0.1, 0.0],  # D-002 (similar to D-001)
            [0.0, 0.0, 1.0],  # D-003 (different)
        ]

        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            return dot / (na * nb) if na and nb else 0.0

        mock_backend.cosine_similarity.side_effect = cosine_sim

        mock_vector_cls = MagicMock(return_value=mock_backend)

        with patch("mind_mem.dedup.VectorBackend", mock_vector_cls, create=True):
            # We need to patch the import inside the function
            import mind_mem.dedup as dedup_mod

            original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

            mock_module = MagicMock()
            mock_module.VectorBackend = mock_vector_cls

            with patch.dict("sys.modules", {"mind_mem.recall_vector": mock_module}):
                kept = layer_vector_cosine_dedup(
                    results, threshold=0.85, workspace="/tmp/ws"
                )

        # D-002 should be removed (too similar to D-001)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["_id"], "D-001")
        self.assertEqual(kept[1]["_id"], "D-003")

        # term-frequency fallback should NOT have been called
        mock_tf_dedup.assert_not_called()

    def test_config_vector_cosine_enabled_default_false(self):
        """vector_cosine_enabled should default to False in DedupConfig."""
        cfg = DedupConfig()
        self.assertFalse(cfg.vector_cosine_enabled)

        cfg_on = DedupConfig({"vector_cosine_enabled": True})
        self.assertTrue(cfg_on.vector_cosine_enabled)


if __name__ == "__main__":
    unittest.main()
