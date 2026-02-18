"""Tests for optional cross-encoder reranker."""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from cross_encoder_reranker import CrossEncoderReranker


class TestCrossEncoderFallback(unittest.TestCase):
    """Test graceful fallback when sentence-transformers not installed."""

    def test_is_available_returns_bool(self):
        result = CrossEncoderReranker.is_available()
        self.assertIsInstance(result, bool)

    def test_import_error_on_missing_package(self):
        """Should raise ImportError when sentence-transformers missing."""
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            # Reset the cached availability
            import cross_encoder_reranker
            cross_encoder_reranker._CE_AVAILABLE = None
            self.assertFalse(cross_encoder_reranker._check_available())


class TestCrossEncoderScoreBlending(unittest.TestCase):
    """Test score blending arithmetic."""

    def test_blend_weight_1_uses_only_ce(self):
        """blend_weight=1.0 means 100% CE score."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1]

        reranker = object.__new__(CrossEncoderReranker)
        reranker._model = mock_model

        candidates = [
            {"content": "high CE", "score": 0.1},
            {"content": "low CE", "score": 0.9},
        ]
        result = reranker.rerank("query", candidates, blend_weight=1.0)
        # First result should be the one with higher CE score
        self.assertEqual(result[0]["content"], "high CE")

    def test_blend_weight_0_uses_only_original(self):
        """blend_weight=0.0 means 100% original score."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1]

        reranker = object.__new__(CrossEncoderReranker)
        reranker._model = mock_model

        candidates = [
            {"content": "low orig", "score": 0.1},
            {"content": "high orig", "score": 0.9},
        ]
        result = reranker.rerank("query", candidates, blend_weight=0.0)
        # First result should be the one with higher original score
        self.assertEqual(result[0]["content"], "high orig")

    def test_empty_candidates(self):
        """Empty input returns empty output."""
        mock_model = MagicMock()
        reranker = object.__new__(CrossEncoderReranker)
        reranker._model = mock_model

        result = reranker.rerank("query", [])
        self.assertEqual(result, [])

    def test_top_k_limits_output(self):
        """Output limited to top_k."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.3, 0.8, 0.1]

        reranker = object.__new__(CrossEncoderReranker)
        reranker._model = mock_model

        candidates = [
            {"content": f"doc{i}", "score": 0.5} for i in range(4)
        ]
        result = reranker.rerank("query", candidates, top_k=2)
        self.assertEqual(len(result), 2)


class TestDeterministicReranking(unittest.TestCase):
    """Test the 3 deterministic reranking features in recall.py."""

    def test_negation_detection(self):
        """Should detect negation patterns."""
        from recall import _detect_negation
        has_neg, terms = _detect_negation("Did Sarah NOT visit Paris?")
        self.assertTrue(has_neg)
        self.assertTrue(len(terms) > 0)

    def test_no_negation(self):
        from recall import _detect_negation
        has_neg, _ = _detect_negation("Where does Sarah live?")
        self.assertFalse(has_neg)

    def test_negation_penalty(self):
        from recall import _negation_penalty
        # Block affirming negated terms should be penalized
        penalty = _negation_penalty("Sarah visited Paris last summer", ["visit", "paris"])
        self.assertLess(penalty, 1.0)
        # Block without negated terms should not be penalized
        no_penalty = _negation_penalty("Sarah lives in New York", ["visit", "paris"])
        self.assertEqual(no_penalty, 1.0)

    def test_date_proximity(self):
        from recall import _date_proximity_score
        # Close dates should score high
        score_close = _date_proximity_score("What happened on 2025-06-15?", "Event on 2025-06-14")
        # Far dates should score lower
        score_far = _date_proximity_score("What happened on 2025-06-15?", "Event on 2020-01-01")
        self.assertGreater(score_close, score_far)

    def test_date_proximity_no_dates(self):
        from recall import _date_proximity_score
        # No dates in query returns 1.0
        score = _date_proximity_score("What is Sarah's hobby?", "She plays tennis")
        self.assertEqual(score, 1.0)

    def test_category_match(self):
        from recall import _category_match_boost
        # Query and block in same category should boost
        boost = _category_match_boost("What food does she like?", "She enjoys Italian cuisine")
        self.assertGreater(boost, 1.0)

    def test_category_no_match(self):
        from recall import _category_match_boost
        # No category overlap
        boost = _category_match_boost("random text 12345", "other random 67890")
        self.assertEqual(boost, 1.0)


if __name__ == "__main__":
    unittest.main()
