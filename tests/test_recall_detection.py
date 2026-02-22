#!/usr/bin/env python3
"""Tests for _recall_detection.py — query type classification and text extraction."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from _recall_detection import (
    _QUERY_TYPE_PARAMS,
    detect_query_type,
    extract_text,
    get_block_type,
    is_skeptical_query,
)


class TestDetectQueryType(unittest.TestCase):
    """Test query type classification heuristics."""

    # -- Temporal -------------------------------------------------------

    def test_when_did_is_temporal(self):
        self.assertEqual(detect_query_type("When did Calvin sell the car?"), "temporal")

    def test_date_reference_is_temporal(self):
        self.assertEqual(detect_query_type("What happened in January 2023?"), "temporal")

    def test_last_week_is_temporal(self):
        self.assertEqual(detect_query_type("What did she do last week?"), "temporal")

    def test_how_long_ago_is_temporal(self):
        self.assertEqual(detect_query_type("How long ago did they move?"), "temporal")

    # -- Adversarial ----------------------------------------------------

    def test_negation_is_adversarial(self):
        self.assertEqual(detect_query_type("Did Caroline never visit Tokyo?"), "adversarial")

    def test_ever_is_adversarial(self):
        self.assertEqual(detect_query_type("Has Dave ever been to Paris?"), "adversarial")

    def test_is_it_true_is_adversarial(self):
        self.assertEqual(detect_query_type("Is it true that Jon likes cooking?"), "adversarial")

    def test_yes_no_opener_is_adversarial(self):
        result = detect_query_type("Is Oscar Melanie's pet?")
        self.assertIn(result, ("adversarial", "single-hop"))

    # -- Multi-hop ------------------------------------------------------

    def test_both_is_multihop(self):
        self.assertEqual(
            detect_query_type("What do both Caroline and Melanie enjoy doing?"),
            "multi-hop",
        )

    def test_comparison_is_multihop(self):
        self.assertEqual(
            detect_query_type("How does Caroline's hobby compare with Melanie's?"),
            "multi-hop",
        )

    def test_long_conjunction_is_multihop(self):
        q = (
            "What activities did Caroline and Melanie both participate in together "
            "and also how did their relationship evolve over time?"
        )
        self.assertEqual(detect_query_type(q), "multi-hop")

    # -- Open-domain / Single-hop ---------------------------------------

    def test_short_what_is_open_domain(self):
        self.assertEqual(detect_query_type("What is Caroline's identity?"), "open-domain")

    def test_who_is_open_domain(self):
        self.assertEqual(detect_query_type("Who is Dave?"), "open-domain")

    def test_describe_is_open_domain(self):
        self.assertEqual(detect_query_type("Describe Caroline's background"), "open-domain")

    # -- Temporal-multi-hop cross-boost ---------------------------------

    def test_when_did_verb_stays_temporal(self):
        """Temporal queries with action verbs still classify as temporal
        (temporal signal is stronger) but get multi-hop boost internally."""
        q = "When did Melanie paint a sunrise?"
        self.assertEqual(detect_query_type(q), "temporal")

    # -- Edge cases -----------------------------------------------------

    def test_empty_query(self):
        result = detect_query_type("")
        self.assertIn(result, ("single-hop", "open-domain"))

    def test_pure_gibberish(self):
        result = detect_query_type("asdf qwerty zxcv")
        self.assertEqual(result, "single-hop")


class TestQueryTypeParams(unittest.TestCase):
    """Test _QUERY_TYPE_PARAMS configuration."""

    def test_all_categories_have_params(self):
        expected = {"temporal", "adversarial", "multi-hop", "single-hop", "open-domain"}
        self.assertEqual(set(_QUERY_TYPE_PARAMS.keys()), expected)

    def test_temporal_extra_limit_factor(self):
        self.assertEqual(_QUERY_TYPE_PARAMS["temporal"]["extra_limit_factor"], 2.0)

    def test_multihop_extra_limit_factor(self):
        self.assertEqual(_QUERY_TYPE_PARAMS["multi-hop"]["extra_limit_factor"], 3.0)

    def test_temporal_date_boost_higher(self):
        self.assertGreater(
            _QUERY_TYPE_PARAMS["temporal"]["date_boost"],
            _QUERY_TYPE_PARAMS["single-hop"]["date_boost"],
        )

    def test_each_category_has_extra_limit_factor(self):
        for cat, params in _QUERY_TYPE_PARAMS.items():
            self.assertIn("extra_limit_factor", params, f"{cat} missing extra_limit_factor")


class TestIsSkepticalQuery(unittest.TestCase):
    """Test skeptical query detection for distractor-prone queries."""

    def test_superlative_is_skeptical(self):
        self.assertTrue(is_skeptical_query("What is her favorite book?"))

    def test_short_vague_is_skeptical(self):
        self.assertTrue(is_skeptical_query("What about them?"))

    def test_specific_query_not_skeptical(self):
        self.assertFalse(is_skeptical_query("What instrument does Caroline play regularly?"))


class TestExtractText(unittest.TestCase):
    """Test text extraction from blocks."""

    def test_basic_decision_block(self):
        block = {"_id": "D-001", "Statement": "Use PostgreSQL"}
        text = extract_text(block)
        self.assertIn("D-001", text)

    def test_empty_block(self):
        text = extract_text({})
        self.assertEqual(text.strip(), "")


class TestGetBlockType(unittest.TestCase):
    """Test block type detection from ID string."""

    def test_decision_block(self):
        self.assertEqual(get_block_type("D-20260101-001"), "decision")

    def test_task_block(self):
        self.assertEqual(get_block_type("T-001"), "task")

    def test_observation_block(self):
        # OBS- prefix may or may not be registered
        result = get_block_type("OBS-001")
        self.assertIsInstance(result, str)

    def test_unknown_prefix(self):
        self.assertEqual(get_block_type("XYZ-001"), "unknown")

    def test_empty_string(self):
        self.assertEqual(get_block_type(""), "unknown")


if __name__ == "__main__":
    unittest.main()
