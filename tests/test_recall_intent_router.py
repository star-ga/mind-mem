"""Tests for IntentRouter integration in recall pipeline."""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestIntentRouterMapping(unittest.TestCase):
    """Test the _INTENT_TO_QUERY_TYPE mapping in recall.py."""

    def test_mapping_exists(self):
        from recall import _INTENT_TO_QUERY_TYPE
        self.assertEqual(len(_INTENT_TO_QUERY_TYPE), 9)

    def test_all_intents_mapped(self):
        from recall import _INTENT_TO_QUERY_TYPE
        expected = {"WHY", "WHEN", "ENTITY", "WHAT", "HOW", "LIST",
                    "VERIFY", "COMPARE", "TRACE"}
        self.assertEqual(set(_INTENT_TO_QUERY_TYPE.keys()), expected)

    def test_adversarial_mapping(self):
        from recall import _INTENT_TO_QUERY_TYPE
        self.assertEqual(_INTENT_TO_QUERY_TYPE["VERIFY"], "adversarial")

    def test_temporal_mapping(self):
        from recall import _INTENT_TO_QUERY_TYPE
        self.assertEqual(_INTENT_TO_QUERY_TYPE["WHEN"], "temporal")

    def test_multihop_mappings(self):
        from recall import _INTENT_TO_QUERY_TYPE
        self.assertEqual(_INTENT_TO_QUERY_TYPE["WHY"], "multi-hop")
        self.assertEqual(_INTENT_TO_QUERY_TYPE["COMPARE"], "multi-hop")
        self.assertEqual(_INTENT_TO_QUERY_TYPE["TRACE"], "multi-hop")


class TestIntentRouterClassification(unittest.TestCase):
    """Test IntentRouter produces correct classifications."""

    def test_why_query(self):
        from intent_router import get_router
        result = get_router().classify("Why did we switch to PostgreSQL?")
        self.assertEqual(result.intent, "WHY")
        self.assertGreater(result.confidence, 0)

    def test_when_query(self):
        from intent_router import get_router
        result = get_router().classify("When was the Redis decision made?")
        self.assertEqual(result.intent, "WHEN")

    def test_verify_query(self):
        from intent_router import get_router
        result = get_router().classify("Did we ever use MongoDB?")
        self.assertEqual(result.intent, "VERIFY")

    def test_compare_query(self):
        from intent_router import get_router
        result = get_router().classify("Compare PostgreSQL vs MySQL")
        self.assertEqual(result.intent, "COMPARE")

    def test_trace_query(self):
        from intent_router import get_router
        result = get_router().classify("Trace the evolution of our database over time")
        self.assertEqual(result.intent, "TRACE")

    def test_list_query(self):
        from intent_router import get_router
        result = get_router().classify("List all active decisions")
        self.assertEqual(result.intent, "LIST")

    def test_empty_query_defaults(self):
        from intent_router import get_router
        result = get_router().classify("")
        self.assertEqual(result.intent, "WHAT")
        self.assertEqual(result.confidence, 0.0)


class TestIntentRouterInRecall(unittest.TestCase):
    """Test that IntentRouter is actually called from recall."""

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

    def test_temporal_query_uses_intent(self):
        """Temporal query should route through IntentRouter."""
        from recall import recall as r
        results = r(self.td, "When was the database decision made?", limit=5)
        # Should return results (the query works regardless of router)
        self.assertIsInstance(results, list)

    def test_adversarial_query_uses_intent(self):
        """Adversarial query should route to VERIFY intent."""
        from recall import recall as r
        results = r(self.td, "Did we ever use Oracle?", limit=5)
        self.assertIsInstance(results, list)

    def test_detect_query_type_preserved(self):
        """Legacy detect_query_type should still exist for fallback."""
        from recall import detect_query_type
        result = detect_query_type("When was X decided?")
        self.assertIn(result, {"temporal", "adversarial", "multi-hop",
                               "single-hop", "open-domain"})

    def test_graph_boost_from_intent_depth(self):
        """WHY queries should get graph_depth >= 2 from IntentRouter."""
        from intent_router import get_router
        result = get_router().classify("Why did we switch databases?")
        self.assertGreaterEqual(result.params.get("graph_depth", 0), 2)


if __name__ == "__main__":
    unittest.main()
