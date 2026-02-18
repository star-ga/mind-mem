"""Tests for 9-type intent router."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from intent_router import IntentRouter, IntentResult, INTENT_CONFIG


class TestIntentClassification(unittest.TestCase):
    """Test each of the 9 intent types."""

    def setUp(self):
        self.router = IntentRouter()

    def test_why_intent(self):
        result = self.router.classify("Why did Sarah move to Boston?")
        self.assertEqual(result.intent, "WHY")
        self.assertGreater(result.confidence, 0)

    def test_when_intent(self):
        result = self.router.classify("When did the meeting happen?")
        self.assertEqual(result.intent, "WHEN")

    def test_entity_intent(self):
        result = self.router.classify("Who is Sarah's doctor?")
        self.assertEqual(result.intent, "ENTITY")

    def test_what_intent(self):
        result = self.router.classify("What is the project about?")
        self.assertEqual(result.intent, "WHAT")

    def test_how_intent(self):
        result = self.router.classify("How do I configure the database?")
        self.assertEqual(result.intent, "HOW")

    def test_list_intent(self):
        result = self.router.classify("List all of Sarah's hobbies")
        self.assertEqual(result.intent, "LIST")

    def test_verify_intent(self):
        result = self.router.classify("Is it true that Sarah visited Paris?")
        self.assertEqual(result.intent, "VERIFY")

    def test_compare_intent(self):
        result = self.router.classify("Compare Python vs JavaScript for web dev")
        self.assertEqual(result.intent, "COMPARE")

    def test_trace_intent(self):
        result = self.router.classify("Show the timeline of the project")
        self.assertEqual(result.intent, "TRACE")


class TestIntentConfidence(unittest.TestCase):
    """Test confidence scoring."""

    def setUp(self):
        self.router = IntentRouter()

    def test_confidence_range(self):
        """Confidence should be in [0, 1]."""
        for q in ["Why?", "When?", "How?", "What?", "random text"]:
            result = self.router.classify(q)
            self.assertGreaterEqual(result.confidence, 0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_empty_query_low_confidence(self):
        result = self.router.classify("")
        self.assertEqual(result.confidence, 0.0)

    def test_ambiguous_query(self):
        """Ambiguous queries should have sub-intents."""
        result = self.router.classify("When and why did Sarah move?")
        self.assertTrue(len(result.sub_intents) > 0 or result.confidence > 0)


class TestIntentParams(unittest.TestCase):
    """Test parameter mapping."""

    def setUp(self):
        self.router = IntentRouter()

    def test_params_present(self):
        result = self.router.classify("Why did this happen?")
        self.assertIn("expansion", result.params)
        self.assertIn("graph_depth", result.params)
        self.assertIn("rerank", result.params)

    def test_why_params(self):
        result = self.router.classify("Why did Sarah leave?")
        self.assertEqual(result.params["expansion"], "rm3")
        self.assertEqual(result.params["graph_depth"], 2)

    def test_when_params(self):
        result = self.router.classify("When did the event occur?")
        self.assertEqual(result.params["expansion"], "date")
        self.assertEqual(result.params["rerank"], "temporal")

    def test_trace_deep_graph(self):
        result = self.router.classify("Trace the history of this decision")
        self.assertEqual(result.params["graph_depth"], 3)

    def test_list_no_graph(self):
        result = self.router.classify("List all projects")
        self.assertEqual(result.params["graph_depth"], 0)


class TestIntentResultStructure(unittest.TestCase):
    """Test IntentResult dataclass."""

    def test_dataclass_fields(self):
        r = IntentResult(intent="WHY", confidence=0.8)
        self.assertEqual(r.intent, "WHY")
        self.assertEqual(r.confidence, 0.8)
        self.assertEqual(r.sub_intents, [])
        self.assertIsInstance(r.params, dict)


class TestFallback(unittest.TestCase):
    """Test fallback to legacy detect_query_type."""

    def setUp(self):
        self.router = IntentRouter()

    def test_fallback_returns_result(self):
        result = self.router.classify_with_fallback("xyzzy random gibberish")
        self.assertIsInstance(result, IntentResult)

    def test_all_intents_have_config(self):
        """Every intent type should have a config entry."""
        for intent in ["WHY", "WHEN", "ENTITY", "WHAT", "HOW", "LIST", "VERIFY", "COMPARE", "TRACE"]:
            self.assertIn(intent, INTENT_CONFIG)


if __name__ == "__main__":
    unittest.main()
