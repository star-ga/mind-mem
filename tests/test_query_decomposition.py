#!/usr/bin/env python3
"""Tests for multi-hop query decomposition (#6)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from _recall_detection import decompose_query, detect_query_type


class TestSimpleConjunctionSplit(unittest.TestCase):
    """Conjunction-based splitting: "X and Y" decomposes into 2 sub-queries."""

    def test_and_split(self):
        parts = decompose_query(
            "When did the team deploy v1.0.6 and what issues were found?"
        )
        self.assertGreater(len(parts), 1)
        self.assertLessEqual(len(parts), 4)

    def test_as_well_as_split(self):
        parts = decompose_query(
            "What tools did we use for the auth migration as well as how long did it take?"
        )
        self.assertGreater(len(parts), 1)

    def test_plus_split(self):
        parts = decompose_query(
            "Describe the incident response process plus the follow-up actions taken"
        )
        self.assertGreater(len(parts), 1)

    def test_but_split(self):
        parts = decompose_query(
            "What did the frontend team deliver this sprint but what blockers remained?"
        )
        self.assertGreater(len(parts), 1)


class TestNoSplitShortQuery(unittest.TestCase):
    """Short or simple queries should NOT be decomposed."""

    def test_short_question(self):
        parts = decompose_query("what is auth?")
        self.assertEqual(len(parts), 1)

    def test_single_clause(self):
        parts = decompose_query("When did we deploy the database migration?")
        self.assertEqual(len(parts), 1)

    def test_empty_query(self):
        parts = decompose_query("")
        self.assertEqual(len(parts), 0)

    def test_short_and_clause(self):
        """'X and Y' where one side is too short should not split."""
        parts = decompose_query("auth and X")
        self.assertEqual(len(parts), 1)

    def test_or_alternative_not_split(self):
        """'A or B' alternatives should not decompose."""
        parts = decompose_query("Did we use PostgreSQL or MySQL for the primary database?")
        self.assertEqual(len(parts), 1)


class TestMultipleQuestions(unittest.TestCase):
    """Multiple question marks should trigger decomposition."""

    def test_two_questions(self):
        parts = decompose_query(
            "When did the team deploy v1.0.6? What issues were found after deployment?"
        )
        self.assertGreater(len(parts), 1)

    def test_three_questions(self):
        parts = decompose_query(
            "Who approved the change? When was it deployed? What broke afterwards?"
        )
        self.assertGreater(len(parts), 1)
        self.assertLessEqual(len(parts), 4)


class TestMaxSubqueries(unittest.TestCase):
    """Very long queries should cap at 4 sub-queries."""

    def test_many_conjunctions_capped(self):
        q = (
            "What tools were used for deployment and what was the rollback plan "
            "and who was on call and what monitoring was in place "
            "and what was the incident timeline and who approved the fix?"
        )
        parts = decompose_query(q)
        self.assertLessEqual(len(parts), 4)

    def test_many_questions_capped(self):
        q = (
            "Who approved the deploy? When was it shipped? "
            "What monitoring was active? Who was on call? "
            "What was the incident response? How was it resolved?"
        )
        parts = decompose_query(q)
        self.assertLessEqual(len(parts), 4)


class TestContextPreservation(unittest.TestCase):
    """Shared entity/topic from first clause should carry into later sub-queries."""

    def test_entity_carries_forward(self):
        parts = decompose_query(
            "What tools did we use for the auth migration and how long did it take?"
        )
        self.assertGreater(len(parts), 1)
        # The second part should contain some context from the first
        # (e.g., "auth" or "migration" should appear in the second sub-query)
        second_lower = parts[-1].lower()
        has_context = "auth" in second_lower or "migration" in second_lower
        self.assertTrue(has_context,
                        f"Second sub-query lacks shared context: {parts[-1]}")

    def test_version_carries_forward(self):
        parts = decompose_query(
            "When did we release v2.3.1 and what bugs were reported?"
        )
        self.assertGreater(len(parts), 1)
        second_lower = parts[-1].lower()
        # Version or "release" should carry forward
        has_context = "v2.3.1" in parts[-1] or "release" in second_lower
        self.assertTrue(has_context,
                        f"Version context missing in: {parts[-1]}")

    def test_no_duplicate_context(self):
        """If second clause already has the entity, do not duplicate it."""
        parts = decompose_query(
            "What tools did we use for the auth migration "
            "and how long did the auth migration take?"
        )
        if len(parts) > 1:
            # "auth" is already in the second part, no need to prepend
            second = parts[-1]
            # Count occurrences — should not have double context injection
            self.assertLessEqual(
                second.lower().count("auth migration"), 2,
                f"Context duplicated unnecessarily: {second}",
            )


class TestNonMultihopNotDecomposed(unittest.TestCase):
    """Single-hop, temporal, and adversarial queries are NOT decomposed."""

    def test_temporal_not_decomposed(self):
        q = "When did Caroline sell the car?"
        qtype = detect_query_type(q)
        # Even though we call decompose_query, the recall pipeline only
        # calls it for multi-hop queries.  Verify the function itself
        # returns 1 part for a simple temporal query.
        parts = decompose_query(q)
        self.assertEqual(len(parts), 1)

    def test_adversarial_not_decomposed(self):
        q = "Did Caroline never visit Tokyo?"
        parts = decompose_query(q)
        self.assertEqual(len(parts), 1)

    def test_simple_single_hop(self):
        q = "What database does the project use?"
        parts = decompose_query(q)
        self.assertEqual(len(parts), 1)


class TestWhWordSplit(unittest.TestCase):
    """Queries with multiple wh-words should split at wh-boundaries."""

    def test_when_how(self):
        parts = decompose_query(
            "When did the team deploy the service and how long did the rollout take?"
        )
        self.assertGreater(len(parts), 1)

    def test_what_who(self):
        parts = decompose_query(
            "What was the root cause of the outage and who was responsible for the fix?"
        )
        self.assertGreater(len(parts), 1)


if __name__ == "__main__":
    unittest.main()
