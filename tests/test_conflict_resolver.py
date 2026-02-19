#!/usr/bin/env python3
"""Tests for conflict_resolver.py — zero external deps (stdlib unittest)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from conflict_resolver import (
    ResolutionStrategy,
    _block_hash,
    _extract_date,
    _get_cs_priority,
    _get_scope_specificity,
    analyze_contradiction,
    generate_resolution_proposals,
    resolve_contradictions,
)


class TestExtractDate(unittest.TestCase):
    def test_from_date_field(self):
        block = {"Date": "2026-02-15"}
        self.assertEqual(_extract_date(block), "2026-02-15")

    def test_from_created_field(self):
        block = {"Created": "2026-01-10T14:30:00"}
        self.assertEqual(_extract_date(block), "2026-01-10")

    def test_from_block_id(self):
        block = {"_id": "D-20260215-001"}
        self.assertEqual(_extract_date(block), "2026-02-15")

    def test_no_date(self):
        block = {"Statement": "No date here"}
        self.assertIsNone(_extract_date(block))

    def test_invalid_date_format(self):
        block = {"Date": "not-a-date"}
        # Falls through to _id check
        block["_id"] = "X-NOPE"
        self.assertIsNone(_extract_date(block))


class TestGetCSPriority(unittest.TestCase):
    def test_no_signatures(self):
        self.assertEqual(_get_cs_priority({}), 0)

    def test_single_signature(self):
        block = {"ConstraintSignatures": [{"priority": 5}]}
        self.assertEqual(_get_cs_priority(block), 5)

    def test_multiple_signatures_returns_max(self):
        block = {"ConstraintSignatures": [{"priority": 2}, {"priority": 7}, {"priority": 3}]}
        self.assertEqual(_get_cs_priority(block), 7)

    def test_non_int_priority_ignored(self):
        block = {"ConstraintSignatures": [{"priority": "high"}]}
        self.assertEqual(_get_cs_priority(block), 0)


class TestGetScopeSpecificity(unittest.TestCase):
    def test_empty_scope(self):
        block = {"ConstraintSignatures": [{"scope": {}}]}
        self.assertEqual(_get_scope_specificity(block), 0)

    def test_string_fields(self):
        block = {"ConstraintSignatures": [{"scope": {"project": "mind-mem", "team": "core"}}]}
        self.assertEqual(_get_scope_specificity(block), 2)

    def test_list_fields(self):
        block = {"ConstraintSignatures": [{"scope": {"files": ["a.py", "b.py", "c.py"]}}]}
        self.assertEqual(_get_scope_specificity(block), 3)

    def test_no_signatures(self):
        self.assertEqual(_get_scope_specificity({}), 0)


class TestBlockHash(unittest.TestCase):
    def test_deterministic(self):
        block = {"Statement": "Use PostgreSQL", "Status": "active"}
        h1 = _block_hash(block)
        h2 = _block_hash(block)
        self.assertEqual(h1, h2)

    def test_different_blocks_different_hash(self):
        block_a = {"Statement": "Use PostgreSQL"}
        block_b = {"Statement": "Use MySQL"}
        self.assertNotEqual(_block_hash(block_a), _block_hash(block_b))

    def test_ignores_underscore_fields(self):
        block = {"Statement": "Test", "_id": "D-20260101-001", "_file": "x.md"}
        h1 = _block_hash(block)
        block2 = {"Statement": "Test", "_id": "DIFFERENT", "_file": "y.md"}
        h2 = _block_hash(block2)
        self.assertEqual(h1, h2)

    def test_hash_length(self):
        block = {"Statement": "Test"}
        self.assertEqual(len(_block_hash(block)), 12)


class TestAnalyzeContradiction(unittest.TestCase):
    def test_confidence_priority_wins(self):
        block_a = {"_id": "D-20260101-001", "ConstraintSignatures": [{"priority": 8}]}
        block_b = {"_id": "D-20260102-001", "ConstraintSignatures": [{"priority": 3}]}
        result = analyze_contradiction(block_a, block_b)
        self.assertEqual(result["strategy"], ResolutionStrategy.CONFIDENCE)
        self.assertEqual(result["winner_id"], "D-20260101-001")
        self.assertEqual(result["confidence"], "high")

    def test_scope_priority_wins(self):
        block_a = {
            "_id": "D-20260101-001",
            "ConstraintSignatures": [{"scope": {"project": "x", "team": "y", "files": ["a", "b"]}}],
        }
        block_b = {
            "_id": "D-20260102-001",
            "ConstraintSignatures": [{"scope": {"project": "x"}}],
        }
        result = analyze_contradiction(block_a, block_b)
        self.assertEqual(result["strategy"], ResolutionStrategy.SCOPE)
        self.assertEqual(result["winner_id"], "D-20260101-001")

    def test_timestamp_priority_wins(self):
        block_a = {"_id": "D-20260201-001", "Date": "2026-02-01"}
        block_b = {"_id": "D-20260115-001", "Date": "2026-01-15"}
        result = analyze_contradiction(block_a, block_b)
        self.assertEqual(result["strategy"], ResolutionStrategy.TIMESTAMP)
        self.assertEqual(result["winner_id"], "D-20260201-001")

    def test_manual_fallback(self):
        block_a = {"_id": "D-20260201-001", "Date": "2026-02-01"}
        block_b = {"_id": "D-20260201-002", "Date": "2026-02-01"}
        result = analyze_contradiction(block_a, block_b)
        self.assertEqual(result["strategy"], ResolutionStrategy.MANUAL)
        self.assertIsNone(result["winner_id"])
        self.assertEqual(result["confidence"], "low")

    def test_confidence_needs_delta_2(self):
        """Priority difference of 1 should NOT trigger confidence strategy."""
        block_a = {"_id": "A", "ConstraintSignatures": [{"priority": 4}]}
        block_b = {"_id": "B", "ConstraintSignatures": [{"priority": 3}], "Date": "2026-01-01"}
        result = analyze_contradiction(block_a, block_b)
        # Should fall through to timestamp or scope, not confidence
        self.assertNotEqual(result["strategy"], ResolutionStrategy.CONFIDENCE)


class TestResolveContradictions(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "intelligence"))
        os.makedirs(os.path.join(self.td, "decisions"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def test_no_contradictions_file(self):
        result = resolve_contradictions(self.td)
        self.assertEqual(result, [])

    def test_empty_contradictions(self):
        with open(os.path.join(self.td, "intelligence", "CONTRADICTIONS.md"), "w") as f:
            f.write("# Contradictions\n\nNone detected.\n")
        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write("")
        result = resolve_contradictions(self.td)
        self.assertEqual(result, [])

    def test_resolves_with_blocks(self):
        # Create two decisions
        with open(os.path.join(self.td, "decisions", "DECISIONS.md"), "w") as f:
            f.write(
                "[D-20260101-001]\n"
                "Statement: Use PostgreSQL\n"
                "Date: 2026-01-01\n"
                "Status: active\n\n---\n\n"
                "[D-20260215-001]\n"
                "Statement: Use MySQL\n"
                "Date: 2026-02-15\n"
                "Status: active\n"
            )
        # Create contradiction referencing them (use ID format that the block
        # parser accepts but _ID_RE won't match, so the first 2 regex hits
        # are the decision IDs)
        with open(os.path.join(self.td, "intelligence", "CONTRADICTIONS.md"), "w") as f:
            f.write(
                "[CONTRA-001]\n"
                "Type: contradiction\n"
                "Blocks: D-20260101-001 vs D-20260215-001\n"
                "Description: Database choice conflict\n"
            )
        result = resolve_contradictions(self.td)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["block_a"], "D-20260101-001")
        self.assertEqual(result[0]["block_b"], "D-20260215-001")
        self.assertEqual(result[0]["strategy"], ResolutionStrategy.TIMESTAMP)


class TestGenerateResolutionProposals(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.td, "intelligence"))
        os.makedirs(os.path.join(self.td, "decisions"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.td, ignore_errors=True)

    def test_no_auto_resolvable(self):
        resolutions = [
            {"strategy": ResolutionStrategy.MANUAL, "winner_id": None, "loser_id": None},
        ]
        count = generate_resolution_proposals(self.td, resolutions)
        self.assertEqual(count, 0)

    def test_generates_proposals(self):
        resolutions = [
            {
                "strategy": ResolutionStrategy.TIMESTAMP,
                "confidence": "medium",
                "winner_id": "D-20260215-001",
                "loser_id": "D-20260101-001",
                "contradiction_id": "C-20260215-001",
                "hash_a": "abc123",
                "hash_b": "def456",
                "rationale": "Newer decision wins",
            },
        ]
        count = generate_resolution_proposals(self.td, resolutions)
        self.assertEqual(count, 1)
        proposed_path = os.path.join(self.td, "intelligence", "proposed", "RESOLUTIONS_PROPOSED.md")
        self.assertTrue(os.path.isfile(proposed_path))
        with open(proposed_path) as f:
            content = f.read()
        self.assertIn("pending-review", content)
        self.assertIn("D-20260215-001", content)
        self.assertIn("timestamp_priority", content)


if __name__ == "__main__":
    unittest.main()
