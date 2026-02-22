#!/usr/bin/env python3
"""Tests for block_parser.py — zero external deps (stdlib unittest)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from block_parser import extract_refs, get_active, get_by_id, parse_blocks


class TestParseBlocks(unittest.TestCase):
    def test_single_block(self):
        text = "[D-20260213-001]\nStatement: Use JWT for auth\nStatus: active\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["_id"], "D-20260213-001")
        self.assertEqual(blocks[0]["Statement"], "Use JWT for auth")
        self.assertEqual(blocks[0]["Status"], "active")

    def test_multiple_blocks(self):
        text = (
            "[D-20260213-001]\nStatement: Decision one\nStatus: active\n\n---\n\n"
            "[D-20260213-002]\nStatement: Decision two\nStatus: superseded\n"
        )
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["_id"], "D-20260213-001")
        self.assertEqual(blocks[1]["_id"], "D-20260213-002")

    def test_block_with_list_field(self):
        text = (
            "[T-20260213-001]\nTitle: Fix bug\nStatus: active\n"
            "Sources:\n- source1\n- source2\n"
        )
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIsInstance(blocks[0]["Sources"], list)
        self.assertEqual(blocks[0]["Sources"], ["source1", "source2"])

    def test_no_match_for_hash_header(self):
        """## [ID] must NOT be parsed — only [ID] on its own line."""
        text = "## [D-20260213-001]\nStatement: Should not parse\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 0)

    def test_line_numbers(self):
        text = "\n\n[T-20260213-001]\nTitle: Test\n"
        blocks = parse_blocks(text)
        self.assertEqual(blocks[0]["_line"], 3)

    def test_empty_input(self):
        self.assertEqual(parse_blocks(""), [])
        self.assertEqual(parse_blocks("\n\n\n"), [])

    def test_field_values_are_strings(self):
        """Top-level fields are stored as raw strings (coercion is for sub-fields only)."""
        text = "[D-20260213-001]\nPriority: 5\nStatus: active\n"
        blocks = parse_blocks(text)
        self.assertEqual(blocks[0]["Priority"], "5")
        self.assertEqual(blocks[0]["Status"], "active")

    def test_ops_section(self):
        text = (
            "[P-20260213-001]\nProposalId: P-20260213-001\n"
            "Ops:\n- op: append_block\n  file: decisions/DECISIONS.md\n  block: test\n"
        )
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("Ops", blocks[0])
        self.assertEqual(blocks[0]["Ops"][0]["op"], "append_block")
        self.assertEqual(blocks[0]["Ops"][0]["file"], "decisions/DECISIONS.md")

    def test_constraint_signatures(self):
        text = (
            "[D-20260213-001]\nStatement: Test\n"
            "ConstraintSignatures:\n"
            "- id: CS-001\n  domain: engineering\n  subject: we\n"
            "  predicate: must_use\n  object: JWT\n"
        )
        blocks = parse_blocks(text)
        sigs = blocks[0].get("ConstraintSignatures", [])
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0]["id"], "CS-001")
        self.assertEqual(sigs[0]["domain"], "engineering")

    def test_block_ids_with_various_prefixes(self):
        for prefix in ["D", "T", "PRJ", "PER", "TOOL", "INC", "C", "SIG"]:
            bid = f"{prefix}-001" if prefix in ("PRJ", "PER", "TOOL") else f"{prefix}-20260213-001"
            text = f"[{bid}]\nStatus: active\n"
            blocks = parse_blocks(text)
            self.assertEqual(len(blocks), 1, f"Failed for prefix {prefix}")
            self.assertEqual(blocks[0]["_id"], bid)


class TestGetActive(unittest.TestCase):
    def test_filters_active(self):
        blocks = [
            {"_id": "D-001", "Status": "active"},
            {"_id": "D-002", "Status": "superseded"},
            {"_id": "D-003", "Status": "active"},
        ]
        active = get_active(blocks)
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["_id"], "D-001")
        self.assertEqual(active[1]["_id"], "D-003")

    def test_empty_list(self):
        self.assertEqual(get_active([]), [])


class TestGetById(unittest.TestCase):
    def test_finds_block(self):
        blocks = [{"_id": "D-001"}, {"_id": "D-002"}]
        self.assertEqual(get_by_id(blocks, "D-002")["_id"], "D-002")

    def test_not_found(self):
        self.assertIsNone(get_by_id([{"_id": "D-001"}], "D-999"))


class TestConstraintSignatureSingularPlural(unittest.TestCase):
    """Parser should accept both ConstraintSignature: and ConstraintSignatures:."""

    def test_singular_form(self):
        text = (
            "[D-20260214-001]\n"
            "Statement: Test decision\n"
            "Status: active\n"
            "ConstraintSignature:\n"
            "- id: CS-db-engine\n"
            "  domain: infrastructure\n"
            "  modality: must\n"
        )
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        sigs = blocks[0].get("ConstraintSignatures", [])
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0]["id"], "CS-db-engine")

    def test_plural_form(self):
        text = (
            "[D-20260214-002]\n"
            "Statement: Test decision\n"
            "Status: active\n"
            "ConstraintSignatures:\n"
            "- id: CS-auth-method\n"
            "  domain: security\n"
            "  modality: must_not\n"
        )
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        sigs = blocks[0].get("ConstraintSignatures", [])
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0]["id"], "CS-auth-method")


class TestParserEdgeCases(unittest.TestCase):
    """Parser hardening — edge cases and error recovery."""

    def test_empty_string(self):
        blocks = parse_blocks("")
        self.assertEqual(blocks, [])

    def test_no_blocks_just_text(self):
        blocks = parse_blocks("This is just regular text\nwith no blocks.\n")
        self.assertEqual(blocks, [])

    def test_block_with_only_id(self):
        blocks = parse_blocks("[D-20260215-001]\n")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["_id"], "D-20260215-001")

    def test_unicode_content(self):
        text = "[D-20260215-001]\nStatement: Использовать UTF-8 кодировку\nStatus: active\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("UTF-8", blocks[0]["Statement"])

    def test_very_long_field_value(self):
        long_val = "x" * 5000
        text = f"[D-20260215-001]\nStatement: {long_val}\nStatus: active\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]["Statement"]), 5000)

    def test_consecutive_separators(self):
        text = "[D-20260215-001]\nStatement: First\n---\n---\n---\n[D-20260215-002]\nStatement: Second\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 2)

    def test_block_after_separator_no_blank(self):
        text = "[D-20260215-001]\nStatement: First\n---\n[D-20260215-002]\nStatement: Second\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 2)

    def test_multiline_continuation(self):
        text = "[D-20260215-001]\nStatement: First line\n  continuation of statement\nStatus: active\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("continuation", blocks[0]["Statement"])

    def test_list_field_with_many_items(self):
        items = "\n".join(f"- Item {i}" for i in range(50))
        text = f"[T-20260215-001]\nTitle: List test\nHistory:\n{items}\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]["History"]), 50)


class TestExtractRefs(unittest.TestCase):
    def test_finds_refs(self):
        blocks = [
            {"_id": "D-001", "Context": "See D-20260213-002 and T-20260213-001"},
        ]
        refs = extract_refs(blocks)
        self.assertIn("D-20260213-002", refs)
        self.assertIn("T-20260213-001", refs)

    def test_no_refs(self):
        blocks = [{"_id": "D-001", "Statement": "No references here"}]
        self.assertEqual(extract_refs(blocks), set())


class TestFactKeyEnrichment(unittest.TestCase):
    """Tests for _entities, _dates, _has_negation fact key fields."""

    def test_entities_extracted(self):
        text = (
            "[D-20260213-001]\n"
            "Statement: See T-20260213-002 and PRJ-abc for details\n"
            "Status: active\n"
        )
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("_entities", blocks[0])
        self.assertIn("T-20260213-002", blocks[0]["_entities"])
        self.assertIn("PRJ-abc", blocks[0]["_entities"])

    def test_entities_empty_when_none(self):
        text = "[D-20260213-001]\nStatement: No entity IDs here\nStatus: active\n"
        blocks = parse_blocks(text)
        self.assertEqual(blocks[0]["_entities"], [])

    def test_entities_deduplicated_and_sorted(self):
        text = (
            "[D-20260213-001]\n"
            "Statement: Ref T-20260213-001 and T-20260213-001 again, also PER-xyz\n"
        )
        blocks = parse_blocks(text)
        entities = blocks[0]["_entities"]
        self.assertEqual(entities, sorted(set(entities)))
        self.assertEqual(entities.count("T-20260213-001"), 1)

    def test_dates_extracted(self):
        text = (
            "[D-20260213-001]\n"
            "Statement: Decided on 2026-02-13 and reviewed 2026-03-01\n"
        )
        blocks = parse_blocks(text)
        self.assertIn("_dates", blocks[0])
        self.assertIn("2026-02-13", blocks[0]["_dates"])
        self.assertIn("2026-03-01", blocks[0]["_dates"])

    def test_dates_empty_when_none(self):
        text = "[D-20260213-001]\nStatement: No dates in content\n"
        blocks = parse_blocks(text)
        self.assertEqual(blocks[0]["_dates"], [])

    def test_dates_deduplicated_and_sorted(self):
        text = (
            "[D-20260213-001]\n"
            "Statement: On 2026-02-13 we decided. Confirmed 2026-02-13.\n"
        )
        blocks = parse_blocks(text)
        self.assertEqual(blocks[0]["_dates"].count("2026-02-13"), 1)

    def test_has_negation_true(self):
        text = "[D-20260213-001]\nStatement: We should never use plaintext passwords\n"
        blocks = parse_blocks(text)
        self.assertTrue(blocks[0]["_has_negation"])

    def test_has_negation_false(self):
        text = "[D-20260213-001]\nStatement: Use JWT tokens for authentication\n"
        blocks = parse_blocks(text)
        self.assertFalse(blocks[0]["_has_negation"])

    def test_negation_patterns(self):
        """Various negation patterns should be detected."""
        negation_phrases = [
            "not allowed",
            "don't use this",
            "won't work",
            "shouldn't deploy",
            "cannot proceed",
            "can't be done",
            "doesn't apply",
            "didn't happen",
        ]
        for phrase in negation_phrases:
            text = f"[D-20260213-001]\nStatement: {phrase}\n"
            blocks = parse_blocks(text)
            self.assertTrue(
                blocks[0]["_has_negation"],
                f"Expected negation for: {phrase}",
            )

    def test_fact_keys_across_separator(self):
        """Fact keys should be present on both blocks separated by ---."""
        text = (
            "[D-20260213-001]\nStatement: See T-20260213-002 on 2026-01-15\n"
            "---\n"
            "[D-20260213-002]\nStatement: We never did this\n"
        )
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertIn("T-20260213-002", blocks[0]["_entities"])
        self.assertIn("2026-01-15", blocks[0]["_dates"])
        self.assertFalse(blocks[0]["_has_negation"])
        self.assertEqual(blocks[1]["_entities"], [])
        self.assertTrue(blocks[1]["_has_negation"])

    def test_fact_keys_from_list_fields(self):
        """Entity IDs in list fields should also be extracted."""
        text = (
            "[D-20260213-001]\nTitle: Test\n"
            "Sources:\n- D-20260213-002\n- PER-admin\n"
        )
        blocks = parse_blocks(text)
        self.assertIn("D-20260213-002", blocks[0]["_entities"])
        self.assertIn("PER-admin", blocks[0]["_entities"])


if __name__ == "__main__":
    unittest.main()
