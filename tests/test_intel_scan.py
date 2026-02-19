#!/usr/bin/env python3
"""Tests for intel_scan.py — contradiction detection, drift analysis, impact graph."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from apply_engine import validate_proposal
from block_parser import parse_file
from intel_scan import (
    IntelReport,
    check_signature_conflict,
    detect_contradictions,
    detect_drift,
    generate_proposals,
    get_axis_key,
    load_all,
    scopes_overlap,
)


class TestIntelReport(unittest.TestCase):
    def test_accumulates_counts(self):
        r = IntelReport()
        r.critical_msg("crit1")
        r.warn("warn1")
        r.warn("warn2")
        r.info_msg("info1")
        self.assertEqual(r.critical, 1)
        self.assertEqual(r.warnings, 2)
        self.assertEqual(r.info, 1)

    def test_text_output(self):
        r = IntelReport()
        r.section("TEST")
        r.ok("All good")
        text = r.text()
        self.assertIn("=== TEST ===", text)
        self.assertIn("OK: All good", text)


class TestScopesOverlap(unittest.TestCase):
    def test_empty_scopes_overlap(self):
        self.assertTrue(scopes_overlap({}, {}))

    def test_disjoint_projects(self):
        s1 = {"projects": ["proj-a"]}
        s2 = {"projects": ["proj-b"]}
        self.assertFalse(scopes_overlap(s1, s2))

    def test_overlapping_projects(self):
        s1 = {"projects": ["proj-a", "proj-b"]}
        s2 = {"projects": ["proj-b", "proj-c"]}
        self.assertTrue(scopes_overlap(s1, s2))

    def test_disjoint_time(self):
        s1 = {"time": {"start": "2026-01-01", "end": "2026-01-31"}}
        s2 = {"time": {"start": "2026-03-01", "end": "2026-03-31"}}
        self.assertFalse(scopes_overlap(s1, s2))

    def test_overlapping_time(self):
        s1 = {"time": {"start": "2026-01-01", "end": "2026-03-01"}}
        s2 = {"time": {"start": "2026-02-01", "end": "2026-04-01"}}
        self.assertTrue(scopes_overlap(s1, s2))


class TestGetAxisKey(unittest.TestCase):
    def test_with_axis(self):
        sig = {"axis": {"key": "auth.jwt"}}
        self.assertEqual(get_axis_key(sig), "auth.jwt")

    def test_fallback_to_domain_subject(self):
        sig = {"domain": "security", "subject": "tokens"}
        self.assertEqual(get_axis_key(sig), "security.tokens")

    def test_empty_sig(self):
        self.assertEqual(get_axis_key({}), "other.unknown")


class TestCheckSignatureConflict(unittest.TestCase):
    def test_no_conflict_different_axis(self):
        s1 = {"id": "CS-001", "axis": {"key": "auth.jwt"}, "modality": "must", "scope": {}}
        s2 = {"id": "CS-002", "axis": {"key": "db.postgres"}, "modality": "must_not", "scope": {}}
        self.assertIsNone(check_signature_conflict(s1, s2))

    def test_modality_conflict(self):
        s1 = {"id": "CS-001", "axis": {"key": "auth.jwt"}, "modality": "must",
               "scope": {}, "predicate": "use", "object": "JWT"}
        s2 = {"id": "CS-002", "axis": {"key": "auth.jwt"}, "modality": "must_not",
               "scope": {}, "predicate": "use", "object": "JWT"}
        result = check_signature_conflict(s1, s2)
        self.assertIsNotNone(result)
        self.assertEqual(result["severity"], "critical")

    def test_composes_with_suppresses(self):
        s1 = {"id": "CS-001", "axis": {"key": "auth.jwt"}, "modality": "must",
               "scope": {}, "composes_with": ["CS-002"]}
        s2 = {"id": "CS-002", "axis": {"key": "auth.jwt"}, "modality": "must_not",
               "scope": {}}
        self.assertIsNone(check_signature_conflict(s1, s2))

    def test_disjoint_scope_no_conflict(self):
        s1 = {"id": "CS-001", "axis": {"key": "auth"}, "modality": "must",
               "scope": {"projects": ["proj-a"]}}
        s2 = {"id": "CS-002", "axis": {"key": "auth"}, "modality": "must_not",
               "scope": {"projects": ["proj-b"]}}
        self.assertIsNone(check_signature_conflict(s1, s2))


class TestCheckSignatureConflictCompeting(unittest.TestCase):
    """Tests for competing hard requirements (same axis, same predicate, different objects)."""

    def test_competing_must_different_objects_is_critical(self):
        """Two 'must use X' vs 'must use Y' on same axis = critical."""
        s1 = {"id": "CS-001", "axis": {"key": "db.primary"}, "modality": "must",
               "scope": {}, "predicate": "use", "object": "PostgreSQL"}
        s2 = {"id": "CS-002", "axis": {"key": "db.primary"}, "modality": "must",
               "scope": {}, "predicate": "use", "object": "MySQL"}
        result = check_signature_conflict(s1, s2)
        self.assertIsNotNone(result)
        self.assertEqual(result["severity"], "critical")
        self.assertIn("competing hard requirements", result["reason"])

    def test_competing_must_not_different_objects_is_compatible(self):
        """Two 'must_not allow X' vs 'must_not allow Y' = compatible (avoid both)."""
        s1 = {"id": "CS-001", "axis": {"key": "auth.session"}, "modality": "must_not",
               "scope": {}, "predicate": "allow", "object": "plaintext_cookies"}
        s2 = {"id": "CS-002", "axis": {"key": "auth.session"}, "modality": "must_not",
               "scope": {}, "predicate": "allow", "object": "insecure_tokens"}
        result = check_signature_conflict(s1, s2)
        self.assertIsNone(result, "must_not + must_not with different objects are compatible")

    def test_competing_should_different_objects_is_low(self):
        """Two soft requirements with different objects = low severity."""
        s1 = {"id": "CS-001", "axis": {"key": "style.format"}, "modality": "should",
               "scope": {}, "predicate": "use", "object": "tabs"}
        s2 = {"id": "CS-002", "axis": {"key": "style.format"}, "modality": "should",
               "scope": {}, "predicate": "use", "object": "spaces"}
        result = check_signature_conflict(s1, s2)
        self.assertIsNotNone(result)
        self.assertEqual(result["severity"], "low")

    def test_same_predicate_same_object_no_conflict(self):
        """Same modality, same predicate, same object = agreement, not conflict."""
        s1 = {"id": "CS-001", "axis": {"key": "db.primary"}, "modality": "must",
               "scope": {}, "predicate": "use", "object": "PostgreSQL"}
        s2 = {"id": "CS-002", "axis": {"key": "db.primary"}, "modality": "must",
               "scope": {}, "predicate": "use", "object": "PostgreSQL"}
        result = check_signature_conflict(s1, s2)
        self.assertIsNone(result)

    def test_non_exclusive_axis_allows_additive_must(self):
        """axis.exclusive=false: two must+must with different objects = no conflict."""
        s1 = {"id": "CS-001", "axis": {"key": "team.hire", "exclusive": False},
               "modality": "must", "scope": {}, "predicate": "hire", "object": "Alice"}
        s2 = {"id": "CS-002", "axis": {"key": "team.hire", "exclusive": False},
               "modality": "must", "scope": {}, "predicate": "hire", "object": "Bob"}
        result = check_signature_conflict(s1, s2)
        self.assertIsNone(result, "Non-exclusive axis should allow additive must constraints")


class TestDetectContradictions(unittest.TestCase):
    def test_no_active_decisions(self):
        report = IntelReport()
        result = detect_contradictions([], report)
        self.assertEqual(result, [])

    def test_detects_contradiction(self):
        decisions = [
            {
                "_id": "D-20260213-001", "Status": "active",
                "ConstraintSignatures": [
                    {"id": "CS-001", "axis": {"key": "auth"}, "modality": "must",
                     "scope": {}, "predicate": "use", "object": "JWT"}
                ]
            },
            {
                "_id": "D-20260213-002", "Status": "active",
                "ConstraintSignatures": [
                    {"id": "CS-002", "axis": {"key": "auth"}, "modality": "must_not",
                     "scope": {}, "predicate": "use", "object": "JWT"}
                ]
            },
        ]
        report = IntelReport()
        result = detect_contradictions(decisions, report)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["severity"], "critical")

    def test_detects_competing_hard_requirements(self):
        """Two active decisions with must+must same predicate, different objects."""
        decisions = [
            {
                "_id": "D-20260214-001", "Status": "active",
                "ConstraintSignatures": [
                    {"id": "CS-010", "axis": {"key": "db.primary"}, "modality": "must",
                     "scope": {}, "predicate": "use", "object": "PostgreSQL", "priority": 5}
                ]
            },
            {
                "_id": "D-20260214-002", "Status": "active",
                "ConstraintSignatures": [
                    {"id": "CS-011", "axis": {"key": "db.primary"}, "modality": "must",
                     "scope": {}, "predicate": "use", "object": "MySQL", "priority": 3}
                ]
            },
        ]
        report = IntelReport()
        result = detect_contradictions(decisions, report)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["severity"], "critical")
        self.assertIn("competing", result[0]["reason"])


class TestDetectDrift(unittest.TestCase):
    def test_no_drift_on_empty(self):
        data = {"decisions": [], "tasks": [], "incidents": []}
        report = IntelReport()
        result = detect_drift(data, report)
        self.assertEqual(result, [])

    def test_detects_blocked_tasks(self):
        data = {
            "decisions": [],
            "tasks": [{"_id": "T-001", "Status": "blocked", "Title": "Stuck task"}],
            "incidents": [],
        }
        report = IntelReport()
        result = detect_drift(data, report)
        stalled = [s for s in result if s["signal"] == "stalled_tasks"]
        self.assertEqual(len(stalled), 1)


class TestLoadAll(unittest.TestCase):
    def test_missing_files_return_empty(self):
        with tempfile.TemporaryDirectory() as td:
            data = load_all(td)
            for key in data:
                self.assertEqual(data[key], [])


class TestE2EProposalToApply(unittest.TestCase):
    """End-to-end: generate_proposals → validate_proposal pipeline."""

    def _scaffold_workspace(self, td):
        """Create minimal workspace structure for proposal generation."""
        os.makedirs(os.path.join(td, "decisions"), exist_ok=True)
        os.makedirs(os.path.join(td, "intelligence/proposed"), exist_ok=True)
        os.makedirs(os.path.join(td, "maintenance"), exist_ok=True)

        # Empty proposed files for generate_proposals to append to
        for fname in ("DECISIONS_PROPOSED.md", "TASKS_PROPOSED.md", "EDITS_PROPOSED.md"):
            with open(os.path.join(td, "intelligence/proposed", fname), "w") as f:
                f.write(f"# Proposed {fname.replace('_PROPOSED.md', '').title()}\n\n")

        # Config with proposal mode enabled
        with open(os.path.join(td, "mind-mem.json"), "w") as f:
            json.dump({
                "mode": "propose",
                "proposal_budget": {"per_run": 3, "per_day": 6, "backlog_limit": 30}
            }, f)

        # Intel state
        with open(os.path.join(td, "intelligence/intel-state.json"), "w") as f:
            json.dump({"mode": "propose", "counters": {}}, f)

        return td

    def test_generated_proposals_pass_validation(self):
        """Proposals from generate_proposals() must pass validate_proposal()."""
        with tempfile.TemporaryDirectory() as td:
            self._scaffold_workspace(td)

            # Create contradictions that generate_proposals will act on
            contradictions = [{
                "sig1": {
                    "decision": "D-20260214-001",
                    "sig": {"id": "CS-010", "axis": {"key": "db"}, "modality": "must",
                            "predicate": "use", "object": "PostgreSQL", "priority": 5}
                },
                "sig2": {
                    "decision": "D-20260214-002",
                    "sig": {"id": "CS-011", "axis": {"key": "db"}, "modality": "must",
                            "predicate": "use", "object": "MySQL", "priority": 3}
                },
                "severity": "critical",
                "reason": "competing hard requirements on axis=db",
            }]

            report = IntelReport()
            intel_state = {"mode": "propose", "counters": {}}

            count = generate_proposals(contradictions, [], td, intel_state, report)
            self.assertEqual(count, 1)

            # Contradiction proposals have type=edit, so they route to EDITS_PROPOSED
            proposed_path = os.path.join(td, "intelligence/proposed/EDITS_PROPOSED.md")
            blocks = parse_file(proposed_path)

            # Should have at least one proposal block
            proposals = [b for b in blocks if b.get("ProposalId")]
            self.assertGreaterEqual(len(proposals), 1)

            # Each proposal must pass validate_proposal and have a Fingerprint
            for p in proposals:
                errors = validate_proposal(p)
                self.assertEqual(errors, [],
                                 f"Proposal {p.get('ProposalId')} failed validation: {errors}")
                self.assertTrue(p.get("Fingerprint"),
                                f"Proposal {p.get('ProposalId')} missing Fingerprint field")

    def test_proposal_budget_limits_output(self):
        """per_run budget limits the number of proposals generated."""
        with tempfile.TemporaryDirectory() as td:
            self._scaffold_workspace(td)

            # Override per_run=1
            with open(os.path.join(td, "mind-mem.json"), "w") as f:
                json.dump({
                    "mode": "propose",
                    "proposal_budget": {"per_run": 1, "per_day": 10, "backlog_limit": 30}
                }, f)

            contradictions = [
                {
                    "sig1": {"decision": "D-001", "sig": {"id": "CS-A", "priority": 5}},
                    "sig2": {"decision": "D-002", "sig": {"id": "CS-B", "priority": 3}},
                    "severity": "critical",
                    "reason": "test contradiction 1",
                },
                {
                    "sig1": {"decision": "D-003", "sig": {"id": "CS-C", "priority": 5}},
                    "sig2": {"decision": "D-004", "sig": {"id": "CS-D", "priority": 3}},
                    "severity": "critical",
                    "reason": "test contradiction 2",
                },
            ]

            report = IntelReport()
            intel_state = {"mode": "propose", "counters": {}}

            count = generate_proposals(contradictions, [], td, intel_state, report)
            # per_run=1 should cap at 1 proposal
            self.assertEqual(count, 1)


class TestProposalIdCollision(unittest.TestCase):
    """Proposal IDs must not collide across multiple runs."""

    def _scaffold(self, td):
        os.makedirs(os.path.join(td, "intelligence/proposed"), exist_ok=True)
        os.makedirs(os.path.join(td, "maintenance"), exist_ok=True)
        for fname in ("DECISIONS_PROPOSED.md", "TASKS_PROPOSED.md", "EDITS_PROPOSED.md"):
            with open(os.path.join(td, "intelligence/proposed", fname), "w") as f:
                f.write(f"# {fname}\n\n")
        with open(os.path.join(td, "mind-mem.json"), "w") as f:
            json.dump({"mode": "propose", "proposal_budget": {"per_run": 5, "per_day": 10, "backlog_limit": 30}}, f)
        with open(os.path.join(td, "intelligence/intel-state.json"), "w") as f:
            json.dump({"mode": "propose", "counters": {}}, f)

    def test_second_run_increments_from_existing(self):
        """Second run should start IDs after existing proposals."""
        with tempfile.TemporaryDirectory() as td:
            self._scaffold(td)
            c1 = [{
                "sig1": {"decision": "D-001", "sig": {"id": "CS-1", "priority": 5}},
                "sig2": {"decision": "D-002", "sig": {"id": "CS-2", "priority": 3}},
                "severity": "critical", "reason": "test contradiction 1",
            }]
            report = IntelReport()
            generate_proposals(c1, [], td, {"mode": "propose", "counters": {}}, report)

            # Second run with different contradiction
            c2 = [{
                "sig1": {"decision": "D-003", "sig": {"id": "CS-3", "priority": 5}},
                "sig2": {"decision": "D-004", "sig": {"id": "CS-4", "priority": 3}},
                "severity": "critical", "reason": "test contradiction 2",
            }]
            report2 = IntelReport()
            generate_proposals(c2, [], td, {"mode": "propose", "counters": {}}, report2)

            # Parse all proposals — IDs must be unique
            blocks = parse_file(os.path.join(td, "intelligence/proposed/EDITS_PROPOSED.md"))
            proposal_ids = [b["ProposalId"] for b in blocks if b.get("ProposalId")]
            self.assertEqual(len(proposal_ids), 2, f"Expected 2 proposals, got {len(proposal_ids)}")
            self.assertEqual(len(set(proposal_ids)), 2, f"Proposal IDs must be unique: {proposal_ids}")


class TestProposalRouting(unittest.TestCase):
    """Proposals must route to correct file based on Type field."""

    def _scaffold(self, td):
        os.makedirs(os.path.join(td, "intelligence/proposed"), exist_ok=True)
        os.makedirs(os.path.join(td, "maintenance"), exist_ok=True)
        for fname in ("DECISIONS_PROPOSED.md", "TASKS_PROPOSED.md", "EDITS_PROPOSED.md"):
            with open(os.path.join(td, "intelligence/proposed", fname), "w") as f:
                f.write(f"# {fname}\n\n")
        with open(os.path.join(td, "mind-mem.json"), "w") as f:
            json.dump({"mode": "propose", "proposal_budget": {"per_run": 5, "per_day": 10, "backlog_limit": 30}}, f)
        with open(os.path.join(td, "intelligence/intel-state.json"), "w") as f:
            json.dump({"mode": "propose", "counters": {}}, f)

    def test_edit_proposals_go_to_edits_file(self):
        """Type=edit proposals route to EDITS_PROPOSED.md."""
        with tempfile.TemporaryDirectory() as td:
            self._scaffold(td)
            contradictions = [{
                "sig1": {"decision": "D-001", "sig": {"id": "CS-1", "priority": 5}},
                "sig2": {"decision": "D-002", "sig": {"id": "CS-2", "priority": 3}},
                "severity": "critical", "reason": "test",
            }]
            report = IntelReport()
            generate_proposals(contradictions, [], td, {"mode": "propose", "counters": {}}, report)

            with open(os.path.join(td, "intelligence/proposed/EDITS_PROPOSED.md")) as f:
                edits = f.read()
            with open(os.path.join(td, "intelligence/proposed/DECISIONS_PROPOSED.md")) as f:
                decisions = f.read()
            self.assertIn("ProposalId:", edits, "Edit proposals should be in EDITS_PROPOSED.md")
            self.assertNotIn("ProposalId:", decisions, "Edit proposals should NOT be in DECISIONS_PROPOSED.md")


class TestParserContinuationLines(unittest.TestCase):
    """Parser should support indented continuation lines for multi-line values."""

    def test_continuation_line_appends(self):
        """Indented text after a field appends to the value with newline."""
        text = (
            "[D-20260215-001]\n"
            "Status: active\n"
            "Statement: This is a long rationale\n"
            "  that continues on the next line\n"
            "  and even a third line\n"
            "Tags: test\n"
        )
        blocks = parse_file.__wrapped__(text) if hasattr(parse_file, '__wrapped__') else None
        # Use parse_blocks directly
        from block_parser import parse_blocks
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("\n", blocks[0]["Statement"])
        self.assertIn("continues on the next line", blocks[0]["Statement"])
        self.assertIn("third line", blocks[0]["Statement"])


class TestParserQuotedInlineList(unittest.TestCase):
    """Parser should handle quoted strings in inline lists."""

    def test_quoted_comma_in_list(self):
        """Quoted strings preserve commas within values."""
        from block_parser import _parse_inline_list
        result = _parse_inline_list('["React, Redux", "Vue"]')
        self.assertEqual(result, ["React, Redux", "Vue"])

    def test_unquoted_list_unchanged(self):
        """Regular lists without quotes work as before."""
        from block_parser import _parse_inline_list
        result = _parse_inline_list('[a, b, c]')
        self.assertEqual(result, ["a", "b", "c"])


class TestParsedInlineDict(unittest.TestCase):
    """Parser should handle quoted strings in inline dicts."""

    def test_quoted_comma_in_dict_value(self):
        """Quoted strings in dict values preserve commas."""
        from block_parser import _parse_inline_dict
        result = _parse_inline_dict('{tags: "frontend, ui", name: foo}')
        self.assertEqual(result["tags"], "frontend, ui")
        self.assertEqual(result["name"], "foo")

    def test_unquoted_dict_unchanged(self):
        """Regular dicts without quotes work as before."""
        from block_parser import _parse_inline_dict
        result = _parse_inline_dict('{key: val, key2: val2}')
        self.assertEqual(result, {"key": "val", "key2": "val2"})


if __name__ == "__main__":
    unittest.main()
