#!/usr/bin/env python3
"""Tests for the lint -> repair-proposal path (mind_mem.lint / lint_autofix).

Gate covered here:
  * three finding classes produce correct proposals (stale date, missing
    metadata, duplicate/contradiction);
  * golden-diff assertions on the staged proposal text;
  * the store is never direct-written — only the proposal file changes;
  * approve_apply round-trips end to end;
  * an unknown finding id raises a typed error, not a traceback.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime

from mind_mem.block_parser import parse_file
from mind_mem.init_workspace import init
from mind_mem.lint import (
    RULE_DUPLICATE_BLOCK,
    RULE_MISSING_METADATA,
    RULE_STALE_DATE,
    LintError,
    find_finding,
    lint,
)
from mind_mem.lint_autofix import (
    PROPOSAL_FILE,
    LintAutofixError,
    NotAutofixableError,
    UnknownFindingError,
    lint_autofix,
)
from mind_mem.v4.feature_flags import FeatureDisabledError

FIXED_NOW = datetime(2026, 8, 27, 12, 0, 0)

# One block per defect class, plus a clean control block.
DECISIONS = """
[D-20260101-001]
Date: 2026-01-01
Status: active
Scope: global
Statement: Default block store backend is SQLite.
Rationale: Zero configuration for a new workspace.
Supersedes: none
Tags: storage
Sources:
- decisions/DECISIONS.md

[D-20260102-002]
Date: 2025-12-30
Status: revoked
Scope: global
Statement: Recall cache is invalidated on every governance event.
Rationale: A stale envelope misleads the next query.
Supersedes: none
Tags: recall
Sources:
- decisions/DECISIONS.md

[D-20260103-003]
Date: 2026-01-03
Status: revoked
Scope:
Statement: Proposals are staged before they are applied.
Rationale: Human review is the gate.
Supersedes: none
Tags: governance
Sources:
- decisions/DECISIONS.md

[D-20260104-004]
Date: 2026-01-04
Status: active
Scope: global
Statement: Default block store backend is SQLite.
Rationale: Restated during onboarding.
Supersedes: none
Tags: storage
Sources:
- decisions/DECISIONS.md
"""

# A required field that is absent entirely and has no content-free
# default — reported, but deliberately not auto-fixable.
DECISIONS_UNFIXABLE = """
[D-20260105-001]
Date: 2026-01-05
Status: active
Scope: global
Statement: Evidence objects are tamper evident.
Supersedes: none
Tags: integrity
Sources:
- decisions/DECISIONS.md
"""


def _make_ws(decisions: str = DECISIONS, *, enable_lint: bool = True, mode: str = "detect_only") -> str:
    ws = tempfile.mkdtemp(prefix="mm_lint_")
    init(ws)
    config_path = os.path.join(ws, "mind-mem.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if enable_lint:
        config["v4"] = {"lint": {"enabled": True}}
    config["governance_mode"] = mode
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    state_path = os.path.join(ws, "memory", "intel-state.json")
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    state["governance_mode"] = mode
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write(decisions)
    return ws


def _digests(ws: str) -> dict[str, str]:
    """sha256 of every corpus file that is NOT the proposal file."""
    out: dict[str, str] = {}
    for root, _dirs, files in os.walk(ws):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, ws).replace(os.sep, "/")
            if rel == PROPOSAL_FILE:
                continue
            with open(path, "rb") as handle:
                out[rel] = hashlib.sha256(handle.read()).hexdigest()
    return out


def _by_rule(ws: str) -> dict[str, object]:
    return {f.rule: f for f in lint(ws)}


def _staged_text(ws: str) -> str:
    with open(os.path.join(ws, PROPOSAL_FILE), encoding="utf-8") as handle:
        return handle.read()


class TestLintFindings(unittest.TestCase):
    def test_reports_exactly_three_classes(self):
        ws = _make_ws()
        findings = lint(ws)
        self.assertEqual([f.rule for f in findings], [RULE_DUPLICATE_BLOCK, RULE_MISSING_METADATA, RULE_STALE_DATE])
        self.assertEqual(
            [f.block_id for f in findings],
            ["D-20260104-004", "D-20260103-003", "D-20260102-002"],
        )
        self.assertTrue(all(f.autofixable for f in findings))
        self.assertTrue(all(f.finding_id.startswith("LF-") and len(f.finding_id) == 11 for f in findings))

    def test_finding_ids_are_stable_across_runs(self):
        ws = _make_ws()
        self.assertEqual([f.finding_id for f in lint(ws)], [f.finding_id for f in lint(ws)])

    def test_flag_off_disables_the_surface(self):
        ws = _make_ws(enable_lint=False)
        with self.assertRaises(FeatureDisabledError):
            lint(ws)
        with self.assertRaises(FeatureDisabledError):
            lint_autofix(ws, "LF-00000000")

    def test_unknown_rule_is_typed(self):
        ws = _make_ws()
        with self.assertRaises(LintError):
            lint(ws, rules=["no_such_rule"])


class TestProposalGoldenDiff(unittest.TestCase):
    """The staged block is byte-checked, fingerprint included."""

    def _stage(self, rule: str) -> str:
        ws = _make_ws()
        before = _staged_text(ws)
        finding = _by_rule(ws)[rule]
        proposal_id = lint_autofix(ws, finding.finding_id, now=FIXED_NOW)  # type: ignore[attr-defined]
        self.assertEqual(proposal_id, "P-20260827-001")
        after = _staged_text(ws)
        self.assertTrue(after.startswith(before), "the proposal file is append-only")
        return after[len(before) :]

    def test_stale_date_golden(self):
        self.assertEqual(
            self._stage(RULE_STALE_DATE),
            "\n[P-20260827-001]\n"
            "ProposalId: P-20260827-001\n"
            "Type: edit\n"
            "TargetBlock: D-20260102-002\n"
            "Risk: low\n"
            "Evidence:\n"
            "- lint stale_date: Date '2025-12-30' drifted from id anchor '2026-01-02'\n"
            "Rollback: restore_snapshot\n"
            "Ops:\n"
            "- op: update_field\n"
            "  file: decisions/DECISIONS.md\n"
            "  target: D-20260102-002\n"
            "  field: Date\n"
            "  value: 2026-01-02\n"
            "Fingerprint: ab127ca961c5d126\n"
            "Status: staged\n"
            "FilesTouched:\n"
            "- decisions/DECISIONS.md\n"
            "Sources:\n"
            "- lint:stale_date\n"
            "- finding:LF-b07a317a\n",
        )

    def test_missing_metadata_golden(self):
        self.assertEqual(
            self._stage(RULE_MISSING_METADATA),
            "\n[P-20260827-001]\n"
            "ProposalId: P-20260827-001\n"
            "Type: edit\n"
            "TargetBlock: D-20260103-003\n"
            "Risk: low\n"
            "Evidence:\n"
            "- lint missing_metadata: required field 'Scope' is empty\n"
            "Rollback: restore_snapshot\n"
            "Ops:\n"
            "- op: update_field\n"
            "  file: decisions/DECISIONS.md\n"
            "  target: D-20260103-003\n"
            "  field: Scope\n"
            "  value: global\n"
            "Fingerprint: 6e1aad597871d2f9\n"
            "Status: staged\n"
            "FilesTouched:\n"
            "- decisions/DECISIONS.md\n"
            "Sources:\n"
            "- lint:missing_metadata\n"
            "- finding:LF-87a4c263\n",
        )

    def test_duplicate_golden(self):
        self.assertEqual(
            self._stage(RULE_DUPLICATE_BLOCK),
            "\n[P-20260827-001]\n"
            "ProposalId: P-20260827-001\n"
            "Type: edit\n"
            "TargetBlock: D-20260104-004\n"
            "Risk: high\n"
            "Evidence:\n"
            "- lint duplicate_block: duplicate statement of D-20260101-001\n"
            "Rollback: restore_snapshot\n"
            "Ops:\n"
            "- op: set_status\n"
            "  file: decisions/DECISIONS.md\n"
            "  target: D-20260104-004\n"
            "  status: superseded\n"
            "Fingerprint: f80c57183228d2f8\n"
            "Status: staged\n"
            "FilesTouched:\n"
            "- decisions/DECISIONS.md\n"
            "Sources:\n"
            "- lint:duplicate_block\n"
            "- finding:LF-d6ba55bd\n",
        )


class TestNeverDirectWrites(unittest.TestCase):
    def test_store_untouched_until_approval(self):
        ws = _make_ws()
        before = _digests(ws)
        for finding in lint(ws):
            lint_autofix(ws, finding.finding_id, now=FIXED_NOW)
        self.assertEqual(_digests(ws), before, "lint_autofix must not touch anything but the proposal file")
        staged = parse_file(os.path.join(ws, PROPOSAL_FILE))
        self.assertEqual(len(staged), 3)
        self.assertTrue(all(b.get("Status") == "staged" for b in staged))
        self.assertEqual(
            [b["ProposalId"] for b in staged],
            ["P-20260827-001", "P-20260827-002", "P-20260827-003"],
        )


class TestTypedErrors(unittest.TestCase):
    def test_identical_repair_is_refused_once_staged(self):
        ws = _make_ws()
        finding = _by_rule(ws)[RULE_STALE_DATE]
        lint_autofix(ws, finding.finding_id, now=FIXED_NOW)  # type: ignore[attr-defined]
        with self.assertRaises(LintAutofixError) as ctx:
            lint_autofix(ws, finding.finding_id, now=FIXED_NOW)  # type: ignore[attr-defined]
        self.assertIn("already staged", str(ctx.exception))
        self.assertEqual(len(parse_file(os.path.join(ws, PROPOSAL_FILE))), 1)

    def test_find_finding_round_trips(self):
        ws = _make_ws()
        wanted = lint(ws)[0]
        self.assertEqual(find_finding(ws, wanted.finding_id), wanted)
        self.assertIsNone(find_finding(ws, "LF-deadbeef"))

    def test_malformed_finding_id(self):
        ws = _make_ws()
        with self.assertRaises(UnknownFindingError) as ctx:
            lint_autofix(ws, "not-a-finding")
        self.assertIn("malformed finding id", str(ctx.exception))

    def test_well_formed_but_absent_finding_id(self):
        ws = _make_ws()
        with self.assertRaises(UnknownFindingError) as ctx:
            lint_autofix(ws, "LF-deadbeef")
        self.assertIn("no such finding", str(ctx.exception))
        self.assertIsInstance(ctx.exception, LintAutofixError)

    def test_missing_workspace(self):
        with self.assertRaises(LintAutofixError):
            lint_autofix(os.path.join(tempfile.gettempdir(), "mm_lint_absent_ws"), "LF-deadbeef")

    def test_not_autofixable_is_typed(self):
        ws = _make_ws(DECISIONS_UNFIXABLE)
        findings = lint(ws)
        self.assertEqual(len(findings), 1)
        self.assertFalse(findings[0].autofixable)
        self.assertIn("Rationale", findings[0].detail)
        with self.assertRaises(NotAutofixableError):
            lint_autofix(ws, findings[0].finding_id, now=FIXED_NOW)


class TestApproveApplyRoundTrip(unittest.TestCase):
    def test_end_to_end(self):
        from unittest.mock import patch

        from mind_mem.mcp.tools import governance

        ws = _make_ws(mode="enforce")
        finding = _by_rule(ws)[RULE_STALE_DATE]
        proposal_id = lint_autofix(ws, finding.finding_id, now=FIXED_NOW)  # type: ignore[attr-defined]

        env = dict(os.environ)
        env["MIND_MEM_WORKSPACE"] = ws
        with patch.dict(os.environ, env, clear=True):
            dry = json.loads(governance.approve_apply.__wrapped__(proposal_id, dry_run=True))
            self.assertEqual(dry["status"], "dry_run_passed", dry)
            with patch("mind_mem.apply_engine.check_preconditions", return_value=(True, ["stubbed"])):
                applied = json.loads(governance.approve_apply.__wrapped__(proposal_id, dry_run=False))
        self.assertEqual(applied["status"], "applied", applied)

        blocks = {b["_id"]: b for b in parse_file(os.path.join(ws, "decisions", "DECISIONS.md"))}
        self.assertEqual(blocks["D-20260102-002"]["Date"], "2026-01-02")
        staged = {b["ProposalId"]: b for b in parse_file(os.path.join(ws, PROPOSAL_FILE))}
        self.assertEqual(staged[proposal_id]["Status"], "applied")
        self.assertEqual(lint(ws, rules=[RULE_STALE_DATE]), ())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
