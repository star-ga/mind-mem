# Copyright 2026 STARGA, Inc.
"""The batch-review surface must never grow an auto-approval path.

Batch review's standing temptation is an "auto-approve the low-risk
ones" fast path. It never ships: a human approves, and the tool only
makes approving fast. These tests fail the build if that path is ever
introduced — structurally (the approval choke point moves or multiplies),
lexically (an identifier named for approving without a human), or
behaviourally (a proposal nobody decided on gets applied).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _review_autoapprove_scan import (  # noqa: E402
    APPROVAL_CHOKE_POINT,
    REJECTION_CHOKE_POINT,
    banned_identifiers,
    governed_call_sites,
    risk_branches,
)
from _review_fixtures import build_workspace, mcp_budget, proposal_status  # noqa: E402,F401


class TestStructure:
    def test_only_one_function_may_call_the_governed_approval(self):
        callers = {fn for _mod, fn, callee in governed_call_sites() if callee == "approve_apply"}
        assert callers == {APPROVAL_CHOKE_POINT}, (
            f"approve_apply is reachable from {sorted(callers)}; the review surface "
            f"must funnel every approval through {APPROVAL_CHOKE_POINT}()."
        )

    def test_only_one_function_may_call_the_governed_rejection(self):
        callers = {fn for _mod, fn, callee in governed_call_sites() if callee == "reject_proposal"}
        assert callers == {REJECTION_CHOKE_POINT}

    def test_the_review_surface_never_calls_the_apply_engine_directly(self):
        direct = [site for site in governed_call_sites() if site[2] == "apply_proposal"]
        assert direct == [], f"review must not bypass approve_apply: {direct}"

    def test_no_identifier_names_an_unattended_approval(self):
        assert banned_identifiers() == ()

    def test_the_review_surface_never_branches_on_risk(self):
        assert risk_branches() == (), (
            "a review front end that branches on Risk is one refactor away from "
            "'auto-approve the low-risk ones'; risk is displayed, never acted on."
        )

    def test_the_scanner_actually_inspects_the_shipped_modules(self):
        """A scanner that reads nothing would pass every check above."""
        from _review_autoapprove_scan import REVIEW_MODULES, module_paths

        paths = module_paths()
        assert len(paths) == len(REVIEW_MODULES)
        for path in paths:
            assert os.path.isfile(path), f"scanner points at a missing module: {path}"
        assert governed_call_sites(), "scanner found no governed call sites at all — it is not reading the source"


class TestScannerCatchesRegressions:
    """The guard is only worth having if it fails on the thing it forbids."""

    def test_scanner_flags_an_injected_auto_approve_identifier(self, tmp_path, monkeypatch):
        import _review_autoapprove_scan as scan

        offender = tmp_path / "review_batch.py"
        offender.write_text("def auto_approve_low_risk(items):\n    return items\n", encoding="utf-8")
        monkeypatch.setattr(scan, "SRC_ROOT", str(tmp_path))
        monkeypatch.setattr(scan, "REVIEW_MODULES", ("review_batch.py",))
        assert ("review_batch.py", "auto_approve_low_risk") in scan.banned_identifiers()

    def test_scanner_flags_an_injected_risk_branch(self, tmp_path, monkeypatch):
        import _review_autoapprove_scan as scan

        offender = tmp_path / "review_batch.py"
        offender.write_text("def f(item):\n    if item.risk == 'low':\n        return True\n", encoding="utf-8")
        monkeypatch.setattr(scan, "SRC_ROOT", str(tmp_path))
        monkeypatch.setattr(scan, "REVIEW_MODULES", ("review_batch.py",))
        assert scan.risk_branches()

    def test_scanner_flags_an_approval_outside_the_choke_point(self, tmp_path, monkeypatch):
        import _review_autoapprove_scan as scan

        offender = tmp_path / "review_batch.py"
        offender.write_text("def sneaky(pid):\n    return approve_apply(pid, dry_run=False)\n", encoding="utf-8")
        monkeypatch.setattr(scan, "SRC_ROOT", str(tmp_path))
        monkeypatch.setattr(scan, "REVIEW_MODULES", ("review_batch.py",))
        assert ("review_batch.py", "sneaky", "approve_apply") in scan.governed_call_sites()

    def test_scanner_ignores_prose_that_forbids_auto_approval(self, tmp_path, monkeypatch):
        import _review_autoapprove_scan as scan

        clean = tmp_path / "review_batch.py"
        clean.write_text('"""Never auto-approve. No auto_approve fast path, ever."""\n', encoding="utf-8")
        monkeypatch.setattr(scan, "SRC_ROOT", str(tmp_path))
        monkeypatch.setattr(scan, "REVIEW_MODULES", ("review_batch.py",))
        assert scan.banned_identifiers() == ()


class TestBehaviour:
    @pytest.fixture
    def workspace(self, tmp_path):
        root = str(tmp_path / "ws")
        os.makedirs(root)
        ids = build_workspace(root, 4)
        return root, ids

    def test_a_proposal_with_no_operator_decision_is_never_applied(self, workspace, monkeypatch):
        from mind_mem.review_batch import ReviewDecision, run_batch

        monkeypatch.setattr(
            "mind_mem.apply_engine.check_no_touch_window",
            lambda ws: (True, "test: window neutralised"),
        )
        root, ids = workspace
        run_batch(root, (ReviewDecision(ids[0], "approve", origin="keypress"),))
        for pid in ids[1:]:
            assert proposal_status(root, pid) == "staged"

    def test_the_review_session_refuses_a_non_operator_origin(self):
        from mind_mem.review_batch import OPERATOR_ORIGINS, ReviewBatchError, ReviewDecision

        assert "policy" not in OPERATOR_ORIGINS
        assert "auto" not in OPERATOR_ORIGINS
        for origin in ("policy", "auto", "scheduler", "daemon", ""):
            with pytest.raises(ReviewBatchError):
                ReviewDecision("P-20260829-001", "approve", origin=origin)
