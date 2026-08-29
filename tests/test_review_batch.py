# Copyright 2026 STARGA, Inc.
"""``review_batch`` — batch approve/reject over the governed apply path.

Two invariants carry the feature:

* every decision is **explicit** — an undecided proposal is never applied;
* atomicity is **per proposal** — a mid-batch failure leaves the already
  applied proposals applied and reports the failure, because a
  half-applied batch that silently rolled back is worse than a slow one.

The 10-minute no-touch window in ``apply_engine`` is a real governance
rate limit that caps batch throughput. ``review_batch`` has no way to
lift it and deliberately offers none; the tests below monkeypatch the
gate itself so they measure batch mechanics rather than the rate limit,
and one test leaves it on to prove the batch reports it per proposal.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _review_fixtures import build_workspace, mcp_budget, proposal_status  # noqa: E402,F401


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A workspace an operator holds admin scope over.

    ``approve_apply`` / ``reject_proposal`` are admin-scope MCP tools.
    Exporting ``MIND_MEM_SCOPE=admin`` is what a reviewing operator
    does; ``mm review`` reports the scope but never sets it.
    """
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    root = str(tmp_path / "ws")
    os.makedirs(root)
    ids = build_workspace(root, 5)
    return root, ids


@pytest.fixture
def no_window(monkeypatch):
    """Neutralise the governance rate limit for the duration of one test."""
    monkeypatch.setattr(
        "mind_mem.apply_engine.check_no_touch_window",
        lambda ws: (True, "test: window neutralised"),
    )


def _approve_all(ids):
    from mind_mem.review_batch import ReviewDecision

    return tuple(ReviewDecision(pid, "approve", origin="keypress") for pid in ids)


class TestDecision:
    def test_action_vocabulary_is_closed(self):
        from mind_mem.review_batch import ReviewBatchError, ReviewDecision

        with pytest.raises(ReviewBatchError):
            ReviewDecision("P-20260829-001", "auto", origin="keypress")

    def test_origin_must_be_an_operator_action(self):
        from mind_mem.review_batch import ReviewBatchError, ReviewDecision

        with pytest.raises(ReviewBatchError):
            ReviewDecision("P-20260829-001", "approve", origin="policy")

    def test_reject_requires_a_written_reason(self):
        from mind_mem.review_batch import ReviewBatchError, ReviewDecision

        with pytest.raises(ReviewBatchError):
            ReviewDecision("P-20260829-001", "reject", origin="keypress", reason="no")

    def test_proposal_id_must_be_well_formed(self):
        from mind_mem.review_batch import ReviewBatchError, ReviewDecision

        with pytest.raises(ReviewBatchError):
            ReviewDecision("../../etc/passwd", "approve", origin="keypress")


class TestBatchApprove:
    def test_applies_every_approved_proposal(self, workspace, no_window):
        from mind_mem.review_batch import run_batch

        root, ids = workspace
        report = run_batch(root, _approve_all(ids))
        assert [o.proposal_id for o in report.applied] == list(ids)
        for pid in ids:
            assert proposal_status(root, pid) == "applied"

    def test_an_undecided_proposal_is_never_touched(self, workspace, no_window):
        from mind_mem.review_batch import run_batch

        root, ids = workspace
        run_batch(root, _approve_all(ids[:2]))
        assert proposal_status(root, ids[2]) == "staged"
        assert proposal_status(root, ids[3]) == "staged"
        assert proposal_status(root, ids[4]) == "staged"

    def test_an_empty_decision_set_applies_nothing(self, workspace, no_window):
        from mind_mem.review_batch import run_batch

        root, ids = workspace
        report = run_batch(root, ())
        assert report.applied == ()
        assert all(proposal_status(root, pid) == "staged" for pid in ids)

    def test_duplicate_decisions_are_refused_before_any_apply(self, workspace, no_window):
        from mind_mem.review_batch import ReviewBatchError, ReviewDecision, run_batch

        root, ids = workspace
        doubled = (
            ReviewDecision(ids[0], "approve", origin="keypress"),
            ReviewDecision(ids[0], "reject", origin="keypress", reason="changed my mind"),
        )
        with pytest.raises(ReviewBatchError):
            run_batch(root, doubled)
        assert proposal_status(root, ids[0]) == "staged"


class TestPerProposalAtomicity:
    def test_a_mid_batch_failure_keeps_earlier_applies_and_reports_it(self, workspace, no_window):
        """Proposal 3 of 5 fails; 1-2 stay applied, 4-5 still run."""
        from mind_mem.review_batch import governed_approve, run_batch

        root, ids = workspace
        failing = ids[2]

        def approve_hook(workspace_root, proposal_id, *, dry_run):
            if proposal_id == failing and not dry_run:
                return False, "injected mid-batch failure"
            return governed_approve(workspace_root, proposal_id, dry_run=dry_run)

        report = run_batch(root, _approve_all(ids), approve_hook=approve_hook)

        assert [o.proposal_id for o in report.applied] == [ids[0], ids[1], ids[3], ids[4]]
        assert [o.proposal_id for o in report.failed] == [failing]
        assert report.failed[0].message == "injected mid-batch failure"

        assert proposal_status(root, ids[0]) == "applied"
        assert proposal_status(root, ids[1]) == "applied"
        assert proposal_status(root, failing) == "staged"
        assert proposal_status(root, ids[3]) == "applied"
        assert proposal_status(root, ids[4]) == "applied"

    def test_a_raising_hook_is_reported_not_propagated(self, workspace, no_window):
        from mind_mem.review_batch import governed_approve, run_batch

        root, ids = workspace

        def approve_hook(workspace_root, proposal_id, *, dry_run):
            if proposal_id == ids[1]:
                raise RuntimeError("backend exploded")
            return governed_approve(workspace_root, proposal_id, dry_run=dry_run)

        report = run_batch(root, _approve_all(ids[:3]), approve_hook=approve_hook)
        assert [o.proposal_id for o in report.failed] == [ids[1]]
        assert "backend exploded" in report.failed[0].message
        assert [o.proposal_id for o in report.applied] == [ids[0], ids[2]]

    def test_the_no_touch_window_is_reported_per_proposal_not_swallowed(self, workspace):
        """The real governance rate limit surfaces as a per-proposal failure."""
        from mind_mem.review_batch import run_batch

        root, ids = workspace
        report = run_batch(root, _approve_all(ids))
        assert len(report.applied) == 1
        assert len(report.failed) == 4
        assert all("No-touch window" in o.message for o in report.failed)


class TestBatchReject:
    def test_rejects_with_the_written_reason(self, workspace, no_window):
        from mind_mem.review_batch import ReviewDecision, run_batch

        root, ids = workspace
        decisions = (ReviewDecision(ids[0], "reject", origin="keypress", reason="superseded upstream"),)
        report = run_batch(root, decisions)
        assert [o.proposal_id for o in report.rejected] == [ids[0]]
        assert proposal_status(root, ids[0]) == "rejected"

    def test_a_rejection_does_not_apply_anything(self, workspace, no_window):
        from mind_mem.review_batch import ReviewDecision, run_batch

        root, ids = workspace
        decisions = (ReviewDecision(ids[0], "reject", origin="keypress", reason="superseded upstream"),)
        run_batch(root, decisions)
        with open(os.path.join(root, "decisions/DECISIONS.md"), encoding="utf-8") as handle:
            assert "reviewed-1" not in handle.read()

    def test_the_reason_reaches_the_proposal_file(self, workspace, no_window):
        from mind_mem.review_batch import ReviewDecision, run_batch

        root, ids = workspace
        run_batch(root, (ReviewDecision(ids[0], "reject", origin="keypress", reason="superseded upstream"),))
        with open(os.path.join(root, "intelligence/proposed/EDITS_PROPOSED.md"), encoding="utf-8") as handle:
            assert "superseded upstream" in handle.read()


class TestScope:
    def test_a_non_admin_scope_is_reported_as_a_failure_not_a_silent_skip(self, workspace, no_window, monkeypatch):
        from mind_mem.review_batch import run_batch

        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        root, ids = workspace
        report = run_batch(root, _approve_all(ids[:1]))
        assert report.applied == ()
        assert len(report.failed) == 1
        assert "admin" in report.failed[0].message.lower()
        assert proposal_status(root, ids[0]) == "staged"

    def test_the_batch_never_elevates_its_own_scope(self, workspace, no_window, monkeypatch):
        """Granting itself admin would be an escalation, not a convenience."""
        from mind_mem.review_batch import run_batch

        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        root, ids = workspace
        run_batch(root, _approve_all(ids[:1]))
        assert os.environ["MIND_MEM_SCOPE"] == "user"


class TestGovernedRouting:
    def test_approve_routes_through_the_mcp_approve_apply_tool(self, workspace, no_window, monkeypatch):
        """``mm review`` must be a front end, never a second write path."""
        from mind_mem.review_batch import run_batch

        seen: list[tuple[str, bool]] = []
        import mind_mem.mcp.tools.governance as gov

        real = gov.approve_apply

        def spy(proposal_id: str, dry_run: bool = True) -> str:
            seen.append((proposal_id, dry_run))
            return real(proposal_id, dry_run=dry_run)

        monkeypatch.setattr(gov, "approve_apply", spy)
        root, ids = workspace
        run_batch(root, _approve_all(ids[:2]))
        assert [pid for pid, _dry in seen] == [ids[0], ids[1]]

    def test_reject_routes_through_the_mcp_reject_proposal_tool(self, workspace, no_window, monkeypatch):
        from mind_mem.review_batch import ReviewDecision, run_batch

        seen: list[str] = []
        import mind_mem.mcp.tools.governance as gov

        real = gov.reject_proposal

        def spy(proposal_id: str, reason: str) -> str:
            seen.append(proposal_id)
            return real(proposal_id, reason)

        monkeypatch.setattr(gov, "reject_proposal", spy)
        root, ids = workspace
        run_batch(root, (ReviewDecision(ids[0], "reject", origin="keypress", reason="superseded upstream"),))
        assert seen == [ids[0]]
