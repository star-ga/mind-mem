# Copyright 2026 STARGA, Inc.
"""Adversarial pass over ``mm review``: the contracts it claims out loud.

Three of these were written after driving the shipped surface against a
real workspace and watching it do the wrong thing, so each one names the
failure it reproduces rather than the code it covers:

* ``preview_diff`` documents "never raises". It contained ``OSError`` and
  ``ValueError`` only, and ``GovernanceBypassError`` — which the gate
  raises on spec drift and on an admission that will not resolve in the
  hash chain — subclasses neither. One such proposal ended the whole
  interactive session and discarded every decision staged before it.
* The governance gates that cap throughput were computed on every run and
  rendered on exactly one path. An operator in ``-i`` pressed thirty keys
  and only then learned that twenty-nine applies were rate-limited.
* The published ``proposals/minute`` measured the applies alone. Deciding
  is most of an operator's session; a throughput number that excludes it
  flatters the surface that publishes it.
"""

from __future__ import annotations

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _review_fixtures import DECISION_FILE, build_workspace, mcp_budget  # noqa: E402,F401


@pytest.fixture
def workspace(tmp_path):
    root = str(tmp_path / "ws")
    os.makedirs(root)
    ids = build_workspace(root, 3)
    return root, ids


@pytest.fixture
def cli_workspace(workspace, monkeypatch):
    """The same workspace, addressed the way the CLI addresses it."""
    root, ids = workspace
    monkeypatch.setenv("MIND_MEM_WORKSPACE", root)
    return root, ids


def _raise_bypass(*_args, **_kwargs):
    from mind_mem.admission import GovernanceBypassError

    raise GovernanceBypassError("spec-hash drifted")


class TestAPreviewFailureCannotEndTheReview:
    """``preview_diff`` says "never raises". Hold it to that."""

    def test_a_governance_error_is_contained_and_reported(self, workspace, monkeypatch):
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        monkeypatch.setattr("mind_mem.apply_engine.execute_op", _raise_bypass)
        result = preview_diff(root, load_queue(root)[0])
        assert not result.available
        assert "GovernanceBypassError" in result.reason
        assert "spec-hash drifted" in result.reason

    def test_an_arbitrary_error_is_contained_too(self, workspace, monkeypatch):
        """The containment is by kind of failure, not by a list of types."""
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace

        def explode(*_args, **_kwargs):
            raise RuntimeError("op executor blew up")

        monkeypatch.setattr("mind_mem.apply_engine.execute_op", explode)
        result = preview_diff(root, load_queue(root)[0])
        assert not result.available
        assert "RuntimeError" in result.reason

    def test_the_sandbox_is_still_removed_when_the_replay_explodes(self, workspace, monkeypatch):
        import tempfile

        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        before = set(os.listdir(tempfile.gettempdir()))
        monkeypatch.setattr("mind_mem.apply_engine.execute_op", _raise_bypass)
        preview_diff(root, load_queue(root)[0])
        appeared = set(os.listdir(tempfile.gettempdir())) - before
        assert not [name for name in appeared if name.startswith("mind-mem-review-")]

    def test_one_broken_preview_does_not_discard_the_other_decisions(self, workspace, monkeypatch):
        """The failure that made this real: a raise mid-session lost everything."""
        from mind_mem.review_queue import load_queue
        from mind_mem.review_session import review_session

        root, ids = workspace
        monkeypatch.setattr("mind_mem.apply_engine.execute_op", _raise_bypass)
        out = io.StringIO()
        decisions = review_session(root, load_queue(root), keys=iter("aaac"), out=out)
        assert [d.proposal_id for d in decisions] == list(ids)


class TestTheGatesAreNamedBeforeTheOperatorWorks:
    """A blocker discovered after thirty keystrokes is a blocker reported late."""

    def test_the_interactive_session_names_the_blockers_up_front(self, cli_workspace, monkeypatch, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        monkeypatch.setattr(sys, "stdin", io.StringIO("q\n"))
        cmd_review(build_parser().parse_args(["review", "-i"]))
        out = capsys.readouterr().out
        assert "BLOCKERS" in out
        assert "MIND_MEM_SCOPE=admin" in out

    def test_a_flag_batch_names_the_blockers_before_it_runs(self, cli_workspace, monkeypatch, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        root, ids = cli_workspace
        cmd_review(build_parser().parse_args(["review", "--approve", ids[0]]))
        out = capsys.readouterr().out
        assert "BLOCKERS" in out

    def test_the_warning_precedes_the_first_proposal(self, cli_workspace, monkeypatch, capsys):
        """Order is the whole point: after the decisions it is an epitaph."""
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        monkeypatch.setattr(sys, "stdin", io.StringIO("q\n"))
        root, ids = cli_workspace
        cmd_review(build_parser().parse_args(["review", "-i"]))
        out = capsys.readouterr().out
        assert out.index("BLOCKERS") < out.index(ids[0])

    def test_a_clean_queue_prints_no_health_banner_at_all(self, cli_workspace, monkeypatch, capsys):
        """No gate, no banner — asserted on the whole panel, not the word.

        Asserting only that ``BLOCKERS`` is absent passes even when the
        banner is printed unconditionally, because the word appears only
        inside the blocker list. The panel's own header is the honest
        witness that nothing was drawn.
        """
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
        monkeypatch.setattr(
            "mind_mem.apply_engine.check_no_touch_window",
            lambda ws: (True, "test: window neutralised"),
        )
        monkeypatch.setattr(sys, "stdin", io.StringIO("q\n"))
        cmd_review(build_parser().parse_args(["review", "-i"]))
        out = capsys.readouterr().out
        assert "BLOCKERS" not in out
        assert "governance_mode:" not in out


class TestPublishedThroughputCoversTheOperatorSession:
    """``proposals/minute`` over the applies alone is a vanity number."""

    def test_the_rate_uses_the_session_span_when_one_is_given(self, workspace):
        import time

        from mind_mem.review_batch import ReviewDecision, run_batch

        root, ids = workspace
        decisions = [ReviewDecision(ids[0], "approve", origin="cli-flag")]
        report = run_batch(
            root,
            decisions,
            approve_hook=lambda *a, **k: (True, "ok"),
            session_started=time.perf_counter() - 60.0,
        )
        assert report.metrics is not None
        assert report.metrics.proposals_per_minute == pytest.approx(1.0, rel=0.05)

    def test_the_apply_span_is_still_reported_separately(self, workspace):
        import time

        from mind_mem.review_batch import ReviewDecision, run_batch

        root, ids = workspace
        report = run_batch(
            root,
            [ReviewDecision(ids[0], "approve", origin="cli-flag")],
            approve_hook=lambda *a, **k: (True, "ok"),
            session_started=time.perf_counter() - 60.0,
        )
        assert report.session_seconds == pytest.approx(60.0, abs=1.0)
        assert 0.0 <= report.elapsed_seconds < 60.0

    def test_without_a_session_span_the_rate_is_unchanged(self, workspace):
        """Default-off: the shipped behaviour is byte-for-byte what it was."""
        from mind_mem.review_batch import ReviewDecision, run_batch

        root, ids = workspace
        report = run_batch(
            root,
            [ReviewDecision(ids[0], "approve", origin="cli-flag")],
            approve_hook=lambda *a, **k: (True, "ok"),
        )
        assert report.metrics is not None
        assert report.session_seconds == pytest.approx(report.elapsed_seconds)

    def test_the_rendered_report_says_which_span_the_rate_covers(self):
        from mind_mem.review_batch import BatchOutcome, BatchReport
        from mind_mem.review_metrics import ApprovalEvent, summarise
        from mind_mem.review_render import render_report

        outcomes = (BatchOutcome("P-20260829-001", "approve", True, "Applied", 3600.0),)
        events = [ApprovalEvent("P-20260829-001", "approve", True, 3600.0, 0.0)]
        report = BatchReport(
            outcomes=outcomes,
            metrics=summarise(events, elapsed_seconds=30.0),
            elapsed_seconds=2.0,
            session_seconds=30.0,
        )
        text = render_report(report)
        assert "over 30.0s" in text
        assert "2.0s applying" in text

    def test_the_cli_measures_the_whole_invocation(self, cli_workspace, monkeypatch, capsys):
        from mind_mem.mm_cli import build_parser
        from mind_mem.review_cli import cmd_review

        monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
        monkeypatch.setattr(
            "mind_mem.apply_engine.check_no_touch_window",
            lambda ws: (True, "test: window neutralised"),
        )
        root, ids = cli_workspace
        cmd_review(build_parser().parse_args(["review", "--approve", ids[0], "--json"]))
        payload = json.loads(capsys.readouterr().out)
        assert payload["session_seconds"] >= payload["apply_seconds"]


class TestATouchedFileMayNotEscapeTheWorkspace:
    """``FilesTouched`` is proposal-supplied; the preview reads what it names."""

    def test_a_posix_traversal_is_dropped(self):
        from mind_mem.review_preview import _targets
        from mind_mem.review_queue import ReviewItem

        item = ReviewItem(
            proposal_id="P-20260829-001",
            source_file="intelligence/proposed/EDITS_PROPOSED.md",
            proposal_type="edit",
            target_block="D-20260801-001",
            risk="low",
            status="staged",
            created="",
            rollback="",
            fingerprint="",
            files_touched=("../../etc/passwd", DECISION_FILE),
        )
        assert _targets(item) == (DECISION_FILE,)

    def test_a_backslash_traversal_is_dropped_too(self):
        """``"..\\\\..\\\\etc"`` has no ``/`` to split on; the old check let it through."""
        from mind_mem.review_preview import _targets
        from mind_mem.review_queue import ReviewItem

        item = ReviewItem(
            proposal_id="P-20260829-001",
            source_file="intelligence/proposed/EDITS_PROPOSED.md",
            proposal_type="edit",
            target_block="D-20260801-001",
            risk="low",
            status="staged",
            created="",
            rollback="",
            fingerprint="",
            files_touched=("..\\..\\etc\\passwd", DECISION_FILE),
        )
        assert _targets(item) == (DECISION_FILE,)

    def test_an_ordinary_relative_path_still_survives(self):
        from mind_mem.review_preview import _targets
        from mind_mem.review_queue import ReviewItem

        item = ReviewItem(
            proposal_id="P-20260829-001",
            source_file="intelligence/proposed/EDITS_PROPOSED.md",
            proposal_type="edit",
            target_block="D-20260801-001",
            risk="low",
            status="staged",
            created="",
            rollback="",
            fingerprint="",
            files_touched=("./decisions/DECISIONS.md",),
            ops=({"op": "update_field", "file": "tasks/TASKS.md"},),
        )
        assert _targets(item) == ("./decisions/DECISIONS.md", "tasks/TASKS.md")
