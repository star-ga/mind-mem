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
import shutil
import stat
import sys
import tempfile

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


class TestThePreviewSandboxIsReallyTornDown:
    """A sandbox is a workspace that lives for one call. Both halves of
    destroying it — the cached gate and the directory — were leaks.

    ``get_gate`` caches one gate per workspace realpath forever, so every
    preview left a gate (holding a chain and a loaded evidence log)
    keyed on a temp path already deleted from disk. Measured before the
    fix: two previews, two cache entries, zero of them still on disk.
    """

    def test_a_preview_leaves_no_gate_behind_for_its_sandbox(self, workspace):
        from mind_mem import governance_gate
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        before = len(governance_gate._gates)
        for _ in range(3):
            preview_diff(root, load_queue(root)[0])

        sandbox_keys = [key for key in governance_gate._gates if "mind-mem-review-" in key]
        assert sandbox_keys == [], f"the gate cache kept a sandbox that no longer exists: {sandbox_keys}"
        # The real workspace gate is legitimately cached; nothing else is.
        assert len(governance_gate._gates) - before <= 1

    def test_a_preview_caches_no_gate_for_a_directory_it_deleted(self, workspace):
        """Scoped to the keys THIS preview adds.

        The cache is process-global and every test that admits a write
        leaves a gate behind for its own ``tmp_path``, which pytest later
        reaps — so "no cached gate names a missing directory" is not a
        property of the whole dict, only of what a preview contributes.
        """
        from mind_mem import governance_gate
        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        before = set(governance_gate._gates)
        preview_diff(root, load_queue(root)[0])
        added = set(governance_gate._gates) - before

        dead = sorted(key for key in added if not os.path.isdir(key))
        assert dead == [], f"the preview cached a gate for a directory it then deleted: {dead}"


class TestARetiredGateRefusesRatherThanForking:
    """Why eviction needs a refusal and not just a ``dict.pop``.

    ``_admit_lock``, ``HashChainV2._lock`` and ``EvidenceChain._lock``
    are per-instance; the ``get_gate`` singleton is the only thing that
    makes them process-wide. Two live gates on one workspace each
    compute the next ``previous_hash`` from their own in-memory evidence
    tail. Measured: three appends across two chains on one file leave
    the JSONL loading as ZERO entries with ``load_integrity_compromised``
    — a fork destroys the whole history, not merely its tail. So a gate
    whose workspace was torn down must fail loudly instead of writing.
    """

    def test_an_evicted_gate_admits_nothing(self, tmp_path):
        from mind_mem.governance_gate import GovernanceBypassError, evict_gate, get_gate

        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        gate = get_gate(ws)
        gate.admit(action="WRITE", block_id="B-1", content="payload", actor="test")

        assert evict_gate(ws) is True
        with pytest.raises(GovernanceBypassError) as caught:
            gate.admit(action="WRITE", block_id="B-2", content="payload", actor="test")
        assert "retired" in str(caught.value)

    def test_evicting_an_uncached_workspace_is_not_an_error(self, tmp_path):
        from mind_mem.governance_gate import evict_gate

        assert evict_gate(str(tmp_path / "never-gated")) is False

    def test_close_is_idempotent_and_a_fresh_gate_still_works(self, tmp_path):
        from mind_mem.governance_gate import evict_gate, get_gate

        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        first = get_gate(ws)
        assert evict_gate(ws) is True
        first.close()  # idempotent
        assert evict_gate(ws) is False

        second = get_gate(ws)
        assert second is not first
        entry = second.admit(action="WRITE", block_id="B-1", content="payload", actor="test")
        assert entry.block_id == "B-1"
        assert second.chain.verify_chain() == (True, -1)


class TestSandboxRemovalIsReportedNotAssumed:
    def test_a_sandbox_that_survives_is_reported(self, workspace, monkeypatch):
        """``sandbox_removed`` was hardcoded True, so it could never be False.

        A field that always claims success is the silent failure the
        module docstring criticises. Removal is forced to fail here by
        neutering rmtree; the sandbox is cleaned up afterwards.
        """
        import shutil as shutil_module

        from mind_mem.review_preview import preview_diff
        from mind_mem.review_queue import load_queue

        root, _ids = workspace
        monkeypatch.setattr("mind_mem.review_preview.shutil.rmtree", lambda *a, **k: None)
        result = preview_diff(root, load_queue(root)[0])
        assert result.sandbox_removed is False

        monkeypatch.undo()
        leftovers = [name for name in os.listdir(tempfile.gettempdir()) if name.startswith("mind-mem-review-")]
        for name in leftovers:
            shutil_module.rmtree(os.path.join(tempfile.gettempdir(), name), ignore_errors=True)

    def test_a_read_only_sandbox_is_still_removed(self, tmp_path):
        """Coverage for the read-only handler, which shipped without any.

        This is not a regression test for the connection fix — it passes
        with or without it. It exists because the handler's docstring
        claimed "a read-only-tree test caught" the parent-directory case
        while no such test was ever committed, and because removing the
        retry loop means ``_clear_readonly`` now has to succeed on the
        first pass. The Windows half — that the read-only ATTRIBUTE
        blocks delete there — is documented behaviour, not covered here.
        """
        from mind_mem.review_preview import _remove_sandbox

        sandbox = tmp_path / "mind-mem-review-fake"
        nested = sandbox / "memory"
        nested.mkdir(parents=True)
        target = nested / "locked.md"
        target.write_text("read only", encoding="utf-8")
        os.chmod(target, stat.S_IRUSR)
        os.chmod(nested, stat.S_IRUSR | stat.S_IXUSR)
        try:
            assert _remove_sandbox(str(sandbox)) is True
            assert not sandbox.exists()
        finally:
            if nested.exists():
                os.chmod(nested, stat.S_IRWXU)
            if sandbox.exists():
                shutil.rmtree(str(sandbox), ignore_errors=True)
