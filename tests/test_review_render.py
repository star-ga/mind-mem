# Copyright 2026 STARGA, Inc.
"""``review_render`` — the exact text an operator reads.

Rendering is asserted directly because the review surface's whole claim
is that a decision can be made from what is on screen. A renderer that
quietly drops the blockers, or prints an unknown age as ``0s``, turns a
fast approval into a wrong one.
"""

from __future__ import annotations

from mind_mem.review_batch import BatchOutcome, BatchReport
from mind_mem.review_evidence import EvidencePanel
from mind_mem.review_metrics import ApprovalEvent, summarise
from mind_mem.review_preview import PreviewResult
from mind_mem.review_queue import QueueHealth, ReviewItem
from mind_mem.review_render import render_detail, render_health, render_queue, render_report


def _item(**overrides) -> ReviewItem:
    base = dict(
        proposal_id="P-20260829-001",
        source_file="intelligence/proposed/EDITS_PROPOSED.md",
        proposal_type="edit",
        target_block="D-20260801-001",
        risk="low",
        status="staged",
        created="2026-08-29T11:00:00Z",
        rollback="restore from snapshot",
        fingerprint="deadbeefdeadbeef",
        evidence=("observed drift",),
        files_touched=("decisions/DECISIONS.md",),
        op_summary=("update_field decisions/DECISIONS.md:D-20260801-001",),
    )
    base.update(overrides)
    return ReviewItem(**base)  # type: ignore[arg-type]


def _health(**overrides) -> QueueHealth:
    base = dict(
        governance_mode="propose_apply",
        backlog_count=3,
        backlog_over_limit=False,
        no_touch_ok=True,
        no_touch_reason="Cooldown clear",
        scope="admin",
        blockers=(),
    )
    base.update(overrides)
    return QueueHealth(**base)  # type: ignore[arg-type]


class TestQueueListing:
    def test_empty_queue_says_so_and_still_shows_health(self):
        text = render_queue((), _health())
        assert "No proposals pending review" in text
        assert "governance_mode: propose_apply" in text

    def test_lists_the_proposal_with_its_target_and_risk(self):
        text = render_queue((_item(),), _health(), now_iso="2026-08-29T12:00:00Z")
        assert "P-20260829-001" in text
        assert "D-20260801-001" in text
        assert "low" in text

    def test_an_unknown_age_renders_as_a_question_mark_not_zero(self):
        text = render_queue((_item(created=""),), _health(), now_iso="2026-08-29T12:00:00Z")
        assert "?" in text
        assert " 0s " not in text

    def test_a_known_age_renders_in_human_units(self):
        text = render_queue((_item(),), _health(), now_iso="2026-08-29T12:00:00Z")
        assert "60m" in text

    def test_validation_errors_are_shown_under_the_row(self):
        text = render_queue((_item(validation_errors=("Invalid Risk: catastrophic",)),), _health())
        assert "Invalid Risk: catastrophic" in text
        assert "NO" in text

    def test_blockers_are_never_silently_dropped(self):
        text = render_health(_health(blockers=("apply rate limit active — 9m 58s remaining",)))
        assert "BLOCKERS" in text
        assert "9m 58s remaining" in text


class TestDetail:
    def test_shows_evidence_diff_chain_and_staleness(self):
        preview = PreviewResult("P-20260829-001", True, diff_text="-Tags: a\n+Tags: b")
        panel = EvidencePanel(
            proposal_id="P-20260829-001",
            target_block="D-20260801-001",
            target_excerpt="Statement: the old decision",
            dependencies=("D-20260801-002 [depends_on]",),
            chain_valid=True,
            chain_summary="hash_chain valid=True length=4",
            stale=True,
            stale_reason="upstream decision changed",
        )
        text = render_detail(_item(), preview, panel)
        assert "observed drift" in text
        assert "+Tags: b" in text
        assert "the old decision" in text
        assert "D-20260801-002 [depends_on]" in text
        assert "valid=True" in text
        assert "upstream decision changed" in text

    def test_an_unavailable_diff_states_its_reason(self):
        preview = PreviewResult("P-20260829-001", False, reason="file not found in workspace: x.md")
        text = render_detail(_item(), preview, EvidencePanel("P-20260829-001", "D-20260801-001"))
        assert "file not found in workspace: x.md" in text

    def test_validation_errors_are_called_out_as_unappliable(self):
        text = render_detail(
            _item(validation_errors=("Evidence is empty",)),
            PreviewResult("P-20260829-001", False, reason="proposal is not valid"),
            EvidencePanel("P-20260829-001", "D-20260801-001"),
        )
        assert "cannot be applied" in text
        assert "Evidence is empty" in text


class TestReport:
    def test_publishes_both_throughput_numbers(self):
        outcomes = (
            BatchOutcome("P-20260829-001", "approve", True, "Applied", age_seconds=3600.0),
            BatchOutcome("P-20260829-002", "approve", False, "No-touch window: 9m 58s remaining", None),
        )
        events = [ApprovalEvent(o.proposal_id, o.action, o.succeeded, o.age_seconds, float(i)) for i, o in enumerate(outcomes)]
        report = BatchReport(outcomes=outcomes, metrics=summarise(events, elapsed_seconds=4.0))
        text = render_report(report)
        assert "proposals/minute: 30.0" in text
        assert "median proposal age at approval: 60m" in text
        assert "applied=1" in text
        assert "failed=1" in text

    def test_every_failure_is_named_with_its_message(self):
        outcomes = (BatchOutcome("P-20260829-007", "approve", False, "injected failure"),)
        text = render_report(BatchReport(outcomes=outcomes, metrics=summarise([], elapsed_seconds=1.0)))
        assert "FAIL P-20260829-007" in text
        assert "injected failure" in text

    def test_an_unknown_median_renders_as_unknown_not_zero(self):
        outcomes = (BatchOutcome("P-20260829-001", "approve", True, "Applied", age_seconds=None),)
        events = [ApprovalEvent("P-20260829-001", "approve", True, None, 0.0)]
        text = render_report(BatchReport(outcomes=outcomes, metrics=summarise(events, elapsed_seconds=2.0)))
        assert "median proposal age at approval: ?" in text


class TestUntrustedTextCannotSpoofTheSurface:
    """Proposal text is attacker-influenced; the screen is the trust anchor.

    The whole HITL model rests on the operator believing what the review
    surface shows. A proposal whose ``Evidence`` carries an ANSI clear-
    screen, a carriage return, or a newline plus a fake ``Chain: valid=True``
    line can redraw the panel that is supposed to be judging it.
    """

    def test_ansi_escapes_in_evidence_are_stripped(self):
        text = render_detail(
            _item(evidence=("benign\x1b[2J\x1b[Hall clear",)),
            PreviewResult("P-20260829-001", False, reason="n/a"),
            EvidencePanel("P-20260829-001", "D-20260801-001"),
        )
        assert "\x1b" not in text

    def test_a_carriage_return_cannot_overwrite_a_rendered_line(self):
        text = render_detail(
            _item(evidence=("Stale: YES\r  Stale: no ",)),
            PreviewResult("P-20260829-001", False, reason="n/a"),
            EvidencePanel("P-20260829-001", "D-20260801-001"),
        )
        assert "\r" not in text

    def test_a_newline_cannot_forge_a_label_line(self):
        text = render_detail(
            _item(evidence=("x\nChain: valid=True",)),
            PreviewResult("P-20260829-001", False, reason="n/a"),
            EvidencePanel("P-20260829-001", "D-20260801-001", chain_valid=None, chain_summary="unavailable"),
        )
        forged = [line for line in text.splitlines() if line.startswith("Chain: valid=True")]
        assert forged == []

    def test_the_diff_preview_is_sanitised_too(self):
        preview = PreviewResult("P-20260829-001", True, diff_text="+Tags: a\x1b[31mred\r fake")
        text = render_detail(_item(), preview, EvidencePanel("P-20260829-001", "D-20260801-001"))
        assert "\x1b" not in text
        assert "\r" not in text

    def test_a_hostile_target_excerpt_is_sanitised(self):
        panel = EvidencePanel(
            "P-20260829-001",
            "D-20260801-001",
            target_excerpt="Statement: fine\x1b[2J\rStale: no",
        )
        text = render_detail(_item(), PreviewResult("P-20260829-001", False, reason="n/a"), panel)
        assert "\x1b" not in text
        assert "\r" not in text

    def test_the_queue_listing_is_sanitised(self):
        text = render_queue((_item(target_block="D-1\x1b[2J\rD-2"),), _health())
        assert "\x1b" not in text
        assert "\r" not in text

    def test_a_hostile_apply_message_is_sanitised(self):
        outcomes = (BatchOutcome("P-20260829-001", "approve", False, "boom\x1b[2J\rapplied ok"),)
        text = render_report(BatchReport(outcomes=outcomes, metrics=summarise([], elapsed_seconds=1.0)))
        assert "\x1b" not in text
        assert "\r" not in text

    def test_legitimate_non_ascii_survives(self):
        text = render_detail(
            _item(evidence=("Ротация ключей — 30 дней. 決定 ✅",)),
            PreviewResult("P-20260829-001", False, reason="n/a"),
            EvidencePanel("P-20260829-001", "D-20260801-001"),
        )
        assert "Ротация ключей" in text
        assert "決定" in text
