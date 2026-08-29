# Copyright 2026 STARGA, Inc.
"""Text rendering for ``mm review``.

Pure functions from data to strings — no I/O, no clock, no decisions.
Splitting rendering out is what lets the queue, the detail view and the
keyboard session share one layout, and lets tests assert on the exact
text an operator sees.

**Every untrusted value is sanitised on the way to the screen.** The
whole HITL model rests on the operator believing the panel in front of
them, and a proposal's fields are attacker-influenced: an ``Evidence``
line carrying an ANSI clear-screen, a carriage return, or a newline plus
a forged ``Chain: valid=True`` can redraw the surface that is judging it.
:func:`sanitize_codepoints` already runs at the *ingest* boundary, but it
is config-gated and does not cover blocks that predate it — so rendering
sanitises again, unconditionally. Defence in depth on the one surface
where being fooled is the whole attack.

Risk is *displayed* and never acted on. There is deliberately no
risk-based styling, ordering or filtering anywhere in this module: a
review front end that treats one risk level differently from another is
one refactor away from approving the low-risk ones by itself.
"""

from __future__ import annotations

from typing import Any, Sequence

from .codepoint_sanitize import sanitize_codepoints
from .review_batch import BatchReport
from .review_evidence import EvidencePanel
from .review_preview import PreviewResult
from .review_queue import QueueHealth, ReviewItem

__all__ = ["render_detail", "render_health", "render_queue", "render_report"]

RULE = "─" * 68


def render_queue(items: Sequence[ReviewItem], health: QueueHealth, *, now_iso: str = "") -> str:
    """The queue listing: one line per proposal, health underneath."""
    if not items:
        return "No proposals pending review.\n\n" + render_health(health)

    lines = [f"{len(items)} proposal(s) pending review", RULE]
    lines.append(f"{'PROPOSAL':<18} {'TYPE':<9} {'RISK':<7} {'AGE':>10}  {'OK':<3} TARGET")
    for item in items:
        age = item.age_seconds(now_iso=now_iso) if now_iso else None
        lines.append(
            f"{_line(item.proposal_id):<18} {_line(item.proposal_type):<9} {_line(item.risk):<7} "
            f"{_age_text(age):>10}  {'yes' if item.applicable else 'NO ':<3} {_line(item.target_block)}"
        )
        for error in item.validation_errors:
            lines.append(f"{'':<18} ! {_line(error)}")
    lines.append(RULE)
    lines.append(render_health(health))
    return "\n".join(lines)


def render_health(health: QueueHealth) -> str:
    """Governance state, and every gate standing between queue and apply."""
    lines = [
        f"governance_mode: {_line(health.governance_mode)}    scope: {_line(health.scope)}    backlog: {health.backlog_count}",
        f"apply rate limit: {_line(health.no_touch_reason)}",
    ]
    if health.blockers:
        lines.append("BLOCKERS — approvals will fail until these clear:")
        lines.extend(f"  * {_line(blocker)}" for blocker in health.blockers)
    return "\n".join(lines)


def render_detail(item: ReviewItem, preview: PreviewResult, panel: EvidencePanel) -> str:
    """One proposal in full: what it is, what it would do, what backs it."""
    lines = [
        RULE,
        f"{_line(item.proposal_id)}   type={_line(item.proposal_type)}   risk={_line(item.risk)}   target={_line(item.target_block)}",
        RULE,
        "Evidence:",
    ]
    lines.extend(f"  - {_line(entry)}" for entry in item.evidence or ("(none)",))
    lines.append(f"Rollback: {_line(item.rollback) or '(none)'}")
    lines.append("Ops:")
    lines.extend(f"  - {_line(entry)}" for entry in item.op_summary or ("(none)",))
    lines.extend(["", "Target block:"])
    lines.extend(f"  {row}" for row in _body(panel.target_excerpt or "(unavailable)"))
    lines.extend(["", "Provenance:"])
    lines.extend(f"  -> {_line(entry)}" for entry in panel.dependencies or ("(no dependency edges)",))
    lines.extend(f"  !! {_line(entry)}" for entry in panel.conflicts)
    lines.extend(_chain_and_staleness(panel))
    lines.extend(["", "Diff (pre-apply preview):"])
    if preview.available:
        lines.extend(f"  {row}" for row in _body(preview.diff_text))
    else:
        lines.append(f"  (unavailable: {_line(preview.reason)})")
    if item.validation_errors:
        lines.extend(["", "VALIDATION ERRORS — this proposal cannot be applied:"])
        lines.extend(f"  ! {_line(error)}" for error in item.validation_errors)
    return "\n".join(lines)


def _chain_and_staleness(panel: EvidencePanel) -> list[str]:
    """The two trust lines, plus any note explaining a missing one."""
    stale = f"YES — {_line(panel.stale_reason)}" if panel.stale else "no"
    lines = [
        "",
        f"Chain: valid={panel.chain_valid}  {_line(panel.chain_summary)}",
        f"Stale: {stale}",
    ]
    lines.extend(f"  note: {_line(note)}" for note in panel.notes)
    return lines


def render_report(report: BatchReport) -> str:
    """Per-proposal outcomes, then the published throughput metric."""
    lines = [RULE, "Batch result", RULE]
    for outcome in report.outcomes:
        mark = "ok  " if outcome.succeeded else "FAIL"
        lines.append(f"  {mark} {_line(outcome.proposal_id)}  {_line(outcome.action)}: {_line(outcome.message)}")
    lines.append(RULE)
    lines.append(f"applied={len(report.applied)}  rejected={len(report.rejected)}  failed={len(report.failed)}")
    metrics = report.metrics
    if metrics:
        lines.append(
            f"proposals/minute: {_num(metrics.proposals_per_minute)}    "
            f"applied/minute: {_num(metrics.applied_per_minute)}    "
            f"(over {report.session_seconds:.1f}s of operator session, {report.elapsed_seconds:.1f}s applying)"
        )
        lines.append(
            f"median proposal age at approval: {_age_text(metrics.median_age_at_approval_seconds)}"
            f"  (sample {metrics.aged_sample}/{metrics.applied}, coverage {metrics.age_coverage:.0%})"
        )
    return "\n".join(lines)


def _clean(text: Any) -> str:
    """Strip control/format codepoints and defang carriage returns.

    ``sanitize_codepoints`` removes ESC, Unicode tag characters, bidi
    controls and zero-width smuggling channels while preserving real
    non-ASCII text. It preserves ``\\r`` by design (line endings), which
    a renderer must not: a bare CR mid-line overwrites what was already
    drawn, so it is escaped rather than passed through.
    """
    return sanitize_codepoints(str(text)).replace("\r", "\\r")


def _line(text: Any) -> str:
    """One display line. Newlines are flattened so a field cannot forge a label."""
    return _clean(text).replace("\n", "\\n")


def _body(text: Any) -> list[str]:
    """A multi-line block: real newlines kept, everything else defanged."""
    return _clean(text).split("\n")


def _age_text(seconds: float | None) -> str:
    """Human age, or ``?`` when unknown. Never a fabricated zero."""
    if seconds is None:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _num(value: float | None) -> str:
    return "?" if value is None else f"{value:.1f}"
