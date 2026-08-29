# Copyright 2026 STARGA, Inc.
"""``resume_brief`` — what session N+1 needs before it does anything.

A coding agent that restarts has to rebuild four things: what the task is,
what has already been tried, what is currently believed, and what remains.
Today it rebuilds them by re-reading files and re-running probes, which is
slow, lossy and — worst of all — silently re-derives conclusions that were
already reached and already recorded.

This module answers those four questions from the active
:class:`~mind_mem.task_frames.TaskFrame`, and attaches the dead ends that
overlap it so the resumed session does not walk back into a known failure.

    >>> brief = resume_brief(workspace)
    >>> brief.goal
    'Close the last two AGI3 floors without a net regression.'
    >>> [w.dead_end.block_id for w in brief.dead_ends]
    ['DE-20260826-001']

Contract
--------
* **Deterministic.**  Every input is a declared block field; every ordering
  is a total order over block IDs.  No clock is read, no model is called,
  no score is computed — two processes on two machines render the same
  bytes.
* **Warnings, not vetoes.**  ``dead_ends`` is evidence attached to the
  brief.  The plan is returned intact next to them, and nothing here
  refuses, filters or exits non-zero.  See
  :func:`mind_mem.dead_ends.match_dead_ends`.
* **Read-only.**  Assembling a brief never writes.  Frames and dead ends
  are authored on the governed proposal route and only ever read back.
* **Degrades to empty.**  A workspace with no frames yields an empty brief
  rather than an error, so a caller can ask unconditionally at session
  start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .dead_ends import DeadEndWarning, load_dead_ends_with_rejections, match_dead_ends
from .frame_fields import FrameSpecError, PlanStep, RejectedBlock
from .task_frames import FramePolicy, TaskFrame, active_frame, load_task_frames_with_rejections

__all__ = [
    "ResumeBrief",
    "render_resume_brief",
    "resume_brief",
]


@dataclass(frozen=True)
class ResumeBrief:
    """The four resume questions, plus the dead ends that overlap them."""

    frame_id: str = ""
    goal: str = ""
    status: str = ""
    steps: tuple[PlanStep, ...] = ()
    tried: tuple[str, ...] = ()
    believed: tuple[str, ...] = ()
    remaining: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    dead_ends: tuple[DeadEndWarning, ...] = ()
    source_file: str = ""
    #: Blocks the corpus declared that could not be read, and why.  An
    #: empty brief with a non-empty ``rejected`` is a *different fact*
    #: from an empty brief with none, and callers must render both.
    rejected: tuple[RejectedBlock, ...] = ()
    #: How many dead ends actually overlapped the frame, before
    #: ``max_warnings`` or the kill switch trimmed ``dead_ends``.
    dead_end_total: int = 0

    @property
    def dead_ends_elided(self) -> int:
        """Warnings that fired but are not in ``dead_ends``. Never negative."""
        return max(0, self.dead_end_total - len(self.dead_ends))

    def is_empty(self) -> bool:
        return not self.frame_id

    def is_silent(self) -> bool:
        """True when there is nothing at all to tell the caller."""
        return self.is_empty() and not self.rejected

    def to_dict(self) -> dict[str, Any]:
        """Deterministic, JSON-safe view.  Key order is fixed by this literal."""
        return {
            "frame_id": self.frame_id,
            "goal": self.goal,
            "status": self.status,
            "steps": [{"status": step.status, "text": step.text} for step in self.steps],
            "tried": list(self.tried),
            "believed": list(self.believed),
            "remaining": list(self.remaining),
            "blockers": list(self.blockers),
            "citations": list(self.citations),
            "dead_ends": [warning.to_dict() for warning in self.dead_ends],
            "dead_end_total": self.dead_end_total,
            "dead_ends_elided": self.dead_ends_elided,
            "rejected": [block.to_dict() for block in self.rejected],
            "source_file": self.source_file,
        }


def _select(frames: Sequence[TaskFrame], frame_id: str) -> TaskFrame | None:
    """Resolve *frame_id* against *frames*, or pick the active frame.

    Raises:
        FrameSpecError: an explicit *frame_id* names no live frame.  Silent
            fallback to a different frame would resume the wrong task.
    """
    if not frame_id:
        return active_frame(frames)
    for frame in frames:
        if frame.block_id == frame_id:
            return frame
    raise FrameSpecError(f"no live task frame with id {frame_id!r}")


def _warnings(
    workspace: str,
    frame: TaskFrame | None,
    policy: FramePolicy,
) -> tuple[tuple[DeadEndWarning, ...], int, tuple[RejectedBlock, ...]]:
    """Match the registry against *frame*, bounded by *policy*.

    Returns ``(shown, total_fired, rejected)``.  ``total_fired`` is the
    count **before** the bound, and it is computed even when the kill
    switch is off: silencing a channel is a policy choice, pretending
    the channel had nothing to say is a lie about the corpus.  The
    registry is read even with no frame, so a refused dead end is still
    reported to a session that has no continuity to resume.
    """
    registry, rejected = load_dead_ends_with_rejections(workspace, policy)
    if frame is None:
        return (), 0, rejected
    matched = match_dead_ends(frame, registry)
    if not policy.enabled or policy.max_warnings <= 0:
        return (), len(matched), rejected
    return matched[: policy.max_warnings], len(matched), rejected


def resume_brief(
    workspace: str,
    frame_id: str = "",
    *,
    policy: FramePolicy | None = None,
    config: Mapping[str, Any] | None = None,
) -> ResumeBrief:
    """Build the resume brief for one task frame.

    Args:
        workspace: Workspace root.
        frame_id: A specific ``TF-...`` id.  Empty selects the active
            frame — the live frame with the highest block ID.
        policy: Bounds + kill switch.  Defaults to
            :meth:`FramePolicy.from_config` over *config*, or the built-in
            defaults when neither is supplied.
        config: Workspace config mapping, read for ``recall.frames``.

    Returns:
        A :class:`ResumeBrief`.  Empty (``frame_id == ""``) when the
        workspace declares no live frame — never an error, so a session
        can ask unconditionally.  An empty brief still carries
        ``rejected``, so "there is no frame" and "there is a frame I
        could not read" never render the same.

    Raises:
        FrameSpecError: *frame_id* was given and names no live frame.
    """
    resolved = policy or FramePolicy.from_config(config)
    frames, refused_frames = load_task_frames_with_rejections(workspace, resolved)
    frame = _select(frames, frame_id)
    dead_ends, total, refused_dead_ends = _warnings(workspace, frame, resolved)
    rejected = tuple(sorted((*refused_frames, *refused_dead_ends), key=RejectedBlock.sort_key))
    if frame is None:
        return ResumeBrief(dead_end_total=total, rejected=rejected)
    return ResumeBrief(
        frame_id=frame.block_id,
        goal=frame.goal,
        status=frame.status,
        steps=frame.steps,
        tried=frame.tried,
        believed=frame.believed,
        remaining=frame.remaining,
        blockers=frame.blockers,
        citations=frame.citations,
        dead_ends=dead_ends,
        dead_end_total=total,
        rejected=rejected,
        source_file=frame.source_file,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_SECTIONS: tuple[tuple[str, str], ...] = (
    ("TRIED", "tried"),
    ("BELIEVED", "believed"),
    ("REMAINING", "remaining"),
    ("BLOCKERS", "blockers"),
    ("REFERENCES", "citations"),
)


def _bullets(title: str, entries: Sequence[str]) -> list[str]:
    if not entries:
        return []
    return [f"{title}:", *(f"  - {entry}" for entry in entries), ""]


def _dead_end_lines(brief: ResumeBrief) -> list[str]:
    """Render the warning channel.  Loud, but explicitly advisory.

    An elision is printed even when nothing is shown: a bound that
    silently drops negative memory is worse than the bound not existing,
    because the reader takes the short list for the whole list.
    """
    warnings = brief.dead_ends
    elided = brief.dead_ends_elided
    if not warnings:
        return [f"DEAD ENDS: {elided} fired but were not shown (raise recall.frames.max_warnings).", ""] if elided else []
    lines = [f"DEAD ENDS ({len(warnings)} of {brief.dead_end_total}) — evidence, not a prohibition:"]
    for warning in warnings:
        dead_end = warning.dead_end
        lines.append(f"  [{dead_end.block_id}] ({dead_end.outcome}) {dead_end.approach}")
        lines.append(f"      why: {dead_end.why_failed}")
        if dead_end.evidence:
            lines.append(f"      evidence: {', '.join(dead_end.evidence)}")
        lines.append(f"      matched: {', '.join(warning.matched)}")
    if elided:
        lines.append(f"  ... {elided} more fired and were not shown (raise recall.frames.max_warnings).")
    lines.append("")
    return lines


def _rejected_lines(rejected: Sequence[RejectedBlock]) -> list[str]:
    """Name every block the corpus declared and this brief could not read."""
    if not rejected:
        return []
    lines = [f"REJECTED BLOCKS ({len(rejected)}) — declared but not loaded:"]
    for block in rejected:
        lines.append(f"  [{block.block_id}] in {block.source_file}")
        lines.append(f"      {block.reason}")
    lines.append("")
    return lines


def render_resume_brief(brief: ResumeBrief) -> str:
    """Render *brief* as plain text for a terminal or a context injection.

    Pure formatting: same brief in, same string out, no clock and no
    locale-dependent formatting.
    """
    if brief.is_silent():
        return "No active task frame.\n"
    if brief.is_empty():
        header = ["No active task frame — but this workspace declared blocks that did not load.", ""]
        return "\n".join([*header, *_rejected_lines(brief.rejected), *_dead_end_lines(brief)])
    lines = [f"TASK FRAME {brief.frame_id}", f"GOAL: {brief.goal}", ""]
    if brief.steps:
        lines.append("STEPS:")
        lines.extend(f"  [{step.status}] {step.text}" for step in brief.steps)
        lines.append("")
    for title, attribute in _SECTIONS:
        lines.extend(_bullets(title, getattr(brief, attribute)))
    lines.extend(_dead_end_lines(brief))
    lines.extend(_rejected_lines(brief.rejected))
    return "\n".join(lines)
