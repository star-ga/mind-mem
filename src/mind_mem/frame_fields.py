# Copyright 2026 STARGA, Inc.
"""Field grammar for TASK-FRAME and DEAD-END blocks — one boundary, two kinds.

:mod:`mind_mem.task_frames` and :mod:`mind_mem.dead_ends` read different
blocks, but they read them the *same way*: bounded entries, prose kept whole,
handles comma-split, one closed step vocabulary.  That grammar lives here so
the two kinds cannot drift apart — the same role
:mod:`mind_mem.guardrail_patterns` plays for guardrail triggers.

Two field shapes, deliberately distinguished:

* **Prose** (``Tried`` / ``Believed`` / ``Remaining`` / ``Blockers`` /
  ``Approach`` / ``WhyFailed``) is **never comma-split** — *"flow bounds
  survive, additive bounds do not"* is one belief, not two.  Use a markdown
  list for multiple entries.
* **Handles** (``References`` / ``Evidence``) **are** comma-split: they are
  paths, ids and URLs, never sentences.

This module is a leaf: it knows about block dicts and nothing about frames,
dead ends, matching or recall.  No clock, no model, no I/O beyond the block
parser it delegates file reading to.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

from .block_parser import parse_file
from .guardrail_patterns import GuardrailSpecError
from .observability import get_logger

__all__ = [
    "MAX_PROSE_ENTRIES",
    "MAX_PROSE_LEN",
    "STEP_STATUSES",
    "FrameSpecError",
    "PlanStep",
    "RejectedBlock",
    "handle_entries",
    "load_blocks",
    "parse_steps",
    "prose_entries",
]

_log = get_logger("frame_fields")

#: Closed plan-step vocabulary, in lifecycle order.  A step line reads
#: ``"<status>: <text>"``; a bare line is a ``todo``.
STEP_STATUSES: tuple[str, ...] = ("todo", "doing", "done", "blocked")

#: Per-field cap on prose entries (bounds the work one block can impose).
MAX_PROSE_ENTRIES = 64

#: Per-entry length cap, so one runaway field cannot dominate a brief.
MAX_PROSE_LEN = 1024

#: Steps in these states still count as remaining work.
_REMAINING_STEP_STATUSES = frozenset({"todo", "doing", "blocked"})


class FrameSpecError(GuardrailSpecError):
    """Raised when a frame / dead-end field cannot be read.

    A subclass of :class:`~mind_mem.guardrail_patterns.GuardrailSpecError`
    so every caller that already fails closed on a malformed declaration
    keeps doing so for these block kinds too.
    """


@dataclass(frozen=True)
class PlanStep:
    """One plan step: a closed-vocabulary status and its text."""

    status: str
    text: str

    def is_remaining(self) -> bool:
        return self.status in _REMAINING_STEP_STATUSES


@dataclass(frozen=True)
class RejectedBlock:
    """A ``[TF-...]`` / ``[DE-...]`` block that could not be read, and why.

    A refused block is *evidence about the corpus*, not an absence of
    one.  Skipping it with only a log line lets a caller reading stdout
    conclude the workspace declares no frame at all — and then re-derive
    the context the frame existed to carry.  So every rejection is
    carried back to the caller alongside the blocks that did load.

    Deterministic like everything else here: the fields are read off the
    block and the parse error, and callers order these by block ID.
    """

    block_id: str
    source_file: str
    reason: str

    def sort_key(self) -> str:
        return self.block_id

    def to_dict(self) -> dict[str, str]:
        return {"block_id": self.block_id, "source_file": self.source_file, "reason": self.reason}


def prose_entries(raw: Any, *, field_name: str, block_id: str) -> tuple[str, ...]:
    """Read a prose field into a bounded tuple of entries.

    A markdown list yields one entry per item; a scalar yields one entry per
    line, which is how the block parser represents an indented continuation.
    Whitespace runs are collapsed so a wrapped line reads as one sentence.

    Raises:
        FrameSpecError: the field is neither a string nor a list, an entry
            exceeds :data:`MAX_PROSE_LEN`, or the field declares more than
            :data:`MAX_PROSE_ENTRIES` entries.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        items: list[str] = raw.split("\n")
    elif isinstance(raw, (list, tuple)):
        items = [str(entry) for entry in raw]
    else:
        raise FrameSpecError(f"{block_id}: {field_name} must be a string or list, got {type(raw).__name__}")

    out: list[str] = []
    for item in items:
        text = " ".join(item.split())
        if not text:
            continue
        if len(text) > MAX_PROSE_LEN:
            raise FrameSpecError(f"{block_id}: {field_name} entry exceeds {MAX_PROSE_LEN} chars")
        out.append(text)
        if len(out) > MAX_PROSE_ENTRIES:
            raise FrameSpecError(f"{block_id}: {field_name} declares more than {MAX_PROSE_ENTRIES} entries")
    return tuple(out)


def handle_entries(raw: Any, *, field_name: str, block_id: str) -> tuple[str, ...]:
    """Read a handle field (``References`` / ``Evidence``) into a tuple."""
    if isinstance(raw, str):
        raw = list(raw.split(","))
    return prose_entries(raw, field_name=field_name, block_id=block_id)


def parse_steps(raw: Any, *, block_id: str) -> tuple[PlanStep, ...]:
    """Read the ``Steps`` field into a bounded tuple of plan steps.

    Raises:
        FrameSpecError: a step declares a status outside
            :data:`STEP_STATUSES`.  The vocabulary is closed so a brief can
            state what remains without interpreting free text — which means
            a bare word before a colon that is *not* a status
            (``"fix: ..."``) is refused rather than silently read as step
            text.  Loud beats quietly-misfiled: write such a step without
            the colon.
    """
    steps: list[PlanStep] = []
    for entry in prose_entries(raw, field_name="Steps", block_id=block_id):
        status, sep, text = entry.partition(":")
        candidate = status.strip().casefold()
        if sep and candidate in STEP_STATUSES:
            steps.append(PlanStep(status=candidate, text=text.strip()))
            continue
        if sep and _looks_like_a_status(status):
            raise FrameSpecError(f"{block_id}: step status {status.strip()!r} is not one of {STEP_STATUSES}")
        steps.append(PlanStep(status="todo", text=entry))
    return tuple(steps)


def _looks_like_a_status(candidate: str) -> bool:
    """A single bare word before the colon was meant to be a status."""
    text = candidate.strip()
    return bool(text) and " " not in text and text.isalpha()


def load_blocks(workspace: str, sources: Sequence[str], prefix: str) -> tuple[dict[str, Any], ...]:
    """Yield every block under *workspace* whose ID starts with *prefix*.

    Shared by frames and dead ends so both obey one containment rule: a
    source that escapes the workspace root is refused, an unreadable source
    is skipped with a warning, and one bad file never takes the rest down.
    Order follows *sources*, then file order — deterministic, no clock.
    """
    workspace_real = os.path.realpath(workspace)
    root = workspace_real + os.sep
    out: list[dict[str, Any]] = []
    for rel_path in sources:
        candidate = os.path.realpath(os.path.join(workspace_real, rel_path))
        if not (candidate == workspace_real or candidate.startswith(root)):
            _log.warning("frame_source_escaped_workspace", source=rel_path)
            continue
        if not os.path.isfile(candidate):
            continue
        try:
            blocks = parse_file(candidate)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            _log.warning("frame_source_parse_failed", source=rel_path, error=str(exc))
            continue
        for raw in blocks:
            if str(raw.get("_id", "")).startswith(prefix):
                enriched = dict(raw)
                enriched.setdefault("_source_file", rel_path)
                out.append(enriched)
    return tuple(out)
