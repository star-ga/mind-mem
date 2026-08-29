# Copyright 2026 STARGA, Inc.
"""TASK-FRAME blocks — what a multi-session task IS, carried across sessions.

A coding agent that stops and restarts re-derives its context from scratch:
it re-reads the same files, re-runs the same probes, and re-discovers the
same conclusions.  Ranked recall does not fix this — "what am I working on"
is not a similarity query, it is a *pointer*, and a pointer that has to win
a relevance contest against the whole corpus is not a pointer.

A ``[TF-...]`` block is that pointer, written down::

    [TF-20260829-001]
    Type: TaskFrame
    Goal: Close the last two AGI3 floors without a net regression.
    Status: active
    Steps:
    - done: rederive the floor count from the live pass
    - doing: pin the SAT-BMC encoding for L5
    - todo: package the bundle for the cloud run
    Tried:
    - explicit BFS over the level graph
    Believed: flow bounds survive multi-carrier levels
    Remaining: fund the cloud run
    Blockers: cloud budget
    References: docs/floors.json
    ApproachTools: Bash
    ApproachIntents: prove_floor
    ApproachPaths: tools/**/*.py

Semantics
---------
* **Four questions, declared not inferred.**  What the task is (``Goal``),
  what has been tried (``Tried``), what is currently believed
  (``Believed``), and what remains (``Remaining``, falling back to the
  steps that are not yet ``done``).  :mod:`mind_mem.resume_brief` reads
  exactly these back.
* **The approach surface.**  ``Approach*`` fields name the tools, commands,
  intent classes and paths this task involves.  They are the *value* side
  of the trigger grammar in :mod:`mind_mem.guardrail_patterns` — the same
  glob syntax, normalisation and per-dimension bounds a ``[GR-...]`` block
  uses for its patterns.  :mod:`mind_mem.dead_ends` matches one against the
  other; nothing here scores, ranks or infers.
* **Deterministic.**  Frames load in block-ID order.  Same corpus ⇒ same
  frames, same order, on every machine, with no clock read.
* **Read-only.**  Nothing in this module writes.  A frame is authored on
  the governed proposal route like any other block and only ever read back
  — see ``docs/task-frames.md`` §Authoring.
* **Provenance-restricted.**  A frame steers an agent's next session, so a
  block carrying external-ingest provenance may never mint one.  The check
  is :func:`mind_mem.guardrails.guardrail_provenance_refusal`, shared
  verbatim with guardrails so the two threat models cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Sequence

from .frame_fields import (
    MAX_PROSE_ENTRIES,
    STEP_STATUSES,
    FrameSpecError,
    PlanStep,
    RejectedBlock,
    handle_entries,
    load_blocks,
    parse_steps,
    prose_entries,
)
from .guardrail_patterns import GuardrailSpecError, coerce_patterns
from .guardrails import GuardrailContext, guardrail_provenance_refusal
from .observability import get_logger

__all__ = [
    "DEFAULT_DEAD_END_FILES",
    "DEFAULT_FRAME_FILES",
    "DEFAULT_MAX_WARNINGS",
    "MAX_PROSE_ENTRIES",
    "STEP_STATUSES",
    "TASK_FRAME_ID_PREFIX",
    "ApproachSurface",
    "FramePolicy",
    "FrameSpecError",
    "PlanStep",
    "RejectedBlock",
    "TaskFrame",
    "active_frame",
    "load_task_frames",
    "load_task_frames_with_rejections",
    "parse_task_frame_block",
]

_log = get_logger("task_frames")

#: Block-ID prefix that marks a block as a task frame.
TASK_FRAME_ID_PREFIX = "TF-"

#: Workspace-relative files scanned for frames / dead ends by default.
DEFAULT_FRAME_FILES: tuple[str, ...] = ("frames/FRAMES.md",)
DEFAULT_DEAD_END_FILES: tuple[str, ...] = ("frames/DEAD-ENDS.md",)

#: Default cap on dead-end warnings attached to one brief.
DEFAULT_MAX_WARNINGS = 5

#: Statuses that keep a frame live.  Anything else is parsed but ignored.
_LIVE_STATUSES = frozenset({"", "active", "wip", "doing"})

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApproachSurface:
    """What a task involves — the *value* side of the trigger grammar.

    The plural counterpart of :class:`~mind_mem.guardrails.GuardrailContext`:
    a guardrail asks "what is the agent doing *right now*" (one tool, one
    command), a frame declares "what does this task involve at all" (every
    tool, every command).  Matching is otherwise identical, so
    :mod:`mind_mem.dead_ends` can serve both from one matcher.
    """

    tools: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()

    @classmethod
    def from_context(cls, context: GuardrailContext) -> "ApproachSurface":
        """Widen a single-action :class:`GuardrailContext` into a surface.

        Lets the about-to-act check and the frame check share one matcher:
        a context is just a surface whose dimensions hold at most one value.
        """
        return cls(
            tools=_singleton(context.tool),
            commands=_singleton(context.command),
            intents=_singleton(context.intent),
            paths=tuple(str(p) for p in context.paths if str(p).strip()),
        )

    def is_empty(self) -> bool:
        return not (self.tools or self.commands or self.intents or self.paths)


@dataclass(frozen=True)
class TaskFrame:
    """A parsed ``[TF-...]`` block: the four resume questions plus a surface."""

    block_id: str
    goal: str
    steps: tuple[PlanStep, ...] = ()
    tried: tuple[str, ...] = ()
    believed: tuple[str, ...] = ()
    remaining: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    approach: ApproachSurface = field(default_factory=ApproachSurface)
    source_file: str = ""
    line: int = 0
    status: str = ""
    block: Mapping[str, Any] = field(default_factory=dict)

    def sort_key(self) -> str:
        """Total order: block ID.  No clocks, no scores."""
        return self.block_id

    def is_live(self) -> bool:
        return self.status.strip().casefold() in _LIVE_STATUSES

    def remaining_steps(self) -> tuple[PlanStep, ...]:
        return tuple(step for step in self.steps if step.is_remaining())


@dataclass(frozen=True)
class FramePolicy:
    """Bounds + kill switch for frames and their dead-end warnings."""

    #: Absolute ceiling on ``max_warnings`` — a misconfigured policy can
    #: never flood a brief with warnings.
    HARD_CAP: ClassVar[int] = 20

    enabled: bool = True
    max_warnings: int = DEFAULT_MAX_WARNINGS
    frame_sources: tuple[str, ...] = DEFAULT_FRAME_FILES
    dead_end_sources: tuple[str, ...] = DEFAULT_DEAD_END_FILES

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "FramePolicy":
        """Read ``recall.frames`` from a workspace config mapping.

        Invalid values fall back to the defaults with a warning rather than
        raising: a typo in ``mind-mem.json`` must not take a resume brief
        down, and must not silently disable the warnings either.
        """
        section: Any = {}
        if isinstance(config, Mapping):
            recall_cfg = config.get("recall")
            if isinstance(recall_cfg, Mapping):
                section = recall_cfg.get("frames", {})
        if not isinstance(section, Mapping):
            _log.warning("frame_config_ignored", reason="recall.frames is not an object")
            return cls()
        try:
            bounded = max(0, min(int(section.get("max_warnings", DEFAULT_MAX_WARNINGS)), cls.HARD_CAP))
        except (TypeError, ValueError):
            _log.warning("frame_config_ignored", reason="max_warnings is not an integer")
            bounded = DEFAULT_MAX_WARNINGS
        return cls(
            enabled=bool(section.get("enabled", True)),
            max_warnings=bounded,
            frame_sources=_sources(section.get("frame_sources"), DEFAULT_FRAME_FILES),
            dead_end_sources=_sources(section.get("dead_end_sources"), DEFAULT_DEAD_END_FILES),
        )


# ---------------------------------------------------------------------------
# Field coercion at the block boundary
# ---------------------------------------------------------------------------


def _singleton(value: Any) -> tuple[str, ...]:
    text = str(value or "").strip()
    return (text,) if text else ()


def _sources(raw: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    """Coerce a configured source list, falling back to *default*."""
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, (list, tuple)):
        if raw is not None:
            _log.warning("frame_config_ignored", reason="sources is not a list")
        return default
    cleaned = tuple(str(s).strip() for s in raw if str(s).strip())
    return cleaned or default


def _approach(block: Mapping[str, Any], block_id: str) -> ApproachSurface:
    """Compile the ``Approach*`` fields with the shared trigger grammar."""
    return ApproachSurface(
        tools=coerce_patterns(block.get("ApproachTools"), field="ApproachTools", block_id=block_id, as_path=False),
        commands=coerce_patterns(block.get("ApproachCommands"), field="ApproachCommands", block_id=block_id, as_path=False),
        intents=coerce_patterns(block.get("ApproachIntents"), field="ApproachIntents", block_id=block_id, as_path=False),
        paths=coerce_patterns(block.get("ApproachPaths"), field="ApproachPaths", block_id=block_id, as_path=True),
    )


# ---------------------------------------------------------------------------
# Parsing + loading
# ---------------------------------------------------------------------------


def parse_task_frame_block(block: Mapping[str, Any]) -> TaskFrame:
    """Read one parsed block dict as a :class:`TaskFrame`.

    Raises:
        FrameSpecError: the block is not a ``TF-`` block, carries
            external-ingest provenance, has no ``Goal``, or declares a
            malformed field.  Provenance is checked before any content
            field, so an untrusted block is refused on origin alone.
    """
    block_id = str(block.get("_id", ""))
    if not block_id.startswith(TASK_FRAME_ID_PREFIX):
        raise FrameSpecError(f"not a task frame block: {block_id!r}")

    refusal = guardrail_provenance_refusal(block)
    if refusal:
        raise FrameSpecError(f"{block_id}: untrusted provenance ({refusal}) — cannot declare a task frame")

    goal = " ".join(str(block.get("Goal") or block.get("Statement") or "").split())
    if not goal:
        raise FrameSpecError(f"{block_id}: task frame has no Goal")

    steps = parse_steps(block.get("Steps"), block_id=block_id)
    remaining = prose_entries(block.get("Remaining"), field_name="Remaining", block_id=block_id)
    if not remaining:
        remaining = tuple(step.text for step in steps if step.is_remaining())

    try:
        line = int(block.get("_line", 0))
    except (TypeError, ValueError):
        line = 0

    return TaskFrame(
        block_id=block_id,
        goal=goal,
        steps=steps,
        tried=prose_entries(block.get("Tried"), field_name="Tried", block_id=block_id),
        believed=prose_entries(block.get("Believed"), field_name="Believed", block_id=block_id),
        remaining=remaining,
        blockers=prose_entries(block.get("Blockers"), field_name="Blockers", block_id=block_id),
        citations=handle_entries(block.get("References"), field_name="References", block_id=block_id),
        approach=_approach(block, block_id),
        source_file=str(block.get("_source_file", "")),
        line=line,
        status=str(block.get("Status", "")),
        block=MappingProxyType(dict(block)),
    )


def load_task_frames_with_rejections(
    workspace: str,
    policy: FramePolicy | None = None,
) -> tuple[tuple[TaskFrame, ...], tuple[RejectedBlock, ...]]:
    """Parse every live task frame, **and** every frame that was refused.

    One bad block never takes the frame set down — but it is not dropped
    on the floor either.  A refusal is returned next to the frames that
    loaded so a caller can say "this workspace declares a frame I could
    not read" instead of "this workspace declares no frame", which are
    opposite facts and lead to opposite behaviour.

    Returns:
        ``(frames, rejected)``, both in block-ID order.  Duplicate frame
        IDs resolve first-source-wins.
    """
    policy = policy or FramePolicy()
    found: dict[str, TaskFrame] = {}
    refused: dict[str, RejectedBlock] = {}
    for raw in load_blocks(workspace, policy.frame_sources, TASK_FRAME_ID_PREFIX):
        source = str(raw.get("_source_file", ""))
        try:
            frame = parse_task_frame_block(raw)
        except GuardrailSpecError as exc:
            block_id = str(raw.get("_id", ""))
            _log.warning("frame_block_rejected", source=source, error=str(exc))
            refused.setdefault(block_id, RejectedBlock(block_id, source, str(exc)))
            continue
        if frame.is_live():
            found.setdefault(frame.block_id, frame)
    return (
        tuple(sorted(found.values(), key=TaskFrame.sort_key)),
        tuple(sorted(refused.values(), key=RejectedBlock.sort_key)),
    )


def load_task_frames(workspace: str, policy: FramePolicy | None = None) -> tuple[TaskFrame, ...]:
    """Parse every live task frame declared under *workspace*.

    The frames only.  Callers that must report what the corpus declared
    but could not be read want :func:`load_task_frames_with_rejections`.

    Returns:
        Frames in block-ID order, duplicates resolved first-source-wins.
    """
    frames, _rejected = load_task_frames_with_rejections(workspace, policy)
    return frames


def active_frame(frames: Sequence[TaskFrame]) -> TaskFrame | None:
    """Return the frame a new session should resume, or ``None``.

    Deterministic tie-break: the highest block ID wins.  Block IDs are
    ``TF-YYYYMMDD-NNN``, so that is the most recently minted frame — read
    off the ID, never off a clock.
    """
    live = [frame for frame in frames if frame.is_live()]
    return max(live, key=TaskFrame.sort_key) if live else None
