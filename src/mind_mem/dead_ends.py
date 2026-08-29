# Copyright 2026 STARGA, Inc.
"""DEAD-END blocks — negative action-space memory, matched declaratively.

The costliest long-horizon agent failure is not forgetting a fact, it is
**re-running an approach that already failed**.  Positive memory cannot fix
that: "we tried X and it did not work" only helps if it surfaces at the
moment the agent is about to try X again, and a ranked corpus will happily
bury it under whatever the current query resembles.

A ``[DE-...]`` block records the failure and the conditions under which it
should be raised::

    [DE-20260826-001]
    Type: DeadEnd
    Approach: Additive per-object assignment lower bound.
    WhyFailed: A co-carrier helper divides the matching by K, so the bound
        lands under the already-proven floor.
    Outcome: refuted
    Evidence: docs/AGI3_FLOOR_METHOD_ARSENAL.md#assignment
    TriggerTools: Bash
    TriggerIntents: prove_floor
    Status: active

Semantics
---------
* **One trigger language.**  ``Trigger*`` fields are parsed by exactly the
  code a ``[GR-...]`` block uses — :func:`~mind_mem.guardrail_patterns.
  coerce_patterns` into a :class:`~mind_mem.guardrails.GuardrailTrigger`.
  Same glob grammar, same normalisation, same per-dimension bounds.  There
  is no second trigger dialect to learn or to keep in sync.
* **Declarative overlap, never similarity.**  A dead end fires when its
  declared patterns overlap the task's declared
  :class:`~mind_mem.task_frames.ApproachSurface`: AND across declared
  dimensions, OR within one, fail-closed on an empty trigger or an empty
  surface.  :func:`match_dead_ends` is a pure function of its two
  arguments — no clock, no model, no learned score, no ranking signal.
  Determinism is the product wedge; a learned detector would forfeit it.
* **A warning, never a veto.**  Firing produces a
  :class:`DeadEndWarning` carrying the reason and the evidence handle.
  Nothing here refuses an action, filters a plan or changes an exit code.
  The operator decides; the memory only makes sure they decide informed.
* **Closed outcome vocabulary.**  ``refuted`` / ``regressed`` / ``blocked``
  / ``inconclusive`` — see :data:`OUTCOME_RANK`.  Free-text outcomes would
  make warning order a judgement call instead of a total order.
* **Read-only.**  Nothing in this module writes.  Dead ends are authored on
  the governed proposal route and only ever read back.
* **Provenance-restricted.**  A dead end steers an agent away from an
  action, so external-ingest content may never mint one — the same refusal
  guardrails use, shared verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .frame_fields import RejectedBlock, handle_entries, load_blocks, prose_entries
from .guardrail_patterns import (
    GuardrailSpecError,
    coerce_patterns,
    exact_or_glob,
    path_match,
    substring_or_glob,
)
from .guardrails import GuardrailTrigger, guardrail_provenance_refusal
from .observability import get_logger
from .task_frames import ApproachSurface, FramePolicy, TaskFrame

__all__ = [
    "DEAD_END_ID_PREFIX",
    "OUTCOME_RANK",
    "TRIGGER_FIELDS",
    "ApproachSurface",
    "DeadEnd",
    "DeadEndSpecError",
    "DeadEndWarning",
    "RejectedBlock",
    "load_dead_ends",
    "load_dead_ends_with_rejections",
    "match_dead_ends",
    "match_surface",
    "overlap",
    "parse_dead_end_block",
]

_log = get_logger("dead_ends")

#: Block-ID prefix that marks a block as a dead end.
DEAD_END_ID_PREFIX = "DE-"

#: Closed outcome vocabulary, ranked most-conclusive first.  A refuted
#: approach is a harder stop than one that was merely blocked.
OUTCOME_RANK: Mapping[str, int] = MappingProxyType({"refuted": 0, "regressed": 1, "blocked": 2, "inconclusive": 3})

_DEFAULT_OUTCOME = "blocked"

#: The trigger dimensions, in match-report order.  Deliberately the same
#: four a guardrail declares: ``(reported name, block field)``.
TRIGGER_FIELDS: tuple[tuple[str, str], ...] = (
    ("tool", "TriggerTools"),
    ("command", "TriggerCommands"),
    ("intent", "TriggerIntents"),
    ("path", "TriggerPaths"),
)

#: Statuses that keep a dead end live.  A superseded dead end is history.
_LIVE_STATUSES = frozenset({"", "active", "confirmed"})


class DeadEndSpecError(GuardrailSpecError):
    """Raised when a ``[DE-...]`` block cannot be read as a dead end."""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeadEnd:
    """A parsed ``[DE-...]`` block plus its compiled trigger."""

    block_id: str
    approach: str
    why_failed: str
    outcome: str
    evidence: tuple[str, ...] = ()
    trigger: GuardrailTrigger = field(default_factory=GuardrailTrigger)
    source_file: str = ""
    line: int = 0
    status: str = ""
    block: Mapping[str, Any] = field(default_factory=dict)

    @property
    def outcome_rank(self) -> int:
        """Rank of this outcome; an unknown one sorts last, never raises.

        :func:`parse_dead_end_block` refuses an outcome outside the closed
        vocabulary, so this default only covers a hand-built record — but a
        total order must stay total whatever it is handed.
        """
        return OUTCOME_RANK.get(self.outcome, len(OUTCOME_RANK))

    def sort_key(self) -> tuple[int, str]:
        """Total order: outcome first, then block ID.  No clocks, no scores."""
        return (self.outcome_rank, self.block_id)

    def is_live(self) -> bool:
        return self.status.strip().casefold() in _LIVE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """Deterministic, JSON-safe view for a brief or an MCP response."""
        return {
            "block_id": self.block_id,
            "approach": self.approach,
            "why_failed": self.why_failed,
            "outcome": self.outcome,
            "evidence": list(self.evidence),
            "source_file": self.source_file,
        }


@dataclass(frozen=True)
class DeadEndWarning:
    """One firing dead end and the dimensions that made it fire.

    Carries no score and no confidence: the match either held or it did
    not, and *matched* says exactly which declarations were responsible so
    an operator can audit the warning instead of trusting it.
    """

    dead_end: DeadEnd
    matched: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = self.dead_end.to_dict()
        payload["matched"] = list(self.matched)
        return payload


# ---------------------------------------------------------------------------
# Parsing + loading
# ---------------------------------------------------------------------------


def _trigger(block: Mapping[str, Any], block_id: str) -> GuardrailTrigger:
    """Compile the ``Trigger*`` fields — the guardrail grammar, verbatim."""
    return GuardrailTrigger(
        tools=coerce_patterns(block.get("TriggerTools"), field="TriggerTools", block_id=block_id, as_path=False),
        commands=coerce_patterns(block.get("TriggerCommands"), field="TriggerCommands", block_id=block_id, as_path=False),
        intents=coerce_patterns(block.get("TriggerIntents"), field="TriggerIntents", block_id=block_id, as_path=False),
        paths=coerce_patterns(block.get("TriggerPaths"), field="TriggerPaths", block_id=block_id, as_path=True),
    )


def parse_dead_end_block(block: Mapping[str, Any]) -> DeadEnd:
    """Read one parsed block dict as a :class:`DeadEnd`.

    Raises:
        DeadEndSpecError: the block is not a ``DE-`` block, carries
            external-ingest provenance, omits ``Approach`` / ``WhyFailed``,
            declares an outcome outside :data:`OUTCOME_RANK`, or declares
            no trigger at all.  A dead end that can never fire is noise,
            not memory, so it is refused rather than stored silently.
    """
    block_id = str(block.get("_id", ""))
    if not block_id.startswith(DEAD_END_ID_PREFIX):
        raise DeadEndSpecError(f"not a dead-end block: {block_id!r}")

    # Origin before content: an untrusted block is refused on provenance
    # alone, never on how well-formed its declaration happens to be.
    refusal = guardrail_provenance_refusal(block)
    if refusal:
        raise DeadEndSpecError(f"{block_id}: untrusted provenance ({refusal}) — cannot declare a dead end")

    approach = _one_line(block.get("Approach") or block.get("Statement"), "Approach", block_id)
    why_failed = _one_line(block.get("WhyFailed") or block.get("Reason"), "WhyFailed", block_id)

    trigger = _trigger(block, block_id)
    if trigger.is_empty():
        raise DeadEndSpecError(
            f"{block_id}: dead end declares no trigger ({', '.join(f for _, f in TRIGGER_FIELDS)}) — it would never fire"
        )

    try:
        line = int(block.get("_line", 0))
    except (TypeError, ValueError):
        line = 0

    return DeadEnd(
        block_id=block_id,
        approach=approach,
        why_failed=why_failed,
        outcome=_outcome(block.get("Outcome"), block_id),
        evidence=handle_entries(block.get("Evidence"), field_name="Evidence", block_id=block_id),
        trigger=trigger,
        source_file=str(block.get("_source_file", "")),
        line=line,
        status=str(block.get("Status", "")),
        block=MappingProxyType(dict(block)),
    )


def _outcome(raw: Any, block_id: str) -> str:
    """Read ``Outcome`` against the closed vocabulary, or refuse the block."""
    outcome = str(raw or _DEFAULT_OUTCOME).strip().casefold()
    if outcome not in OUTCOME_RANK:
        raise DeadEndSpecError(f"{block_id}: Outcome {outcome!r} is not one of {tuple(OUTCOME_RANK)}")
    return outcome


def _one_line(raw: Any, field_name: str, block_id: str) -> str:
    """Collapse a required prose field to one line, or refuse the block."""
    entries = prose_entries(raw, field_name=field_name, block_id=block_id)
    text = " ".join(entries)
    if not text:
        raise DeadEndSpecError(f"{block_id}: dead end has no {field_name}")
    return text


def load_dead_ends_with_rejections(
    workspace: str,
    policy: FramePolicy | None = None,
) -> tuple[tuple[DeadEnd, ...], tuple[RejectedBlock, ...]]:
    """Parse every live dead end, **and** every one that was refused.

    A refused dead end is the worst thing in this module to lose
    quietly: the registry exists to stop an agent re-running a known
    failure, so a warning that vanishes leaves the agent *more*
    confident than if the block had never been written.  Refusals come
    back next to the registry and every caller publishes them.

    Returns:
        ``(dead_ends, rejected)``, both in block-ID order.  Warning
        order is a separate concern; :func:`match_surface` ranks by
        :meth:`DeadEnd.sort_key`.
    """
    policy = policy or FramePolicy()
    found: dict[str, DeadEnd] = {}
    refused: dict[str, RejectedBlock] = {}
    for raw in load_blocks(workspace, policy.dead_end_sources, DEAD_END_ID_PREFIX):
        source = str(raw.get("_source_file", ""))
        try:
            dead_end = parse_dead_end_block(raw)
        except GuardrailSpecError as exc:
            block_id = str(raw.get("_id", ""))
            _log.warning("dead_end_block_rejected", source=source, error=str(exc))
            refused.setdefault(block_id, RejectedBlock(block_id, source, str(exc)))
            continue
        if dead_end.is_live():
            found.setdefault(dead_end.block_id, dead_end)
    return (
        tuple(sorted(found.values(), key=lambda d: d.block_id)),
        tuple(sorted(refused.values(), key=RejectedBlock.sort_key)),
    )


def load_dead_ends(workspace: str, policy: FramePolicy | None = None) -> tuple[DeadEnd, ...]:
    """Parse every live dead end declared under *workspace*.

    The registry only.  Callers that must publish what the corpus
    declared but could not be read want
    :func:`load_dead_ends_with_rejections`.

    Returns:
        Dead ends in block-ID order — the registry view.
    """
    dead_ends, _rejected = load_dead_ends_with_rejections(workspace, policy)
    return dead_ends


# ---------------------------------------------------------------------------
# The deterministic overlap test
# ---------------------------------------------------------------------------


def overlap(trigger: GuardrailTrigger, surface: ApproachSurface) -> tuple[str, ...]:
    """Return the dimensions where *trigger* overlaps *surface*, or ``()``.

    AND across declared dimensions, OR within one — the guardrail rule,
    lifted to plural values on the surface side.  Fail-closed: an empty
    trigger or an empty surface never overlaps.  Pure: reads only its two
    arguments, calls only the shared literal/glob matchers, and returns
    dimension names in :data:`TRIGGER_FIELDS` order so reports never vary.
    """
    if trigger.is_empty() or surface.is_empty():
        return ()
    checks = (
        ("tool", trigger.tools, surface.tools, exact_or_glob),
        ("command", trigger.commands, surface.commands, substring_or_glob),
        ("intent", trigger.intents, surface.intents, exact_or_glob),
    )
    matched: list[str] = []
    for name, patterns, values, matcher in checks:
        if not patterns:
            continue
        if not any(matcher(patterns, value) for value in values):
            return ()
        matched.append(name)
    if trigger.paths:
        if not path_match(trigger.paths, surface.paths):
            return ()
        matched.append("path")
    return tuple(matched)


def match_surface(surface: ApproachSurface, dead_ends: Sequence[DeadEnd]) -> tuple[DeadEndWarning, ...]:
    """Return every dead end overlapping *surface*, most conclusive first.

    Ordered by :meth:`DeadEnd.sort_key` — ``(outcome, block_id)`` — so the
    same inputs always produce the same warnings in the same order, in any
    process, on any machine.
    """
    hits = [DeadEndWarning(dead_end=dead_end, matched=matched) for dead_end in dead_ends if (matched := overlap(dead_end.trigger, surface))]
    hits.sort(key=lambda warning: warning.dead_end.sort_key())
    return tuple(hits)


def match_dead_ends(frame: TaskFrame, dead_ends: Sequence[DeadEnd]) -> tuple[DeadEndWarning, ...]:
    """Return the dead ends that overlap *frame*'s declared approach.

    A pure function of ``(frame, dead-end records)``: it reads no clock,
    calls no model, computes no similarity and mutates neither argument.
    Feeding it the same two arguments twice — in two processes, on two
    machines — yields byte-identical output.

    The result is **evidence, not a prohibition**.  A dead end can only
    warn: it never blocks an action, never filters the frame's plan and
    never changes an exit code.  The operator reads the reason and the
    evidence handle and decides; re-running a known-failed approach is
    sometimes exactly right, and this function has no standing to refuse
    it.

    Args:
        frame: The task being resumed.  Its ``Approach*`` fields are the
            value side of the match; a frame declaring no approach
            surface matches nothing.
        dead_ends: The registry to test against, in any order.

    Returns:
        Warnings ordered by ``(outcome, block_id)``.
    """
    return match_surface(frame.approach, dead_ends)
