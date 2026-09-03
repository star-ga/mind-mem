#!/usr/bin/env python3
"""The placebo arm: same shape, same length, same framing — wrong corpus.

The memory arm's prompt is longer than the control's by construction, and the
``<evidence>`` framing widened that gap further. So a positive memory delta is
consistent with two different stories:

    1. the recalled content helped, or
    2. ~1300 extra tokens of *anything* helped.

The control arm cannot separate them: it has no prefix at all. Only a third
arm can — one that is identical to the memory arm in every respect a token
counter or a formatter can see, and differs only in *which task the corpus
came from*. That is what this builds.

Matched on all three axes that could otherwise explain a delta:

* **Rendering** — built through the same ``pack_to_budget`` +
  ``AgentFormatter.inject`` path the memory arm uses, so the placebo carries
  the identical framing preamble and ``<evidence>`` wrappers. Matching a
  pre-framing rendering against a framed one would reintroduce the confound
  it exists to remove.
* **Length** — packed to the real section's own token count, within a stated
  tolerance, and the achieved gap is reported rather than assumed.
* **Provenance** — the corpus is seeded and recalled for a DONOR task, chosen
  mechanically (the next task in the artifact's own order, wrapping), never
  hand-picked. Its blocks must not overlap the real arm's.

Status: the builder and its invariants are here and tested. Wiring a third arm
into the run loop needs ``run_suite``/``build_prompt`` in
``src/mind_mem/bench/`` to admit one, which this file deliberately does not
touch. Until that lands, ``benchmarks/memory_ab_analysis.py`` reports
``placebo_arm_present: False`` and every delta stays length-confounded.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mind_mem.bench.ab_arms import RECALL_LIMIT, Budget  # noqa: E402
from mind_mem.bench.ab_task import Task  # noqa: E402

#: Placebo length may differ from the real section by at most this fraction.
#: Not zero: the packer works in whole blocks, so an exact token match is not
#: generally reachable. Stated and asserted rather than left to chance.
DEFAULT_TOLERANCE = 0.15

ARM_PLACEBO = "placebo"


@dataclass(frozen=True)
class PlaceboBuild:
    """A placebo section and the evidence that it is a fair match."""

    section: str
    tokens: int
    target_tokens: int
    block_ids: tuple[str, ...]
    donor_task_id: str
    within_tolerance: bool
    tolerance: float

    @property
    def length_gap(self) -> int:
        return self.tokens - self.target_tokens

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": ARM_PLACEBO,
            "donor_task_id": self.donor_task_id,
            "tokens": self.tokens,
            "target_tokens": self.target_tokens,
            "length_gap_tokens": self.length_gap,
            "within_tolerance": self.within_tolerance,
            "tolerance": self.tolerance,
            "block_ids": list(self.block_ids),
        }


def choose_donor(tasks: tuple[Task, ...], task: Task) -> Task:
    """The next task in the artifact's own order, wrapping. Mechanical.

    Deliberately not "the most dissimilar task" or anything else that would
    let the donor be chosen to flatter a result: the rule is stated, it is a
    function of the artifact's fixed order, and it is reproducible.
    """
    if len(tasks) < 2:
        raise ValueError("a placebo arm needs at least two tasks to draw a donor from")
    index = next((i for i, t in enumerate(tasks) if t.task_id == task.task_id), None)
    if index is None:
        raise ValueError(f"task {task.task_id!r} is not in the given task set")
    return tasks[(index + 1) % len(tasks)]


def build_placebo_section(
    donor: Task,
    donor_workspace: str,
    target_tokens: int,
    budget: Budget,
    tolerance: float = DEFAULT_TOLERANCE,
) -> PlaceboBuild:
    """Render a donor-sourced section matched to ``target_tokens``.

    Uses the same three shipped surfaces the memory arm uses -- ``recall``
    with the donor's pinned scoring instant, ``pack_to_budget``, and
    ``AgentFormatter.inject`` -- so the result is framed exactly as the real
    section is. Only the corpus differs.
    """
    from mind_mem.agent_bridge import AgentFormatter
    from mind_mem.cognitive_forget import estimate_tokens, pack_to_budget
    from mind_mem.recall import recall

    if target_tokens <= 0:
        return PlaceboBuild("", 0, target_tokens, (), donor.task_id, True, tolerance)

    hits = recall(donor_workspace, donor.recall_query, limit=RECALL_LIMIT, scoring_instant=donor.scoring_date)
    # Pack to the REAL section's size, not to the memory sub-budget: the
    # placebo's job is to match the length that actually shipped.
    packed = pack_to_budget(hits, max_tokens=min(target_tokens, budget.memory_tokens))
    if not packed.included:
        return PlaceboBuild("", 0, target_tokens, (), donor.task_id, False, tolerance)

    section = AgentFormatter(max_blocks=len(packed.included)).inject("generic", donor.recall_query, packed.included)
    section = f"{section.rstrip()}\n\n"
    tokens = estimate_tokens(section)
    within = abs(tokens - target_tokens) <= max(1, int(round(tolerance * target_tokens)))
    return PlaceboBuild(
        section=section,
        tokens=tokens,
        target_tokens=target_tokens,
        block_ids=tuple(str(b.get("_id", "")) for b in packed.included),
        donor_task_id=donor.task_id,
        within_tolerance=within,
        tolerance=tolerance,
    )


def assert_placebo_fair(placebo: PlaceboBuild, memory_block_ids: tuple[str, ...], memory_section: str) -> tuple[str, ...]:
    """Refuse a placebo that is not actually a fair match.

    Raises rather than returning a flag: a placebo that shares blocks with the
    real arm, or that is a different length, is not a control -- it is a second
    memory arm with a misleading label, and it would make a null look like a
    refutation of memory.
    """
    from mind_mem.data_marking import DATA_OPEN

    overlap = sorted(set(placebo.block_ids) & set(memory_block_ids))
    if overlap:
        raise AssertionError(f"placebo shares blocks with the memory arm: {overlap}")
    if not placebo.within_tolerance:
        raise AssertionError(
            f"placebo is {placebo.tokens} tokens against a target of {placebo.target_tokens} "
            f"(gap {placebo.length_gap}); outside the {placebo.tolerance:.0%} tolerance"
        )
    if memory_section and DATA_OPEN in memory_section and DATA_OPEN not in placebo.section:
        raise AssertionError("memory section is framed but the placebo is not; the rendering is not matched")
    return ("no_block_overlap_with_memory_arm", "length_matched_within_tolerance", "same_framed_rendering")
