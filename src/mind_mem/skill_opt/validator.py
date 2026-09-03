# Copyright 2026 STARGA, Inc.
"""Mutation validation and governance submission."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any

from ..enums import INITIAL_STATUS, IngestTier
from ..observability import get_logger
from ._types import Mutation, SkillScore, SkillSpec, TestCase, ValidationResult
from .analyzer import aggregate_analysis, analyze_skill
from .config import SkillOptConfig
from .fleet_bridge import FleetBridge
from .scorer import aggregate_critiques
from .test_runner import run_tests

_log = get_logger("skill_opt.validator")

# Per-rubric baseline scores for the ORIGINAL skill may be carried on
# ``Mutation.critic_consensus`` under this prefix (``pre_rubric:safety``).
# Without them the pre-mutation rubric breakdown is simply not available
# — ``Mutation`` only ever carries the single overall consensus scalar.
RUBRIC_BASELINE_PREFIX = "pre_rubric:"


def _rubric_baseline(mutation: Mutation, original_score: SkillScore | None) -> dict[str, float]:
    """Per-rubric scores of the ORIGINAL skill, or {} when unavailable."""
    if original_score is not None and original_score.by_rubric:
        return dict(original_score.by_rubric)
    baseline: dict[str, float] = {}
    for key, value in mutation.critic_consensus.items():
        if key.startswith(RUBRIC_BASELINE_PREFIX):
            try:
                baseline[key[len(RUBRIC_BASELINE_PREFIX) :]] = float(value)
            except (TypeError, ValueError):
                continue
    return baseline


async def validate_mutation(
    original: SkillSpec,
    mutation: Mutation,
    test_cases: list[TestCase],
    fleet: FleetBridge,
    config: SkillOptConfig,
    *,
    original_score: SkillScore | None = None,
) -> ValidationResult:
    """Re-run tests with the mutated skill and compare scores.

    Args:
        original_score: Scores of the ORIGINAL skill. Supplying it is what
            makes the per-rubric regression check possible: without a
            per-rubric baseline (here, or as ``pre_rubric:*`` entries on
            ``mutation.critic_consensus``) the mutation's per-rubric means
            have nothing like-for-like to be compared against, and no
            regression is claimed either way.

    ``improved`` additionally requires the critic panel to have reached
    ``config.min_critics`` distinct models — a rewrite is never called an
    improvement on a panel smaller than the configured minimum.
    """
    mutated_spec = SkillSpec(
        skill_id=original.skill_id,
        system=original.system,
        source_path=original.source_path,
        format=original.format,
        name=original.name,
        description=original.description,
        content=mutation.proposed_content,
        metadata=original.metadata,
    )

    results = await run_tests(mutated_spec, test_cases, fleet)
    critiques = await analyze_skill(
        mutated_spec,
        results,
        fleet,
        min_critics=config.min_critics,
        test_cases=test_cases,
    )

    now = datetime.now(timezone.utc).isoformat()
    new_score = aggregate_critiques(
        original.skill_id,
        hashlib.sha256(mutation.proposed_content.encode()).hexdigest(),
        critiques,
        timestamp=now,
    )

    analysis = aggregate_analysis(critiques, min_critics=config.min_critics)
    critics_sufficient = bool(analysis["critics_sufficient"])
    if not critics_sufficient:
        _log.warning(
            "mutation_evidence_below_min_critics",
            skill_id=original.skill_id,
            mutation_id=mutation.mutation_id,
            required=config.min_critics,
            observed=analysis["n_critics"],
        )

    pre_score = mutation.critic_consensus.get("pre_mutation_score", 0.0)
    improved = critics_sufficient and new_score.overall - pre_score >= config.improvement_threshold

    # Compare the mutation's per-rubric means against the ORIGINAL's
    # per-rubric means. Comparing them against the original's single
    # OVERALL score instead flags every intrinsically-low category as a
    # regression and misses a category that genuinely collapsed.
    baseline_by_rubric = _rubric_baseline(mutation, original_score)
    regression_categories: list[str] = []
    if baseline_by_rubric:
        for key, new_val in new_score.by_rubric.items():
            old_val = baseline_by_rubric.get(key)
            if old_val is not None and new_val < old_val - config.regression_threshold:
                regression_categories.append(key)
        regression_categories.sort()
    else:
        _log.warning(
            "rubric_baseline_unavailable",
            skill_id=original.skill_id,
            mutation_id=mutation.mutation_id,
            detail="per-rubric regression not assessed: no pre-mutation rubric scores",
        )

    critic_votes: dict[str, bool] = {}
    for model in {c.critic_model for c in critiques}:
        model_scores = [c.overall_score for c in critiques if c.critic_model == model]
        if model_scores:
            avg = sum(model_scores) / len(model_scores)
            critic_votes[model] = avg > pre_score

    return ValidationResult(
        mutation_id=mutation.mutation_id,
        skill_id=original.skill_id,
        score_before=pre_score,
        score_after=new_score.overall,
        improved=improved,
        regression_categories=tuple(regression_categories),
        critic_votes=critic_votes,
    )


class SkillGovernanceError(RuntimeError):
    """A skill mutation could not be staged for governance review.

    Raised rather than returned, and rather than degraded to a warning. The
    caller (``mm skill-optimize``) records the returned id as the mutation's
    governance handle, so a submission that half-happened would be recorded
    as a submission that happened. Mirrors ``LintAutofixError`` and
    ``ImportQuarantineError``, which refuse the same way for the same reason.
    """


#: The tier a staged skill mutation is admitted under, and the status it
#: therefore declares. AUTO_CAPTURE is what this is: a machine-proposed
#: edit nobody has reviewed. Its ``INITIAL_STATUS`` row is ``PENDING``, so
#: the block is withheld from recall until a governance release admits it --
#: which is the whole point of "submitting to governance".
SIGNAL_TIER = IngestTier.AUTO_CAPTURE
_SIGNAL_STATUS_OR_NONE = INITIAL_STATUS[SIGNAL_TIER]
if _SIGNAL_STATUS_OR_NONE is None:  # pragma: no cover - import-time invariant
    # NOT an ``assert``: ``python -O`` strips those. An UNSTATED status is
    # SERVABLE, so a ``None`` here would publish every unreviewed skill
    # mutation straight into recall.
    raise RuntimeError(
        f"skill mutations are staged under tier {SIGNAL_TIER!r}, which mints no initial "
        "status; an unstated status is SERVABLE, so this would publish them"
    )
SIGNAL_STATUS = _SIGNAL_STATUS_OR_NONE

#: Block-id prefix for captured signals. ``corpus_registry.CORPUS_TABLE``
#: routes it to ``intelligence/SIGNALS.md``; that row landed in 5.0.2, and
#: before it existed ``write_block`` refused every ``SIG`` id -- which is
#: the mechanical reason this function used to splice the file by hand.
_SIGNAL_PREFIX = "SIG"


def _next_signal_id(store: Any, today_compact: str) -> str:
    """The next free ``SIG-<date>-###`` id, read from the STORE.

    Read through the store rather than off the file for the same reason
    ``intel_scan._recorded_findings`` does: on a non-Markdown backend the
    canonical file stays empty, so a file-only scan would restart the daily
    counter and hand a new signal an id an existing one already holds.
    """
    stamp = f"{_SIGNAL_PREFIX}-{today_compact}-"
    used: list[int] = []
    for block in store.get_all(active_only=False):
        block_id = str(block.get("_id") or "")
        tail = block_id[len(stamp) :]
        if block_id.startswith(stamp) and tail.isdigit():
            used.append(int(tail))
    return f"{stamp}{max(used, default=0) + 1:03d}"


def submit_to_governance(
    mutation: Mutation,
    validation: ValidationResult,
    workspace: str,
) -> str:
    """Stage this mutation for governance review as a governed signal block.

    Returns the block id, and it is a real id: ``get_by_id`` resolves it,
    and recall withholds it until a governance release admits it.

    This used to append a ``## SKILL-<mutation_id>`` markdown heading to
    ``intelligence/SIGNALS.md`` with a bare ``open(..., "a")``. Two things
    were wrong with that and only one of them was the bypass. The heading
    is not block syntax, so the id it returned -- recorded by ``mm
    skill-optimize`` as the mutation's ``governance_signal`` -- resolved to
    nothing; and the write reached a file recall serves with no admission,
    no evidence row and no chain row. "Submitted to governance" named a
    write governance never saw.

    Raises:
        SkillGovernanceError: an identical mutation is already staged.
    """
    from ..governance_gate import get_gate
    from ..storage import get_block_store

    store = get_block_store(workspace)
    now = datetime.now(timezone.utc)
    today_compact = now.strftime("%Y%m%d")

    preimage = f"{mutation.skill_id}\x1f{mutation.mutation_id}\x1f{mutation.proposed_content}"
    fingerprint = hashlib.sha256(preimage.encode()).hexdigest()[:16]
    for block in store.get_all(active_only=False):
        if str(block.get("_id") or "").startswith(f"{_SIGNAL_PREFIX}-") and str(block.get("ContentHash") or "") == fingerprint:
            raise SkillGovernanceError(
                f"an identical skill mutation is already staged as {block.get('_id')} "
                f"(fingerprint {fingerprint}); review or reject it before staging another"
            )

    signal_id = _next_signal_id(store, today_compact)
    verdict = "accepted" if validation.accepted else "rejected"
    statement = (
        f"Skill mutation {mutation.mutation_id} for {mutation.skill_id}: "
        f"score {validation.score_before:.4f} → {validation.score_after:.4f} ({verdict})."
    )
    block = {
        "_id": signal_id,
        "Statement": statement,
        "Date": now.strftime("%Y-%m-%d"),
        "Status": SIGNAL_STATUS.value,
        "Type": "auto-capture-skill-mutation",
        "Subject": mutation.skill_id,
        "Object": mutation.mutation_id,
        "Rationale": mutation.rationale,
        # Provenance that is TRUE. The block says it came from the skill
        # optimiser, because it did; it does not borrow ``capture``'s
        # ``memory/<date>.md:<line>`` shape, which would claim a daily-log
        # origin this content never had.
        "Source": "skill_opt.validator",
        "ContentHash": fingerprint,
        "Confidence": f"{validation.score_after:.4f}",
        "Evidence": [
            f"score_before: {validation.score_before:.4f}",
            f"score_after: {validation.score_after:.4f}",
            f"improved: {validation.improved}",
            f"accepted: {validation.accepted}",
            f"critic_votes: {validation.critic_votes}",
        ],
        "Action": "Review and apply the mutation, or reject it.",
    }

    # Admit BEFORE the bytes land, and let a refusal propagate.
    with get_gate(workspace).admit_block(
        action="WRITE",
        block_id=signal_id,
        content=statement,
        tier=SIGNAL_TIER,
        actor="skill_opt",
        target_file=os.path.join("intelligence", "SIGNALS.md"),
        metadata={"skill_id": mutation.skill_id, "mutation_id": mutation.mutation_id, "fingerprint": fingerprint},
    ):
        store.write_block(block)

    _log.info(
        "skill_mutation_staged",
        signal_id=signal_id,
        skill_id=mutation.skill_id,
        mutation_id=mutation.mutation_id,
        status=SIGNAL_STATUS.value,
    )
    return signal_id
