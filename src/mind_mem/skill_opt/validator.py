# Copyright 2026 STARGA, Inc.
"""Mutation validation and governance submission."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

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


def submit_to_governance(
    mutation: Mutation,
    validation: ValidationResult,
    workspace: str,
) -> str:
    """Write a governance proposal for this mutation via SIGNALS.md.

    Returns the signal block ID for tracking.
    """
    signal_id = f"SKILL-{mutation.mutation_id}"
    signal_block = {
        "signal_id": signal_id,
        "type": "edit",
        "source": "skill_opt",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "skill_id": mutation.skill_id,
        "mutation_id": mutation.mutation_id,
        "rationale": mutation.rationale,
        "score_before": validation.score_before,
        "score_after": validation.score_after,
        "improved": validation.improved,
        "accepted": validation.accepted,
        "critic_votes": validation.critic_votes,
    }

    signals_dir = os.path.join(workspace, "intelligence")
    os.makedirs(signals_dir, exist_ok=True)
    signals_path = os.path.join(signals_dir, "SIGNALS.md")

    entry = (
        f"\n## {signal_id}\n"
        f"- **Type:** skill mutation\n"
        f"- **Skill:** {mutation.skill_id}\n"
        f"- **Score:** {validation.score_before:.4f} → {validation.score_after:.4f}\n"
        f"- **Status:** {'accepted' if validation.accepted else 'rejected'}\n"
        f"- **Rationale:** {mutation.rationale}\n"
        f"- **Timestamp:** {signal_block['timestamp']}\n"
    )

    with open(signals_path, "a", encoding="utf-8") as f:
        f.write(entry)

    return signal_id
