# Copyright 2026 STARGA, Inc.
"""Mutation validation and governance submission."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from ._types import Mutation, SkillSpec, TestCase, ValidationResult
from .analyzer import aggregate_analysis, analyze_skill
from .config import SkillOptConfig
from .fleet_bridge import FleetBridge
from .scorer import aggregate_critiques
from .test_runner import run_tests


async def validate_mutation(
    original: SkillSpec,
    mutation: Mutation,
    test_cases: list[TestCase],
    fleet: FleetBridge,
    config: SkillOptConfig,
) -> ValidationResult:
    """Re-run tests with the mutated skill and compare scores."""
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
    critiques = await analyze_skill(mutated_spec, results, fleet, min_critics=config.min_critics)

    now = datetime.now(timezone.utc).isoformat()
    new_score = aggregate_critiques(
        original.skill_id,
        hashlib.sha256(mutation.proposed_content.encode()).hexdigest(),
        critiques,
        timestamp=now,
    )

    pre_score = mutation.critic_consensus.get("pre_mutation_score", 0.0)
    improved = new_score.overall - pre_score >= config.improvement_threshold

    regression_categories: list[str] = []
    for key, old_val in new_score.by_rubric.items():
        if old_val < pre_score - config.regression_threshold:
            regression_categories.append(key)

    analysis = aggregate_analysis(critiques)
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
