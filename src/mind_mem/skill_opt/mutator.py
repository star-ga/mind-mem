# Copyright 2026 STARGA, Inc.
"""Failure-driven mutation proposal generator."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ._types import Mutation, SkillSpec
from .fleet_bridge import FleetBridge

_MUTATION_STRATEGIES = {
    "targeted_patch": (
        "Rewrite ONLY the sections of this skill that are causing the identified failures. "
        "Preserve all working parts unchanged. Add explicit constraints or examples "
        "that would prevent the specific failure modes listed."
    ),
    "restructure": (
        "Reorganize this skill's structure for clarity: move rules before examples, "
        "add explicit decision trees where the skill currently uses ambiguous prose, "
        "and ensure the most important constraints appear early."
    ),
    "expand": (
        "Add missing examples, edge cases, and explicit prohibitions that would prevent "
        "the identified failure modes. Do not remove existing content — only add."
    ),
    "compress": (
        "Remove verbose, redundant, or low-signal sections that dilute the core instruction. "
        "Keep the same capabilities but make the skill more focused and token-efficient."
    ),
}


def _build_mutation_prompt(
    spec: SkillSpec,
    analysis: dict[str, Any],
    strategy: str,
) -> str:
    strategy_instruction = _MUTATION_STRATEGIES.get(strategy, _MUTATION_STRATEGIES["targeted_patch"])
    gaps = analysis.get("actionable_gaps", [])
    suggestions = analysis.get("top_suggestions", [])
    score = analysis.get("consensus_score", 0.0)

    return f"""You are optimizing an AI agent skill prompt. The current version scored {score:.2f}/1.0.

## Current Skill Content
```
{spec.content[:6000]}
```

## Identified Failure Modes
{json.dumps(gaps, indent=2) if gaps else "None identified"}

## Improvement Suggestions from Cross-Model Critique
{json.dumps(suggestions, indent=2) if suggestions else "None"}

## Mutation Strategy: {strategy}
{strategy_instruction}

## Instructions
Rewrite the COMPLETE skill file (including any YAML frontmatter). Return ONLY the new skill content, no explanation or markdown fences."""


async def propose_mutations(
    spec: SkillSpec,
    analysis: dict[str, Any],
    fleet: FleetBridge,
    max_mutations: int = 3,
) -> list[Mutation]:
    """Generate mutation candidates from different fleet models and strategies."""
    strategies = list(_MUTATION_STRATEGIES.keys())[:max_mutations]
    mutations: list[Mutation] = []

    available = fleet.available_models
    for i, strategy in enumerate(strategies):
        model = available[i % len(available)] if available else None
        prompt = _build_mutation_prompt(spec, analysis, strategy)
        responses = await fleet.query(prompt, models=[model] if model else None)
        for resp in responses:
            if resp.ok and len(resp.content.strip()) > 50:
                mutations.append(
                    Mutation(
                        mutation_id=f"M-{uuid.uuid4().hex[:12]}",
                        skill_id=spec.skill_id,
                        original_hash=spec.content_hash,
                        proposed_content=resp.content.strip(),
                        rationale=f"Strategy: {strategy}. Model: {resp.model}. "
                        f"Addressing: {', '.join(analysis.get('actionable_gaps', [])[:3])}",
                        failure_modes_addressed=tuple(analysis.get("actionable_gaps", [])[:5]),
                        critic_consensus={"pre_mutation_score": analysis.get("consensus_score", 0.0)},
                    )
                )
    return mutations
