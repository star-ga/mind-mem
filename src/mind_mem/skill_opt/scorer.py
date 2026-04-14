# Copyright 2026 STARGA, Inc.
"""Rubric definitions and score aggregation for skill evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._types import CritiqueReport, SkillScore


@dataclass(frozen=True)
class RubricItem:
    key: str
    description: str
    weight: float = 1.0


RUBRICS: dict[str, tuple[RubricItem, ...]] = {
    "coding": (
        RubricItem("correctness", "Does the skill produce correct code/analysis?", 0.3),
        RubricItem("safety", "Does the skill enforce security best practices?", 0.2),
        RubricItem("completeness", "Does the output cover all requested aspects?", 0.2),
        RubricItem("format_compliance", "Does output match expected format?", 0.15),
        RubricItem("edge_case_handling", "Are edge cases handled gracefully?", 0.15),
    ),
    "tool": (
        RubricItem("command_accuracy", "Are generated commands syntactically correct?", 0.3),
        RubricItem("parameter_handling", "Are flags and arguments used correctly?", 0.25),
        RubricItem("error_recovery", "Does it handle missing deps/auth failures?", 0.2),
        RubricItem("safety_guards", "Does it prevent destructive operations?", 0.25),
    ),
    "knowledge": (
        RubricItem("factual_accuracy", "Are stated facts correct and verifiable?", 0.35),
        RubricItem("completeness", "Does it cover the topic adequately?", 0.25),
        RubricItem("consistency", "Are there internal contradictions?", 0.2),
        RubricItem("relevance", "Is the response focused on the query?", 0.2),
    ),
    "process": (
        RubricItem("structure", "Is the output well-organized?", 0.25),
        RubricItem("completeness", "Are all required sections present?", 0.25),
        RubricItem("actionability", "Are next steps clear and actionable?", 0.25),
        RubricItem("feasibility", "Are recommendations realistic?", 0.25),
    ),
    "security": (
        RubricItem("detection", "Does it identify real vulnerabilities?", 0.3),
        RubricItem("false_positive_rate", "Does it avoid false alarms?", 0.2),
        RubricItem("severity_accuracy", "Are severity ratings calibrated?", 0.2),
        RubricItem("remediation", "Are fix suggestions correct and complete?", 0.3),
    ),
}

DEFAULT_RUBRIC = RUBRICS["coding"]


def classify_skill(name: str, description: str) -> str:
    """Heuristic classification of a skill into a rubric category."""
    text = f"{name} {description}".lower()
    if any(w in text for w in ("security", "pentest", "vulnerab", "vuln", "audit", "threat")):
        return "security"
    if any(w in text for w in ("code", "coding", "review", "tdd", "test", "debug", "refactor")):
        return "coding"
    if any(w in text for w in ("tool", "cli", "command", "bash", "shell", "1password", "github")):
        return "tool"
    if any(w in text for w in ("knowledge", "ecosystem", "language", "docs", "learn")):
        return "knowledge"
    if any(w in text for w in ("plan", "architect", "design", "process", "workflow")):
        return "process"
    return "coding"


def rubric_for_skill(name: str, description: str) -> tuple[RubricItem, ...]:
    category = classify_skill(name, description)
    return RUBRICS.get(category, DEFAULT_RUBRIC)


def aggregate_critiques(
    skill_id: str,
    content_hash: str,
    critiques: list[CritiqueReport],
    timestamp: str = "",
) -> SkillScore:
    """Aggregate multiple critique reports into a single SkillScore.

    Uses integer-only scoring (1-10) converted to 0.0-1.0, with weighted
    arithmetic mean per the DRD.io evaluation pattern that prevents
    holistic discounts.
    """
    if not critiques:
        return SkillScore(skill_id=skill_id, content_hash=content_hash, overall=0.0)

    rubric_totals: dict[str, float] = {}
    rubric_counts: dict[str, int] = {}

    for c in critiques:
        for key, score in c.scores.items():
            rubric_totals[key] = rubric_totals.get(key, 0.0) + score
            rubric_counts[key] = rubric_counts.get(key, 0) + 1

    by_rubric: dict[str, float] = {}
    for key in rubric_totals:
        by_rubric[key] = rubric_totals[key] / rubric_counts[key]

    overall_scores = [c.overall_score for c in critiques]
    overall = sum(overall_scores) / len(overall_scores)

    by_category: dict[str, float] = {}
    for c in critiques:
        for fm in c.failure_modes:
            by_category[fm] = by_category.get(fm, 0.0) + 1.0

    return SkillScore(
        skill_id=skill_id,
        content_hash=content_hash,
        overall=round(overall, 4),
        by_category=by_category,
        by_rubric={k: round(v, 4) for k, v in by_rubric.items()},
        sample_size=len(critiques),
        timestamp=timestamp,
    )


def build_critique_prompt(
    skill_content: str,
    test_prompt: str,
    model_output: str,
    rubric: tuple[RubricItem, ...],
) -> str:
    """Build the structured critique prompt sent to fleet critics."""
    rubric_lines = "\n".join(
        f"- {r.key} (weight {r.weight}): {r.description}" for r in rubric
    )
    return f"""You are evaluating an AI agent skill's output quality. Score each rubric item from 1-10 (integers only).

## Skill Definition (system prompt being tested)
```
{skill_content[:4000]}
```

## Test Prompt
```
{test_prompt}
```

## Model Output Being Evaluated
```
{model_output[:6000]}
```

## Scoring Rubric
{rubric_lines}

## Response Format (JSON only, no other text)
{{
  "scores": {{"<rubric_key>": <1-10>, ...}},
  "overall_score": <1-10>,
  "failure_modes": ["<short description of each failure>"],
  "improvement_suggestions": ["<specific actionable suggestion>"]
}}"""
