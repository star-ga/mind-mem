# Copyright 2026 STARGA, Inc.
"""Cross-model critique engine — the model that executed NEVER critiques itself."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ._types import CritiqueReport, SkillSpec, TestResult
from .fleet_bridge import FleetBridge
from .scorer import build_critique_prompt, rubric_for_skill


def _parse_critique(raw: str, critic_model: str, test_id: str) -> CritiqueReport:
    """Parse a JSON critique response from a fleet model."""
    text = raw.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return CritiqueReport(
            critic_model=critic_model,
            test_id=test_id,
            overall_score=0.0,
            raw_response=raw[:2000],
        )
    raw_scores = data.get("scores", {})
    scores: dict[str, float] = {}
    for k, v in raw_scores.items():
        try:
            scores[str(k)] = float(v) / 10.0
        except (TypeError, ValueError):
            pass
    try:
        overall = float(data.get("overall_score", 0)) / 10.0
    except (TypeError, ValueError):
        overall = 0.0

    return CritiqueReport(
        critic_model=critic_model,
        test_id=test_id,
        scores=scores,
        overall_score=round(overall, 4),
        failure_modes=tuple(str(f) for f in data.get("failure_modes", [])),
        improvement_suggestions=tuple(str(s) for s in data.get("improvement_suggestions", [])),
        raw_response=raw[:2000],
    )


async def analyze_skill(
    spec: SkillSpec,
    results: list[TestResult],
    fleet: FleetBridge,
    min_critics: int = 3,
) -> list[CritiqueReport]:
    """Send test results to fleet critics (excluding execution models)."""
    execution_models = {r.model for r in results}
    rubric = rubric_for_skill(spec.name, spec.description)

    critiques: list[CritiqueReport] = []
    for result in results:
        prompt = build_critique_prompt(
            skill_content=spec.content,
            test_prompt=result.output[:500],
            model_output=result.output,
            rubric=rubric,
        )
        responses = await fleet.query_excluding(prompt, exclude=execution_models)
        for resp in responses:
            if resp.ok:
                c = _parse_critique(resp.content, resp.model, result.test_id)
                critiques.append(c)
    return critiques


def aggregate_analysis(critiques: list[CritiqueReport]) -> dict[str, Any]:
    """Compute consensus analysis from multiple critique reports."""
    if not critiques:
        return {"consensus_score": 0.0, "failure_modes": {}, "actionable_gaps": [], "inter_rater_agreement": 0.0}

    overall_scores = [c.overall_score for c in critiques]
    consensus = sum(overall_scores) / len(overall_scores)

    fm_counter: Counter[str] = Counter()
    suggestions: Counter[str] = Counter()
    for c in critiques:
        for fm in c.failure_modes:
            fm_counter[fm] += 1
        for s in c.improvement_suggestions:
            suggestions[s] += 1

    n_critics = len({c.critic_model for c in critiques})
    majority = max(1, n_critics // 2)

    actionable = [fm for fm, count in fm_counter.most_common() if count >= majority]

    scores_by_key: dict[str, list[float]] = {}
    for c in critiques:
        for k, v in c.scores.items():
            scores_by_key.setdefault(k, []).append(v)
    agreement = _inter_rater_agreement(scores_by_key) if scores_by_key else 0.0

    return {
        "consensus_score": round(consensus, 4),
        "failure_modes": dict(fm_counter.most_common()),
        "actionable_gaps": actionable,
        "top_suggestions": [s for s, _ in suggestions.most_common(5)],
        "inter_rater_agreement": round(agreement, 4),
        "n_critics": n_critics,
        "n_critiques": len(critiques),
    }


def _inter_rater_agreement(scores_by_key: dict[str, list[float]]) -> float:
    """Simplified agreement metric: mean std dev across rubric items (lower=better)."""
    if not scores_by_key:
        return 0.0
    stds: list[float] = []
    for vals in scores_by_key.values():
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        stds.append(variance**0.5)
    if not stds:
        return 1.0
    avg_std = sum(stds) / len(stds)
    return max(0.0, 1.0 - avg_std)
