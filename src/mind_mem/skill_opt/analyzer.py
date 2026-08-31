# Copyright 2026 STARGA, Inc.
"""Cross-model critique engine — the model that executed NEVER critiques itself."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ..observability import get_logger
from ._types import CritiqueReport, SkillSpec, TestCase, TestResult
from .fleet_bridge import FleetBridge
from .scorer import build_critique_prompt, rubric_for_skill

_log = get_logger("skill_opt.analyzer")

# Sent to a critic when the caller could not supply the prompt that produced
# the output under review. Echoing the output back in the prompt slot would
# make every prompt/response rubric item (relevance, completeness,
# format_compliance, command_accuracy) trivially self-consistent, so an
# answer that ignored the user entirely could never be scored irrelevant.
PROMPT_UNAVAILABLE = "(the prompt that produced this output was not recorded)"


class CritiqueParseError(ValueError):
    """A critic's response could not be read as a critique.

    Raised instead of returning a zero-scored report: an unparseable
    answer is missing evidence, not evidence of a bad skill, and
    counting it as 0.0 silently drags the consensus down.
    """


def _extract_json_object(text: str) -> dict[str, Any]:
    """Return the JSON object in *text*, tolerating fences and prose.

    Raises:
        CritiqueParseError: when no JSON object can be recovered.
    """
    stripped = text.strip()
    candidates = [stripped]
    # Models routinely wrap the object in ``` fences or lead with prose
    # ("Sure, here is my critique: {...}"), so always try the outermost
    # brace slice as well — not only when a fence happens to be present.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        sliced = stripped[start : end + 1]
        if sliced != stripped:
            candidates.append(sliced)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
        raise CritiqueParseError(f"critique response is a {type(data).__name__}, not a JSON object")
    raise CritiqueParseError("critique response contains no parseable JSON object")


def _parse_critique(raw: str, critic_model: str, test_id: str) -> CritiqueReport:
    """Parse a JSON critique response from a fleet model.

    Raises:
        CritiqueParseError: when the response is not a JSON object. The
            caller drops the critique (and says so) rather than folding a
            fabricated 0.0 into the consensus.
    """
    data = _extract_json_object(raw)

    raw_scores = data.get("scores", {})
    scores: dict[str, float] = {}
    if isinstance(raw_scores, dict):
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
    test_cases: list[TestCase] | None = None,
) -> list[CritiqueReport]:
    """Send test results to fleet critics (excluding execution models).

    Args:
        spec: The skill whose outputs are being judged.
        results: Test outputs to critique.
        fleet: Bridge used to query critics.
        min_critics: Size the critic panel is expected to reach. A
            shortfall is logged; callers that gate on evidence read it
            back from :func:`aggregate_analysis` via ``critics_sufficient``.
        test_cases: The cases ``results`` answered. Used to show each
            critic the ACTUAL prompt; without them the prompt slot says
            so explicitly (:data:`PROMPT_UNAVAILABLE`) instead of
            echoing the model's own output back as its prompt.

    Unparseable critic responses are dropped with a warning, not scored 0.
    """
    execution_models = {r.model for r in results}
    rubric = rubric_for_skill(spec.name, spec.description)
    prompts = {tc.test_id: tc.prompt for tc in (test_cases or ())}

    critiques: list[CritiqueReport] = []
    dropped = 0
    for result in results:
        prompt = build_critique_prompt(
            skill_content=spec.content,
            test_prompt=prompts.get(result.test_id) or PROMPT_UNAVAILABLE,
            model_output=result.output,
            rubric=rubric,
        )
        responses = await fleet.query_excluding(prompt, exclude=execution_models)
        for resp in responses:
            if not resp.ok:
                continue
            try:
                critiques.append(_parse_critique(resp.content, resp.model, result.test_id))
            except CritiqueParseError as exc:
                dropped += 1
                _log.warning(
                    "critique_unparseable",
                    skill_id=spec.skill_id,
                    critic_model=resp.model,
                    test_id=result.test_id,
                    error=str(exc),
                )

    if dropped:
        _log.warning("critiques_dropped", skill_id=spec.skill_id, dropped=dropped, kept=len(critiques))

    n_critics = len({c.critic_model for c in critiques})
    if n_critics < min_critics:
        _log.warning(
            "insufficient_critics",
            skill_id=spec.skill_id,
            required=min_critics,
            observed=n_critics,
            unparseable=dropped,
        )
    return critiques


def aggregate_analysis(critiques: list[CritiqueReport], min_critics: int = 1) -> dict[str, Any]:
    """Compute consensus analysis from multiple critique reports.

    ``critics_sufficient`` reports whether the panel that produced these
    critiques reached ``min_critics`` distinct models — the check the
    configured minimum exists for. ``inter_rater_agreement`` is ``None``
    when no rubric item was scored by two or more critics.
    """
    n_critics = len({c.critic_model for c in critiques})
    if not critiques:
        return {
            "consensus_score": 0.0,
            "failure_modes": {},
            "actionable_gaps": [],
            "top_suggestions": [],
            "inter_rater_agreement": None,
            "n_critics": 0,
            "n_critiques": 0,
            "min_critics": min_critics,
            "critics_sufficient": 0 >= min_critics,
        }

    overall_scores = [c.overall_score for c in critiques]
    consensus = sum(overall_scores) / len(overall_scores)

    fm_counter: Counter[str] = Counter()
    suggestions: Counter[str] = Counter()
    for c in critiques:
        for fm in c.failure_modes:
            fm_counter[fm] += 1
        for s in c.improvement_suggestions:
            suggestions[s] += 1

    majority = max(1, n_critics // 2)

    actionable = [fm for fm, count in fm_counter.most_common() if count >= majority]

    scores_by_key: dict[str, list[float]] = {}
    for c in critiques:
        for k, v in c.scores.items():
            scores_by_key.setdefault(k, []).append(v)
    agreement = _inter_rater_agreement(scores_by_key)

    return {
        "consensus_score": round(consensus, 4),
        "failure_modes": dict(fm_counter.most_common()),
        "actionable_gaps": actionable,
        "top_suggestions": [s for s, _ in suggestions.most_common(5)],
        "inter_rater_agreement": round(agreement, 4) if agreement is not None else None,
        "n_critics": n_critics,
        "n_critiques": len(critiques),
        "min_critics": min_critics,
        "critics_sufficient": n_critics >= min_critics,
    }


def _inter_rater_agreement(scores_by_key: dict[str, list[float]]) -> float | None:
    """Simplified agreement metric: mean std dev across rubric items (lower=better).

    Returns ``None`` when no rubric item was scored by two or more
    critics — with nothing to compare, agreement is UNDEFINED. Reporting
    1.0 there would claim the strongest possible cross-model consensus
    for a run in which no two raters ever scored the same item.
    """
    stds: list[float] = []
    for vals in scores_by_key.values():
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        stds.append(variance**0.5)
    if not stds:
        return None
    avg_std = sum(stds) / len(stds)
    return max(0.0, 1.0 - avg_std)
