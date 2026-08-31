# Copyright 2026 STARGA, Inc.
"""Synthetic test case generation and fleet-based execution."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from ..observability import get_logger
from ._types import SkillSpec, TestCase, TestResult
from .fleet_bridge import FleetBridge
from .scorer import classify_skill

_log = get_logger("skill_opt.test_runner")

#: How many fleet seats :func:`generate_test_cases` asks before giving up.
#: One seat is a single point of failure — a timeout or an unparseable
#: answer yields zero cases — and the whole fleet is needless spend for a
#: generation step.
_MAX_GENERATION_SEATS = 3


class TestGenerationError(RuntimeError):
    """No fleet seat produced a usable test case.

    Raised instead of returning an empty list: zero cases run cleanly all
    the way to a "completed" run scoring the skill 0.0, which is a verdict
    on a skill that was never tested once.
    """

    # Pytest collects any class named ``Test*`` that appears in a test
    # module's namespace, including imported ones. This is its opt-out.
    __test__ = False


class FleetExecutionError(RuntimeError):
    """Test cases were supplied but no fleet seat answered any of them."""


# Test generation prompt templates per skill category
_GEN_TEMPLATES: dict[str, str] = {
    "coding": (
        "Generate {n} synthetic test prompts for an AI coding agent skill.\n"
        "Skill name: {name}\nSkill description: {description}\n\n"
        "Include: 1 correctness test, 1 security edge case, 1 format compliance test, "
        "and the rest as medium-difficulty coding tasks the skill should handle.\n\n"
        "Return JSON array: [{{"
        '"test_id": "<unique>", "category": "<correctness|safety|edge-case|format-compliance>", '
        '"prompt": "<the user prompt to send>", "expected_behavior": "<what good output looks like>", '
        '"rubric": ["<evaluation criterion>", ...], "difficulty": "<easy|medium|hard>"'
        "}}]"
    ),
    "tool": (
        "Generate {n} synthetic test prompts for an AI tool/CLI agent skill.\n"
        "Skill name: {name}\nSkill description: {description}\n\n"
        "Include: 1 correct command test, 1 missing dependency edge case, 1 destructive operation guard test.\n\n"
        "Return JSON array: [{{"
        '"test_id": "<unique>", "category": "<correctness|safety|edge-case|format-compliance>", '
        '"prompt": "<the user prompt>", "expected_behavior": "<expected>", '
        '"rubric": ["<criterion>", ...], "difficulty": "<easy|medium|hard>"'
        "}}]"
    ),
    "knowledge": (
        "Generate {n} factual recall / consistency test prompts for a knowledge skill.\n"
        "Skill name: {name}\nSkill description: {description}\n\n"
        "Include: 1 factual accuracy test, 1 contradiction detection test, 1 relevance test.\n\n"
        "Return JSON array with same schema as above."
    ),
    "process": (
        "Generate {n} test prompts for a process/planning agent skill.\n"
        "Skill name: {name}\nSkill description: {description}\n\n"
        "Include: 1 structure test, 1 completeness test, 1 actionability test.\n\n"
        "Return JSON array with same schema as above."
    ),
    "security": (
        "Generate {n} test prompts for a security audit agent skill.\n"
        "Skill name: {name}\nSkill description: {description}\n\n"
        "Include: 1 real vulnerability detection test, 1 false positive avoidance test, 1 remediation test.\n\n"
        "Return JSON array with same schema as above."
    ),
}


def _loads_array(text: str) -> list | None:
    """Parse *text* as a JSON array, or None if it is not one."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def _parse_test_cases(raw: str, skill_id: str) -> list[TestCase]:
    """Parse JSON test cases from a model response.

    The array is salvaged from surrounding prose whenever the response as a
    whole does not parse — not only when it carries a markdown fence. A
    model answering ``Here are 5 test prompts: [...]`` is ordinary, and
    gating the salvage on ``` threw away every case in that batch.
    """
    text = raw.strip()
    items = _loads_array(text)
    if items is None:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            items = _loads_array(text[start : end + 1])
    if items is None:
        return []
    cases: list[TestCase] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            # A case with no prompt cannot exercise anything; counting it
            # would pad the batch with a test that is never really run.
            continue
        raw_rubric = item.get("rubric", [])
        rubric = tuple(str(r) for r in raw_rubric) if isinstance(raw_rubric, (list, tuple)) else ()
        cases.append(
            TestCase(
                test_id=str(item.get("test_id", uuid.uuid4().hex[:12])),
                skill_id=skill_id,
                category=str(item.get("category", "correctness")),
                prompt=prompt,
                expected_behavior=str(item.get("expected_behavior", "")),
                rubric=rubric,
                difficulty=str(item.get("difficulty", "medium")),
            )
        )
    return cases


async def generate_test_cases(
    spec: SkillSpec,
    fleet: FleetBridge,
    count: int = 5,
) -> list[TestCase]:
    """Generate synthetic test cases for a skill using a fleet model.

    Seats are tried in order until one answers with at least one usable
    case, up to ``_MAX_GENERATION_SEATS``.

    Raises:
        TestGenerationError: no seat produced a usable case. Returning an
            empty list instead would run the whole pipeline over nothing —
            zero tests, zero critiques, consensus 0.0 — and record that
            0.0 against the skill as a completed run.
    """
    category = classify_skill(spec.name, spec.description)
    template = _GEN_TEMPLATES.get(category, _GEN_TEMPLATES["coding"])
    prompt = template.format(n=count, name=spec.name, description=spec.description)

    failures: list[str] = []
    for seat in fleet.available_models[:_MAX_GENERATION_SEATS]:
        for resp in await fleet.query(prompt, models=[seat]):
            if not resp.ok:
                reason = resp.error or "empty response"
            else:
                cases = _parse_test_cases(resp.content, spec.skill_id)
                if cases:
                    return cases[:count]
                reason = "no usable test case in response"
            failures.append(f"{resp.model}: {reason}")
            _log.warning(
                "skill_opt_test_generation_seat_failed",
                skill_id=spec.skill_id,
                model=resp.model,
                reason=reason,
            )
    detail = "; ".join(failures) if failures else "no fleet seat available"
    raise TestGenerationError(f"no usable test case generated for skill {spec.skill_id!r} ({detail})")


async def run_tests(
    spec: SkillSpec,
    cases: list[TestCase],
    fleet: FleetBridge,
    models: list[str] | None = None,
) -> list[TestResult]:
    """Execute test cases against fleet models with the skill as system prompt.

    Only a seat that actually answered yields a :class:`TestResult`. A
    transport failure is never laundered into one carrying ``ERROR: ...``
    as ``output``: :class:`TestResult` has no field marking itself failed,
    so the critics would score the transport error as the skill's answer
    and average that near-zero into the skill's consensus — letting a flaky
    network condemn a good skill and accept the rewrite whose only merit is
    that the retry did not time out. Failed seats are logged and dropped.

    Raises:
        FleetExecutionError: cases were supplied but no seat answered any of
            them. An empty result set scores 0.0 downstream, which would be
            a verdict on a skill the fleet never actually ran.
    """
    targets = models or fleet.available_models[:2]
    results: list[TestResult] = []
    failures: list[str] = []
    for case in cases:
        system_prompt = f"You are an AI agent with the following skill:\n\n{spec.content[:6000]}"
        full_prompt = f"[System]\n{system_prompt}\n\n[User]\n{case.prompt}"
        responses = await fleet.query(full_prompt, models=targets)
        now = datetime.now(timezone.utc).isoformat()
        for resp in responses:
            if not resp.ok:
                reason = resp.error or "empty response"
                failures.append(f"{resp.model}: {reason}")
                _log.warning(
                    "skill_opt_test_seat_failed",
                    skill_id=spec.skill_id,
                    test_id=case.test_id,
                    model=resp.model,
                    reason=reason,
                )
                continue
            results.append(
                TestResult(
                    test_id=case.test_id,
                    skill_id=spec.skill_id,
                    model=resp.model,
                    output=resp.content,
                    latency_ms=resp.latency_ms,
                    timestamp=now,
                )
            )
    if cases and not results:
        detail = "; ".join(failures) if failures else "no fleet seat available"
        raise FleetExecutionError(f"no fleet seat executed any test case for skill {spec.skill_id!r} ({detail})")
    return results
