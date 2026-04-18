# Copyright 2026 STARGA, Inc.
"""Synthetic test case generation and fleet-based execution."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from ._types import SkillSpec, TestCase, TestResult
from .fleet_bridge import FleetBridge
from .scorer import classify_skill

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


def _parse_test_cases(raw: str, skill_id: str) -> list[TestCase]:
    """Parse JSON test cases from LLM response, tolerating markdown fences."""
    text = raw.strip()
    if "```" in text:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    try:
        items = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(items, list):
        return []
    cases: list[TestCase] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cases.append(
            TestCase(
                test_id=str(item.get("test_id", uuid.uuid4().hex[:12])),
                skill_id=skill_id,
                category=str(item.get("category", "correctness")),
                prompt=str(item.get("prompt", "")),
                expected_behavior=str(item.get("expected_behavior", "")),
                rubric=tuple(str(r) for r in item.get("rubric", [])),
                difficulty=str(item.get("difficulty", "medium")),
            )
        )
    return cases


async def generate_test_cases(
    spec: SkillSpec,
    fleet: FleetBridge,
    count: int = 5,
) -> list[TestCase]:
    """Generate synthetic test cases for a skill using a fleet model."""
    category = classify_skill(spec.name, spec.description)
    template = _GEN_TEMPLATES.get(category, _GEN_TEMPLATES["coding"])
    prompt = template.format(n=count, name=spec.name, description=spec.description)

    responses = await fleet.query(prompt, models=fleet.available_models[:1])
    cases: list[TestCase] = []
    for resp in responses:
        if resp.ok:
            cases.extend(_parse_test_cases(resp.content, spec.skill_id))
    return cases[:count]


async def run_tests(
    spec: SkillSpec,
    cases: list[TestCase],
    fleet: FleetBridge,
    models: list[str] | None = None,
) -> list[TestResult]:
    """Execute test cases against fleet models with the skill as system prompt."""
    targets = models or fleet.available_models[:2]
    results: list[TestResult] = []
    for case in cases:
        system_prompt = f"You are an AI agent with the following skill:\n\n{spec.content[:6000]}"
        full_prompt = f"[System]\n{system_prompt}\n\n[User]\n{case.prompt}"
        responses = await fleet.query(full_prompt, models=targets)
        now = datetime.now(timezone.utc).isoformat()
        for resp in responses:
            results.append(
                TestResult(
                    test_id=case.test_id,
                    skill_id=spec.skill_id,
                    model=resp.model,
                    output=resp.content if resp.ok else f"ERROR: {resp.error}",
                    latency_ms=resp.latency_ms,
                    timestamp=now,
                )
            )
    return results
