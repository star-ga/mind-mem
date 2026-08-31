# Copyright 2026 STARGA, Inc.
"""Regression tests for the critique analyzer and the mutation validator.

Each test here pins a behaviour that the previous implementation got
wrong: a fabricated 0.0 for an unparseable critique, the model's own
output echoed back as the prompt it was answering, a perfect
inter-rater-agreement claim from a single rater, a configured minimum
critic panel that constrained nothing, and a per-rubric regression gate
that compared per-rubric means against a single overall score.
"""

from __future__ import annotations

import asyncio

import pytest

from mind_mem.skill_opt._types import (
    CritiqueReport,
    Mutation,
    SkillScore,
    SkillSpec,
)
from mind_mem.skill_opt._types import TestCase as _Case  # aliased: pytest collects Test* names
from mind_mem.skill_opt._types import TestResult as _Result
from mind_mem.skill_opt.analyzer import (
    PROMPT_UNAVAILABLE,
    CritiqueParseError,
    _parse_critique,
    aggregate_analysis,
    analyze_skill,
)
from mind_mem.skill_opt.config import SkillOptConfig
from mind_mem.skill_opt.fleet_bridge import FleetResponse

# ── helpers ─────────────────────────────────────────────────────


def _spec(content: str = "# Skill\nDo the thing.") -> SkillSpec:
    return SkillSpec(
        skill_id="test:skill",
        system="test",
        source_path="/tmp/skill.md",
        format="agent-md",
        name="reviewer",
        description="Reviews code",
        content=content,
    )


def _result(output: str, test_id: str = "T-1") -> _Result:
    return _Result(
        test_id=test_id,
        skill_id="test:skill",
        model="executor",
        output=output,
        latency_ms=1.0,
        timestamp="2026-01-01T00:00:00+00:00",
    )


def _case(prompt: str, test_id: str = "T-1") -> _Case:
    return _Case(
        test_id=test_id,
        skill_id="test:skill",
        category="correctness",
        prompt=prompt,
        expected_behavior="answers the question",
    )


class _StubFleet:
    """Returns canned critic responses and records the prompts it saw."""

    def __init__(self, responses: list[FleetResponse]) -> None:
        self._responses = responses
        self.prompts: list[str] = []

    async def query_excluding(self, prompt: str, exclude: set[str]) -> list[FleetResponse]:
        self.prompts.append(prompt)
        return list(self._responses)


def _critique_json(overall: int, **scores: int) -> str:
    items = ", ".join(f'"{k}": {v}' for k, v in scores.items())
    return f'{{"scores": {{{items}}}, "overall_score": {overall}}}'


# ── _parse_critique ─────────────────────────────────────────────


class TestParseCritique:
    def test_prose_prefixed_json_is_salvaged(self) -> None:
        """Unfenced JSON behind a preamble used to score a flat 0.0."""
        report = _parse_critique(
            'Sure, here is my critique: {"scores": {"correctness": 9}, "overall_score": 9}',
            "critic-a",
            "T-1",
        )
        assert report.overall_score == 0.9
        assert report.scores == {"correctness": 0.9}

    def test_fenced_json_still_parsed(self) -> None:
        report = _parse_critique(
            '```json\n{"scores": {"safety": 8}, "overall_score": 8}\n```',
            "critic-a",
            "T-1",
        )
        assert report.overall_score == 0.8

    def test_unparseable_response_raises(self) -> None:
        """An unreadable answer is missing evidence, not a score of zero."""
        with pytest.raises(CritiqueParseError):
            _parse_critique("I am unable to evaluate this.", "critic-a", "T-1")

    def test_non_object_json_raises(self) -> None:
        with pytest.raises(CritiqueParseError):
            _parse_critique("[1, 2, 3]", "critic-a", "T-1")


# ── analyze_skill ───────────────────────────────────────────────


class TestAnalyzeSkill:
    def test_unparseable_critique_is_dropped_not_scored_zero(self) -> None:
        fleet = _StubFleet(
            [
                FleetResponse(model="critic-a", content=_critique_json(9, correctness=9), latency_ms=1.0),
                FleetResponse(model="critic-b", content=_critique_json(9, correctness=9), latency_ms=1.0),
                FleetResponse(model="critic-c", content="sorry, I cannot", latency_ms=1.0),
            ]
        )
        critiques = asyncio.run(analyze_skill(_spec(), [_result("out")], fleet, min_critics=1))
        assert [c.critic_model for c in critiques] == ["critic-a", "critic-b"]
        assert aggregate_analysis(critiques)["consensus_score"] == 0.9

    def test_critic_sees_the_real_test_prompt(self) -> None:
        fleet = _StubFleet([FleetResponse(model="critic-a", content=_critique_json(8), latency_ms=1.0)])
        output = "Paris is a city in Europe. " * 40
        asyncio.run(
            analyze_skill(
                _spec(),
                [_result(output)],
                fleet,
                min_critics=1,
                test_cases=[_case("What is the capital of France?")],
            )
        )
        prompt = fleet.prompts[0]
        assert "## Test Prompt\n```\nWhat is the capital of France?\n```" in prompt
        assert output[:500] not in prompt.split("## Model Output Being Evaluated")[0]

    def test_missing_test_case_says_so_instead_of_echoing_output(self) -> None:
        fleet = _StubFleet([FleetResponse(model="critic-a", content=_critique_json(8), latency_ms=1.0)])
        output = "an answer that ignored the user entirely. " * 20
        asyncio.run(analyze_skill(_spec(), [_result(output)], fleet, min_critics=1))
        header = fleet.prompts[0].split("## Model Output Being Evaluated")[0]
        assert PROMPT_UNAVAILABLE in header
        assert output[:500] not in header


# ── aggregate_analysis ──────────────────────────────────────────


class TestAggregateAnalysis:
    def test_single_rater_agreement_is_undefined_not_perfect(self) -> None:
        critiques = [CritiqueReport(critic_model="critic-a", test_id="T-1", scores={"correctness": 0.9}, overall_score=0.9)]
        assert aggregate_analysis(critiques)["inter_rater_agreement"] is None

    def test_non_overlapping_rubric_keys_are_undefined(self) -> None:
        critiques = [
            CritiqueReport(critic_model="critic-a", test_id="T-1", scores={"correctness": 0.9}, overall_score=0.9),
            CritiqueReport(critic_model="critic-b", test_id="T-1", scores={"Correctness": 0.2}, overall_score=0.2),
        ]
        assert aggregate_analysis(critiques)["inter_rater_agreement"] is None

    def test_two_raters_on_one_item_do_get_a_number(self) -> None:
        critiques = [
            CritiqueReport(critic_model="critic-a", test_id="T-1", scores={"correctness": 0.9}, overall_score=0.9),
            CritiqueReport(critic_model="critic-b", test_id="T-1", scores={"correctness": 0.9}, overall_score=0.9),
        ]
        assert aggregate_analysis(critiques)["inter_rater_agreement"] == 1.0

    def test_reports_shortfall_against_min_critics(self) -> None:
        critiques = [CritiqueReport(critic_model="critic-a", test_id="T-1", overall_score=0.9)]
        out = aggregate_analysis(critiques, min_critics=3)
        assert out["n_critics"] == 1
        assert out["min_critics"] == 3
        assert out["critics_sufficient"] is False

    def test_no_critiques_is_not_sufficient(self) -> None:
        out = aggregate_analysis([], min_critics=1)
        assert out["critics_sufficient"] is False
        assert out["inter_rater_agreement"] is None


# ── validate_mutation ───────────────────────────────────────────


def _mutation(consensus: dict[str, float]) -> Mutation:
    return Mutation(
        mutation_id="M-1",
        skill_id="test:skill",
        original_hash="h0",
        proposed_content="# Skill\nDo the thing, carefully.",
        rationale="tighten",
        critic_consensus=consensus,
    )


def _run_validate(monkeypatch, critiques: list[CritiqueReport], mutation: Mutation, config: SkillOptConfig, **kwargs):
    from mind_mem.skill_opt import validator as validator_mod

    async def _fake_run_tests(spec, cases, fleet):
        return [_result("mutated output")]

    async def _fake_analyze(spec, results, fleet, min_critics=3, test_cases=None):
        return list(critiques)

    monkeypatch.setattr(validator_mod, "run_tests", _fake_run_tests)
    monkeypatch.setattr(validator_mod, "analyze_skill", _fake_analyze)
    return asyncio.run(validator_mod.validate_mutation(_spec(), mutation, [_case("q")], object(), config, **kwargs))


class TestValidateMutation:
    """The per-rubric gate must compare like with like.

    Original: overall 0.50, safety 0.90, edge_case_handling 0.30.
    Mutation: safety collapses to 0.55, edge_case_handling untouched.
    """

    _CONFIG = SkillOptConfig(min_critics=1, improvement_threshold=0.05, regression_threshold=0.10)
    _CRITIQUES = [
        CritiqueReport(
            critic_model="critic-a",
            test_id="T-1",
            scores={"safety": 0.55, "edge_case_handling": 0.30},
            overall_score=0.70,
        )
    ]
    _BASELINE = SkillScore(
        skill_id="test:skill",
        content_hash="h0",
        overall=0.50,
        by_rubric={"safety": 0.90, "edge_case_handling": 0.30},
    )

    def test_collapsed_category_is_caught_with_a_baseline(self, monkeypatch) -> None:
        result = _run_validate(
            monkeypatch,
            self._CRITIQUES,
            _mutation({"pre_mutation_score": 0.50}),
            self._CONFIG,
            original_score=self._BASELINE,
        )
        assert result.regression_categories == ("safety",)
        assert result.accepted is False

    def test_untouched_low_category_is_not_a_regression(self, monkeypatch) -> None:
        """edge_case_handling sits below the ORIGINAL overall but never moved."""
        result = _run_validate(
            monkeypatch,
            self._CRITIQUES,
            _mutation({"pre_mutation_score": 0.50}),
            self._CONFIG,
            original_score=self._BASELINE,
        )
        assert "edge_case_handling" not in result.regression_categories

    def test_no_regression_claimed_without_a_baseline(self, monkeypatch) -> None:
        result = _run_validate(
            monkeypatch,
            self._CRITIQUES,
            _mutation({"pre_mutation_score": 0.50}),
            self._CONFIG,
        )
        assert result.regression_categories == ()

    def test_baseline_can_ride_on_critic_consensus(self, monkeypatch) -> None:
        result = _run_validate(
            monkeypatch,
            self._CRITIQUES,
            _mutation({"pre_mutation_score": 0.50, "pre_rubric:safety": 0.90}),
            self._CONFIG,
        )
        assert result.regression_categories == ("safety",)

    def test_improvement_needs_the_configured_critic_panel(self, monkeypatch) -> None:
        """One critic cannot carry a decision configured to need three."""
        config = SkillOptConfig(min_critics=3, improvement_threshold=0.05, regression_threshold=0.10)
        result = _run_validate(
            monkeypatch,
            self._CRITIQUES,
            _mutation({"pre_mutation_score": 0.50}),
            config,
        )
        assert result.score_after > result.score_before
        assert result.improved is False
        assert result.accepted is False

    def test_full_panel_still_accepts_a_clean_improvement(self, monkeypatch) -> None:
        critiques = [CritiqueReport(critic_model=f"critic-{i}", test_id="T-1", scores={"safety": 0.95}, overall_score=0.90) for i in "abc"]
        config = SkillOptConfig(min_critics=3, improvement_threshold=0.05, regression_threshold=0.10)
        result = _run_validate(
            monkeypatch,
            critiques,
            _mutation({"pre_mutation_score": 0.50, "pre_rubric:safety": 0.90}),
            config,
            original_score=self._BASELINE,
        )
        assert result.improved is True
        assert result.regression_categories == ()
        assert result.accepted is True
