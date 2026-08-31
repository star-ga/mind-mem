# Copyright 2026 STARGA, Inc.
"""Tests for skill_opt.test_runner — generation parsing and seat failures."""

from __future__ import annotations

import asyncio

import pytest

from mind_mem.skill_opt._types import SkillSpec
from mind_mem.skill_opt.fleet_bridge import FleetResponse
from mind_mem.skill_opt.test_runner import (
    FleetExecutionError,
    TestGenerationError,
    _parse_test_cases,
    generate_test_cases,
    run_tests,
)

_CASES_JSON = (
    '[{"test_id": "t1", "category": "correctness", "prompt": "do the thing", '
    '"expected_behavior": "it works", "rubric": ["clear"], "difficulty": "easy"}]'
)


class _FakeFleet:
    """Duck-typed stand-in for FleetBridge with scripted per-seat answers."""

    def __init__(self, answers: dict[str, list[FleetResponse]]) -> None:
        # answers: model -> queue of responses, popped one per query
        self._answers = {m: list(r) for m, r in answers.items()}
        self.queried: list[list[str]] = []

    @property
    def available_models(self) -> list[str]:
        return list(self._answers)

    async def query(self, prompt: str, models: list[str] | None = None) -> list[FleetResponse]:
        targets = list(self._answers) if models is None else models
        self.queried.append(list(targets))
        out: list[FleetResponse] = []
        for m in targets:
            queue = self._answers.get(m)
            if queue:
                out.append(queue.pop(0))
        return out


def _spec() -> SkillSpec:
    return SkillSpec(
        skill_id="S-1",
        system="hub",
        source_path="/tmp/s.md",
        format="markdown",
        name="demo",
        description="a demo skill",
        content="do things well",
    )


def _ok(model: str, content: str) -> FleetResponse:
    return FleetResponse(model=model, content=content, latency_ms=1.0)


def _dead(model: str, error: str = "timeout") -> FleetResponse:
    return FleetResponse(model=model, content="", latency_ms=1.0, error=error)


class TestParseTestCases:
    def test_salvages_array_from_unfenced_prose(self) -> None:
        """Regression: salvage used to be gated on a markdown fence, so an
        answer with a plain prose preamble parsed to zero cases."""
        cases = _parse_test_cases(f"Here are 5 test prompts: {_CASES_JSON}", "S-1")
        assert [c.test_id for c in cases] == ["t1"]
        assert cases[0].prompt == "do the thing"

    def test_still_parses_bare_array_and_fenced_array(self) -> None:
        assert len(_parse_test_cases(_CASES_JSON, "S-1")) == 1
        assert len(_parse_test_cases(f"```json\n{_CASES_JSON}\n```", "S-1")) == 1

    def test_unparseable_response_yields_no_cases(self) -> None:
        assert _parse_test_cases("no json at all", "S-1") == []

    def test_drops_cases_with_no_prompt(self) -> None:
        raw = '[{"test_id": "t1", "prompt": "   "}, {"test_id": "t2", "prompt": "real"}]'
        assert [c.test_id for c in _parse_test_cases(raw, "S-1")] == ["t2"]

    def test_non_list_rubric_does_not_explode_into_characters(self) -> None:
        raw = '[{"test_id": "t1", "prompt": "p", "rubric": "clear"}]'
        assert _parse_test_cases(raw, "S-1")[0].rubric == ()


class TestGenerateTestCases:
    def test_raises_when_no_seat_produces_a_case(self) -> None:
        """Regression: an unusable answer used to return [], and the whole
        pipeline then scored the skill 0.0 without running a single test."""
        fleet = _FakeFleet({"a": [_ok("a", "sorry, I cannot help with that")]})
        with pytest.raises(TestGenerationError) as exc:
            asyncio.run(generate_test_cases(_spec(), fleet, count=3))
        assert "S-1" in str(exc.value)

    def test_raises_when_every_seat_is_dead(self) -> None:
        fleet = _FakeFleet({"a": [_dead("a")], "b": [_dead("b", "http 500")]})
        with pytest.raises(TestGenerationError) as exc:
            asyncio.run(generate_test_cases(_spec(), fleet, count=3))
        assert "timeout" in str(exc.value)

    def test_raises_when_fleet_has_no_seats(self) -> None:
        with pytest.raises(TestGenerationError):
            asyncio.run(generate_test_cases(_spec(), _FakeFleet({}), count=3))

    def test_falls_through_to_the_next_seat(self) -> None:
        """Regression: only ``available_models[:1]`` was ever asked."""
        fleet = _FakeFleet({"a": [_dead("a")], "b": [_ok("b", _CASES_JSON)]})
        cases = asyncio.run(generate_test_cases(_spec(), fleet, count=3))
        assert [c.test_id for c in cases] == ["t1"]
        assert fleet.queried == [["a"], ["b"]]

    def test_first_usable_seat_stops_the_walk(self) -> None:
        fleet = _FakeFleet({"a": [_ok("a", _CASES_JSON)], "b": [_ok("b", _CASES_JSON)]})
        asyncio.run(generate_test_cases(_spec(), fleet, count=3))
        assert fleet.queried == [["a"]]


class TestRunTests:
    def test_failed_seat_is_not_recorded_as_model_output(self) -> None:
        """Regression: a dead seat produced TestResult(output='ERROR: ...'),
        which the critics then scored as the skill's own answer."""
        fleet = _FakeFleet({"a": [_dead("a")], "b": [_ok("b", "a real answer")]})
        cases = _parse_test_cases(_CASES_JSON, "S-1")
        results = asyncio.run(run_tests(_spec(), cases, fleet, models=["a", "b"]))
        assert [r.model for r in results] == ["b"]
        assert [r.output for r in results] == ["a real answer"]
        assert not any(r.output.startswith("ERROR:") for r in results)

    def test_raises_when_no_seat_answers(self) -> None:
        fleet = _FakeFleet({"a": [_dead("a")], "b": [_dead("b", "http 500")]})
        cases = _parse_test_cases(_CASES_JSON, "S-1")
        with pytest.raises(FleetExecutionError) as exc:
            asyncio.run(run_tests(_spec(), cases, fleet, models=["a", "b"]))
        assert "http 500" in str(exc.value)

    def test_no_cases_is_not_an_execution_failure(self) -> None:
        fleet = _FakeFleet({"a": [_ok("a", "unused")]})
        assert asyncio.run(run_tests(_spec(), [], fleet, models=["a"])) == []
