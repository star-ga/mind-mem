"""``self_consistency`` confidence must be votes over samples REQUESTED.

The docstring promised ``votes / samples``, but the divisor was the
number of samples that survived: an answerer that raised (or returned a
blank / non-string answer) was dropped from both the numerator and the
denominator. With ``samples=5`` and four failures the single surviving
answer scored 1.0 — indistinguishable from a genuine 5/5, and strictly
more confident than a real 3/5 — so a ``confidence >= 0.8`` acceptance
gate fired hardest exactly when sampling had collapsed. Nothing in the
result exposed the shortfall either.
"""

from __future__ import annotations

import pytest

from mind_mem.answer_quality import self_consistency


def _flaky(fail_seeds: set[int], answer: str = "7 May 2023"):
    def answerer(question: str, evidence: list[dict], seed: int) -> str:
        if seed in fail_seeds:
            raise RuntimeError("transient")
        return answer

    return answerer


@pytest.mark.unit
def test_a_collapsed_run_is_not_reported_as_certain() -> None:
    r = self_consistency("q", [], answerer=_flaky({1, 2, 3, 4}), samples=5)
    assert r.votes == 1
    assert r.total_samples == 1
    assert r.requested_samples == 5
    assert r.dropped_samples == 4
    assert r.confidence == pytest.approx(0.2)


@pytest.mark.unit
def test_a_collapsed_run_scores_below_a_real_majority() -> None:
    outputs = ["A", "A", "A", "B", "B"]

    def three_of_five(question: str, evidence: list[dict], seed: int) -> str:
        return outputs[seed % len(outputs)]

    genuine = self_consistency("q", [], answerer=three_of_five, samples=5)
    collapsed = self_consistency("q", [], answerer=_flaky({1, 2, 3, 4}), samples=5)
    assert genuine.confidence == pytest.approx(0.6)
    assert collapsed.confidence < genuine.confidence


@pytest.mark.unit
def test_a_blank_or_non_string_answer_counts_as_a_dropped_sample() -> None:
    def answerer(question: str, evidence: list[dict], seed: int):
        if seed == 0:
            return "   "
        if seed == 1:
            return None
        return "gamma"

    r = self_consistency("q", [], answerer=answerer, samples=4)
    assert r.winner == "gamma"
    assert r.votes == 2
    assert r.total_samples == 2
    assert r.requested_samples == 4
    assert r.dropped_samples == 2
    assert r.confidence == pytest.approx(0.5)


@pytest.mark.unit
def test_an_intact_unanimous_run_is_still_one() -> None:
    r = self_consistency("q", [], answerer=_flaky(set()), samples=5)
    assert r.confidence == pytest.approx(1.0)
    assert r.dropped_samples == 0
    assert r.requested_samples == 5


@pytest.mark.unit
def test_a_total_failure_reports_the_requested_count() -> None:
    def always_fails(question: str, evidence: list[dict], seed: int) -> str:
        raise RuntimeError("always")

    r = self_consistency("q", [], answerer=always_fails, samples=3)
    assert (r.winner, r.votes, r.total_samples, r.confidence) == ("", 0, 0, 0.0)
    assert r.requested_samples == 3
    assert r.dropped_samples == 3
