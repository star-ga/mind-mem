"""Tests for the v3.3.0 answer-quality shims."""

from __future__ import annotations

import pytest

from mind_mem.answer_quality import (
    classify_question_category,
    prompt_for_category,
    self_consistency,
    verify_answer,
)


class TestVerifyAnswer:
    def test_empty_answer_fails(self) -> None:
        assert verify_answer("").passes is False
        assert verify_answer("   ").passes is False

    def test_matching_date_passes(self) -> None:
        r = verify_answer("7 May 2023", expected_pattern="date")
        assert r.passes is True
        assert "matched" in r.reason

    def test_missing_expected_pattern_fails(self) -> None:
        r = verify_answer("I don't remember exactly.", expected_pattern="date")
        assert r.passes is False
        assert r.suggested_pattern == "date"

    def test_infer_pattern_from_gold_sample(self) -> None:
        # Gold is "7 May 2023" — inferred pattern is "date".
        r = verify_answer(
            "She arrived on 7 May 2023.",
            gold_sample="7 May 2023",
        )
        assert r.passes is True

    def test_no_expected_pattern_no_gold_is_noop(self) -> None:
        r = verify_answer("Some free-text answer.")
        assert r.passes is True
        assert r.reason == "no_expected_pattern"

    def test_yes_no_pattern(self) -> None:
        assert verify_answer("No", expected_pattern="yes_no").passes is True
        assert verify_answer("Maybe", expected_pattern="yes_no").passes is False


class TestSelfConsistency:
    def test_zero_samples_rejected(self) -> None:
        with pytest.raises(ValueError):
            self_consistency("q", [], answerer=lambda q, e, s: "x", samples=0)

    def test_unanimous_answers(self) -> None:
        def answerer(question: str, evidence: list[dict], seed: int) -> str:
            return "7 May 2023"

        r = self_consistency("q", [], answerer=answerer, samples=5)
        assert r.winner == "7 May 2023"
        assert r.votes == 5
        assert r.confidence == pytest.approx(1.0)
        assert r.total_samples == 5

    def test_plurality_wins(self) -> None:
        # 3× "A", 2× "B"
        outputs = ["A", "B", "A", "B", "A"]

        def answerer(question: str, evidence: list[dict], seed: int) -> str:
            return outputs[seed % len(outputs)]

        r = self_consistency("q", [], answerer=answerer, samples=5)
        assert r.winner == "A"
        assert r.votes == 3
        assert r.confidence == pytest.approx(0.6)

    def test_case_insensitive_bucketing(self) -> None:
        outputs = ["PostgreSQL", "postgresql", "PostgreSQL", "postgresql", "PostgreSQL"]

        def answerer(question: str, evidence: list[dict], seed: int) -> str:
            return outputs[seed % len(outputs)]

        r = self_consistency("q", [], answerer=answerer, samples=5)
        assert r.votes == 5  # all bucketed together

    def test_failing_sample_is_skipped(self) -> None:
        def answerer(question: str, evidence: list[dict], seed: int) -> str:
            if seed == 2:
                raise RuntimeError("transient")
            return "answer"

        r = self_consistency("q", [], answerer=answerer, samples=5)
        # 4 successful samples, 1 skipped.
        assert r.total_samples == 4
        assert r.votes == 4

    def test_all_samples_fail(self) -> None:
        def answerer(question: str, evidence: list[dict], seed: int) -> str:
            raise RuntimeError("always fail")

        r = self_consistency("q", [], answerer=answerer, samples=3)
        assert r.winner == ""
        assert r.votes == 0
        assert r.confidence == 0.0


class TestCategoryPrompts:
    def test_temporal_template(self) -> None:
        p = prompt_for_category("temporal", "When did X happen?", timeline="2023-01-01 X happened")
        assert "TEMPORAL" in p
        assert "2023-01-01" in p

    def test_adversarial_template_includes_refusal_scaffold(self) -> None:
        p = prompt_for_category("adversarial", "Was X purple?", facts="none")
        assert "ADVERSARIAL" in p
        assert "No information" in p

    def test_multi_hop_template_requires_subqueries(self) -> None:
        p = prompt_for_category(
            "multi-hop",
            "What did A say after B left?",
            subqueries="1. What did A say?\n2. When did B leave?",
        )
        assert "MULTI-HOP" in p
        assert "B leave" in p

    def test_unknown_category_falls_back(self) -> None:
        p = prompt_for_category("mystery-category", "Q?")
        # Falls back to single-hop template.
        assert "SINGLE-HOP" in p

    def test_case_insensitive_category(self) -> None:
        p = prompt_for_category("TEMPORAL", "Q?")
        assert "TEMPORAL" in p


class TestClassifyCategory:
    def test_empty_question_returns_single_hop(self) -> None:
        assert classify_question_category("") == "single-hop"

    def test_temporal_detected(self) -> None:
        assert classify_question_category("When did it happen?") == "temporal"

    def test_multi_hop_detected(self) -> None:
        assert classify_question_category("What is the relationship between X and Y?") == "multi-hop"
