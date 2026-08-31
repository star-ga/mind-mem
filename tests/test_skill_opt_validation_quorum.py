"""The skill-mutation acceptance vote needs an actual electorate.

``ValidationResult.accepted`` gates whether a mutated skill is put up for
governance. Its "2/3 supermajority" had no floor on the panel: zero
votes satisfied ``0 >= 0`` and one vote was 100% of the electorate, so a
mutation could be accepted on the say-so of a single model — or of none.
"""

from __future__ import annotations

from mind_mem.skill_opt._types import ValidationResult


def _result(votes: dict[str, bool]) -> ValidationResult:
    return ValidationResult(
        mutation_id="m1",
        skill_id="s1",
        score_before=0.5,
        score_after=0.6,
        improved=True,
        regression_categories=(),
        critic_votes=votes,
    )


def test_no_critics_is_not_a_supermajority():
    assert _result({}).accepted is False


def test_one_critic_is_not_a_supermajority():
    assert _result({"critic-a": True}).accepted is False


def test_two_unanimous_critics_pass():
    assert _result({"critic-a": True, "critic-b": True}).accepted is True


def test_two_critics_split_one_one_fail():
    assert _result({"critic-a": True, "critic-b": False}).accepted is False


def test_two_of_three_pass():
    assert _result({"critic-a": True, "critic-b": True, "critic-c": False}).accepted is True


def test_two_of_four_fail():
    votes = {"critic-a": True, "critic-b": True, "critic-c": False, "critic-d": False}
    assert _result(votes).accepted is False


def test_three_of_four_pass():
    votes = {"critic-a": True, "critic-b": True, "critic-c": True, "critic-d": False}
    assert _result(votes).accepted is True


def test_quorum_does_not_rescue_an_unimproved_mutation():
    result = ValidationResult(
        mutation_id="m2",
        skill_id="s1",
        score_before=0.5,
        score_after=0.6,
        improved=False,
        critic_votes={"critic-a": True, "critic-b": True},
    )
    assert result.accepted is False


def test_quorum_does_not_rescue_a_regression():
    result = ValidationResult(
        mutation_id="m3",
        skill_id="s1",
        score_before=0.5,
        score_after=0.6,
        improved=True,
        regression_categories=("safety",),
        critic_votes={"critic-a": True, "critic-b": True},
    )
    assert result.accepted is False
