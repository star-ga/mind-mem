# Copyright 2026 STARGA, Inc.
"""Computed abstention -- the pure core (statistic, bands, policy, gating).

``recall`` returns a ranked list and leaves "is any of it a clear match?" to
the caller's discipline. :mod:`mind_mem.recall_abstention` derives that
judgement from the score distribution the engine already computed.

These tests pin the contract the roadmap item states:

* one statistic per SET, scale-free, in ``[0, 1]`` -- how separated the top
  candidate is from the runner-up, not a per-block score;
* a first-class verdict that tells "five weak matches" from "one clear match";
* three bands (``act`` / ``proceed_with_caution`` / ``do_not_act``) whose
  boundary scales with what the caller intends to do with the answer;
* a request probe that is invisible when the answer is "no".

Negative assertions carry a positive control; the grid and threshold tests are
the ones a loosened comparison or a swapped use table turns red.
"""

from __future__ import annotations

import builtins
import copy
import random
import time
from typing import Any

import pytest

from mind_mem.recall_abstention import (
    BAND_ACT,
    BAND_CAUTION,
    BAND_DO_NOT_ACT,
    BANDS,
    DEFAULT_THRESHOLDS,
    DEFAULT_USE,
    INTENDED_USES,
    UNOPPOSED_SEPARATION,
    USE_ANSWER,
    USE_GOVERNED_WRITE,
    USE_SUMMARY,
    abstention_requested,
    assess,
    resolve_policy,
    separation_statistic,
    validate_intended_use,
)

_RANK = {BAND_DO_NOT_ACT: 0, BAND_CAUTION: 1, BAND_ACT: 2}


def _hits(*scores: Any, **extra: Any) -> list[dict[str, Any]]:
    return [{"_id": f"D-20260101-{i:03d}", "score": s, **extra} for i, s in enumerate(scores, 1)]


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------


def test_empty_set_has_zero_separation() -> None:
    assert separation_statistic([]) == 0.0


def test_single_candidate_is_the_unopposed_prior() -> None:
    """One candidate has nothing to be separated FROM -- neither evidence of a
    win nor of ambiguity -- so the statistic is the declared neutral prior."""
    assert separation_statistic([17.9]) == UNOPPOSED_SEPARATION
    assert 0.0 < UNOPPOSED_SEPARATION < 1.0


def test_known_answer_two_candidates() -> None:
    # (24.885 - 3.994) / 24.885 -- the top candidate dwarfs the runner-up.
    assert separation_statistic([24.885, 3.994]) == pytest.approx(0.8395, abs=1e-4)


def test_known_answer_is_rounded_to_four_places() -> None:
    value = separation_statistic([3.0, 1.0])  # 2/3
    assert value == 0.6667
    assert round(value, 4) == value


def test_exact_tie_is_zero_separation() -> None:
    assert separation_statistic([5.0, 5.0]) == 0.0
    assert separation_statistic([5.0, 5.0, 1.0]) == 0.0


def test_flat_distribution_is_nearly_zero() -> None:
    flat = [10.0 - 0.05 * i for i in range(10)]
    assert separation_statistic(flat) < 0.01


def test_concentrated_distribution_is_high() -> None:
    assert separation_statistic([10.0, 1.0, 0.9, 0.8, 0.7]) >= 0.9


def test_only_the_top_two_matter_for_the_gap() -> None:
    """The statistic is the top-vs-runner-up gap: a long tail below the
    runner-up cannot move it."""
    assert separation_statistic([9.0, 3.0]) == separation_statistic([9.0, 3.0, 2.9, 0.1, 0.0])


def test_order_independent() -> None:
    scores = [4.0, 11.5, 2.25, 11.0, 0.5, 7.75]
    expected = separation_statistic(scores)
    rng = random.Random(1234)
    for _ in range(25):
        shuffled = list(scores)
        rng.shuffle(shuffled)
        assert separation_statistic(shuffled) == expected


def test_scale_invariant() -> None:
    """A BM25 number and an RRF number describe the same shape the same way."""
    base = [12.0, 7.0, 3.0]
    ref = separation_statistic(base)
    for factor in (1e-6, 0.033, 1.0, 1000.0):
        assert separation_statistic([s * factor for s in base]) == pytest.approx(ref, abs=1e-4)


def test_bounded_in_unit_interval_over_a_deterministic_sweep() -> None:
    rng = random.Random(7)
    for _ in range(500):
        n = rng.randint(0, 12)
        scores = [rng.choice([0.0, rng.random() * 50, rng.random() * 0.05]) for _ in range(n)]
        value = separation_statistic(scores)
        assert 0.0 <= value <= 1.0, (scores, value)


def test_all_zero_scores_carry_no_evidence() -> None:
    assert separation_statistic([0.0, 0.0, 0.0]) == 0.0


@pytest.mark.parametrize(
    "junk",
    [float("nan"), float("inf"), float("-inf"), None, "9.9", True, False, object(), [1.0]],
)
def test_unusable_scores_are_dropped_not_counted(junk: Any) -> None:
    """A score that is not a finite real number is not a candidate.

    The base set is small on purpose (0.5, 0.2): ``True`` would count as 1.0 and
    ``"9.9"`` as 9.9, and either would take the top slot and move the statistic,
    so a junk value that is wrongly admitted cannot hide behind a larger score.
    """
    base = separation_statistic([0.5, 0.2])
    assert base == pytest.approx(0.6, abs=1e-4)  # positive control: a real, non-trivial value
    assert separation_statistic([0.5, 0.2, junk]) == base
    # ...and it is not counted as a candidate either (``nan`` cannot move the
    # statistic, so the count is the only thing that catches it).
    assert assess(_hits(0.5, 0.2, junk), intended_use=USE_ANSWER)["candidates"] == 2


def test_negative_scores_clamp_to_zero_rather_than_inflating() -> None:
    assert separation_statistic([3.0, -5.0]) == 1.0
    assert separation_statistic([-1.0, -2.0]) == 0.0


def test_none_input_is_an_empty_set() -> None:
    assert separation_statistic(None) == 0.0  # type: ignore[arg-type]


def test_large_set_is_linear_and_deterministic() -> None:
    rng = random.Random(99)
    scores = [rng.random() * 100 for _ in range(50_000)]
    start = time.perf_counter()
    first = separation_statistic(scores)
    elapsed = time.perf_counter() - start
    assert first == separation_statistic(list(reversed(scores)))
    assert elapsed < 2.0, f"top-2 selection over 50k scores took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Bands and intended use
# ---------------------------------------------------------------------------


def test_vocabulary_is_three_bands_and_three_uses_strictest_last() -> None:
    assert BANDS == (BAND_ACT, BAND_CAUTION, BAND_DO_NOT_ACT)
    assert INTENDED_USES == (USE_SUMMARY, USE_ANSWER, USE_GOVERNED_WRITE)
    assert DEFAULT_USE == USE_ANSWER
    assert set(DEFAULT_THRESHOLDS) == set(INTENDED_USES)


def test_empty_set_abstains_for_every_use() -> None:
    for use in INTENDED_USES:
        result = assess([], intended_use=use)
        assert result["band"] == BAND_DO_NOT_ACT
        assert result["abstain"] is True
        assert result["candidates"] == 0
        assert result["separation"] == 0.0
        assert result["reason"] == "no_candidates"


def test_clear_winner_acts_for_every_use() -> None:
    for use in INTENDED_USES:
        result = assess(_hits(30.0, 2.0, 1.5), intended_use=use)
        assert result["band"] == BAND_ACT, use
        assert result["abstain"] is False
        assert result["reason"] == "top_candidate_clearly_separated"


def test_flat_set_is_do_not_act_for_every_use() -> None:
    flat = _hits(*[10.0 - 0.02 * i for i in range(8)])
    for use in INTENDED_USES:
        result = assess(flat, intended_use=use)
        assert result["band"] == BAND_DO_NOT_ACT, use
        assert result["abstain"] is True
        assert result["candidates"] == 8
        assert result["reason"] == "no_clear_winner"


def test_five_weak_matches_differ_from_one_clear_match() -> None:
    """The exact distinction the roadmap item names: both are a list of five
    today; the response must tell them apart."""
    clear = assess(_hits(40.0, 5.0, 4.0, 3.0, 2.0), intended_use=USE_ANSWER)
    weak = assess(_hits(8.1, 8.0, 7.9, 7.8, 7.7), intended_use=USE_ANSWER)
    assert clear["candidates"] == weak["candidates"] == 5
    assert clear["band"] == BAND_ACT and clear["abstain"] is False
    assert weak["band"] == BAND_DO_NOT_ACT and weak["abstain"] is True
    assert clear["separation"] > weak["separation"]


def test_single_candidate_is_caution_for_answers_and_writes() -> None:
    """An unopposed hit cannot show it is a clear winner among alternatives."""
    one = _hits(18.0)
    assert assess(one, intended_use=USE_SUMMARY)["band"] == BAND_ACT
    for use in (USE_ANSWER, USE_GOVERNED_WRITE):
        result = assess(one, intended_use=use)
        assert result["band"] == BAND_CAUTION, use
        assert result["reason"] == "single_candidate"
        assert result["separation"] == UNOPPOSED_SEPARATION


def test_zero_top_score_is_do_not_act_even_with_candidates() -> None:
    result = assess(_hits(0.0, 0.0), intended_use=USE_SUMMARY)
    assert result["band"] == BAND_DO_NOT_ACT
    assert result["reason"] == "no_positive_score"
    assert result["candidates"] == 2


@pytest.mark.parametrize("use", INTENDED_USES)
def test_thresholds_are_inclusive_at_the_act_and_caution_edges(use: str) -> None:
    act_min, caution_min = DEFAULT_THRESHOLDS[use]

    def at(separation: float) -> dict[str, Any]:
        # top=10000 so (top - second)/top has a clean 4-place value.
        second = round(10000 * (1.0 - separation))
        return assess(_hits(10000.0, float(second), 1.0), intended_use=use)

    assert at(act_min)["band"] == BAND_ACT
    assert at(act_min - 0.0001)["band"] == BAND_CAUTION
    assert at(caution_min)["band"] == BAND_CAUTION
    assert at(caution_min - 0.0001)["band"] == BAND_DO_NOT_ACT


def test_stricter_use_is_never_looser_over_the_whole_range() -> None:
    """A recall feeding a governed write needs at least as much separation as
    one feeding a summary -- at EVERY point, not just the defaults' edges."""
    saw_difference = False
    for step in range(0, 10001):
        separation = step / 10000
        second = 10000 * (1.0 - separation)
        hits = _hits(10000.0, second, 0.5)
        ranks = [_RANK[assess(hits, intended_use=use)["band"]] for use in INTENDED_USES]
        assert ranks == sorted(ranks, reverse=True), (separation, ranks)
        if ranks[0] > ranks[-1]:
            saw_difference = True
    # Positive control: the sweep is not vacuously monotone because every use
    # gives the same answer.
    assert saw_difference


def test_assess_reports_the_thresholds_it_applied() -> None:
    result = assess(_hits(10.0, 5.0), intended_use=USE_GOVERNED_WRITE)
    act_min, caution_min = DEFAULT_THRESHOLDS[USE_GOVERNED_WRITE]
    assert result["thresholds"] == {"act": act_min, "caution": caution_min}
    assert result["intended_use"] == USE_GOVERNED_WRITE


def test_default_use_applies_when_none_is_given() -> None:
    result = assess(_hits(10.0, 5.0))
    assert result["intended_use"] == DEFAULT_USE


def test_assess_shape_is_exact() -> None:
    result = assess(_hits(10.0, 5.0), intended_use=USE_SUMMARY)
    assert set(result) == {
        "abstain",
        "band",
        "separation",
        "candidates",
        "intended_use",
        "thresholds",
        "reason",
        "basis",
    }
    assert isinstance(result["abstain"], bool)
    assert result["band"] in BANDS
    assert isinstance(result["candidates"], int)


def test_assess_does_not_mutate_its_input() -> None:
    hits = _hits(10.0, 5.0, 4.0)
    snapshot = copy.deepcopy(hits)
    assess(hits, intended_use=USE_GOVERNED_WRITE)
    assert hits == snapshot


def test_guardrail_hits_are_constraints_not_evidence() -> None:
    """A force-surfaced guardrail is prepended whatever its score, so it says
    nothing about whether the question was answered."""
    evidence = _hits(9.0, 8.9, 8.8)
    guardrail = [{"_id": "G-1", "score": 500.0, "guardrail": True}]
    with_guardrail = assess(guardrail + evidence, intended_use=USE_ANSWER)
    without = assess(evidence, intended_use=USE_ANSWER)
    assert with_guardrail["separation"] == without["separation"]
    assert with_guardrail["band"] == without["band"] == BAND_DO_NOT_ACT
    assert with_guardrail["candidates"] == without["candidates"] == 3
    # Positive control: if the guardrail WERE counted it would flip the verdict.
    counted = assess(_hits(500.0, 9.0, 8.9, 8.8), intended_use=USE_ANSWER)
    assert counted["band"] == BAND_ACT


def test_non_dict_and_scoreless_hits_are_ignored() -> None:
    hits: list[Any] = [None, "x", 7, {"_id": "no-score"}, {"_id": "bad", "score": "n/a"}, *_hits(12.0, 3.0)]
    result = assess(hits, intended_use=USE_ANSWER)
    assert result["candidates"] == 2
    assert result["separation"] == pytest.approx(0.75, abs=1e-4)


def test_unicode_and_hostile_ids_do_not_matter() -> None:
    hits = [{"_id": "D-‮\U0001f600'; DROP TABLE blocks;--", "score": 9.0}, {"_id": "\x00", "score": 2.0}]
    result = assess(hits, intended_use=USE_ANSWER)
    assert result["candidates"] == 2
    assert result["band"] == BAND_ACT


def test_basis_is_rrf_when_hits_are_rrf_fused() -> None:
    fused = _hits(0.0328, 0.0161, fusion="rrf")
    assert assess(fused, intended_use=USE_SUMMARY)["basis"] == "rrf"
    assert assess(_hits(10.0, 5.0), intended_use=USE_SUMMARY, basis="sqlite")["basis"] == "sqlite"
    assert assess(_hits(10.0, 5.0), intended_use=USE_SUMMARY)["basis"] == "unspecified"


def test_rrf_agreement_structure_is_scored_like_any_other_shape() -> None:
    """Both legs ranking one hit first while the runner-up appears in a single
    leg yields ~0.51 -- clear for a summary, cautious for answers and writes."""
    both = 2 / 61
    single = 1 / 62
    hits = _hits(both, single, 1 / 63, fusion="rrf")
    assert assess(hits, intended_use=USE_SUMMARY)["band"] == BAND_ACT
    assert assess(hits, intended_use=USE_ANSWER)["band"] == BAND_CAUTION
    assert assess(hits, intended_use=USE_GOVERNED_WRITE)["band"] == BAND_CAUTION
    # Adjacent ranks in the SAME legs are a near-tie -- no clear winner.
    tie = _hits(2 / 61, 2 / 62, 2 / 63, fusion="rrf")
    assert assess(tie, intended_use=USE_SUMMARY)["band"] == BAND_DO_NOT_ACT


# ---------------------------------------------------------------------------
# Policy (config)
# ---------------------------------------------------------------------------


def test_defaults_when_config_absent_or_empty() -> None:
    for cfg in (None, {}, {"abstention": {}}):
        policy = resolve_policy(cfg)
        assert policy.enabled is False
        assert policy.default_use == DEFAULT_USE
        assert policy.warnings == ()
        for use in INTENDED_USES:
            assert policy.thresholds[use] == DEFAULT_THRESHOLDS[use]


def test_partial_override_replaces_only_what_it_names() -> None:
    cfg = {"abstention": {"bands": {"governed_write": {"act": 0.9}}}}
    policy = resolve_policy(cfg)
    assert policy.warnings == ()
    assert policy.thresholds[USE_GOVERNED_WRITE][0] == 0.9
    assert policy.thresholds[USE_GOVERNED_WRITE][1] == DEFAULT_THRESHOLDS[USE_GOVERNED_WRITE][1]
    assert policy.thresholds[USE_SUMMARY] == DEFAULT_THRESHOLDS[USE_SUMMARY]
    # The override is applied, not merely parsed.
    hits = _hits(10.0, 2.0)  # separation 0.8
    assert assess(hits, intended_use=USE_GOVERNED_WRITE)["band"] == BAND_ACT
    assert assess(hits, intended_use=USE_GOVERNED_WRITE, policy=policy)["band"] == BAND_CAUTION


def test_zero_caution_edge_never_turns_no_evidence_into_caution() -> None:
    """A policy may legally set caution to 0.0; an empty or all-zero set must
    still read as "do not act" -- the guard is explicit, not an accident of the
    shipped numbers."""
    policy = resolve_policy({"abstention": {"bands": {"summary": {"act": 0.0, "caution": 0.0}}}})
    assert policy.warnings == ()
    assert assess([], intended_use=USE_SUMMARY, policy=policy)["band"] == BAND_DO_NOT_ACT
    assert assess(_hits(0.0, 0.0), intended_use=USE_SUMMARY, policy=policy)["band"] == BAND_DO_NOT_ACT
    # Positive control: the same policy does act on real evidence.
    assert assess(_hits(5.0, 5.0), intended_use=USE_SUMMARY, policy=policy)["band"] == BAND_ACT


def test_default_use_can_be_configured() -> None:
    policy = resolve_policy({"abstention": {"default_use": "governed_write"}})
    assert policy.default_use == USE_GOVERNED_WRITE
    assert assess(_hits(10.0, 5.0), policy=policy)["intended_use"] == USE_GOVERNED_WRITE


@pytest.mark.parametrize(
    "bad_cfg",
    [
        # a governed write looser than a summary: the one ordering the item forbids
        {"abstention": {"bands": {"governed_write": {"act": 0.1, "caution": 0.05}}}},
        {"abstention": {"bands": {"summary": {"act": 0.95}}}},
        # caution above act -- chosen so the monotonic-use rule is NOT also violated
        # (0.72/0.80 are both >= the answer edges), i.e. this case isolates the
        # caution<=act check instead of passing because a different check fires
        {"abstention": {"bands": {"governed_write": {"act": 0.72, "caution": 0.8}}}},
        {"abstention": {"bands": {"summary": {"act": 0.2, "caution": 0.3}}}},
        # out of range / wrong type
        {"abstention": {"bands": {"answer": {"act": 1.5}}}},
        {"abstention": {"bands": {"answer": {"act": -0.1}}}},
        {"abstention": {"bands": {"answer": {"act": "high"}}}},
        {"abstention": {"bands": {"answer": {"act": True}}}},
        {"abstention": {"bands": {"answer": {"act": float("nan")}}}},
        {"abstention": {"bands": "strict"}},
        {"abstention": {"bands": {"answer": 0.5}}},
        {"abstention": {"default_use": "yolo"}},
        {"abstention": "on"},
    ],
)
def test_invalid_policy_falls_back_to_defaults_and_says_so(bad_cfg: dict[str, Any]) -> None:
    policy = resolve_policy(bad_cfg)
    assert policy.warnings, bad_cfg
    for use in INTENDED_USES:
        assert policy.thresholds[use] == DEFAULT_THRESHOLDS[use], (use, bad_cfg)
    assert policy.default_use == DEFAULT_USE


def test_unknown_use_keys_are_reported_not_silently_honoured() -> None:
    policy = resolve_policy({"abstention": {"bands": {"paranoid": {"act": 0.99, "caution": 0.9}}}})
    assert policy.warnings
    assert "paranoid" not in policy.thresholds


def test_default_thresholds_are_themselves_valid_and_ordered() -> None:
    previous = (0.0, 0.0)
    for use in INTENDED_USES:
        act_min, caution_min = DEFAULT_THRESHOLDS[use]
        assert 0.0 <= caution_min <= act_min <= 1.0
        assert act_min >= previous[0] and caution_min >= previous[1]
        previous = (act_min, caution_min)
    # The unopposed prior must not clear the strictest act edge: a lone hit may
    # never be "act" for a governed write.
    assert UNOPPOSED_SEPARATION < DEFAULT_THRESHOLDS[USE_GOVERNED_WRITE][0]


# ---------------------------------------------------------------------------
# Request gating -- invisible when the answer is "no"
# ---------------------------------------------------------------------------


def test_requested_truth_table() -> None:
    assert abstention_requested(None, None) is False
    assert abstention_requested({}, None) is False
    assert abstention_requested({"abstention": {"enabled": False}}, None) is False
    assert abstention_requested({"abstention": {"enabled": True}}, None) is True
    # Only a literal true enables: "true", 1 and "yes" are not opt-ins.
    for loose in ("true", 1, "yes", [True], {"x": 1}):
        assert abstention_requested({"abstention": {"enabled": loose}}, None) is False, loose
    # An explicit intended use is itself the request.
    assert abstention_requested({}, "summary") is True
    assert abstention_requested(None, "governed_write") is True
    # Empty strings are "not given".
    assert abstention_requested({}, "") is False


@pytest.mark.parametrize("cfg", [None, 5, "x", [], {"abstention": None}, {"abstention": 3}, {"abstention": []}])
def test_malformed_config_reads_as_off_without_raising(cfg: Any) -> None:
    assert abstention_requested(cfg, None) is False


def test_the_off_probe_touches_nothing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Rule 2 of the wiring discipline: a feature probe must not log, read a
    file or read a clock when the answer is no."""

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the OFF probe performed I/O or read a clock")

    monkeypatch.setattr(builtins, "open", boom)
    monkeypatch.setattr(time, "time", boom)
    monkeypatch.setattr(time, "monotonic", boom)
    assert abstention_requested({"abstention": {"enabled": False}}, None) is False
    assert abstention_requested(None, None) is False
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_validate_intended_use() -> None:
    assert validate_intended_use(None) is None
    assert validate_intended_use("") is None
    assert validate_intended_use("   ") is None
    assert validate_intended_use(" Governed_Write ") == USE_GOVERNED_WRITE
    assert validate_intended_use("SUMMARY") == USE_SUMMARY
    for bad in ("delete_everything", "write", "gov-ernance", 7, ["summary"], b"summary"):
        with pytest.raises(ValueError) as err:
            validate_intended_use(bad)  # type: ignore[arg-type]
        for use in INTENDED_USES:
            assert use in str(err.value)


def test_validation_error_echoes_a_bounded_excerpt() -> None:
    with pytest.raises(ValueError) as err:
        validate_intended_use("x" * 5000)
    assert len(str(err.value)) < 300


def test_deterministic_across_repeats() -> None:
    rng = random.Random(5)
    hits = _hits(*[rng.random() * 20 for _ in range(30)])
    first = assess(hits, intended_use=USE_ANSWER)
    for _ in range(5):
        assert assess(hits, intended_use=USE_ANSWER) == first
