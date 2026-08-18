"""Regression gate for benchmarks/feedback_success_bench.py (Group I item 3).

Proves the 48-episode feedback-quality -> downstream-success bench is
byte-for-byte deterministic, clears conservative separation floors (set
below the authored-fixture result so this is a real regression gate, not
a flake), and that the flag-off surface stays honestly inert.

Run:
    pytest tests/test_feedback_success_bench.py -x
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

# Make the benchmarks/ helper importable without polluting sys.path
# permanently for the rest of the suite (same idiom as
# tests/test_train_mind_mem_4b.py and tests/test_grid_search.py).
_BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH_DIR))

import feedback_success_bench as bench  # noqa: E402

from mind_mem.retrieval_graph import feedback_quality_credit, recall_sufficiency  # noqa: E402

# Conservative floors, ~20% under the authored-fixture result committed in
# benchmarks/feedback_success_results.json (accuracy 1.0, separation
# 0.8096, success_rate_predicted_sufficient 1.0,
# success_rate_predicted_starved 0.0) -- a real regression gate, not a
# flake, in the LoCoMo-gate style (see test_recall_quality_locomo.py).
_FLOOR_ACCURACY = 0.9
_FLOOR_SEPARATION = 0.3
_FLOOR_SUCCESS_RATE_SUFFICIENT = 0.9
_CEILING_SUCCESS_RATE_STARVED = 0.2


def test_feedback_success_bench_deterministic_and_separating() -> None:
    """The bench must (1) run byte-identically twice, (2) clear the
    separation floors, and (3) prove the flag-off path stays inert."""
    # 1. Run twice, in-process.
    result_a = bench.run_bench()
    result_b = bench.run_bench()

    # 2. Deterministic: exact byte equality of the sorted JSON encoding.
    json_a = json.dumps(result_a, sort_keys=True)
    json_b = json.dumps(result_b, sort_keys=True)
    assert json_a == json_b, "run_bench() is not byte-for-byte deterministic across calls"

    # 3. Separates: conservative floors.
    assert result_a["accuracy"] >= _FLOOR_ACCURACY, f"accuracy {result_a['accuracy']} < floor {_FLOOR_ACCURACY}"
    assert result_a["separation"] >= _FLOOR_SEPARATION, f"separation {result_a['separation']} < floor {_FLOOR_SEPARATION}"
    assert result_a["success_rate_predicted_sufficient"] >= _FLOOR_SUCCESS_RATE_SUFFICIENT, (
        f"success_rate_predicted_sufficient {result_a['success_rate_predicted_sufficient']} < floor {_FLOOR_SUCCESS_RATE_SUFFICIENT}"
    )
    assert result_a["success_rate_predicted_starved"] <= _CEILING_SUCCESS_RATE_STARVED, (
        f"success_rate_predicted_starved {result_a['success_rate_predicted_starved']} > ceiling {_CEILING_SUCCESS_RATE_STARVED}"
    )

    # 4. Flag-off honesty: with feedback_credit disabled, recall_sufficiency
    # returns None for every episode -- the flagged path is what the bench
    # measures, and the flag-off surface stays a byte-identical no-op.
    cfg_disabled = {"feedback_credit": {"enabled": False}}
    for episode in bench.EPISODES:
        hits = copy.deepcopy(list(episode.hits))
        feedback_quality_credit(hits, "unused://flag-off-check", cfg_disabled)
        assert recall_sufficiency(hits, episode.intent_type) is None, (
            f"{episode.episode_id}: recall_sufficiency should be None with feedback_credit disabled"
        )


def test_episode_grid_shape() -> None:
    """Sanity: exactly 8 intent classes x 6 families = 48 episodes."""
    assert len(bench.EPISODES) == 48
    assert len({ep.intent_type for ep in bench.EPISODES}) == 8
    assert len({ep.family for ep in bench.EPISODES}) == 6
    per_family_counts = {family: 0 for family in {ep.family for ep in bench.EPISODES}}
    for episode in bench.EPISODES:
        per_family_counts[episode.family] += 1
    assert all(count == 8 for count in per_family_counts.values())


def test_run_bench_matches_committed_reference() -> None:
    """The committed benchmarks/feedback_success_results.json must match
    a fresh run() exactly -- catches silent drift between the shipped
    reference artifact and the live bench."""
    reference_path = _BENCH_DIR / "feedback_success_results.json"
    with reference_path.open(encoding="utf-8") as fh:
        committed = json.load(fh)

    fresh = bench.run_bench()
    assert json.dumps(fresh, sort_keys=True) == json.dumps(committed, sort_keys=True), (
        "benchmarks/feedback_success_results.json is stale — regenerate with `python benchmarks/feedback_success_bench.py`"
    )
