"""Reps reduction, context poisoning, and disclosure — checked on known inputs.

Every number here is one a reader would act on, so each is tested against a
hand-built case whose answer is obvious by inspection, and each metric is
paired with a case where it comes out NON-zero. A poisoning rate that is
structurally always 0.0 would look like excellent news and mean nothing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from benchmarks.memory_ab_analysis import REDUCERS, build, load_runs, poisoning, reduce_cell, rep_instability


def _artifact(rows: list[tuple[str, bool, bool]]) -> dict[str, Any]:
    return {
        "results": [
            {"task_id": tid, "excluded": None, "outcome": {"memory_success": m, "control_success": c}}
            for tid, m, c in rows
        ]
    }


def _write(tmp_path: Any, name: str, rows: list[tuple[str, bool, bool]]) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(_artifact(rows)), encoding="utf-8")
    return str(path)


# -- reduction --------------------------------------------------------------


def test_majority_needs_a_strict_majority() -> None:
    """A cell that passes half the time is not a pass."""
    assert reduce_cell([True, True, False], "majority") is True
    assert reduce_cell([True, False], "majority") is False, "a 1-of-2 tie must not count as a pass"
    assert reduce_cell([False, False, True], "majority") is False
    assert reduce_cell([True], "majority") is True


def test_all_and_any_are_the_strict_and_generous_readings() -> None:
    assert reduce_cell([True, False], "all") is False
    assert reduce_cell([True, False], "any") is True
    assert reduce_cell([True, True], "all") is True
    assert reduce_cell([False, False], "any") is False


def test_unknown_reducer_and_empty_cell_are_refused() -> None:
    with pytest.raises(ValueError):
        reduce_cell([True], "mean")
    with pytest.raises(ValueError):
        reduce_cell([], "majority")


def test_the_reducer_actually_changes_the_answer(tmp_path: Any) -> None:
    """Positive control: if every reducer agreed, the choice would be cosmetic."""
    rows = [("t1", True, False)]
    a = _write(tmp_path, "r1.json", rows)
    b = _write(tmp_path, "r2.json", [("t1", False, False)])
    strict = build([a, b], reduce="all")
    generous = build([a, b], reduce="any")
    assert strict["summary"]["memory_successes"] == 0
    assert generous["summary"]["memory_successes"] == 1


# -- context poisoning ------------------------------------------------------


def test_poisoning_rate_counts_only_tasks_the_control_could_do() -> None:
    pairs = [
        (False, True),  # poisoned: control passed, memory failed
        (True, True),  # both passed
        (True, False),  # rescued
        (False, False),  # neither
    ]
    got = poisoning(pairs)
    assert got["passed_without_memory"] == 2
    assert got["poisoned"] == 1
    assert got["context_poisoning_rate"] == 0.5
    assert got["failed_without_memory"] == 2
    assert got["rescued"] == 1
    assert got["memory_rescue_rate"] == 0.5


def test_poisoning_rate_is_zero_when_nothing_is_poisoned() -> None:
    got = poisoning([(True, True), (True, False)])
    assert got["poisoned"] == 0
    assert got["context_poisoning_rate"] == 0.0


def test_poisoning_rate_is_none_when_the_control_passed_nothing() -> None:
    """No denominator is not a rate of zero, and must not be reported as one."""
    got = poisoning([(True, False), (False, False)])
    assert got["context_poisoning_rate"] is None
    assert got["passed_without_memory"] == 0


# -- rep stability ----------------------------------------------------------


def test_rep_instability_finds_disagreeing_reps() -> None:
    cells = {
        "t1": {"memory": [True, False], "control": [True, True]},
        "t2": {"memory": [True, True], "control": [False, False]},
    }
    got = rep_instability(cells)
    assert got["cells_with_multiple_reps"] == 4
    assert got["cells_that_disagreed_across_reps"] == 1
    assert got["unstable_fraction"] == 0.25
    assert got["reps_per_cell_observed"] == [2]


def test_single_rep_reports_no_stability_evidence() -> None:
    """With R=1 there is nothing to compare, and the field says so."""
    got = rep_instability({"t1": {"memory": [True], "control": [False]}})
    assert got["cells_with_multiple_reps"] == 0
    assert got["unstable_fraction"] is None
    assert got["reps_per_cell_observed"] == [1]


# -- loading and end to end -------------------------------------------------


def test_load_runs_groups_reps_by_task(tmp_path: Any) -> None:
    a = _write(tmp_path, "a.json", [("t1", True, False), ("t2", False, False)])
    b = _write(tmp_path, "b.json", [("t1", False, False)])
    cells = load_runs([a, b])
    assert cells["t1"]["memory"] == [True, False]
    assert cells["t2"]["memory"] == [False]


def test_excluded_records_are_dropped(tmp_path: Any) -> None:
    path = tmp_path / "x.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    {"task_id": "t1", "excluded": "setup_failed", "outcome": {"memory_success": True, "control_success": True}},
                    {"task_id": "t2", "excluded": None, "outcome": {"memory_success": False, "control_success": False}},
                ]
            }
        ),
        encoding="utf-8",
    )
    cells = load_runs([str(path)])
    assert "t1" not in cells
    assert "t2" in cells


def test_build_reports_the_paired_test_and_the_disclosure(tmp_path: Any) -> None:
    rows = [("t1", True, False), ("t2", False, True), ("t3", True, True), ("t4", False, False)]
    payload = build([_write(tmp_path, "one.json", rows)], reduce="majority", agent_cutoff="2026-04")
    assert payload["n_tasks"] == 4
    assert payload["summary"]["memory_successes"] == 2
    assert payload["summary"]["control_successes"] == 2
    assert payload["summary"]["n_discordant"] == 2
    assert payload["disclosure"]["agent_cutoff"] == "2026-04"
    assert payload["reduction"]["reducer"] == "majority"
    assert set(payload["reduction"]["known_reducers"]) == set(REDUCERS)


def test_an_unstated_cutoff_is_labelled_not_blank(tmp_path: Any) -> None:
    payload = build([_write(tmp_path, "one.json", [("t1", True, True)])])
    assert payload["disclosure"]["agent_cutoff"] == "UNSTATED"


# -- harness fingerprint and the length confound ----------------------------


def test_a_harness_change_after_the_first_artifact_is_flagged(tmp_path: Any) -> None:
    """The guard that stops two harness states being pooled as one experiment.

    Several seats share this tree, so "these runs all used the same harness"
    has to be checked, not promised. Touching a prompt-shaping file after an
    artifact exists must raise the flag.
    """
    import os
    import time

    from benchmarks.memory_ab_analysis import PROMPT_SHAPING_FILES, harness_fingerprint

    repo = tmp_path / "repo"
    for rel in PROMPT_SHAPING_FILES:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    artifact = tmp_path / "run.json"
    artifact.write_text("{}", encoding="utf-8")

    # Artifact written after the harness: the ordinary, clean case.
    now = time.time()
    for rel in PROMPT_SHAPING_FILES:
        os.utime(repo / rel, (now - 100, now - 100))
    os.utime(artifact, (now, now))
    clean = harness_fingerprint(str(repo), [str(artifact)])
    assert clean["harness_changed_after_first_artifact"] is False

    # A prompt-shaping file touched after the artifact: the flagged case.
    os.utime(repo / PROMPT_SHAPING_FILES[0], (now + 100, now + 100))
    dirty = harness_fingerprint(str(repo), [str(artifact)])
    assert dirty["harness_changed_after_first_artifact"] is True, "a mid-run harness change went unflagged"


def test_fingerprint_records_a_digest_per_prompt_shaping_file(tmp_path: Any) -> None:
    from benchmarks.memory_ab_analysis import PROMPT_SHAPING_FILES, harness_fingerprint

    repo = tmp_path / "repo"
    (repo / PROMPT_SHAPING_FILES[0]).parent.mkdir(parents=True, exist_ok=True)
    (repo / PROMPT_SHAPING_FILES[0]).write_text("content", encoding="utf-8")
    got = harness_fingerprint(str(repo), [])
    assert got["files"][PROMPT_SHAPING_FILES[0]]["present"] is True
    assert len(got["files"][PROMPT_SHAPING_FILES[0]]["sha256_16"]) == 16
    # A file that is not there is recorded as absent, never silently skipped.
    assert got["files"][PROMPT_SHAPING_FILES[1]]["present"] is False


def test_prompt_lengths_measures_the_gap_a_placebo_must_match(tmp_path: Any) -> None:
    """The length confound as a number, and always flagged as unresolved."""
    from benchmarks.memory_ab_analysis import prompt_lengths

    path = tmp_path / "len.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "task_id": "t1",
                        "arms": {
                            "memory": {"prompt": {"prompt_tokens": 1500}},
                            "control": {"prompt": {"prompt_tokens": 300}},
                        },
                    },
                    {
                        "task_id": "t2",
                        "arms": {
                            "memory": {"prompt": {"prompt_tokens": 1100}},
                            "control": {"prompt": {"prompt_tokens": 300}},
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    got = prompt_lengths([str(path)])
    assert got["n"] == 2
    assert got["mean_gap_tokens"] == 1000.0
    assert got["min_gap_tokens"] == 800
    assert got["max_gap_tokens"] == 1200
    assert got["placebo_required"] is True, "the placebo must never be reported as satisfied by this metric"


def test_prompt_lengths_still_demands_a_placebo_with_no_data(tmp_path: Any) -> None:
    """No measurement is not evidence that length does not matter."""
    from benchmarks.memory_ab_analysis import prompt_lengths

    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"results": []}), encoding="utf-8")
    got = prompt_lengths([str(path)])
    assert got["n"] == 0
    assert got["placebo_required"] is True
