"""The placebo arm must be a fair match, or it is a second memory arm.

A placebo that is shorter than the real section, unframed, or drawn from the
same corpus does not control for anything — and it fails in the dangerous
direction: it makes a length effect look like a memory effect, or makes a real
memory effect look like noise. So every fairness property is asserted, and each
assertion is paired with a case that violates it.
"""

from __future__ import annotations

import pytest

from benchmarks.memory_ab_placebo import (
    ARM_PLACEBO,
    DEFAULT_TOLERANCE,
    PlaceboBuild,
    assert_placebo_fair,
    choose_donor,
)
from mind_mem.bench.ab_task import Task


def _task(task_id: str) -> Task:
    return Task(
        task_id=task_id,
        sha="a" * 40,
        parent_sha="b" * 40,
        memory_cutoff="2026-01-01T00:00:00Z",
        scoring_instant="2026-01-01",
        subject=f"subject {task_id}",
        task_statement="do the thing",
        tests_to_run=("tests/test_thing.py",),
        test_patch_paths=("tests/test_thing.py",),
        fail_to_pass=("tests/test_thing.py::test_it",),
    )


def _build(**kw: object) -> PlaceboBuild:
    base: dict = {
        "section": "NOTE: ... <evidence>donor text</evidence>\n\n",
        "tokens": 100,
        "target_tokens": 100,
        "block_ids": ("D-DONOR-1",),
        "donor_task_id": "t2",
        "within_tolerance": True,
        "tolerance": DEFAULT_TOLERANCE,
    }
    base.update(kw)
    return PlaceboBuild(**base)  # type: ignore[arg-type]


# -- donor selection --------------------------------------------------------


def test_donor_is_the_next_task_and_wraps() -> None:
    tasks = (_task("t1"), _task("t2"), _task("t3"))
    assert choose_donor(tasks, tasks[0]).task_id == "t2"
    assert choose_donor(tasks, tasks[2]).task_id == "t1", "donor selection must wrap, not fall off the end"


def test_donor_is_never_the_task_itself() -> None:
    tasks = (_task("t1"), _task("t2"), _task("t3"))
    for t in tasks:
        assert choose_donor(tasks, t).task_id != t.task_id


def test_donor_needs_a_set_and_a_member() -> None:
    with pytest.raises(ValueError):
        choose_donor((_task("t1"),), _task("t1"))
    with pytest.raises(ValueError):
        choose_donor((_task("t1"), _task("t2")), _task("absent"))


# -- fairness assertions ----------------------------------------------------


def test_a_fair_placebo_passes_and_names_what_it_checked() -> None:
    checks = assert_placebo_fair(_build(), ("D-REAL-1",), "NOTE: ... <evidence>real</evidence>")
    assert checks == (
        "no_block_overlap_with_memory_arm",
        "length_matched_within_tolerance",
        "same_framed_rendering",
    )


def test_sharing_a_block_with_the_memory_arm_is_refused() -> None:
    """Overlap means the placebo carries real signal — not a control."""
    with pytest.raises(AssertionError, match="shares blocks"):
        assert_placebo_fair(_build(block_ids=("D-REAL-1",)), ("D-REAL-1",), "<evidence>x</evidence>")


def test_a_length_mismatch_is_refused() -> None:
    """The whole point is matching length; an unmatched one is worse than none."""
    with pytest.raises(AssertionError, match="outside the"):
        assert_placebo_fair(_build(within_tolerance=False, tokens=20), ("D-REAL-1",), "<evidence>x</evidence>")


def test_an_unframed_placebo_against_a_framed_memory_arm_is_refused() -> None:
    """Matching a pre-framing rendering to a framed one restores the confound."""
    with pytest.raises(AssertionError, match="not matched"):
        assert_placebo_fair(_build(section="plain donor text, no framing"), ("D-REAL-1",), "<evidence>real</evidence>")


def test_framing_check_is_skipped_when_the_memory_arm_is_unframed() -> None:
    """If nothing is framed, an unframed placebo is the correct match."""
    checks = assert_placebo_fair(_build(section="plain donor text"), ("D-REAL-1",), "plain real text")
    assert "same_framed_rendering" in checks


def test_placebo_reports_its_own_length_gap() -> None:
    got = _build(tokens=112, target_tokens=100).as_dict()
    assert got["arm"] == ARM_PLACEBO
    assert got["length_gap_tokens"] == 12
    assert got["donor_task_id"] == "t2"
