"""``repeats`` below two silently deleted the determinism guarantee.

``_arm`` compares ``results[1:]`` against ``results[0]``, so with one run per
arm ``stable`` is ``all()`` over an empty sequence — True unconditionally — and
the ``nondeterministic_grading`` drop in ``validate`` becomes structurally
unreachable. ``--repeats`` is operator-settable and its value is interpolated
verbatim into the published artifact, which then reads "Each arm runs 1 times
… dropped as nondeterministic_grading": prose asserting a check the run did not
perform. ``--repeats 0`` was worse again — the ``max(1, repeats)`` floor
executed one run while the artifact recorded ``"repeats": 0``.

``validate`` now refuses the value instead of producing a hollow task set.
"""

from __future__ import annotations

from typing import Any

import pytest

from mind_mem.bench import repo_task_mining as mining
from mind_mem.bench import repo_task_validation as validation


def _candidate(sha: str = "a" * 40) -> Any:
    return mining.Candidate(
        sha=sha,
        parent_sha="b" * 40,
        committed_at="2026-01-01T00:00:00Z",
        parent_committed_at="2025-12-31T00:00:00Z",
        subject="fix(x): y",
        added_test_files=("tests/test_x.py",),
        test_patch_paths=("tests/test_x.py",),
        src_changed=("src/mind_mem/x.py",),
        files_changed=("src/mind_mem/x.py", "tests/test_x.py"),
    )


@pytest.fixture()
def never_runs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Trip-wire on the expensive path: a refusal must not extract any tree."""
    calls: list[int] = []
    monkeypatch.setattr(validation, "_arm", lambda *a, **k: calls.append(1) or (None, True))
    monkeypatch.setattr(validation, "extract_tree", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(validation, "run_pytest", lambda *a, **k: calls.append(1))
    return calls


@pytest.mark.parametrize("repeats", [1, 0, -3])
def test_validate_refuses_a_repeat_count_that_cannot_check_determinism(
    tmp_path: Any,
    never_runs: list[int],
    repeats: int,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        validation.validate("repo", _candidate(), str(tmp_path / "wd"), "py", repeats=repeats)
    assert "nondeterministic_grading" in str(excinfo.value)
    assert str(validation.MIN_REPEATS) in str(excinfo.value)
    assert never_runs == [], "the refusal must come before any tree is built"


def test_the_documented_minimum_is_two() -> None:
    """Two is the smallest count at which ``results[1:]`` is non-empty."""
    assert validation.MIN_REPEATS == 2


def test_the_default_is_accepted(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The floor must not reject the value every real caller passes."""
    parent = validation.RunResult(1, False, {"t::a": "FAILED"}, "")
    task = validation.RunResult(0, False, {"t::a": "PASSED"}, "")
    calls = iter([(parent, True), (task, True)])
    monkeypatch.setattr(validation, "_arm", lambda *a, **k: next(calls))
    result = validation.validate("repo", _candidate(), str(tmp_path / "wd"), "py")
    assert result.well_formed
    assert result.drop_reason is None


def test_a_single_run_arm_reports_stable_vacuously(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mechanism itself, pinned so the floor cannot be removed as cosmetic.

    ``_arm`` is a lower-level helper and one run there is legitimate; what is
    not legitimate is *publishing* a determinism verdict derived from it. This
    documents why the boundary check lives in ``validate``.
    """
    monkeypatch.setattr(validation, "extract_tree", lambda *a, **k: None)
    monkeypatch.setattr(validation, "shutil", validation.shutil)
    pending = iter([validation.RunResult(1, False, {"t::a": "FAILED"}, "")])
    monkeypatch.setattr(validation, "run_pytest", lambda *a, **k: next(pending))
    _first, stable = validation._arm("repo", "sha", ["tests/t.py"], "/tmp/nowhere", "x", "py", 10, 1)
    assert stable is True, "one run has nothing to disagree with — this is the vacuous True"
