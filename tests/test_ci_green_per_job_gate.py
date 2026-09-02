"""The CI-green gate must read JOBS, not just the workflow run's conclusion.

The hole this file closes
-------------------------
``scripts/check_ci_green.py`` used to read only the workflow RUN conclusion. A
job marked ``continue-on-error`` fails visibly red while the run it belongs to
still concludes ``success`` -- so a red matrix row read as a pass, and the
operator's rule ("a red row is NOT green and must block a version bump") was
unenforced.

Positive controls, in both directions, on real captured data:

* run 33511997100 / ba4cd4e1 -- run ``success`` and all 31 jobs ``success``:
  the gate MUST pass it, or every rejection below is vacuous;
* run 33579619488 / 1ec7f63 (the current HEAD) -- run ``failure`` with
  ``test (windows-latest, 3.10)`` red: still rejected;
* run 33566252435 / a085081 (the commit 5.0.1 shipped from) -- five red Windows
  rows: still rejected;
* the constructed advisory-red run -- run ``success`` with one red
  ``continue-on-error`` job: the OLD run-level logic accepts it and the NEW
  gate rejects it. ``test_the_old_run_level_logic_accepted_what_the_gate_now_rejects``
  asserts BOTH halves, so it fails if the new leg is removed AND if the fixture
  ever stops being dangerous.

Why the dangerous case is constructed rather than captured: it does not occur
naturally in this repository's recorded history. All 83 success-concluded CI
runs in the last 200 (2026-05-24 .. 2026-09-02) have every one of their jobs
green, so there was nothing to capture. Its job list is nevertheless the real,
unaltered list from green run 33511997100 with exactly one row set to the
conclusion a real red ``continue-on-error`` row carries -- measured in run
33386877502, where ``test (windows-latest, 3.14)`` reports
``conclusion: failure`` at the job level.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
RUNS_FIXTURE = FIXTURES / "ci_runs_by_sha.json"
JOBS_FIXTURE = FIXTURES / "ci_jobs_by_run.json"
ADVISORY_FIXTURE = FIXTURES / "ci_jobs_advisory_red.json"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

SHA_GREEN = "ba4cd4e12de5091b38d971b0ee921f47effd2acd"  # run 33511997100, every job success
SHA_HEAD_RED = "1ec7f63e011443eca2e63299b08c51241fa09915"  # run 33579619488, windows 3.10 red
SHA_5_0_1 = "a0850815782277b17fbd1243fd0f084d51a0e651"  # run 33566252435, five windows rows red
SHA_NO_RUN = "313134b66d99b9ebaed8af77fb0092a6b5115c53"  # on main, zero runs recorded
SHA_ADVISORY = "c0ffeec0ffeec0ffeec0ffeec0ffeec0ffeeabcd"  # constructed: run green, one job red

RUN_GREEN = 33511997100
RUN_HEAD_RED = 33579619488
RUN_ADVISORY = 90000000001

ADVISORY_RED_JOB = "test (ubuntu-latest, 3.14)"


def _load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / f"{name}.py"
    assert path.is_file(), f"missing release gate script: {path}"
    spec = importlib.util.spec_from_file_location(f"_per_job_gate_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_script("check_ci_green")


def _runs(path: Path, sha: str) -> list[Any]:
    return gate._load_runs_file(str(path), sha)


def _jobs(path: Path, run_id: int) -> list[Any]:
    return gate._load_jobs_file(str(path), run_id)


def _provider(path: Path):
    """A jobs provider that reads a captured payload instead of the network."""

    def provide(run: dict[str, Any]) -> Any:
        return gate._load_jobs_file(str(path), run.get("id", "?"))

    return provide


def _evaluate(runs_path: Path, jobs_path: Path, sha: str) -> tuple[bool, str]:
    return gate.evaluate(_runs(runs_path, sha), sha, _provider(jobs_path))


class TestFixturesAreWhatTheyClaim:
    """A negative assertion needs proof the thing it looks for exists.

    Every rejection test below is meaningless if the fixtures are empty, or if
    the "dangerous" one is not actually dangerous.
    """

    def test_the_green_fixture_really_is_all_green(self) -> None:
        jobs = _jobs(JOBS_FIXTURE, RUN_GREEN)
        assert len(jobs) == 31, "capture drifted; the positive control no longer covers the whole workflow"
        assert [j["name"] for j in jobs if j["conclusion"] != "success"] == []

    def test_the_head_fixture_really_has_a_red_job(self) -> None:
        jobs = _jobs(JOBS_FIXTURE, RUN_HEAD_RED)
        red = sorted(j["name"] for j in jobs if j["conclusion"] != "success")
        assert red == ["test (windows-latest, 3.10)"]

    def test_the_advisory_fixture_really_is_the_dangerous_shape(self) -> None:
        """Run says success; exactly one job says failure; it is an advisory row."""
        runs = _runs(ADVISORY_FIXTURE, SHA_ADVISORY)
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"
        assert runs[0]["conclusion"] == "success", "the fixture stops being dangerous if the run is not green"
        jobs = _jobs(ADVISORY_FIXTURE, RUN_ADVISORY)
        red = sorted(j["name"] for j in jobs if j["conclusion"] != "success")
        assert red == [ADVISORY_RED_JOB], "exactly one red job, and it must be a continue-on-error matrix row"
        assert "3.14" in ADVISORY_RED_JOB

    def test_every_fixture_job_carries_the_fields_the_gate_reads(self) -> None:
        for path, run_id in ((JOBS_FIXTURE, RUN_GREEN), (ADVISORY_FIXTURE, RUN_ADVISORY)):
            for job in _jobs(path, run_id):
                assert set(("name", "status", "conclusion")) <= set(job), job


class TestTheRunVersusJobHole:
    """The whole point: a green run can contain a red job."""

    def test_the_old_run_level_logic_accepted_what_the_gate_now_rejects(self) -> None:
        """Both halves in one test, so neither can rot unnoticed.

        ``gate.decide`` is the run-level leg, unchanged from the version that
        shipped before this fix -- it is literally the code that used to be the
        whole gate. It still says green here. ``gate.evaluate`` is the gate as
        it now stands, and it says red.
        """
        runs = _runs(ADVISORY_FIXTURE, SHA_ADVISORY)

        old_green, old_detail = gate.decide(runs, SHA_ADVISORY)
        assert old_green is True, "the fixture must be one the run-level check passes, or it proves nothing"
        assert "success" in old_detail

        new_green, new_detail = gate.evaluate(runs, SHA_ADVISORY, _provider(ADVISORY_FIXTURE))
        assert new_green is False
        assert ADVISORY_RED_JOB in new_detail
        assert "continue-on-error" in new_detail

    def test_the_cli_rejects_the_advisory_red_run(self) -> None:
        rc = gate.main(["--sha", SHA_ADVISORY, "--runs-file", str(ADVISORY_FIXTURE), "--jobs-file", str(ADVISORY_FIXTURE)])
        assert rc == 1


class TestEveryExistingBehaviourSurvives:
    """Widening the gate must not have dropped anything it already caught."""

    def test_accepts_the_real_green_commit(self) -> None:
        green, detail = _evaluate(RUNS_FIXTURE, JOBS_FIXTURE, SHA_GREEN)
        assert green is True, detail
        assert "33511997100" in detail
        assert "31 jobs" in detail

    def test_rejects_the_current_head(self) -> None:
        green, detail = _evaluate(RUNS_FIXTURE, JOBS_FIXTURE, SHA_HEAD_RED)
        assert green is False
        assert "33579619488" in detail

    def test_rejects_the_commit_5_0_1_shipped_from(self) -> None:
        green, detail = _evaluate(RUNS_FIXTURE, JOBS_FIXTURE, SHA_5_0_1)
        assert green is False
        assert "33566252435" in detail

    def test_no_recorded_run_is_still_a_failure(self) -> None:
        green, detail = _evaluate(RUNS_FIXTURE, JOBS_FIXTURE, SHA_NO_RUN)
        assert green is False
        assert "no CI run" in detail

    def test_an_unfinished_run_is_still_undeterminable(self) -> None:
        runs = copy.deepcopy(_runs(RUNS_FIXTURE, SHA_GREEN))
        runs[0]["status"] = "in_progress"
        runs[0]["conclusion"] = None
        green, detail = gate.evaluate(runs, SHA_GREEN, _provider(JOBS_FIXTURE))
        assert green is False
        assert "not finished" in detail

    def test_an_unreadable_run_payload_still_fails_closed(self) -> None:
        with pytest.raises(gate.GateError):
            gate.evaluate(["not-an-object"], SHA_GREEN, _provider(JOBS_FIXTURE))

    def test_cli_exit_codes_match_the_decision(self) -> None:
        args = ["--runs-file", str(RUNS_FIXTURE), "--jobs-file", str(JOBS_FIXTURE)]
        assert gate.main(["--sha", SHA_GREEN, *args]) == 0
        assert gate.main(["--sha", SHA_HEAD_RED, *args]) == 1
        assert gate.main(["--sha", SHA_5_0_1, *args]) == 1
        assert gate.main(["--sha", SHA_NO_RUN, *args]) == 1


class TestJobConclusions:
    """Only ``success`` is evidence. ``skipped`` and ``cancelled`` are not."""

    @pytest.mark.parametrize(
        "conclusion",
        [
            pytest.param("skipped", id="skipped-ran-nothing"),
            pytest.param("cancelled", id="cancelled-never-answered"),
            "failure",
            "timed_out",
            "neutral",
            "action_required",
            "stale",
            "startup_failure",
            None,
        ],
    )
    def test_a_non_success_job_fails_the_gate(self, conclusion: str | None) -> None:
        jobs = copy.deepcopy(_jobs(JOBS_FIXTURE, RUN_GREEN))
        jobs[0]["conclusion"] = conclusion
        green, detail = gate.decide_jobs(jobs, RUN_GREEN)
        assert green is False
        assert jobs[0]["name"] in detail
        assert repr(conclusion) in detail

    def test_an_all_success_job_list_passes(self) -> None:
        """Positive control for the parametrised rejections above."""
        green, detail = gate.decide_jobs(_jobs(JOBS_FIXTURE, RUN_GREEN), RUN_GREEN)
        assert green is True
        assert "31" in detail

    def test_an_unfinished_job_is_undeterminable(self) -> None:
        jobs = copy.deepcopy(_jobs(JOBS_FIXTURE, RUN_GREEN))
        jobs[3]["status"] = "in_progress"
        jobs[3]["conclusion"] = None
        green, detail = gate.decide_jobs(jobs, RUN_GREEN)
        assert green is False
        assert "have not finished" in detail
        assert jobs[3]["name"] in detail

    def test_a_queued_job_does_not_read_as_a_missing_conclusion_error(self) -> None:
        jobs = copy.deepcopy(_jobs(JOBS_FIXTURE, RUN_GREEN))
        del jobs[2]["conclusion"]
        jobs[2]["status"] = "queued"
        green, _ = gate.decide_jobs(jobs, RUN_GREEN)
        assert green is False

    def test_a_run_with_no_jobs_is_not_a_pass(self) -> None:
        """The empty-search trap: no red jobs found because nothing was found."""
        green, detail = gate.decide_jobs([], RUN_GREEN)
        assert green is False
        assert "no jobs" in detail

    def test_every_red_job_is_named_not_just_the_first(self) -> None:
        jobs = _jobs(JOBS_FIXTURE, 33566252435)
        green, detail = gate.decide_jobs(jobs, 33566252435)
        assert green is False
        for name in ("3.10", "3.11", "3.12", "3.13", "3.14"):
            assert f"test (windows-latest, {name})" in detail

    @pytest.mark.parametrize(
        "jobs",
        [
            pytest.param({"jobs": []}, id="not-an-array"),
            pytest.param(["not-an-object"], id="entry-not-an-object"),
            pytest.param([{"status": "completed", "conclusion": "success"}], id="entry-missing-name"),
            pytest.param([{"name": "x", "status": "completed"}], id="completed-without-a-conclusion-field"),
        ],
    )
    def test_unreadable_job_payloads_fail_closed(self, jobs: Any) -> None:
        with pytest.raises(gate.GateError):
            gate.decide_jobs(jobs, RUN_GREEN)

    def test_a_jobs_provider_that_cannot_answer_fails_closed(self) -> None:
        """A provider that raises must not be swallowed into a pass."""

        def broken(run: dict[str, Any]) -> Any:
            raise gate.GateError("api unreadable")

        with pytest.raises(gate.GateError):
            gate.evaluate(_runs(RUNS_FIXTURE, SHA_GREEN), SHA_GREEN, broken)

    def test_the_cli_fails_closed_when_the_job_payload_is_missing(self, tmp_path: Path) -> None:
        rc = gate.main(["--sha", SHA_GREEN, "--runs-file", str(RUNS_FIXTURE), "--jobs-file", str(tmp_path / "nope.json")])
        assert rc == 1


class TestOfflineModeCannotDropALeg:
    """``--runs-file`` alone would silently restore the run-level-only gate."""

    def test_runs_file_without_jobs_file_is_refused(self) -> None:
        assert gate.main(["--sha", SHA_GREEN, "--runs-file", str(RUNS_FIXTURE)]) == 1

    def test_jobs_file_without_runs_file_is_refused(self) -> None:
        assert gate.main(["--sha", SHA_GREEN, "--jobs-file", str(JOBS_FIXTURE)]) == 1


class TestRunSelectionIsShared:
    """The judged run and the fetched job list must be the same run."""

    def test_select_run_picks_the_run_decide_reports(self) -> None:
        runs = copy.deepcopy(_runs(RUNS_FIXTURE, SHA_GREEN))
        later = copy.deepcopy(runs[0])
        later.update({"id": 99999999999, "conclusion": "failure", "run_attempt": 2, "updated_at": "2030-01-01T00:00:00Z"})
        runs.append(later)
        chosen = gate.select_run(runs, SHA_GREEN)
        assert chosen is not None and chosen["id"] == 99999999999
        green, detail = gate.decide(runs, SHA_GREEN)
        assert green is False
        assert "99999999999" in detail

    def test_select_run_is_none_when_no_run_matches(self) -> None:
        assert gate.select_run(_runs(RUNS_FIXTURE, SHA_GREEN), SHA_5_0_1) is None


class TestCiWorkflowShape:
    """ci.yml itself: a red row must not be forgiven, and lint must see it all."""

    @pytest.fixture
    def ci_text(self) -> str:
        return CI_WORKFLOW.read_text(encoding="utf-8")

    def test_no_job_is_allowed_to_fail_quietly(self, ci_text: str) -> None:
        """``continue-on-error`` is what turns a red row into a green run."""
        offenders = [line.strip() for line in ci_text.splitlines() if re.match(r"^\s*continue-on-error\s*:", line)]
        assert offenders == [], f"ci.yml forgives a failing job: {offenders}"

    def test_the_matrix_does_not_silently_resolve_prereleases(self, ci_text: str) -> None:
        """``allow-prereleases`` lets a row test an interpreter nobody advertised."""
        assert not re.search(r"^\s*allow-prereleases\s*:", ci_text, re.MULTILINE)

    def test_the_python_matrix_still_covers_every_advertised_row(self, ci_text: str) -> None:
        """Positive control: the two assertions above must not be satisfiable by
        deleting the matrix."""
        assert re.search(r'python-version:\s*\["3\.10", "3\.11", "3\.12", "3\.13", "3\.14"\]', ci_text)
        assert "os: [ubuntu-latest, macos-latest, windows-latest]" in ci_text

    @staticmethod
    def _run_commands(ci_text: str) -> list[str]:
        """Every executable ``run:`` line, comments excluded.

        Matching against the raw file would let a passing assertion be satisfied
        by a sentence in a comment -- and this workflow's comments quote the old
        commands verbatim.
        """
        return [line.strip() for line in ci_text.splitlines() if re.match(r"^\s*-?\s*run:", line)]

    def test_lint_covers_the_whole_repository(self, ci_text: str) -> None:
        """The narrow ``src/ tests/`` scope hid 9 real errors in conftest_trace.py."""
        commands = self._run_commands(ci_text)
        assert any(re.fullmatch(r"- run: ruff check \.", command) for command in commands), commands
        narrow = [c for c in commands if "ruff check" in c and "ruff check ." not in c]
        assert narrow == [], f"ci.yml lints a subset of the repository again: {narrow}"

    def test_the_format_check_is_scoped_by_exclusion_not_by_an_include_list(self, ci_text: str) -> None:
        """An include-list never grows when new directories do -- that is the
        blind spot that produced the unlinted 62 files in the first place."""
        commands = self._run_commands(ci_text)
        formats = [c for c in commands if "ruff format" in c]
        assert formats == ["run: python3 -m ruff format --check . --exclude benchmarks --exclude train"], formats

    def test_the_release_gate_script_is_the_one_being_tested(self) -> None:
        """Wiring: the module under test is the file the workflow invokes."""
        release = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        assert "scripts/check_ci_green.py" in release


class TestJobsFixtureLoader:
    """The loader is part of the gate; a broken one silently changes the answer."""

    def test_a_bare_job_array_is_accepted(self) -> None:
        assert gate._load_jobs_file(str(JOBS_FIXTURE), RUN_GREEN)

    def test_provenance_keys_are_not_mistaken_for_runs(self) -> None:
        payload = json.loads(JOBS_FIXTURE.read_text(encoding="utf-8"))
        assert "_provenance" in payload, "the fixture must record where it came from"
        with pytest.raises(gate.GateError):
            gate._load_jobs_file(str(JOBS_FIXTURE), "_provenance")

    def test_an_unknown_run_id_fails_closed(self) -> None:
        with pytest.raises(gate.GateError):
            gate._load_jobs_file(str(JOBS_FIXTURE), 1)
