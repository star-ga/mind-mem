"""Tests for the release-preflight gates.

A gate nobody has watched fail is not a gate. These tests exercise both
release-preflight checkers against payloads captured verbatim from the real
index and the real Actions API, and they include positive controls in both
directions:

* a case that MUST pass (5.2.0 absent from the index; run 33511997100 green),
  so a future edit that hardcodes a failure goes red;
* cases that MUST fail (5.1.0 yanked; runs 33566252435 and 33579619488 failed;
  no run recorded at all), so a future edit that neuters the check -- or that
  reintroduces a resolver-shaped "can pip see it?" probe, which cannot see a
  yanked version -- also goes red.

The third block asserts the workflow's own job graph, because a checker that is
written and never wired is the exact failure that let the GitHub Release for
5.1.0 outlive the version on the index.
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
INDEX_FIXTURE = FIXTURES / "pypi_mind_mem_releases.json"
RUNS_FIXTURE = FIXTURES / "ci_runs_by_sha.json"
# The CI-green gate reads per-JOB conclusions as well as the run conclusion
# (a continue-on-error job fails red inside a run that concludes success), so
# its CLI needs both captured payloads or it refuses to answer. The per-job
# behaviour itself is covered in tests/test_ci_green_per_job_gate.py.
JOBS_FIXTURE = FIXTURES / "ci_jobs_by_run.json"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# Real shas from this repository's history. The comments are the measured
# ground truth these fixtures were captured from.
SHA_FAILED_ALL_WINDOWS = "a0850815782277b17fbd1243fd0f084d51a0e651"  # run 33566252435, conclusion failure
SHA_FAILED_ONE_WINDOWS = "1ec7f63e011443eca2e63299b08c51241fa09915"  # run 33579619488, conclusion failure
SHA_GREEN = "ba4cd4e12de5091b38d971b0ee921f47effd2acd"  # run 33511997100, conclusion success
SHA_NO_RUN = "313134b66d99b9ebaed8af77fb0092a6b5115c53"  # on main, zero runs recorded


def _load_script(name: str) -> ModuleType:
    """Import a scripts/*.py checker as a module."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    assert path.is_file(), f"missing release gate script: {path}"
    spec = importlib.util.spec_from_file_location(f"_release_gate_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


index_gate = _load_script("check_index_absence")
ci_gate = _load_script("check_ci_green")


@pytest.fixture
def index_payload() -> dict[str, Any]:
    """The real PyPI ``releases`` map, trimmed to four versions."""
    payload = json.loads(INDEX_FIXTURE.read_text(encoding="utf-8"))
    # Positive control on the fixture itself: if the captured data ever stops
    # containing a yanked 5.1.0, the rejection tests below would pass
    # vacuously.
    assert all(entry["yanked"] is True for entry in payload["releases"]["5.1.0"])
    assert all(entry["yanked"] is False for entry in payload["releases"]["5.0.1"])
    assert "5.2.0" not in payload["releases"]
    return payload


@pytest.fixture
def runs_by_sha() -> dict[str, Any]:
    """Real ``/actions/workflows/ci.yml/runs`` responses, keyed by head sha."""
    return json.loads(RUNS_FIXTURE.read_text(encoding="utf-8"))


def _runs(runs_by_sha: dict[str, Any], sha: str) -> list[Any]:
    assert sha in runs_by_sha, f"fixture has no captured runs for {sha}"
    return list(runs_by_sha[sha]["workflow_runs"])


class TestIndexAbsenceGate:
    """Gate (d): the version must be absent from the index in ANY state."""

    def test_rejects_the_yanked_version(self, index_payload: dict[str, Any]) -> None:
        """5.1.0 was published then yanked, so the number is spent."""
        state, detail = index_gate.classify(index_payload, "5.1.0")
        assert state == index_gate.YANKED
        assert "YANKED" in detail

    def test_accepts_the_next_unused_version(self, index_payload: dict[str, Any]) -> None:
        """Positive control: the gate must be able to pass, or it proves nothing."""
        state, detail = index_gate.classify(index_payload, "5.2.0")
        assert state == index_gate.ABSENT
        assert "absent" in detail

    def test_rejects_the_current_live_version(self, index_payload: dict[str, Any]) -> None:
        state, _ = index_gate.classify(index_payload, "5.0.1")
        assert state == index_gate.LIVE

    def test_rejects_an_equivalent_spelling_of_the_yanked_version(self, index_payload: dict[str, Any]) -> None:
        """``5.1`` and ``5.1.0`` are the same version, so both must be refused."""
        state, _ = index_gate.classify(index_payload, "5.1")
        assert state == index_gate.YANKED

    def test_a_present_version_with_no_files_is_still_spent(self, index_payload: dict[str, Any]) -> None:
        """Deleting the files does not return the number, so this is not 'absent'."""
        payload = copy.deepcopy(index_payload)
        payload["releases"]["5.2.0"] = []
        state, detail = index_gate.classify(payload, "5.2.0")
        assert state == index_gate.FILELESS
        assert "burned" in detail

    def test_partial_yank_counts_as_yanked(self, index_payload: dict[str, Any]) -> None:
        payload = copy.deepcopy(index_payload)
        payload["releases"]["5.2.0"] = [
            {"filename": "a.whl", "yanked": True},
            {"filename": "a.tar.gz", "yanked": False},
        ]
        state, _ = index_gate.classify(payload, "5.2.0")
        assert state == index_gate.YANKED

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param("not a dict", id="payload-not-an-object"),
            pytest.param({"info": {}}, id="no-releases-key"),
            pytest.param({"releases": []}, id="releases-not-a-mapping"),
            pytest.param({"releases": {}}, id="releases-empty"),
        ],
    )
    def test_unreadable_payloads_fail_closed(self, payload: Any) -> None:
        with pytest.raises(index_gate.GateError):
            index_gate.classify(payload, "5.2.0")

    @pytest.mark.parametrize(
        "files",
        [
            pytest.param("not-a-list", id="files-not-a-list"),
            pytest.param(["not-an-object"], id="file-entry-not-an-object"),
            pytest.param([{"filename": "a.whl"}], id="file-entry-missing-yanked"),
        ],
    )
    def test_malformed_file_lists_fail_closed(self, index_payload: dict[str, Any], files: Any) -> None:
        payload = copy.deepcopy(index_payload)
        payload["releases"]["5.2.0"] = files
        with pytest.raises(index_gate.GateError):
            index_gate.classify(payload, "5.2.0")

    def test_canonical_agrees_with_pep_440_equivalence(self) -> None:
        """Each pair below is one version under PEP 440, so it must fold together."""
        for left, right in [
            ("v5.2.0", "5.2"),
            ("5.2.0", "5.2.0.0"),
            ("5.0.0", "5"),
            ("5.2.0rc1", "5.2rc1"),
        ]:
            assert index_gate.canonical(left) == index_gate.canonical(right), (left, right)

    def test_canonical_deviations_all_err_toward_failing(self) -> None:
        """Where the fold is coarser than PEP 440, it can only add refusals."""
        # PEP 440 keeps a local segment distinct; this folds it away. The index
        # rejects local versions outright, so treating ``5.2.0+local`` as
        # ``5.2.0`` can only make the gate refuse more, never less.
        assert index_gate.canonical("5.2.0+local") == index_gate.canonical("5.2.0")
        # An unrecognised shape is returned as-is rather than guessed at, and
        # ``classify`` still compares raw strings, so it cannot be smuggled past.
        assert index_gate.canonical("not-a-version") == "not-a-version"
        assert index_gate.canonical("  V5.2.0  ") == "5.2"

    def test_cli_exit_codes_match_the_decision(self) -> None:
        """The decision function must actually be the thing the CLI reports."""
        assert index_gate.main(["--version", "5.2.0", "--payload-file", str(INDEX_FIXTURE)]) == 0
        assert index_gate.main(["--version", "5.1.0", "--payload-file", str(INDEX_FIXTURE)]) == 1
        assert index_gate.main(["--version", "5.0.1", "--payload-file", str(INDEX_FIXTURE)]) == 1

    def test_cli_fails_closed_on_a_missing_payload(self, tmp_path: Path) -> None:
        assert index_gate.main(["--version", "5.2.0", "--payload-file", str(tmp_path / "nope.json")]) == 1

    def test_cli_fails_closed_on_an_unreachable_index(self) -> None:
        rc = index_gate.main(
            ["--version", "5.2.0", "--api", "http://127.0.0.1:9/{project}/json", "--timeout", "2"],
        )
        assert rc == 1


class TestCiGreenGate:
    """Gate (c): CI must have concluded success for the exact commit."""

    def test_rejects_the_commit_5_0_1_shipped_from(self, runs_by_sha: dict[str, Any]) -> None:
        """a085081 / run 33566252435 failed all five Windows matrix rows."""
        green, detail = ci_gate.decide(_runs(runs_by_sha, SHA_FAILED_ALL_WINDOWS), SHA_FAILED_ALL_WINDOWS)
        assert green is False
        assert "33566252435" in detail
        assert "failure" in detail

    def test_rejects_a_partial_failure(self, runs_by_sha: dict[str, Any]) -> None:
        """1ec7f63 / run 33579619488 failed only windows 3.10; still a failure."""
        green, detail = ci_gate.decide(_runs(runs_by_sha, SHA_FAILED_ONE_WINDOWS), SHA_FAILED_ONE_WINDOWS)
        assert green is False
        assert "33579619488" in detail

    def test_accepts_a_green_run(self, runs_by_sha: dict[str, Any]) -> None:
        """Positive control: ba4cd4e1 / run 33511997100 concluded success."""
        green, detail = ci_gate.decide(_runs(runs_by_sha, SHA_GREEN), SHA_GREEN)
        assert green is True
        assert "33511997100" in detail

    def test_no_recorded_run_is_a_failure_not_a_pass(self, runs_by_sha: dict[str, Any]) -> None:
        """313134b is on main with zero runs: a batched push runs only its tip."""
        runs = _runs(runs_by_sha, SHA_NO_RUN)
        assert runs == [], "fixture must capture the genuinely empty case"
        green, detail = ci_gate.decide(runs, SHA_NO_RUN)
        assert green is False
        assert "no CI run" in detail

    def test_a_green_run_for_a_different_commit_does_not_count(self, runs_by_sha: dict[str, Any]) -> None:
        """The 'CI is green on main' mistake, reproduced: right branch, wrong commit."""
        green, detail = ci_gate.decide(_runs(runs_by_sha, SHA_GREEN), SHA_FAILED_ALL_WINDOWS)
        assert green is False
        assert "no CI run" in detail

    def test_an_unfinished_run_is_undeterminable(self, runs_by_sha: dict[str, Any]) -> None:
        runs = copy.deepcopy(_runs(runs_by_sha, SHA_GREEN))
        runs[0]["status"] = "in_progress"
        runs[0]["conclusion"] = None
        green, detail = ci_gate.decide(runs, SHA_GREEN)
        assert green is False
        assert "not finished" in detail

    def test_a_later_failure_overrides_an_earlier_success(self, runs_by_sha: dict[str, Any]) -> None:
        """A re-run that went red must not be masked by the original green."""
        runs = copy.deepcopy(_runs(runs_by_sha, SHA_GREEN))
        later = copy.deepcopy(runs[0])
        later.update(
            {
                "id": 99999999999,
                "conclusion": "failure",
                "run_attempt": 2,
                "updated_at": "2030-01-01T00:00:00Z",
            }
        )
        green, detail = ci_gate.decide([*runs, later], SHA_GREEN)
        assert green is False
        assert "99999999999" in detail

    @pytest.mark.parametrize(
        "conclusion",
        ["failure", "cancelled", "timed_out", "skipped", "neutral", "startup_failure", None],
    )
    def test_only_success_passes(self, runs_by_sha: dict[str, Any], conclusion: str | None) -> None:
        runs = copy.deepcopy(_runs(runs_by_sha, SHA_GREEN))
        runs[0]["conclusion"] = conclusion
        green, _ = ci_gate.decide(runs, SHA_GREEN)
        assert green is False

    @pytest.mark.parametrize(
        "runs",
        [
            pytest.param({"workflow_runs": []}, id="not-an-array"),
            pytest.param(["not-an-object"], id="entry-not-an-object"),
            pytest.param([{"id": 1, "status": "completed", "conclusion": "success"}], id="entry-missing-head-sha"),
        ],
    )
    def test_unreadable_payloads_fail_closed(self, runs: Any) -> None:
        with pytest.raises(ci_gate.GateError):
            ci_gate.decide(runs, SHA_GREEN)

    def test_completed_run_without_a_conclusion_field_fails_closed(self, runs_by_sha: dict[str, Any]) -> None:
        runs = copy.deepcopy(_runs(runs_by_sha, SHA_GREEN))
        del runs[0]["conclusion"]
        with pytest.raises(ci_gate.GateError):
            ci_gate.decide(runs, SHA_GREEN)

    def test_short_shas_are_refused_rather_than_prefix_matched(self, runs_by_sha: dict[str, Any]) -> None:
        with pytest.raises(ci_gate.GateError):
            ci_gate.decide(_runs(runs_by_sha, SHA_GREEN), "ba4cd")

    def test_empty_sha_fails_closed(self, runs_by_sha: dict[str, Any]) -> None:
        with pytest.raises(ci_gate.GateError):
            ci_gate.decide(_runs(runs_by_sha, SHA_GREEN), "   ")

    def test_a_green_run_containing_a_red_advisory_job_is_rejected(self) -> None:
        """The run-versus-job hole: a ``continue-on-error`` job fails red while
        its run still concludes ``success``, so a run-level-only gate passes a
        red row.

        Both halves are asserted so neither can rot: ``decide`` is the run-level
        leg -- all this gate used to be -- and it must still say green here, or
        the fixture has stopped being dangerous and the rejection below proves
        nothing. The full per-job battery lives in
        tests/test_ci_green_per_job_gate.py.
        """
        advisory = FIXTURES / "ci_jobs_advisory_red.json"
        sha = "c0ffeec0ffeec0ffeec0ffeec0ffeec0ffeeabcd"
        runs = ci_gate._load_runs_file(str(advisory), sha)
        assert ci_gate.decide(runs, sha)[0] is True, "fixture must be one the run-level leg passes"
        assert ci_gate.main(["--sha", sha, "--runs-file", str(advisory), "--jobs-file", str(advisory)]) == 1

    def test_cli_exit_codes_match_the_decision(self) -> None:
        offline = ["--runs-file", str(RUNS_FIXTURE), "--jobs-file", str(JOBS_FIXTURE)]
        assert ci_gate.main(["--sha", SHA_GREEN, *offline]) == 0
        assert ci_gate.main(["--sha", SHA_FAILED_ALL_WINDOWS, *offline]) == 1
        assert ci_gate.main(["--sha", SHA_FAILED_ONE_WINDOWS, *offline]) == 1
        assert ci_gate.main(["--sha", SHA_NO_RUN, *offline]) == 1

    def test_cli_fails_closed_on_an_unknown_sha(self) -> None:
        assert ci_gate.main(["--sha", "0" * 40, "--runs-file", str(RUNS_FIXTURE), "--jobs-file", str(JOBS_FIXTURE)]) == 1

    def test_cli_fails_closed_on_a_missing_runs_file(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "nope.json")
        assert ci_gate.main(["--sha", SHA_GREEN, "--runs-file", missing, "--jobs-file", str(JOBS_FIXTURE)]) == 1


class TestVersionTripleGate:
    """Gate (a): the tag joins pyproject.toml / __init__.py / CHANGELOG.md."""

    def test_programmatic_main_does_not_inherit_the_process_command_line(self) -> None:
        """Regression: ``main()`` once read ``sys.argv``, so calling it from a
        test made the enclosing pytest arguments look like gate arguments."""
        from mind_mem.check_version import main as version_main

        assert version_main() == 0

    def test_the_tag_is_a_real_fourth_member_of_the_set(self) -> None:
        from mind_mem.check_version import main as version_main

        assert version_main(["--expect", "999.999.999"]) == 1

    @pytest.mark.parametrize("argv", [["--expect"], ["--expect", ""], ["--expect", "   "], ["--unknown"]])
    def test_a_malformed_expectation_fails_rather_than_dropping_the_leg(self, argv: list[str]) -> None:
        """An unset shell variable must not silently remove the gate it added."""
        from mind_mem.check_version import main as version_main

        assert version_main(argv) == 1


def _job_needs(workflow_text: str) -> dict[str, list[str]]:
    """Extract ``{job: [needs...]}`` from a workflow file.

    Hand-rolled rather than pyyaml-based on purpose: pyyaml is not in the
    ``test`` extra, and an ``importorskip`` here would let this whole structural
    check disappear on a runner that happens to lack it -- which is the one
    thing a wiring test must never do.
    """
    jobs: dict[str, list[str]] = {}
    current: str | None = None
    in_jobs = False
    for raw in workflow_text.splitlines():
        if raw.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if raw and not raw.startswith(" ") and not raw.startswith("#"):
            in_jobs = False
            continue
        job = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", raw)
        if job:
            current = job.group(1)
            jobs[current] = []
            continue
        needs = re.match(r"^    needs:\s*(.+?)\s*$", raw)
        if needs and current is not None:
            value = needs.group(1)
            if value.startswith("["):
                jobs[current] = [item.strip() for item in value.strip("[]").split(",") if item.strip()]
            else:
                jobs[current] = [value]
    assert jobs, "no jobs parsed out of the workflow — the parser, not the workflow, is wrong"
    return jobs


class TestReleaseWorkflowWiring:
    """The gates above are worthless unless the release path actually runs them."""

    @pytest.fixture
    def release_text(self) -> str:
        return RELEASE_WORKFLOW.read_text(encoding="utf-8")

    @pytest.fixture
    def graph(self, release_text: str) -> dict[str, list[str]]:
        return _job_needs(release_text)

    def test_preflight_job_exists(self, graph: dict[str, list[str]]) -> None:
        assert "release-preflight" in graph

    def test_nothing_is_built_or_published_without_preflight(self, graph: dict[str, list[str]]) -> None:
        """Every job that produces or ships an artifact is downstream of preflight."""
        upstream: dict[str, set[str]] = {}

        def resolve(job: str, seen: frozenset[str] = frozenset()) -> set[str]:
            assert job not in seen, f"cycle in the release job graph at {job}"
            if job in upstream:
                return upstream[job]
            found: set[str] = set()
            for parent in graph.get(job, []):
                assert parent in graph, f"job {job!r} needs unknown job {parent!r}"
                found.add(parent)
                found |= resolve(parent, seen | {job})
            upstream[job] = found
            return found

        for job in ("build", "sign", "sbom", "publish-pypi", "github-release", "verify-published"):
            assert job in graph, f"release.yml lost the {job!r} job"
            assert "release-preflight" in resolve(job), f"{job} does not depend on release-preflight"

    def test_github_release_waits_for_the_pypi_upload(self, graph: dict[str, list[str]]) -> None:
        """The 5.1.0 divergence: index yanked, GitHub Release still serving it."""
        assert "publish-pypi" in graph["github-release"]

    def test_post_publish_proof_runs_after_publishing(self, graph: dict[str, list[str]]) -> None:
        assert "publish-pypi" in graph["verify-published"]
        assert "release-preflight" in graph["verify-published"]

    def test_the_job_graph_is_acyclic(self, graph: dict[str, list[str]]) -> None:
        remaining = {job: set(parents) for job, parents in graph.items()}
        ordered: list[str] = []
        while remaining:
            ready = sorted(job for job, parents in remaining.items() if not parents - set(ordered))
            assert ready, f"cycle among {sorted(remaining)}"
            for job in ready:
                ordered.append(job)
                del remaining[job]
        assert set(ordered) == set(graph)

    def test_publish_step_does_not_skip_existing_versions(self, release_text: str) -> None:
        """A collision means the tag is re-attempting a spent number: be loud."""
        assert "skip-existing: false" in release_text
        assert re.search(r"^\s*skip-existing:\s*true\s*$", release_text, re.MULTILINE) is None

    @pytest.mark.parametrize(
        "invocation",
        [
            "python src/mind_mem/check_version.py --expect",
            "scripts/check_ci_green.py",
            "scripts/check_index_absence.py",
            "git merge-base --is-ancestor",
        ],
    )
    def test_every_gate_is_invoked(self, release_text: str, invocation: str) -> None:
        assert invocation in release_text, f"release.yml no longer runs: {invocation}"

    def test_release_runs_the_same_unit_selector_as_ci(self, release_text: str) -> None:
        """A release must not test less than CI does."""
        pattern = re.compile(r"^\s*run: (python3 -m pytest tests/ --ignore=tests/integration .*)$", re.MULTILINE)
        release_commands = pattern.findall(release_text)
        assert len(release_commands) == 1, f"expected exactly one release unit-test command, found {release_commands}"
        ci_text = CI_WORKFLOW.read_text(encoding="utf-8")
        assert release_commands[0] in ci_text, (
            "the release unit-test selector has drifted from ci.yml; keep them identical so a release cannot test less"
        )
