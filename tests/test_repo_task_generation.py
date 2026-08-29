"""Tests for the real-repo A/B task generator.

The generator's whole value is that nobody can fabricate its ground truth,
so these tests check the two things that could quietly break that: the
selection rule (mechanical, stated, outcome-blind) and the red->green proof
(executed, not inferred).  The end-to-end case builds a throw-away git
repository with a genuine defect and a genuine fix and asserts the harness
finds the flip.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from mind_mem.bench import repo_task_cli as cli
from mind_mem.bench import repo_task_mining as mining
from mind_mem.bench import repo_task_validation as validation


def _run(cwd: str, *args: str) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8")


def _write(root: str, rel: str, text: str) -> None:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


@pytest.fixture
def toy_repo(tmp_path) -> str:
    """A two-commit repo: a defect, then a fix that ships its own test."""
    root = str(tmp_path / "toy")
    os.makedirs(root)
    _run(root, "git", "init", "-q", "-b", "main")
    _run(root, "git", "config", "user.email", "noreply@star.ga")
    _run(root, "git", "config", "user.name", "STARGA Inc")
    _write(root, "src/toy/__init__.py", "def add(a, b):\n    return a - b\n")
    _write(root, "tests/test_existing.py", "def test_placeholder():\n    assert True\n")
    _run(root, "git", "add", "-A")
    _run(root, "git", "commit", "-q", "-m", "feat: seed the toy package")
    _write(root, "src/toy/__init__.py", "def add(a, b):\n    return a + b\n")
    _write(root, "tests/test_add.py", "from toy import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    _run(root, "git", "add", "-A")
    _run(root, "git", "commit", "-q", "-m", "fix(toy): addition subtracted")
    return root


class TestSelectionRule:
    """The rule must be mechanical and must not read outcomes."""

    def test_test_and_src_path_predicates(self) -> None:
        assert mining.is_test_path("tests/test_recall.py")
        assert not mining.is_test_path("tests/helpers.py")
        assert not mining.is_test_path("src/mind_mem/test_recall.py")
        assert mining.is_src_path("src/mind_mem/recall.py")
        assert not mining.is_src_path("benchmarks/x.py")

    def test_test_infra_covers_root_conftest(self) -> None:
        assert mining.is_test_infra_path("tests/conftest.py")
        assert mining.is_test_infra_path("conftest.py")
        assert not mining.is_test_infra_path("src/mind_mem/recall.py")

    @pytest.mark.parametrize("word", ["psycopg", "Postgres", "POSTGRESQL", "redis", "Ollama"])
    def test_shared_service_pattern_matches(self, word: str) -> None:
        assert mining.SHARED_SERVICE_PATTERN.search(f"import {word}\n")

    def test_shared_service_pattern_is_not_trigger_happy(self) -> None:
        assert not mining.SHARED_SERVICE_PATTERN.search("from mind_mem.recall import recall\n")

    def test_utc_iso_is_offset_free(self) -> None:
        assert mining._utc_iso("0") == "1970-01-01T00:00:00Z"

    def test_toy_repo_yields_exactly_the_fix_commit(self, toy_repo: str) -> None:
        selected, stats = mining.select_candidates(toy_repo, "HEAD", limit=10)
        assert stats.commits_scanned == 2
        assert [c.subject for c in selected] == ["fix(toy): addition subtracted"]
        assert selected[0].added_test_files == ("tests/test_add.py",)
        assert selected[0].src_changed == ("src/toy/__init__.py",)


class TestSandboxIsolation:
    """A run must not inherit anything that could reach a live service."""

    def test_environment_is_constructed_not_inherited(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("MIND_MEM_PG_DSN", "postgresql://127.0.0.1:5432/mindmem")
        monkeypatch.setenv("MIND_MEM_DSN", "postgresql://127.0.0.1:5432/mindmem")
        env = validation.sandbox_env(str(tmp_path), str(tmp_path / "home"))
        assert "MIND_MEM_PG_DSN" not in env
        assert "MIND_MEM_DSN" not in env

    def test_determinism_knobs_are_pinned(self, tmp_path) -> None:
        env = validation.sandbox_env(str(tmp_path), str(tmp_path / "home"))
        assert env["TZ"] == "UTC"
        assert env["PYTHONHASHSEED"] == "0"
        assert env["LC_ALL"] == "C.UTF-8"
        assert env["PYTHONPATH"] == os.path.join(str(tmp_path), "src")


class TestHypothesisSeeding:
    """Property-based randomness is removed at the source, when present."""

    @staticmethod
    def _flags_for(monkeypatch, returncode: int) -> tuple:
        validation._hypothesis_flags.cache_clear()
        monkeypatch.setattr(
            validation.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=returncode),
        )
        try:
            return validation._hypothesis_flags(f"python-{returncode}")
        finally:
            validation._hypothesis_flags.cache_clear()

    def test_seed_is_pinned_when_plugin_present(self, monkeypatch) -> None:
        assert self._flags_for(monkeypatch, 0) == ("--hypothesis-seed=0",)

    def test_flag_is_omitted_when_plugin_absent(self, monkeypatch) -> None:
        assert self._flags_for(monkeypatch, 1) == ()


class TestOutcomeParsing:
    """Node-level verdicts come from pytest's own summary, nothing inferred."""

    def test_parses_short_summary(self) -> None:
        text = "PASSED tests/a.py::test_one\nFAILED tests/a.py::test_two\nERROR tests/b.py\n"
        assert validation._parse_statuses(text) == {
            "tests/a.py::test_one": "PASSED",
            "tests/a.py::test_two": "FAILED",
            "tests/b.py": "ERROR",
        }

    def test_collection_error_is_distinguished_from_test_failure(self) -> None:
        collected = validation.RunResult(1, False, {"tests/a.py::test_one": "FAILED"}, "")
        not_collected = validation.RunResult(2, False, {"tests/a.py": "ERROR"}, "")
        assert collected.imported
        assert not not_collected.imported

    def test_durations_and_sandbox_paths_are_scrubbed(self) -> None:
        text = "1 passed in 0.27s\n/tmp/mmtask_ab/parent/tests/a.py ok\n"
        cleaned = validation.scrub_nondeterminism(text, "/tmp/mmtask_ab/parent")
        assert "0.27s" not in cleaned
        assert "<elapsed>s" in cleaned
        assert "/tmp/mmtask_ab/parent" not in cleaned
        assert "<sandbox>" in cleaned

    def test_scrubber_is_idempotent(self) -> None:
        once = validation.scrub_nondeterminism("2 passed in 1.50s", "/tmp/x")
        assert validation.scrub_nondeterminism(once, "/tmp/x") == once

    def test_xpass_does_not_count_as_a_pass(self) -> None:
        result = validation.RunResult(0, False, {"tests/a.py::test_one": "XPASS"}, "")
        assert result.passed_nodes == frozenset()


class TestRepeatStability:
    """Grading that disagrees with itself is caught, not recorded."""

    @staticmethod
    def _arm_with(monkeypatch, results: list) -> tuple:
        monkeypatch.setattr(validation, "extract_tree", lambda *a, **k: None)
        pending = iter(results)
        monkeypatch.setattr(validation, "run_pytest", lambda *a, **k: next(pending))
        return validation._arm("repo", "sha", ["tests/t.py"], "/tmp/nowhere", "x", "py", 10, len(results))

    def test_agreeing_repeats_are_stable(self, monkeypatch) -> None:
        same = [validation.RunResult(1, False, {"t::a": "FAILED"}, "") for _ in range(2)]
        _, stable = self._arm_with(monkeypatch, same)
        assert stable

    def test_disagreeing_repeats_are_unstable(self, monkeypatch) -> None:
        flaky = [
            validation.RunResult(1, False, {"t::a": "FAILED"}, ""),
            validation.RunResult(0, False, {"t::a": "PASSED"}, ""),
        ]
        _, stable = self._arm_with(monkeypatch, flaky)
        assert not stable

    def test_unstable_arm_drops_the_task(self, monkeypatch, tmp_path) -> None:
        parent = validation.RunResult(1, False, {"t::a": "FAILED"}, "")
        task = validation.RunResult(0, False, {"t::a": "PASSED"}, "")
        calls = iter([(parent, True), (task, False)])
        monkeypatch.setattr(validation, "_arm", lambda *a, **k: next(calls))
        result = validation.validate("repo", _candidate(), str(tmp_path / "wd"), "py")
        assert not result.well_formed
        assert result.drop_reason == "nondeterministic_grading"


def _candidate(sha: str = "a" * 40) -> mining.Candidate:
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


class TestVerdict:
    """Well-formedness is a transition, not an opinion."""

    def test_red_then_green_is_well_formed(self) -> None:
        parent = validation.RunResult(1, False, {"tests/test_x.py::test_a": "FAILED"}, "")
        task = validation.RunResult(0, False, {"tests/test_x.py::test_a": "PASSED"}, "")
        result = validation._verdict(_candidate(), parent, task)
        assert result.well_formed
        assert result.fail_to_pass == ("tests/test_x.py::test_a",)
        assert result.tier == "behavioral"

    def test_missing_module_at_parent_is_api_construction(self) -> None:
        parent = validation.RunResult(2, False, {"tests/test_x.py": "ERROR"}, "")
        task = validation.RunResult(0, False, {"tests/test_x.py::test_a": "PASSED"}, "")
        assert validation._verdict(_candidate(), parent, task).tier == "api_construction"

    def test_already_green_at_parent_is_dropped(self) -> None:
        parent = validation.RunResult(0, False, {"tests/test_x.py::test_a": "PASSED"}, "")
        task = validation.RunResult(0, False, {"tests/test_x.py::test_a": "PASSED"}, "")
        result = validation._verdict(_candidate(), parent, task)
        assert not result.well_formed
        assert result.drop_reason == "already_green_at_parent"

    def test_red_at_its_own_commit_is_dropped(self) -> None:
        parent = validation.RunResult(1, False, {"tests/test_x.py::test_a": "FAILED"}, "")
        task = validation.RunResult(1, False, {"tests/test_x.py::test_a": "FAILED"}, "")
        result = validation._verdict(_candidate(), parent, task)
        assert not result.well_formed
        assert result.drop_reason == "task_sha_not_green"

    def test_timeout_at_task_is_dropped(self) -> None:
        parent = validation.RunResult(1, False, {"tests/test_x.py::test_a": "FAILED"}, "")
        task = validation.RunResult(-1, True, {}, "TIMEOUT")
        assert validation._verdict(_candidate(), parent, task).drop_reason == "task_sha_not_green"


class TestPromptDoesNotLeak:
    """The statement is a specification, never the answer."""

    def test_statement_carries_subject_and_tests(self) -> None:
        text = cli.derive_task_statement(_candidate())
        assert "fix(x): y" in text
        assert "tests/test_x.py" in text

    def test_statement_omits_the_reference_fix(self) -> None:
        text = cli.derive_task_statement(_candidate())
        assert "src/mind_mem/x.py" not in text
        assert _candidate().sha not in text


class TestEndToEnd:
    """The transition is executed on a real repository, never assumed."""

    def test_toy_repo_task_is_validated_red_to_green(self, toy_repo: str, tmp_path) -> None:
        payload = cli.run(toy_repo, "HEAD", limit=10, jobs=1, python=sys.executable, timeout=180)
        assert payload["validation"]["well_formed"] == 1
        task = payload["tasks"][0]
        assert task["parent_returncode"] != 0
        assert task["task_returncode"] == 0
        assert task["fail_to_pass"] == ["tests/test_add.py::test_add"]
        assert task["memory_cutoff"] == task["parent_committed_at"]
        assert task["scoring_instant"] == task["parent_committed_at"][:10]

    def test_artifact_is_byte_identical_across_runs(self, toy_repo: str, tmp_path) -> None:
        first = str(tmp_path / "a.json")
        second = str(tmp_path / "b.json")
        for out in (first, second):
            assert cli.main(["--repo", toy_repo, "--limit", "10", "--jobs", "1", "--out", out]) == 0
        with open(first, encoding="utf-8") as fa, open(second, encoding="utf-8") as fb:
            assert fa.read() == fb.read()

    def test_schema_documents_every_task_field(self, toy_repo: str) -> None:
        payload = cli.run(toy_repo, "HEAD", limit=10, jobs=1, python=sys.executable, timeout=180)
        documented = set(payload["schema"])
        emitted = set(payload["tasks"][0])
        assert emitted <= documented, f"undocumented fields: {sorted(emitted - documented)}"
        assert json.dumps(payload)


class TestNotAnOrphan:
    """The generator is reachable as a shipped entry point, not dead code."""

    def test_console_script_is_declared(self) -> None:
        with open(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"), encoding="utf-8") as handle:
            assert 'mind-mem-bench-tasks = "mind_mem.bench.repo_task_cli:main"' in handle.read()

    def test_package_exports_the_generator(self) -> None:
        import mind_mem.bench as bench

        assert bench.generate_repo_tasks is cli.run
        assert "select_candidates" in bench.__all__
