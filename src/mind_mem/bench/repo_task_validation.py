"""Execute a mined commit to prove it is a real red->green task.

A candidate is only a task if the machine says so.  For each one we build
two throw-away trees with ``git archive`` -- never a checkout, a stash or a
worktree, so a concurrently-edited working tree is untouched -- and run the
added tests in two arms:

  * **parent tree + test patch** -- the pre-task state with the commit's
    ``tests/`` delta laid on top.  The named tests must NOT pass here.
  * **task tree** -- the commit itself.  The named tests must pass here.

The flip set (``fail_to_pass``) is the node ids that pass in the second run
and do not pass in the first.  A candidate with an empty flip set is not a
task and is dropped with a counted reason.

Each arm runs ``repeats`` times in independent trees.  A candidate whose own
grading disagrees with itself between two identical runs is dropped as
``nondeterministic_grading`` -- it cannot be a benchmark task, because the
score would move without the agent doing anything.

Determinism: the child environment is *constructed*, never inherited, so a
stray ``MIND_MEM_PG_DSN`` in the parent shell cannot reach a run and cannot
point it at the production Postgres.  ``TZ=UTC`` and ``PYTHONHASHSEED=0``
are pinned because this repo has a known clock-and-hash sensitivity on the
scoring path, and a benchmark must not be the thing that reintroduces it.

The two subprocesses -- ``git archive`` and ``pytest`` -- both run with a
fixed argv and ``shell=False``.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import site
import subprocess  # nosec B404
import tarfile
from dataclasses import dataclass, replace
from typing import Sequence

from .repo_task_mining import Candidate, git

_STATUS_LINE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)")

#: pytest closes every run with "N passed in 0.27s".  That elapsed time is
#: wall-clock, and storing it verbatim made the generated artifact differ
#: between two runs over identical inputs -- caught by hashing two runs, not
#: by assuming.  Durations are scrubbed before anything is recorded.
_DURATION = re.compile(r"\d+\.\d+s\b")

#: Statuses that count as "the test passed".  ``XPASS`` is deliberately not
#: here: an unexpectedly-passing xfail is not a demonstrated fix.
_PASSING = frozenset({"PASSED"})


@dataclass(frozen=True)
class RunResult:
    """One pytest invocation: exit code plus per-node outcomes."""

    returncode: int
    timed_out: bool
    node_status: dict[str, str]
    tail: str

    @property
    def passed_nodes(self) -> frozenset[str]:
        return frozenset(n for n, s in self.node_status.items() if s in _PASSING)

    @property
    def imported(self) -> bool:
        """True if pytest reported at least one *test-level* outcome.

        A collection failure reports the file (``ERROR tests/x.py``) with no
        ``::node``; a module that imported reports per-test node ids.  That
        is the discriminator between "this API does not exist yet" and
        "this API exists and behaves differently".
        """
        return any("::" in node for node in self.node_status)


@dataclass(frozen=True)
class Validation:
    """The verdict for one candidate, with the raw transition recorded."""

    sha: str
    well_formed: bool
    drop_reason: str | None
    parent_returncode: int
    task_returncode: int
    parent_mode: str
    tier: str
    fail_to_pass: tuple[str, ...]
    task_passed_count: int
    parent_passed_count: int
    parent_tail: str
    task_tail: str


def sandbox_env(tree: str, home: str) -> dict[str, str]:
    """Build the child environment from scratch -- nothing is inherited."""
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": home,
        # User site is derived from HOME; pin the real base so the installed
        # test dependencies stay importable under the sandboxed HOME.
        "PYTHONUSERBASE": site.getuserbase(),
        "PYTHONPATH": os.path.join(tree, "src"),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "MIND_MEM_DISABLE_TELEMETRY": "1",
    }


def extract_tree(repo: str, sha: str, dest: str) -> None:
    """Materialise ``sha`` into ``dest`` via ``git archive`` (read-only)."""
    os.makedirs(dest, exist_ok=True)
    # Fixed argv, shell=False, read-only git archive.
    proc = subprocess.Popen(  # nosec B603 B607
        ["git", "-C", repo, "archive", "--format=tar", sha],
        stdout=subprocess.PIPE,
    )
    if proc.stdout is None:  # pragma: no cover - defensive
        raise RuntimeError(f"git archive produced no stream for {sha}")
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            tar.extractall(dest, filter="data")
    finally:
        proc.stdout.close()
        if proc.wait() != 0:  # pragma: no cover - defensive
            raise RuntimeError(f"git archive failed for {sha}")


def apply_test_patch(repo: str, sha: str, paths: Sequence[str], dest: str) -> None:
    """Copy the commit's test-side files into an already-extracted tree."""
    for path in paths:
        target = os.path.join(dest, path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(git(repo, "show", f"{sha}:{path}"))


def scrub_nondeterminism(text: str, *volatile_paths: str) -> str:
    """Remove wall-clock durations and per-run paths from captured output.

    Diagnostics are kept for the reader, but nothing that varies between two
    runs over the same inputs may survive into the artifact -- otherwise the
    task set has a different hash every time and "re-runnable" is a slogan.
    """
    cleaned = text
    for path in sorted(volatile_paths, key=len, reverse=True):
        if path:
            cleaned = cleaned.replace(path, "<sandbox>")
    return _DURATION.sub("<elapsed>s", cleaned)


def _parse_statuses(text: str) -> dict[str, str]:
    """Read pytest's ``-rA`` short summary into ``node id -> status``."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = _STATUS_LINE.match(line.strip())
        if match:
            found[match.group(2)] = match.group(1)
    return found


@functools.lru_cache(maxsize=8)
def _hypothesis_flags(python: str) -> tuple[str, ...]:
    """``--hypothesis-seed=0`` when the interpreter has hypothesis installed.

    Property-based tests draw fresh examples every run.  One in this
    repository flipped between two generations and changed the artifact's
    hash; pinning the seed removes the randomness at the source instead of
    hoping the repeat check happens to catch it.  Empty when the plugin is
    absent, so the flag is never passed to a pytest that would reject it.
    """
    probe = subprocess.run(  # nosec B603
        [python, "-c", "import hypothesis"],
        capture_output=True,
        check=False,
        timeout=60,
    )
    return ("--hypothesis-seed=0",) if probe.returncode == 0 else ()


def run_pytest(python: str, tree: str, home: str, tests: Sequence[str], timeout: int, nice: int = 15) -> RunResult:
    """Run ``tests`` inside ``tree`` and return the per-node outcome."""
    os.makedirs(home, exist_ok=True)
    cmd = [
        "nice",
        "-n",
        str(nice),
        python,
        "-m",
        "pytest",
        *tests,
        "-p",
        "no:cacheprovider",
        "-q",
        "--tb=no",
        "-rA",
        f"--timeout={max(10, timeout // 2)}",
        *_hypothesis_flags(python),
    ]
    try:
        # argv is built above from repository-relative test paths; shell=False.
        proc = subprocess.run(  # nosec B603
            cmd,
            cwd=tree,
            env=sandbox_env(tree, home),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RunResult(returncode=-1, timed_out=True, node_status={}, tail="TIMEOUT")
    output = scrub_nondeterminism(proc.stdout + proc.stderr, tree, home)
    return RunResult(
        returncode=proc.returncode,
        timed_out=False,
        node_status=_parse_statuses(output),
        tail="\n".join(output.splitlines()[-6:]),
    )


def _verdict(candidate: Candidate, parent: RunResult, task: RunResult) -> Validation:
    """Decide well-formedness from the two raw runs. No judging, only counts."""
    flips = tuple(sorted(task.passed_nodes - parent.passed_nodes))
    if parent.timed_out:
        mode, tier = "timeout", "unknown"
    elif parent.imported:
        mode, tier = "test_failure", "behavioral"
    else:
        mode, tier = "collection_error", "api_construction"
    reason: str | None = None
    if task.timed_out or task.returncode != 0 or not task.passed_nodes:
        reason = "task_sha_not_green"
    elif parent.timed_out:
        reason = "parent_timed_out"
    elif parent.returncode == 0:
        reason = "already_green_at_parent"
    elif not flips:
        reason = "no_fail_to_pass_nodes"
    return Validation(
        sha=candidate.sha,
        well_formed=reason is None,
        drop_reason=reason,
        parent_returncode=parent.returncode,
        task_returncode=task.returncode,
        parent_mode=mode,
        tier=tier,
        fail_to_pass=flips,
        task_passed_count=len(task.passed_nodes),
        parent_passed_count=len(parent.passed_nodes),
        parent_tail=parent.tail,
        task_tail=task.tail,
    )


def _arm(
    repo: str,
    sha: str,
    tests: Sequence[str],
    workdir: str,
    label: str,
    python: str,
    timeout: int,
    repeats: int,
    patch_sha: str = "",
    patch_paths: Sequence[str] = (),
) -> tuple[RunResult, bool]:
    """Run one arm ``repeats`` times in independent trees; report stability.

    Each repeat gets a freshly extracted tree and a fresh HOME so nothing a
    run leaves behind (a property-test example database, a cache) can steer
    the next one.  ``stable`` is False if any repeat disagreed.
    """
    results: list[RunResult] = []
    for index in range(max(1, repeats)):
        tree = os.path.join(workdir, f"{label}_{index}")
        extract_tree(repo, sha, tree)
        if patch_sha:
            apply_test_patch(repo, patch_sha, patch_paths, tree)
        results.append(run_pytest(python, tree, os.path.join(workdir, f"home_{label}_{index}"), tests, timeout))
        shutil.rmtree(tree, ignore_errors=True)
    first = results[0]
    stable = all(r.node_status == first.node_status and r.returncode == first.returncode for r in results[1:])
    return first, stable


def validate(
    repo: str,
    candidate: Candidate,
    workdir: str,
    python: str,
    timeout: int = 600,
    repeats: int = 2,
) -> Validation:
    """Build both trees, run both arms ``repeats`` times, return the verdict.

    A candidate whose own grading disagrees with itself between two identical
    runs is dropped.  Without this the task set is not reproducible: a
    property-based fuzz test in this repository flipped between generations
    and changed the artifact's hash, which is exactly the failure the file
    claims not to have.
    """
    tests = list(candidate.added_test_files)
    try:
        parent, parent_stable = _arm(
            repo,
            candidate.parent_sha,
            tests,
            workdir,
            "parent",
            python,
            timeout,
            repeats,
            patch_sha=candidate.sha,
            patch_paths=candidate.test_patch_paths,
        )
        task, task_stable = _arm(repo, candidate.sha, tests, workdir, "task", python, timeout, repeats)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    result = _verdict(candidate, parent, task)
    if result.well_formed and not (parent_stable and task_stable):
        return replace(result, well_formed=False, drop_reason="nondeterministic_grading")
    return result
