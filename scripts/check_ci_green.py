#!/usr/bin/env python3
"""Release gate: CI must have concluded success for the EXACT commit being released.

What this would have caught
---------------------------
5.0.1 shipped from a commit whose CI run failed all five Windows matrix rows.
The release workflow's only gate at the time was the code-scanning alert count,
which has nothing to say about whether the tests passed, so the tag sailed
through. This gate reads the recorded conclusion of the CI workflow for the
precise head sha the tag points at, AND the conclusion of every job inside it.

Two real examples from this repository, both of which this gate rejects:

  * run 33566252435 on a085081 -- ``conclusion: failure`` (all five Windows rows)
  * run 33579619488 on 1ec7f63 -- ``conclusion: failure`` (windows 3.10 only,
    with 3.11 through 3.14 green: a partial failure is still a failure)

and one it accepts: run 33511997100 on ba4cd4e1 -- ``conclusion: success``, all
31 jobs ``success``.

Why the RUN conclusion alone is not enough
------------------------------------------
A job marked ``continue-on-error`` fails visibly red while the run it belongs to
still concludes ``success`` -- that is the whole point of the setting. Measured
on this repository's own history: in run 33386877502 the job
``test (windows-latest, 3.14)`` reports ``conclusion: failure`` at the job level
while the ubuntu and macos 3.14 rows report ``success``; ci.yml carried
``continue-on-error: ${{ matrix.python-version == '3.14' }}`` at the time. A
run-level-only gate reads that red row as a pass. The operator rule is explicit:
a red row is NOT green and must block a version bump. So this gate enumerates
``/actions/runs/{id}/jobs`` and requires EVERY job to have concluded ``success``.

That leg is not made redundant by dropping ``continue-on-error`` from ci.yml.
The setting can be reintroduced, a reusable workflow can carry its own, and
``if:`` conditions can retire a job silently; the gate must hold whatever the
workflow says.

How each job conclusion is treated, and why
-------------------------------------------
Only ``success`` passes. Every other value fails, because this gate asks "did
every job demonstrate what it exists to demonstrate?", not "did anything shout?".
The two the operator asked to be called out by name:

  * ``skipped`` -- FAILS. A skipped job ran nothing and proved nothing. A job is
    skipped when an ``if:`` condition excluded it or a ``needs:`` dependency
    failed; in both cases the evidence the release is relying on does not exist.
    Reading "nothing failed here" as a pass is the same mistake as reading "no
    run recorded" as a pass. If ci.yml ever gains a deliberately conditional
    job, this gate should be given an explicit, named allowlist for it -- a
    decision someone makes on purpose, not a default that silently forgives.
  * ``cancelled`` -- FAILS. A cancelled job was stopped before it could answer.
    ci.yml sets ``cancel-in-progress: true``, so superseded runs are cancelled
    routinely; a cancelled job in an otherwise-successful run is exactly the
    "we never actually measured it" case.

``neutral``, ``timed_out``, ``action_required``, ``stale``, ``startup_failure``
and a null conclusion fail for the same reason.

Why the exact sha matters
-------------------------
"CI is green on main" is a claim about a branch tip, not about the commit being
released, and the two drift apart constantly. Worse, a batched push produces a
single CI run for its tip only: commits 313134b and 4c869f7 are both on main and
both have ZERO recorded CI runs. A branch-level check would call a tag on either
of them green; this gate reports no-run-recorded as a FAILURE.

Fail-closed contract
--------------------
The only pass is: at least one CI run exists for this sha, every run for it has
finished, the most recently updated one concluded ``success``, that run has at
least one job, every job finished, and every job concluded ``success``. No run,
an unfinished run, a non-``success`` run conclusion, a run with no jobs, an
unfinished job, any non-``success`` job conclusion, an unreadable API response,
a partially-read job list, or a payload whose shape is not what this gate
expects all exit non-zero.

Usage:
    python scripts/check_ci_green.py --sha "$GITHUB_SHA"
    python scripts/check_ci_green.py --sha ba4cd4e1... \
        --runs-file tests/fixtures/ci_runs_by_sha.json \
        --jobs-file tests/fixtures/ci_jobs_by_run.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any, Callable

DEFAULT_REPO = "star-ga/mind-mem"
DEFAULT_WORKFLOW = "ci.yml"
DEFAULT_TIMEOUT = 120.0

# The one conclusion that counts as evidence. Everything else -- including
# ``skipped`` and ``cancelled`` -- is the absence of evidence, which is not a
# pass. See the module docstring for the per-value reasoning.
PASSING_CONCLUSION = "success"

# ``/actions/runs/{id}/jobs`` pages at 100. This workflow has 31 jobs; the cap
# exists so a malformed ``total_count`` cannot spin forever, and being hit is
# itself a failure rather than a silent truncation.
JOBS_PER_PAGE = 100
MAX_JOB_PAGES = 20


class GateError(Exception):
    """The gate could not determine the answer, so it fails."""


def _timestamp(run: dict[str, Any]) -> str:
    """Best available ordering key for a run.

    ``updated_at`` moves when a run is re-run, so it orders re-attempts
    correctly; the other two are fallbacks for a payload that omits it. All
    three are ISO-8601 UTC strings, which sort lexicographically.
    """
    for field in ("updated_at", "run_started_at", "created_at"):
        value = run.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _runs_for_sha(runs: Any, sha: str) -> list[dict[str, Any]]:
    """Every recorded run whose head sha is ``sha``.

    Shared by ``decide`` and ``select_run`` on purpose: two copies of the
    matching rule would eventually judge one run and fetch another one's jobs.
    """
    if not isinstance(runs, list):
        raise GateError(f"run list is not a JSON array (got {type(runs).__name__})")

    wanted = sha.strip().lower()
    if not wanted:
        raise GateError("no commit sha supplied")

    mine = []
    for run in runs:
        if not isinstance(run, dict):
            raise GateError("run list contains a non-object entry")
        head = run.get("head_sha")
        if not isinstance(head, str) or not head:
            raise GateError(f"run {run.get('id', '?')} has no 'head_sha'; cannot confirm it is this commit")
        # Compare full shas both ways so a short sha on either side still
        # matches, but never let a 1-2 character prefix match anything.
        head = head.strip().lower()
        if len(wanted) < 7 or len(head) < 7:
            raise GateError(f"refusing to match on a sha shorter than 7 characters ({head!r} vs {wanted!r})")
        if head.startswith(wanted) or wanted.startswith(head):
            mine.append(run)
    return mine


def select_run(runs: Any, sha: str) -> dict[str, Any] | None:
    """The run ``decide`` judges: the most recently updated one for ``sha``."""
    mine = _runs_for_sha(runs, sha)
    if not mine:
        return None
    return max(mine, key=_timestamp)


def decide(runs: Any, sha: str) -> tuple[bool, str]:
    """Return ``(is_green, detail)`` for the CI runs recorded against ``sha``.

    This is the RUN-level leg only. It is necessary and not sufficient: see
    ``decide_jobs`` for the per-job leg, and ``evaluate`` for the composition
    that the CLI actually reports.

    Raises ``GateError`` if the payload cannot be interpreted -- an unreadable
    response must not be read as "no failures found".
    """
    mine = _runs_for_sha(runs, sha)

    if not mine:
        return False, (
            f"no CI run is recorded for {sha}. A commit with no run has no evidence, "
            "not a pass — batched pushes leave every non-tip commit in exactly this state."
        )

    unfinished = [run for run in mine if run.get("status") != "completed"]
    if unfinished:
        listed = ", ".join(f"run {run.get('id', '?')} status={run.get('status')!r}" for run in unfinished)
        return False, f"a CI run for {sha} has not finished ({listed}); the answer is not yet determinable"

    for run in mine:
        if "conclusion" not in run:
            raise GateError(f"completed run {run.get('id', '?')} has no 'conclusion' field")

    newest = max(mine, key=_timestamp)
    conclusion = newest.get("conclusion")
    described = f"run {newest.get('id', '?')} attempt {newest.get('run_attempt', '?')} conclusion={conclusion!r}"
    if conclusion == PASSING_CONCLUSION:
        return True, f"CI concluded success for {sha} ({described}, {len(mine)} run(s) recorded)"
    return False, f"CI did not pass for {sha} ({described})"


def decide_jobs(jobs: Any, run_id: Any) -> tuple[bool, str]:
    """Return ``(all_green, detail)`` for the jobs of one workflow run.

    Every job must have finished and concluded ``success``. See the module
    docstring for why ``skipped`` and ``cancelled`` are failures here.
    """
    if not isinstance(jobs, list):
        raise GateError(f"job list for run {run_id} is not a JSON array (got {type(jobs).__name__})")

    if not jobs:
        # Not a GateError: the payload was readable, it simply contains no
        # evidence. An empty list is the "the search never happened" shape, and
        # must never read as "no failures found".
        return False, f"run {run_id} has no jobs recorded; a run that ran nothing proves nothing"

    unfinished: list[str] = []
    red: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            raise GateError(f"job list for run {run_id} contains a non-object entry")
        name = job.get("name")
        if not isinstance(name, str) or not name:
            raise GateError(f"a job of run {run_id} has no 'name'; cannot report which job failed")
        status = job.get("status")
        if status != "completed":
            unfinished.append(f"{name} status={status!r}")
            continue
        if "conclusion" not in job:
            raise GateError(f"completed job {name!r} of run {run_id} has no 'conclusion' field")
        conclusion = job.get("conclusion")
        if conclusion != PASSING_CONCLUSION:
            red.append(f"{name} conclusion={conclusion!r}")

    if unfinished or red:
        parts = []
        if red:
            parts.append(f"{len(red)} job(s) did not pass: " + "; ".join(sorted(red)))
        if unfinished:
            parts.append(f"{len(unfinished)} job(s) have not finished: " + "; ".join(sorted(unfinished)))
        return False, f"run {run_id} is not green — " + " | ".join(parts)

    return True, f"all {len(jobs)} jobs of run {run_id} concluded success"


JobsProvider = Callable[[dict[str, Any]], Any]


def evaluate(runs: Any, sha: str, jobs_provider: JobsProvider) -> tuple[bool, str]:
    """The full gate: the run-level leg, then the per-job leg on the same run.

    ``jobs_provider`` receives the selected run object and returns its job list,
    so the caller decides whether that comes from the API or from a fixture.
    Anything it raises is allowed to propagate: a job list that cannot be read
    is an unanswerable question, not a pass.
    """
    run_green, run_detail = decide(runs, sha)
    if not run_green:
        return False, run_detail

    run = select_run(runs, sha)
    if run is None:  # pragma: no cover - decide() already returned False in this case
        raise GateError(f"internal: the run-level leg passed for {sha} but no run could be selected")

    jobs_green, jobs_detail = decide_jobs(jobs_provider(run), run.get("id", "?"))
    if not jobs_green:
        return False, (
            f"{jobs_detail}. The RUN concluded 'success' anyway — that is what a job marked "
            "continue-on-error does, and it is exactly why this gate reads jobs and not just runs."
        )
    return True, f"{run_detail}; {jobs_detail}"


def _gh_json(command: list[str], timeout: float) -> Any:
    """Run a ``gh api`` command and parse its stdout as JSON.

    ``gh`` is already the API client this workflow uses (see the code-scanning
    alert gate), so this adds no new credential path: it inherits ``GH_TOKEN``.
    """
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError as exc:
        raise GateError("the 'gh' CLI is not available; cannot read CI conclusions") from exc
    except subprocess.TimeoutExpired as exc:
        raise GateError(f"'gh api' timed out after {timeout}s") from exc
    if completed.returncode != 0:
        raise GateError(f"'gh api' failed with exit {completed.returncode}: {completed.stderr.strip()[:500]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise GateError(f"'gh api' did not return valid JSON: {exc}") from exc


def fetch_runs(repo: str, workflow: str, sha: str, timeout: float) -> list[Any]:
    """Read the CI runs recorded for ``sha`` via the ``gh`` CLI."""
    payload = _gh_json(
        [
            "gh",
            "api",
            "-X",
            "GET",
            f"repos/{repo}/actions/workflows/{workflow}/runs",
            "-f",
            f"head_sha={sha}",
            "-f",
            "per_page=100",
        ],
        timeout,
    )
    if not isinstance(payload, dict) or "workflow_runs" not in payload:
        raise GateError("'gh api' response has no 'workflow_runs' key")
    runs = payload["workflow_runs"]
    if not isinstance(runs, list):
        raise GateError(f"'workflow_runs' is not an array (got {type(runs).__name__})")
    return runs


def fetch_jobs(repo: str, run_id: Any, timeout: float) -> list[Any]:
    """Read every job of ``run_id``, following pagination to the end.

    ``filter=latest`` is the API default and is passed explicitly: after a
    re-run it returns the newest attempt of each job, which is the attempt the
    run's own conclusion reflects.

    A partially-read list is refused rather than judged. Checking 100 of 140
    jobs and reporting "all green" is the same class of error as a test selector
    that matched zero tests.
    """
    if run_id in (None, "", "?"):
        raise GateError("cannot fetch jobs: the run has no usable 'id'")
    collected: list[Any] = []
    total: int | None = None
    for page in range(1, MAX_JOB_PAGES + 1):
        payload = _gh_json(
            [
                "gh",
                "api",
                "-X",
                "GET",
                f"repos/{repo}/actions/runs/{run_id}/jobs",
                "-f",
                "filter=latest",
                "-f",
                f"per_page={JOBS_PER_PAGE}",
                "-f",
                f"page={page}",
            ],
            timeout,
        )
        if not isinstance(payload, dict) or "jobs" not in payload:
            raise GateError(f"'gh api' response for run {run_id} jobs has no 'jobs' key")
        jobs = payload["jobs"]
        if not isinstance(jobs, list):
            raise GateError(f"'jobs' for run {run_id} is not an array (got {type(jobs).__name__})")
        if total is None:
            total = payload.get("total_count")
            if not isinstance(total, int) or isinstance(total, bool) or total < 0:
                raise GateError(f"'total_count' for run {run_id} is not a count (got {payload.get('total_count')!r})")
        collected.extend(jobs)
        if not jobs or len(collected) >= total:
            break
    if total is None or len(collected) != total:
        raise GateError(f"read {len(collected)} of {total} jobs for run {run_id}; refusing to judge a partial job list")
    return collected


def _load_runs_file(path: str, sha: str) -> list[Any]:
    """Read runs from a captured payload.

    Accepts either a bare list, a raw API response (``{"workflow_runs": [...]}``),
    or a by-sha mapping of such responses (the shape of the test fixture).
    """
    with open(path, "rb") as handle:
        payload = json.loads(handle.read().decode("utf-8"))
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise GateError(f"runs file {path} is neither an array nor an object")
    if "workflow_runs" in payload:
        runs = payload["workflow_runs"]
        if not isinstance(runs, list):
            raise GateError("'workflow_runs' is not an array")
        return runs
    wanted = sha.strip().lower()
    for key, value in payload.items():
        if key.startswith("_"):  # provenance / documentation keys
            continue
        if not (key.lower().startswith(wanted) or wanted.startswith(key.lower())):
            continue
        if not isinstance(value, dict) or "workflow_runs" not in value:
            raise GateError(f"runs file entry {key!r} has no 'workflow_runs' key")
        runs = value["workflow_runs"]
        if not isinstance(runs, list):
            raise GateError(f"runs file entry {key!r} 'workflow_runs' is not an array")
        return runs
    raise GateError(f"runs file {path} has no entry for {sha}")


def _load_jobs_file(path: str, run_id: Any) -> list[Any]:
    """Read one run's jobs from a captured payload.

    Accepts a bare list, a raw API response (``{"jobs": [...]}``), or a by-run-id
    mapping of such responses. Keys beginning with ``_`` are provenance notes.
    """
    with open(path, "rb") as handle:
        payload = json.loads(handle.read().decode("utf-8"))
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise GateError(f"jobs file {path} is neither an array nor an object")
    if "jobs" in payload:
        jobs = payload["jobs"]
        if not isinstance(jobs, list):
            raise GateError(f"jobs file {path} has a 'jobs' key that is not an array")
        return jobs
    entry = payload.get(str(run_id))
    if entry is None:
        raise GateError(f"jobs file {path} has no entry for run {run_id}")
    if not isinstance(entry, dict) or "jobs" not in entry:
        raise GateError(f"jobs file entry {str(run_id)!r} has no 'jobs' key")
    jobs = entry["jobs"]
    if not isinstance(jobs, list):
        raise GateError(f"jobs file entry {str(run_id)!r} 'jobs' is not an array")
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sha", required=True, help="the exact commit sha being released")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"owner/name (default: {DEFAULT_REPO})")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW, help=f"workflow file (default: {DEFAULT_WORKFLOW})")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="gh api timeout in seconds")
    parser.add_argument("--runs-file", help="read a captured runs payload instead of calling gh (for tests/demos)")
    parser.add_argument("--jobs-file", help="read a captured jobs payload instead of calling gh (for tests/demos)")
    args = parser.parse_args(argv)

    try:
        # Offline mode is all-or-nothing. Supplying only one of the two files
        # would leave the other leg reaching for the network mid-check, and the
        # tempting "just skip the job leg then" is precisely the hole this gate
        # was widened to close.
        if bool(args.runs_file) != bool(args.jobs_file):
            raise GateError("--runs-file and --jobs-file must be supplied together; one alone would drop a leg of the gate")

        if args.runs_file:
            runs = _load_runs_file(args.runs_file, args.sha)

            def provider(run: dict[str, Any]) -> Any:
                return _load_jobs_file(args.jobs_file, run.get("id", "?"))

        else:
            runs = fetch_runs(args.repo, args.workflow, args.sha, args.timeout)

            def provider(run: dict[str, Any]) -> Any:
                return fetch_jobs(args.repo, run.get("id", "?"), args.timeout)

        green, detail = evaluate(runs, args.sha, provider)
    except GateError as exc:
        print(f"FAIL: ci-green gate could not determine the answer: {exc}")
        print("FAIL: a gate that cannot answer must not pass — refusing to release.")
        return 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"FAIL: ci-green gate could not read the payload: {exc}")
        return 1

    if green:
        print(f"OK: {detail}")
        return 0
    print(f"FAIL: {detail}")
    print("FAIL: release the commit CI actually passed, or fix CI and tag the fixed commit.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
