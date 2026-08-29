"""Generate ``benchmarks/tasks/real_repo_tasks.json`` -- the A/B task set.

Run it::

    mind-mem-bench-tasks --limit 120 --out benchmarks/tasks/real_repo_tasks.json

Mining rule and exclusions live in :mod:`mind_mem.bench.repo_task_mining`;
the red->green proof lives in :mod:`mind_mem.bench.repo_task_validation`.
This module only joins the two, derives the prompt an agent is given, and
serialises the result.

The output carries no wall-clock stamp anywhere.  Re-running at the same
HEAD reproduces the file byte for byte, which is the whole point: an A/B
number is only worth reading if the task set under it cannot drift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .repo_task_mining import Candidate, MiningStats, git, select_candidates
from .repo_task_validation import Validation, validate

SCHEMA_VERSION = 1

#: Field-by-field meaning, written into the artifact so the file documents
#: itself and a reader never has to guess what a column means.
SCHEMA_DOC: dict[str, str] = {
    "task_id": "Stable id for the task: 'mm-' + the first 12 hex of sha.",
    "sha": "The commit that fixed the defect. Ground truth; never shown to the agent.",
    "parent_sha": "The pre-task state. The agent starts from a tree extracted at this commit.",
    "committed_at": "UTC ISO-8601 commit time of sha.",
    "parent_committed_at": "UTC ISO-8601 commit time of parent_sha.",
    "memory_cutoff": (
        "Equals parent_committed_at. The memory arm may be seeded ONLY with material "
        "authored strictly before this instant. Seeding at or after it leaks the answer "
        "and the resulting delta measures the leak, not the memory."
    ),
    "scoring_instant": (
        "Date (YYYY-MM-DD) of parent_committed_at, to be passed to recall(scoring_instant=...) "
        "so recency scoring is a pure function of the corpus instead of the wall clock."
    ),
    "subject": "The commit's single-line subject. The body is never read (it leaks the fix).",
    "task_statement": "The prompt an agent is given. Derived from subject only; identical in both arms.",
    "tests_to_run": "Test files executed to grade the attempt (the commit's newly added test files).",
    "test_patch_paths": "Test-side files laid onto the parent tree before grading, so imports resolve.",
    "fail_to_pass": "Node ids that do NOT pass at parent_sha and DO pass at sha. Non-empty by construction.",  # nosec B105
    "tier": "'behavioral' = module imported at parent and behaved differently. 'api_construction' = module did not import at parent.",
    "parent_mode": "Raw pytest outcome class at the parent: test_failure | collection_error | timeout.",
    "parent_returncode": "pytest exit code at parent_sha with the test patch applied (must be non-zero).",
    "task_returncode": "pytest exit code at sha (must be 0).",
    "parent_passed_count": "Node ids passing at parent.",
    "task_passed_count": "Node ids passing at sha.",
    "n_src_files": "How many files under src/ the reference fix touched.",
    "n_tests_to_run": "How many test files are executed to grade the attempt.",
    "n_fail_to_pass": "Size of the flip set.",  # nosec B105
    "src_added_lines": "Lines added under src/ by the reference fix.",
    "src_deleted_lines": "Lines deleted under src/ by the reference fix.",
    "size_bucket": (
        "Mechanical size class from n_src_files alone: 1='single_file', 2-3='small', 4-8='medium', "
        ">8='large'. Recorded so a harness can stratify against a threshold it states up front. "
        "It is NOT used to select tasks -- dropping the large ones during generation would be "
        "curating the set for outcome."
    ),
    "src_changed": "Files under src/ the reference fix touched. Metadata only; not in the prompt.",
    "files_changed": "Every path the commit touched. Metadata only; not in the prompt.",
}

#: Mechanical size classes, keyed on the reference fix's src-file count.
_SIZE_BUCKETS: tuple[tuple[int, str], ...] = ((1, "single_file"), (3, "small"), (8, "medium"))


def size_bucket(n_src_files: int) -> str:
    """Classify a task by how many source files the reference fix touched."""
    for ceiling, name in _SIZE_BUCKETS:
        if n_src_files <= ceiling:
            return name
    return "large"


_STATEMENT = """\
Repository: mind-mem (Python, src layout; the package lives under src/mind_mem/).
You are working from a clean checkout of commit {parent_short}.

Reported issue
--------------
{subject}

Definition of done
------------------
These test files must pass, unchanged:
{test_list}

Rules
-----
- Fix the cause in the source. Do not edit, weaken, skip or delete any test.
- Do not edit files under tests/ or conftest.py.
- Confine changes to src/ (documentation may also be updated).
"""


def derive_task_statement(candidate: Candidate) -> str:
    """Build the agent-facing prompt from the subject line alone.

    Two things are deliberately withheld: the diff, and the commit body.
    The body is where the reasoning for a fix is written, which is exactly
    what a memory arm is meant to supply -- putting it in the prompt would
    hand both arms the answer and flatten the measurement.
    """
    listing = "\n".join(f"  - {path}" for path in candidate.added_test_files)
    return _STATEMENT.format(
        parent_short=candidate.parent_sha[:12],
        subject=candidate.subject,
        test_list=listing,
    )


def _task_record(candidate: Candidate, result: Validation) -> dict[str, Any]:
    """Serialise one validated candidate into the artifact's task shape."""
    return {
        "task_id": f"mm-{candidate.sha[:12]}",
        "sha": candidate.sha,
        "parent_sha": candidate.parent_sha,
        "committed_at": candidate.committed_at,
        "parent_committed_at": candidate.parent_committed_at,
        "memory_cutoff": candidate.parent_committed_at,
        "scoring_instant": candidate.parent_committed_at[:10],
        "subject": candidate.subject,
        "task_statement": derive_task_statement(candidate),
        "tests_to_run": list(candidate.added_test_files),
        "test_patch_paths": list(candidate.test_patch_paths),
        "fail_to_pass": list(result.fail_to_pass),
        "tier": result.tier,
        "parent_mode": result.parent_mode,
        "parent_returncode": result.parent_returncode,
        "task_returncode": result.task_returncode,
        "parent_passed_count": result.parent_passed_count,
        "task_passed_count": result.task_passed_count,
        "n_src_files": len(candidate.src_changed),
        "n_tests_to_run": len(candidate.added_test_files),
        "n_fail_to_pass": len(result.fail_to_pass),
        "src_added_lines": candidate.src_added_lines,
        "src_deleted_lines": candidate.src_deleted_lines,
        "size_bucket": size_bucket(len(candidate.src_changed)),
        "src_changed": list(candidate.src_changed),
        "files_changed": list(candidate.files_changed),
    }


def _drop_record(candidate: Candidate, result: Validation) -> dict[str, Any]:
    """Every rejected candidate is recorded, never silently discarded."""
    return {
        "sha": candidate.sha,
        "subject": candidate.subject,
        "drop_reason": result.drop_reason,
        "parent_returncode": result.parent_returncode,
        "task_returncode": result.task_returncode,
        "parent_mode": result.parent_mode,
        "task_tail": result.task_tail,
    }


def _validate_one(repo: str, candidate: Candidate, python: str, timeout: int, repeats: int) -> tuple[Candidate, Validation]:
    workdir = tempfile.mkdtemp(prefix=f"mmtask_{candidate.sha[:8]}_")
    return candidate, validate(repo, candidate, workdir, python, timeout=timeout, repeats=repeats)


def run(repo: str, head: str, limit: int, jobs: int, python: str, timeout: int, repeats: int = 2) -> dict[str, Any]:
    """Mine, validate, and assemble the full artifact payload."""
    candidates, stats = select_candidates(repo, head, limit=limit)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        outcomes = list(pool.map(lambda c: _validate_one(repo, c, python, timeout, repeats), candidates))
    outcomes.sort(key=lambda pair: (pair[0].committed_at, pair[0].sha), reverse=True)
    tasks = [_task_record(c, v) for c, v in outcomes if v.well_formed]
    drops = [_drop_record(c, v) for c, v in outcomes if not v.well_formed]
    return _assemble(repo, head, python, repeats, stats, tasks, drops)


def _assemble(
    repo: str,
    head: str,
    python: str,
    repeats: int,
    stats: MiningStats,
    tasks: list[dict[str, Any]],
    drops: list[dict[str, Any]],
) -> dict[str, Any]:
    """Wrap tasks + drops in the self-documenting artifact envelope."""
    head_sha = git(repo, "rev-parse", head).strip()
    reasons: dict[str, int] = {}
    tiers: dict[str, int] = {}
    buckets: dict[str, int] = {}
    for drop in drops:
        reasons[str(drop["drop_reason"])] = reasons.get(str(drop["drop_reason"]), 0) + 1
    for task in tasks:
        tiers[str(task["tier"])] = tiers.get(str(task["tier"]), 0) + 1
        buckets[str(task["size_bucket"])] = buckets.get(str(task["size_bucket"]), 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA_DOC,
        "generator": {
            "module": "mind_mem.bench.repo_task_cli",
            "entry_point": "mind-mem-bench-tasks",
            "selection_rule": (
                "Non-merge commits on first-parent history from HEAD that BOTH add at least one new "
                "tests/test_*.py file AND add or modify at least one src/**/*.py file; ordered newest "
                "first by (committer_date, sha); the most recent --limit are validated. No commit is "
                "selected, reordered or dropped on the basis of its text, its author, or any expectation "
                "about whether memory would help."
            ),
            "pre_exclusion": (
                "A candidate whose added test source names a shared production service "
                "(psycopg|postgres|postgresql|redis|ollama) is excluded before execution, because this "
                "host runs a production Postgres and a GPU-pinned model. Every other exclusion is decided "
                "by executing the candidate."
            ),
            "validation_rule": (
                "Extract parent_sha and sha with git archive; lay the commit's test-side files onto the "
                "parent tree; run the added test files in both. Well-formed iff pytest exits non-zero at "
                "the parent, exits 0 at the commit, and at least one node id flips to passing. "
                f"Each arm runs {repeats} times in independent trees; a candidate whose own grading "
                "disagrees with itself between runs is dropped as nondeterministic_grading."
            ),
            "repeats": repeats,
            "test_delivery": (
                "provided_before_run: the commit's test-side files are laid onto the parent tree BEFORE "
                "the agent runs, in both arms. Withholding them is not a viable alternative here -- a "
                "task whose tier is api_construction requires names that exist nowhere in the parent "
                "tree, so a withheld-test run would fail in both arms for a reason unrelated to memory "
                "and would report a null result caused by the harness. Delivery is identical in both "
                "arms, so it cannot bias the delta."
            ),
            "determinism": (
                "Child environment is constructed, not inherited (TZ=UTC, PYTHONHASHSEED=0, LC_ALL=C.UTF-8, "
                "no MIND_MEM_* DSN reaches a run). No wall-clock value is written to this file; the same "
                "HEAD reproduces it byte for byte."
            ),
        },
        "provenance": {"repo": os.path.basename(os.path.abspath(repo)), "head_sha": head_sha, "python": python},
        "selection": stats.as_dict(),
        "validation": {
            "validated": len(tasks) + len(drops),
            "well_formed": len(tasks),
            "dropped": len(drops),
            "drop_reasons": dict(sorted(reasons.items())),
            "tiers": dict(sorted(tiers.items())),
            "size_buckets": dict(sorted(buckets.items())),
        },
        "dropped": drops,
        "tasks": tasks,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the real-repo A/B task set from git history.")
    parser.add_argument("--repo", default=os.getcwd(), help="Repository to mine (default: cwd).")
    parser.add_argument("--head", default="HEAD", help="Ref the task set is anchored to.")
    parser.add_argument("--limit", type=int, default=120, help="Most recent N candidates to validate.")
    parser.add_argument("--jobs", type=int, default=3, help="Parallel validations (this box is shared; keep it small).")
    parser.add_argument("--python", default=sys.executable, help="Interpreter used for the sandboxed pytest runs.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-pytest-run wall-clock ceiling, seconds.")
    parser.add_argument("--repeats", type=int, default=2, help="Runs per arm; disagreement between them drops the task.")
    parser.add_argument("--out", default="benchmarks/tasks/real_repo_tasks.json", help="Artifact path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Prints machine-greppable count and yield lines."""
    args = _parse_args(argv)
    payload = run(args.repo, args.head, args.limit, args.jobs, args.python, args.timeout, args.repeats)
    out = args.out if os.path.isabs(args.out) else os.path.join(args.repo, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, ensure_ascii=False)
        handle.write("\n")
    validation = payload["validation"]
    total = max(1, int(validation["validated"]))
    print(json.dumps({"selection": payload["selection"] | {"excluded_detail": "omitted"}, "validation": validation}, indent=2))
    print(f"real_repo_tasks_count: {validation['well_formed']}")
    print(f"real_repo_tasks_yield: {validation['well_formed'] / total:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry
    raise SystemExit(main())
