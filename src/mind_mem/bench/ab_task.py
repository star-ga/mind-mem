"""The unit of the with-memory versus without-memory comparison.

A task is one row of ``benchmarks/tasks/real_repo_tasks.json``: a commit
from this repository's own history whose added test file is red at the
parent commit and green at the commit itself.  The generator proved that
transition by *executing* it (see :mod:`mind_mem.bench.repo_task_validation`),
so success here is decided by pytest, never by a judge.

Only the fields the A/B harness is allowed to use are lifted into
:class:`Task`.  The reference fix's diff is deliberately absent: it is
ground truth for grading the *generator*, not material either arm may see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

#: Keys a task record must carry.  A task set missing any of them is
#: rejected loudly rather than silently degrading a run.
REQUIRED_FIELDS: tuple[str, ...] = (
    "task_id",
    "sha",
    "parent_sha",
    "memory_cutoff",
    "scoring_instant",
    "subject",
    "task_statement",
    "tests_to_run",
    "test_patch_paths",
    "fail_to_pass",
)


class TaskSetError(ValueError):
    """The task set on disk is not usable as a benchmark input."""


@dataclass(frozen=True)
class Task:
    """One machine-checkable repository task, identical in both arms."""

    task_id: str
    sha: str
    parent_sha: str
    memory_cutoff: str
    scoring_instant: str
    subject: str
    task_statement: str
    tests_to_run: tuple[str, ...]
    test_patch_paths: tuple[str, ...]
    fail_to_pass: tuple[str, ...]
    tier: str = "unknown"
    size_bucket: str = "unknown"

    @property
    def scoring_date(self) -> date:
        """The instant recall scores against, so recency is not wall-clock."""
        return date.fromisoformat(self.scoring_instant)

    @property
    def recall_query(self) -> str:
        """The query the memory arm asks with.

        Derived only from material the agent already holds -- the reported
        issue and the names of the tests it must turn green.  Nothing here
        comes from the fix, so the query cannot smuggle the answer in.
        """
        names = " ".join(path.rsplit("/", 1)[-1][5:-3].replace("_", " ") for path in self.tests_to_run)
        return f"{self.subject} {names}".strip()


def task_from_record(record: dict) -> Task:
    """Lift one artifact row into a :class:`Task`, or refuse it."""
    missing = [key for key in REQUIRED_FIELDS if key not in record]
    if missing:
        raise TaskSetError(f"task record is missing required field(s): {', '.join(missing)}")
    return Task(
        task_id=str(record["task_id"]),
        sha=str(record["sha"]),
        parent_sha=str(record["parent_sha"]),
        memory_cutoff=str(record["memory_cutoff"]),
        scoring_instant=str(record["scoring_instant"]),
        subject=str(record["subject"]),
        task_statement=str(record["task_statement"]),
        tests_to_run=tuple(str(p) for p in record["tests_to_run"]),
        test_patch_paths=tuple(str(p) for p in record["test_patch_paths"]),
        fail_to_pass=tuple(str(n) for n in record["fail_to_pass"]),
        tier=str(record.get("tier", "unknown")),
        size_bucket=str(record.get("size_bucket", "unknown")),
    )


def load_task_set(path: str) -> tuple[Task, ...]:
    """Read the generated task set, preserving its recorded order."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise TaskSetError(f"{path} carries no tasks; regenerate it with mind-mem-bench-tasks")
    return tuple(task_from_record(row) for row in rows)


def select_tasks(tasks: tuple[Task, ...], spec: str) -> tuple[Task, ...]:
    """Apply a stated, mechanical selection to the task set.

    ``all`` | ``first:<n>`` | ``bucket:<name>`` | ``bucket:<name>:<n>`` |
    ``task:<id>``.  Order is always the artifact's own order (newest
    first), so a selection is a prefix of a stated stratum and never a
    hand-picked set.  The spec string is recorded in the run artifact.
    """
    if spec == "all":
        return tasks
    head, _, rest = spec.partition(":")
    if head == "first" and rest.isdigit():
        return tasks[: int(rest)]
    if head == "task":
        chosen = tuple(t for t in tasks if t.task_id == rest)
        if not chosen:
            raise TaskSetError(f"no task with id {rest!r} in the task set")
        return chosen
    if head == "bucket":
        name, _, count = rest.partition(":")
        subset = tuple(t for t in tasks if t.size_bucket == name)
        if not subset:
            raise TaskSetError(f"no tasks in size bucket {name!r}")
        return subset[: int(count)] if count.isdigit() else subset
    raise TaskSetError(f"unrecognised selection spec {spec!r}")
