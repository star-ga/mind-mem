"""Machine-checked grading. A task passes iff its named tests pass.

No model reads the attempt, no rubric is applied and no partial credit is
awarded.  The verdict is a function of two things: pytest's exit code, and
whether every node id the generator proved red-at-parent is now green.

Tampering is graded too.  The task set delivers the commit's test files
into the tree before the agent runs, so an agent that edits a test can
make it pass without fixing anything.  Every test-side file is hashed
before and after the attempt; any change voids the attempt with an
explicit ``tampered`` verdict rather than a silent pass.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from .ab_task import Task
from .repo_task_validation import RunResult, run_pytest

#: Paths an attempt may not touch. Mirrors the rule stated in the prompt.
PROTECTED_PREFIX = "tests/"
PROTECTED_FILES = ("conftest.py",)

#: Directories excluded from the integrity snapshot: build artefacts, not source.
_IGNORED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis"})


@dataclass(frozen=True)
class Verdict:
    """One graded attempt."""

    success: bool
    reason: str
    returncode: int
    timed_out: bool
    passed_required: tuple[str, ...]
    missing_required: tuple[str, ...]
    tampered_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    tail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "reason": self.reason,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "n_passed_required": len(self.passed_required),
            "missing_required": list(self.missing_required),
            "tampered_paths": list(self.tampered_paths),
            "changed_paths": list(self.changed_paths),
            "tail": self.tail,
        }


def snapshot_tree(tree: str) -> dict[str, str]:
    """Hash every tracked file in ``tree`` so any edit is visible."""
    digests: dict[str, str] = {}
    for root, dirs, files in os.walk(tree):
        dirs[:] = sorted(d for d in dirs if d not in _IGNORED_DIRS)
        for name in sorted(files):
            path = os.path.join(root, name)
            # POSIX separators: `rel` is a KEY in the returned snapshot, and
            # changed_paths() publishes those keys into the run artifact. An
            # os-native separator would make the same edit read as a different
            # path on Windows than on Linux, and would not match PROTECTED_FILES
            # (is_protected normalises, but the published list would not).
            rel = os.path.relpath(path, tree).replace(os.sep, "/")
            try:
                with open(path, "rb") as handle:
                    digests[rel] = hashlib.sha256(handle.read()).hexdigest()
            except OSError:  # pragma: no cover - unreadable transient file
                digests[rel] = "unreadable"
    return digests


def changed_paths(before: dict[str, str], after: dict[str, str]) -> tuple[str, ...]:
    """Paths added, removed or modified between two snapshots."""
    return tuple(sorted(set(before) ^ set(after) | {k for k in set(before) & set(after) if before[k] != after[k]}))


def is_protected(path: str) -> bool:
    """True for a path the attempt was told not to touch."""
    normalised = path.replace(os.sep, "/")
    return normalised.startswith(PROTECTED_PREFIX) or normalised in PROTECTED_FILES


def grade(task: Task, tree: str, home: str, python: str, before: dict[str, str], timeout: int) -> Verdict:
    """Run the named tests and decide, from counts alone, whether it passed."""
    after = snapshot_tree(tree)
    changed = changed_paths(before, after)
    tampered = tuple(p for p in changed if is_protected(p))
    if tampered:
        return _verdict_from(task, RunResult(returncode=-2, timed_out=False, node_status={}, tail=""), changed, tampered, "tampered")
    result = run_pytest(python, tree, home, task.tests_to_run, timeout)
    return _verdict_from(task, result, changed, (), "")


def _verdict_from(task: Task, result: RunResult, changed: tuple[str, ...], tampered: tuple[str, ...], forced: str) -> Verdict:
    """Turn a raw pytest run into a verdict. Pure counting, no judgement."""
    passed = result.passed_nodes
    required = tuple(sorted(task.fail_to_pass))
    missing = tuple(node for node in required if node not in passed)
    if forced:
        reason, success = forced, False
    elif result.timed_out:
        reason, success = "timeout", False
    elif result.returncode != 0:
        reason, success = "tests_failed", False
    elif missing:
        reason, success = "required_nodes_not_passing", False
    else:
        reason, success = "all_required_nodes_passing", True
    return Verdict(
        success=success,
        reason=reason,
        returncode=result.returncode,
        timed_out=result.timed_out,
        passed_required=tuple(node for node in required if node in passed),
        missing_required=missing,
        tampered_paths=tampered,
        changed_paths=changed,
        tail=result.tail,
    )
