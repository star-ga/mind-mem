"""Mine this repository's own git history for machine-checkable agent tasks.

WHY THIS EXISTS
---------------
The product claim is that governed memory makes a coding agent *better*.
Testing that needs tasks whose success is decided by a machine, not by a
judge, and whose ground truth cannot be fabricated.  A commit that fixed a
defect and shipped the test proving it is exactly that: the test it added
must go red at the parent commit and green at the commit itself.  Git
history is the ground truth and nobody can retrofit it.

THE SELECTION RULE (stated up front; no curation for outcome)
-------------------------------------------------------------
A *candidate* is a commit reachable from the generation HEAD along
**first-parent** history, excluding merges, that BOTH

  (a) **adds** at least one new test file matching ``tests/test_*.py``
      (git ``--name-status`` letter ``A``), and
  (b) adds or modifies at least one file matching ``src/**/*.py``.

Candidates are ordered newest-first by ``(committer_date, sha)`` and the
most recent ``limit`` are handed to validation.  No commit is selected,
reordered, or dropped on the basis of its subject text, its author, or any
expectation about whether memory would help on it.

THE ONLY PRE-EXCLUSION (safety, not outcome)
--------------------------------------------
A candidate whose **test-side patch** names a **shared production service**
is excluded before anything is executed -- see ``SHARED_SERVICE_PATTERN``.
This box runs a production Postgres and a pinned GPU model; the benchmark
must not touch either.  Every other exclusion is decided by *executing*
the task, never by guessing from its text.

The scanned set is deliberately the *executed* set: every path in
``test_patch_paths`` (the whole ``tests/`` delta plus the root conftest, all
of which the validation harness lays onto the parent tree, and which pytest
imports at collection time).  Scanning only the added ``tests/test_*.py``
files left the hole this exclusion exists to close -- a commit that adds a
benign test and *modifies* ``tests/conftest.py`` to add a psycopg fixture
passed the exclusion and then ran that fixture against the live server.

Every git call here is read-only plumbing with a fixed argv and
``shell=False``; nothing this module builds is ever handed to a shell.

WHAT IS DELIBERATELY NOT USED
-----------------------------
The commit **body** is never read.  It routinely contains the reasoning for
the fix, which is precisely the thing a memory arm is supposed to supply --
feeding it to the agent would leak the answer into the prompt.  Only the
single-line subject is used, as the "reported issue" statement.
"""

from __future__ import annotations

import re
import subprocess  # nosec B404
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

#: Test-side source naming a shared production service.  Matching candidates
#: are excluded before execution: this host runs a production Postgres on
#: 127.0.0.1:5432 and a GPU-pinned local model, and a benchmark that mutates
#: either is not a benchmark, it is an outage.
SHARED_SERVICE_PATTERN = re.compile(r"(?i)\b(psycopg|postgres|postgresql|redis|ollama)\b")

_FIELD_SEP = "\x01"
_COMMIT_MARK = "@@@"


@dataclass(frozen=True)
class Candidate:
    """One commit that satisfies the selection rule, before validation."""

    sha: str
    parent_sha: str
    committed_at: str
    parent_committed_at: str
    subject: str
    added_test_files: tuple[str, ...]
    test_patch_paths: tuple[str, ...]
    src_changed: tuple[str, ...]
    files_changed: tuple[str, ...]
    src_added_lines: int = 0
    src_deleted_lines: int = 0
    excluded_reason: str | None = None


@dataclass
class MiningStats:
    """Counts for every stage of the rule, so the yield is auditable."""

    commits_scanned: int = 0
    rule_matched: int = 0
    excluded_shared_service: int = 0
    eligible: int = 0
    selected: int = 0
    excluded_detail: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "commits_scanned": self.commits_scanned,
            "rule_matched": self.rule_matched,
            "excluded_shared_service": self.excluded_shared_service,
            "eligible": self.eligible,
            "selected": self.selected,
            "excluded_detail": self.excluded_detail,
        }


def git(repo: str, *args: str) -> str:
    """Run a read-only git command in ``repo`` and return stdout."""
    # Fixed argv, shell=False, read-only git subcommands.
    proc = subprocess.run(  # nosec B603 B607
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return proc.stdout


def is_test_path(path: str) -> bool:
    """True for ``tests/test_*.py`` (the only files the rule counts as tests)."""
    return path.startswith("tests/") and path.rsplit("/", 1)[-1].startswith("test_") and path.endswith(".py")


def is_src_path(path: str) -> bool:
    """True for a Python source file under ``src/``."""
    return path.startswith("src/") and path.endswith(".py")


def is_test_infra_path(path: str) -> bool:
    """True for any test-side file: the ``tests/`` tree and the root conftest.

    The whole test-side delta is laid onto the parent tree before the parent
    run, the way a fail-to-pass harness applies a test patch.  Without it a
    new test importing a helper added in the same commit would fail at the
    parent for a bookkeeping reason rather than for the defect.
    """
    return path.startswith("tests/") or path == "conftest.py"


def _parse_name_status(raw: str) -> dict[str, list[tuple[str, str]]]:
    """Split one ``git log --name-status`` stream into per-sha file lists."""
    per: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for line in raw.splitlines():
        if line.startswith(_COMMIT_MARK):
            current = line[len(_COMMIT_MARK) :].strip()
            per.setdefault(current, [])
        elif line and current and "\t" in line:
            status, path = line.split("\t", 1)
            per[current].append((status[0], path.strip()))
    return per


def _utc_iso(epoch: str) -> str:
    """Format a git unix timestamp as UTC ISO-8601.

    Deliberately not ``%cI``: that renders the commit's *authoring* offset,
    so the same commit reads differently depending on where it was made and
    the generated task file would not be byte-stable across hosts.
    """
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def commit_time_index(repo: str, head: str) -> dict[str, str]:
    """sha -> UTC ISO commit time, over every ancestor of ``head``."""
    raw = git(repo, "log", f"--format=%H{_FIELD_SEP}%ct", head)
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if _FIELD_SEP in line:
            sha, epoch = line.split(_FIELD_SEP, 1)
            out[sha] = _utc_iso(epoch)
    return out


def _commit_headers(repo: str, head: str) -> list[tuple[str, str, str, str]]:
    """(sha, parent_sha, committer_time_utc, subject) newest-first."""
    fmt = _FIELD_SEP.join(["%H", "%P", "%ct", "%s"])
    raw = git(repo, "log", "--first-parent", "--no-merges", f"--format={fmt}", head)
    rows: list[tuple[str, str, str, str]] = []
    for line in raw.splitlines():
        if not line:
            continue
        sha, parents, epoch, subject = line.split(_FIELD_SEP, 3)
        parent = parents.split(" ")[0] if parents else ""
        rows.append((sha, parent, _utc_iso(epoch), subject))
    return rows


def _file_index(repo: str, head: str) -> dict[str, list[tuple[str, str]]]:
    """One pass over history producing per-commit ``(status, path)`` pairs."""
    raw = git(
        repo,
        "log",
        "--first-parent",
        "--no-merges",
        f"--format={_COMMIT_MARK}%H",
        "--name-status",
        "--no-renames",
        head,
    )
    return _parse_name_status(raw)


def _numstat_index(repo: str, head: str) -> dict[str, dict[str, tuple[int, int]]]:
    """sha -> path -> (added, deleted) line counts.

    Size is recorded, never used to select.  A benchmark that quietly kept
    only the small commits would be curating for outcome; recording the
    size lets a harness stratify against a threshold it has to state.
    """
    raw = git(repo, "log", "--first-parent", "--no-merges", f"--format={_COMMIT_MARK}%H", "--numstat", "--no-renames", head)
    per: dict[str, dict[str, tuple[int, int]]] = {}
    current: str | None = None
    for line in raw.splitlines():
        if line.startswith(_COMMIT_MARK):
            current = line[len(_COMMIT_MARK) :].strip()
            per.setdefault(current, {})
        elif line and current and line.count("\t") >= 2:
            added, deleted, path = line.split("\t", 2)
            if added.isdigit() and deleted.isdigit():
                per[current][path.strip()] = (int(added), int(deleted))
    return per


def _test_patch_sources(repo: str, sha: str, patch_paths: Iterable[str]) -> str:
    """Concatenate every file of the test-side patch as it exists *at* ``sha``.

    This is the text the shared-service exclusion reads, and it must cover
    exactly what the harness later executes: the whole ``test_patch_paths``
    set is copied onto the parent tree, and pytest imports ``conftest.py``
    from it at collection.  Reading only the *added* ``test_*.py`` files
    scanned a strict subset of what runs.
    """
    chunks = []
    for path in patch_paths:
        try:
            chunks.append(git(repo, "show", f"{sha}:{path}"))
        except subprocess.CalledProcessError:  # pragma: no cover - defensive
            continue
    return "\n".join(chunks)


def _build_candidate(
    repo: str,
    header: tuple[str, str, str, str],
    files: Sequence[tuple[str, str]],
    times: dict[str, str],
    numstat: dict[str, tuple[int, int]],
) -> Candidate | None:
    """Apply rule (a)+(b) to one commit, then the shared-service exclusion."""
    sha, parent, when, subject = header
    added_tests = tuple(sorted(p for st, p in files if st == "A" and is_test_path(p)))
    patch_paths = tuple(sorted(p for st, p in files if st in ("A", "M") and is_test_infra_path(p)))
    src_changed = tuple(sorted(p for st, p in files if st in ("A", "M") and is_src_path(p)))
    if not added_tests or not src_changed or not parent:
        return None
    # Scan the executed set, not the selected-on set: `patch_paths` is a
    # superset of `added_tests` (every added tests/test_*.py is test infra)
    # and is precisely what repo_task_validation writes into the extracted
    # tree before the run.
    source = _test_patch_sources(repo, sha, patch_paths)
    hit = SHARED_SERVICE_PATTERN.search(source)
    churn = [numstat.get(path, (0, 0)) for path in src_changed]
    return Candidate(
        sha=sha,
        parent_sha=parent,
        committed_at=when,
        parent_committed_at=times.get(parent, ""),
        subject=subject,
        added_test_files=added_tests,
        test_patch_paths=patch_paths,
        src_changed=src_changed,
        files_changed=tuple(sorted(p for _, p in files)),
        src_added_lines=sum(a for a, _ in churn),
        src_deleted_lines=sum(d for _, d in churn),
        excluded_reason=f"shared_service:{hit.group(1).lower()}" if hit else None,
    )


def select_candidates(repo: str, head: str = "HEAD", limit: int = 80) -> tuple[list[Candidate], MiningStats]:
    """Apply the stated rule and return the most recent ``limit`` candidates."""
    headers = _commit_headers(repo, head)
    index = _file_index(repo, head)
    times = commit_time_index(repo, head)
    numstat = _numstat_index(repo, head)
    stats = MiningStats(commits_scanned=len(headers))
    eligible: list[Candidate] = []
    for header in headers:
        candidate = _build_candidate(repo, header, index.get(header[0], []), times, numstat.get(header[0], {}))
        if candidate is None:
            continue
        stats.rule_matched += 1
        if candidate.excluded_reason:
            stats.excluded_shared_service += 1
            stats.excluded_detail.append({"sha": candidate.sha, "reason": candidate.excluded_reason})
            continue
        eligible.append(candidate)
    stats.eligible = len(eligible)
    eligible.sort(key=lambda c: (c.committed_at, c.sha), reverse=True)
    selected = eligible[:limit]
    stats.selected = len(selected)
    return selected, stats
