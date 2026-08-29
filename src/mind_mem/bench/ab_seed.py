"""Seed the memory arm from material that existed BEFORE the task's commit.

WHY THIS FILE IS THE HONESTY LOAD-BEARING ONE
---------------------------------------------
The cheapest way to fake this benchmark is to seed memory with the answer
and then measure that memory returns the answer.  The defence here is not
a convention or a code-review note; it is git ancestry, checked in code:

* The seed corpus is enumerated by ``git log <parent_sha>`` -- exactly
  the commits **reachable from the parent**.  A commit cannot be an
  ancestor of its own parent, so the task's own commit, and every commit
  after it, is structurally unreachable.  This is a DAG property, not a
  filter someone can forget to apply.
* Every seeded record's committer instant is then re-checked against
  ``memory_cutoff`` independently of reachability, so a rewritten or
  grafted history cannot smuggle a later commit in through the first check.
* The rendered corpus is scanned for the task commit's own identifier.

Any of the three failing raises :class:`SeedLeakError` and the run stops.
A benchmark that silently seeded the answer would report a flattering
number nobody could reproduce, which is worse than no number at all.

WHAT IS SEEDED (one rule, no per-task curation)
-----------------------------------------------
Every pre-cutoff commit's subject and body, rendered as one memory block
each.  That is the project's own rationale record -- the "why" a governed
memory is supposed to hold -- and it is applied identically to every task.
Nothing is picked because it looks likely to help on a particular task.

The extracted work tree has no ``.git`` (it is produced by ``git archive``),
so this history is genuinely unavailable to the arm that has no memory.

HOW IT IS WRITTEN
-----------------
Through ``GovernanceGate.admit_proposal`` and ``BlockStore.write_block`` --
the same governed path the product claims every write takes -- and not by
appending to the corpus file.  An earlier draft did append directly; this
repository's own structural invariant
(``tests/test_governed_write_paths.py``) refused it, which is the invariant
working.  A benchmark whose memory arm was seeded by bypassing governance
would be measuring something the product does not ship.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .ab_task import Task
from .repo_task_mining import git

_RECORD_SEP = "\x1e"
_FIELD_SEP = "\x1f"

#: C0 controls minus tab/newline; stripped so a commit body can never
#: inject a field break into the rendered corpus.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

#: Conventional-commit prefix, used only to derive tags.
_CC_PREFIX = re.compile(r"^([a-z]+)(?:\(([^)]+)\))?!?:")

#: Deterministic cap on one rendered statement.  Truncation is marked, so
#: a reader can tell a trimmed block from a short one.
MAX_STATEMENT_CHARS = 1000

#: Block-id prefix. ``D`` is what routes a block to ``decisions/DECISIONS.md``
#: through the store's own prefix map, so the corpus lands where recall reads
#: it without this module hard-coding a path.
BLOCK_PREFIX = "D"


class SeedLeakError(AssertionError):
    """Seeding would have exposed material from at or after the cutoff."""


@dataclass(frozen=True)
class SeedRecord:
    """One pre-cutoff commit, as it will be offered to recall."""

    sha: str
    committed_at: str
    subject: str
    body: str


@dataclass(frozen=True)
class SeedReport:
    """What was seeded, and the checks that licensed it."""

    workspace: str
    blocks: int
    corpus_bytes: int
    newest_seeded_at: str
    cutoff: str
    checks: tuple[str, ...]
    #: Commits whose message could not be split into the four expected
    #: fields (a body containing the record separator).  They are omitted
    #: from the corpus, so the count is published rather than dropped --
    #: a seed that is quietly smaller than it claims is still a seed
    #: nobody can reproduce.
    skipped_malformed: int = 0


def _utc_iso(epoch: str) -> str:
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def collect_history(repo: str, rev: str) -> tuple[tuple[SeedRecord, ...], int]:
    """Every commit reachable from ``rev``, plus the count it could not parse."""
    fmt = _FIELD_SEP.join(["%H", "%ct", "%s", "%b"]) + _RECORD_SEP
    raw = git(repo, "log", f"--format={fmt}", rev)
    records: list[SeedRecord] = []
    malformed = 0
    for chunk in raw.split(_RECORD_SEP):
        parts = chunk.strip("\n").split(_FIELD_SEP)
        if not chunk.strip():
            continue
        if len(parts) != 4 or not parts[0].strip():
            malformed += 1
            continue
        sha, epoch, subject, body = parts
        records.append(SeedRecord(sha=sha.strip(), committed_at=_utc_iso(epoch), subject=subject, body=body))
    return tuple(records), malformed


def assert_no_leak(records: tuple[SeedRecord, ...], task: Task) -> tuple[str, ...]:
    """Refuse a corpus that reaches the task commit or anything after it.

    Returns the names of the checks that passed, so the run artifact can
    record *which* guarantees were actually evaluated rather than asserting
    that some unnamed guarantee held.
    """
    if any(record.sha == task.sha for record in records):
        raise SeedLeakError(f"{task.task_id}: the task commit {task.sha[:12]} is inside its own seed corpus")
    late = [r.sha[:12] for r in records if r.committed_at > task.memory_cutoff]
    if late:
        raise SeedLeakError(f"{task.task_id}: {len(late)} seed record(s) at or after the cutoff {task.memory_cutoff}: {late[:5]}")
    # Case-insensitive: block ids carry the sha upper-cased, so a lower-case
    # comparison alone would miss a record that names the task commit in the
    # other case.
    marker = task.sha[:12].lower()
    hits = [r.sha[:12] for r in records if marker in f"{r.subject}\n{r.body}".lower()]
    if hits:
        raise SeedLeakError(f"{task.task_id}: seed text names the task commit {marker} (in {hits[:5]})")
    return ("task_commit_unreachable", "every_record_at_or_before_cutoff", "task_commit_id_absent_from_text")


def _clean(text: str) -> str:
    """Collapse a commit message to one injection-safe line."""
    return " ".join(_CONTROL.sub(" ", text).split())


def _tags(subject: str) -> str:
    match = _CC_PREFIX.match(subject.strip())
    if not match:
        return "commit"
    kind, scope = match.group(1), match.group(2)
    return f"commit, {kind}, {scope}" if scope else f"commit, {kind}"


def statement_of(record: SeedRecord) -> str:
    """One injection-safe line: the subject, then the body that explains it.

    The body is where a commit records *why*, which is the whole reason a
    memory would be worth having.  It is collapsed to a single line so no
    body can forge a block header or a field, and capped so one essay
    cannot crowd the corpus.
    """
    statement = _clean(f"{record.subject} -- {record.body}" if record.body.strip() else record.subject)
    if len(statement) > MAX_STATEMENT_CHARS:
        statement = statement[:MAX_STATEMENT_CHARS] + " [truncated]"
    return statement


def block_record(record: SeedRecord) -> dict[str, str]:
    """One commit as a block, ready for the governed write path."""
    return {
        "_id": f"{BLOCK_PREFIX}-{record.sha[:12].upper()}",
        "Type": "Decision",
        "Statement": statement_of(record),
        "Status": "active",
        "Date": record.committed_at[:10],
        "Tags": _tags(record.subject),
    }


def seed_blocks(records: tuple[SeedRecord, ...]) -> tuple[dict[str, str], ...]:
    """The whole pre-cutoff corpus as blocks, oldest first and deterministic."""
    ordered = sorted(records, key=lambda r: (r.committed_at, r.sha))
    return tuple(block_record(record) for record in ordered)


def write_seed(workspace: str, blocks: tuple[dict[str, str], ...], task_id: str) -> int:
    """Write the seed through the governed path and return the corpus size.

    One admission covers the whole seed, exactly as applying one approved
    proposal covers the blocks it writes.  ``write_block`` refuses a write
    with no receipt open, so this is the gate enforcing the rule rather
    than this module remembering it.
    """
    from ..governance_gate import get_gate
    from ..storage import get_block_store

    store = get_block_store(workspace)
    with get_gate(workspace).admit_proposal(
        f"P-bench-ab-seed-{task_id}",
        "\n".join(block["_id"] for block in blocks),
        actor="bench_ab_seed",
        target_file=os.path.join("decisions", "DECISIONS.md"),
        metadata={"benchmark": "memory_ab", "task_id": task_id, "blocks": str(len(blocks))},
    ):
        for block in blocks:
            store.write_block(dict(block))
    path = os.path.join(workspace, "decisions", "DECISIONS.md")
    return os.path.getsize(path) if os.path.isfile(path) else 0


def seed_workspace(repo: str, task: Task, workspace: str) -> SeedReport:
    """Build a mind-mem workspace holding only pre-cutoff history."""
    from ..init_workspace import init

    records, malformed = collect_history(repo, task.parent_sha)
    checks = assert_no_leak(records, task)
    blocks = seed_blocks(records)
    joined = "\n".join(block["Statement"] for block in blocks).lower()
    if task.sha[:12].lower() in joined:  # pragma: no cover - defence in depth
        raise SeedLeakError(f"{task.task_id}: rendered corpus names the task commit")
    os.makedirs(workspace, exist_ok=True)
    init(workspace)
    size = write_seed(workspace, blocks, task.task_id)
    return SeedReport(
        workspace=workspace,
        blocks=len(blocks),
        corpus_bytes=size,
        newest_seeded_at=max((r.committed_at for r in records), default=""),
        cutoff=task.memory_cutoff,
        checks=(*checks, "written_through_governance_gate"),
        skipped_malformed=malformed,
    )
