"""The right-hand side of the served/outcome join (RA.1).

ROADMAP (RA.1 — served-set ledger): "**Still open:** ``report_outcome(run_id=…)`` — the
right-hand side of the join; ``run_id`` appears nowhere in ``outcome_attribution.py``."

The served ledger records WHAT was served, stably keyed by a content-derived ``run_id``.
Nothing recorded whether that answer HELPED, so the ledger could say "these ids were
served under this run" while no query could ever ask "did serving them work".

**Why a separate append-only sidecar and not a column on the calibration table.** RA.1's
structural rail is that nothing on the scoring path may import the ledger, in any
spelling, at either laziness level, nor through ``importlib``. Putting the ledger's key
into the calibration DB — which the scoring path reads — would turn that rail from a
property into a convention. This file is append-only and hash-chained like the ledger it
joins, and the scoring path does not import it either.

**An unknown ``run_id`` is REFUSED, never recorded.** An orphan row makes the join
silently drop: the caller believes the outcome was attributed, every later view
under-counts, and nothing anywhere says so. The failure belongs on the caller who can
still fix it.

**Absence stays legal and stays visible.** Most reporters never saw a recall run, so
``run_id`` is optional at the ``report_outcome`` boundary — but an unattributed outcome
must never be countable as an attributed one.

No clock and no randomness: ``seq`` orders the file and the row hash is derived, so two
identical joins produce identical rows and the chain verifies on replay.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "JoinRefused",
    "OutcomeJoin",
    "append_outcome_join",
    "join_path",
    "join_rows",
    "outcomes_for_run",
    "row_hash",
]

#: Chain seed. A literal, so the first row's ``prev_row_hash`` is a value rather than an
#: absence — an empty first link cannot be told apart from a truncated file.
GENESIS = "0" * 64

_FILENAME = "served_outcome_join.jsonl"


class JoinRefused(ValueError):
    """A join was refused rather than recorded as an orphan."""


@dataclass(frozen=True)
class OutcomeJoin:
    """One fact: outcome ``outcome_id`` was reported for served run ``run_id``."""

    seq: int
    prev_row_hash: str
    run_id: str
    outcome_id: str


def join_path(workspace: str | Path) -> str:
    return os.path.join(str(workspace), "intelligence", _FILENAME)


def row_hash(row: OutcomeJoin) -> str:
    """Derived, never stored. Length-prefixed so no field boundary is ambiguous.

    Concatenating variable-length fields with a separator lets two different rows
    collide by moving the separator into a value; prefixing each field with its length
    makes the encoding injective.
    """
    parts = (str(row.seq), row.prev_row_hash, row.run_id, row.outcome_id)
    preimage = "MM_OUTCOME_JOIN_v1\0" + "".join(f"{len(p)}:{p}\0" for p in parts)
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=16)
def _known_run_ids(workspace: str) -> frozenset[str]:
    """Run ids the served ledger actually holds.

    Read from the ledger rather than trusted from the caller: the whole value of the
    refusal below is that it consults the authority. Cached per workspace so validating
    a batch of outcomes does not re-read the ledger per row.
    """
    try:
        from .served_ledger import read_served_runs

        return frozenset(str(r.run_id) for r in read_served_runs(workspace))
    except Exception:  # noqa: BLE001 — an unreadable ledger knows no run ids
        return frozenset()


def join_rows(workspace: str | Path) -> tuple[OutcomeJoin, ...]:
    """Every join row, in file order. A missing sidecar reads as empty, never an error.

    A malformed line is SKIPPED rather than fatal — one bad row must not make the whole
    join unreadable — and skipping is safe here because the chain makes tampering
    detectable separately.
    """
    path = join_path(workspace)
    if not os.path.exists(path):
        return ()
    rows: list[OutcomeJoin] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                rows.append(
                    OutcomeJoin(
                        seq=int(data["seq"]),
                        prev_row_hash=str(data["prev_row_hash"]),
                        run_id=str(data["run_id"]),
                        outcome_id=str(data["outcome_id"]),
                    )
                )
            except (ValueError, KeyError, TypeError):
                continue
    return tuple(rows)


def outcomes_for_run(workspace: str | Path, run_id: str) -> tuple[str, ...]:
    """Outcome ids reported for *run_id*, in file order.

    A read never refuses: asking about a run with no outcomes is a normal question whose
    answer is "none".
    """
    wanted = str(run_id or "")
    if not wanted:
        return ()
    return tuple(r.outcome_id for r in join_rows(workspace) if r.run_id == wanted)


def append_outcome_join(
    workspace: str | Path, *, run_id: str, outcome_id: str
) -> Optional[OutcomeJoin]:
    """Record that *outcome_id* was reported for served run *run_id*.

    Returns the new row, or ``None`` when the exact pair is already recorded —
    idempotent, because one outcome for one run is ONE fact and a second row would
    double-count it in every derived view. A view that double-counts is worse than no
    view, because it looks like evidence.

    Raises :class:`JoinRefused` when *run_id* is not in the served ledger.
    """
    run = str(run_id or "").strip()
    out = str(outcome_id or "").strip()
    if not run or not out:
        raise JoinRefused("both run_id and outcome_id are required to record a join")

    if run not in _known_run_ids(str(workspace)):
        raise JoinRefused(
            f"run_id {run[:12]}… is not in the served ledger, so this outcome cannot be "
            f"attributed to a served run; recording it would make the join silently "
            f"drop rows while every later view under-counts"
        )

    existing = join_rows(workspace)
    if any(r.run_id == run and r.outcome_id == out for r in existing):
        return None

    prev = row_hash(existing[-1]) if existing else GENESIS
    row = OutcomeJoin(seq=len(existing) + 1, prev_row_hash=prev, run_id=run, outcome_id=out)

    path = join_path(workspace)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "seq": row.seq,
                    "prev_row_hash": row.prev_row_hash,
                    "run_id": row.run_id,
                    "outcome_id": row.outcome_id,
                },
                sort_keys=True,
            )
            + "\n"
        )
    return row
