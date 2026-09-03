# Copyright 2026 STARGA, Inc.
"""RA — replay: proving one recall served exactly what its attestation says.

A recall publishes an attestation on its envelope, and the served-set ledger
records what was served. Each half is already self-checking: the attestation
hashes its own bound fields (:func:`~mind_mem.recall_attestation.verify_recall_attestation`)
and the ledger's row chain proves no row was altered or removed
(:func:`~mind_mem.served_ledger.verify_served_chain`). Neither half checks the
*other*, and until this module there was no surface that did — both verifiers
were reachable from tests and a docs paragraph and from nothing an operator
could run.

This is the join, and the join is where the claim lives:

    the attestation says "this question, this pipeline, this answer digest,
    this many results"; the ledger says "these ids, in this order". Agreeing
    on the run id means they are describing the same answer. Agreeing on the
    *count* means the attestation did not overstate it.

WHAT THE COUNT CHECK CATCHES, and why it is not redundant. ``run_id`` is
``SHA256(MM_RUN_v1\\0 ‖ query_hash ‖ served_digest ‖ pipeline_hash)`` — it binds
the answer *digest*, never the cardinality. ``result_count`` is bound into the
attestation hash but nothing outside the attestation can check it, so a record
can claim ``result_count = 9`` over a two-block answer and stay internally
consistent forever. The ledger row holds the ids themselves, and
:func:`~mind_mem.served_ledger.verify_served_chain` has already proven those
ids hash to the row's ``served_digest``. So ``len(row.ids)`` is an
*independent* witness of the cardinality, and comparing the two is the one
check neither half can make alone.

FOUR OUTCOMES, and the two that are both honest passes:

``REPLAYED``
    A ledger row carries this run id **and** the same ``index_anchor`` and
    ``scoring_instant``. This exact run was recorded.
``ANSWER_RECORDED``
    A row carries this run id but under a different anchor or instant. That is
    not a failure and must not be reported as one: ``run_id`` deliberately
    excludes both, so it names THE ANSWER rather than one serving of it, and
    the same answer served again tomorrow is a legitimate second row.
``NOT_RECORDED``
    No row carries this run id — and the reason says which kind of silence it
    is: "this workspace opted out" is a different fact from "recording and
    this run is missing". The ledger proves nothing
    about completeness (an append that never happened leaves no gap), so this
    is never evidence that the run did not occur.
``MISMATCH`` / ``CHAIN_BROKEN`` / ``UNVERIFIABLE``
    The two halves disagree; the ledger cannot be used as evidence at all; or
    the record does not hash to its own fields, so nothing it claims is worth
    checking against anything.

DELIBERATELY ABSENT: the served ids. The verdict reports how many and under
what digest, never the list. The ids on a ledger row were admitted when they
were served and may not be admitted now, and a verb that reprinted them would
be a read surface around the admission gate — the exact shape
:mod:`mind_mem.accountability_views` refuses in its waste view. The ids are on
the row for an operator who wants them; a *verdict* does not need to carry
them to be evidence.

Nothing here writes. It opens no database, creates no file, reads no clock,
and — like every accountability surface — is recomputed on demand rather than
stored, so there is no verdict for a later ranking to read back.

    from mind_mem.replay_check import replay_check
    verdict = replay_check(workspace, envelope["attestation"])
    assert verdict.replayable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .observability import get_logger
from .recall_attestation import RecallAttestation, verify_recall_attestation
from .recall_digests import served_set_digest
from .served_ledger import ServedRun, ledger_enabled, ledger_path, read_served_runs, verify_served_chain

_log = get_logger("replay_check")

#: This exact run is on the ledger — same answer, same anchor, same instant.
VERDICT_REPLAYED: Final = "REPLAYED"

#: The same answer is on the ledger, served under a different anchor/instant.
#: A pass: ``run_id`` names the answer, not one serving of it.
VERDICT_ANSWER_RECORDED: Final = "ANSWER_RECORDED"

#: No row carries this run id. Never proof the run did not happen.
VERDICT_NOT_RECORDED: Final = "NOT_RECORDED"

#: A row carries this run id and contradicts the attestation.
VERDICT_MISMATCH: Final = "MISMATCH"

#: The ledger failed its own chain check, so no row in it is evidence.
VERDICT_CHAIN_BROKEN: Final = "CHAIN_BROKEN"

#: The record is not a well-formed, internally consistent v2 attestation.
VERDICT_UNVERIFIABLE: Final = "UNVERIFIABLE"

#: Every verdict this module can return, most- to least-conclusive. Closed, so
#: a caller switching on it cannot be surprised by a seventh.
VERDICTS: Final[tuple[str, ...]] = (
    VERDICT_REPLAYED,
    VERDICT_ANSWER_RECORDED,
    VERDICT_NOT_RECORDED,
    VERDICT_MISMATCH,
    VERDICT_CHAIN_BROKEN,
    VERDICT_UNVERIFIABLE,
)

#: The two verdicts that mean "the ledger corroborates this attestation".
PASSING_VERDICTS: Final[frozenset[str]] = frozenset({VERDICT_REPLAYED, VERDICT_ANSWER_RECORDED})

__all__ = [
    "PASSING_VERDICTS",
    "VERDICTS",
    "VERDICT_ANSWER_RECORDED",
    "VERDICT_CHAIN_BROKEN",
    "VERDICT_MISMATCH",
    "VERDICT_NOT_RECORDED",
    "VERDICT_REPLAYED",
    "VERDICT_UNVERIFIABLE",
    "ReplayVerdict",
    "main",
    "replay_check",
]


@dataclass(frozen=True)
class ReplayVerdict:
    """What the ledger says about one attestation. Derived, never stored."""

    verdict: str
    replayable: bool
    reason: str
    run_id: str
    record_consistent: bool
    chain_ok: bool
    ledger_enabled: bool
    ledger_present: bool
    rows_examined: int
    matched_seqs: tuple[int, ...]
    exact_seqs: tuple[int, ...]
    attested_count: int
    recorded_counts: tuple[int, ...]
    served_digest: str
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "replayable": self.replayable,
            "reason": self.reason,
            "run_id": self.run_id,
            "record_consistent": self.record_consistent,
            "chain_ok": self.chain_ok,
            "ledger_enabled": self.ledger_enabled,
            "ledger_present": self.ledger_present,
            "rows_examined": self.rows_examined,
            "matched_seqs": list(self.matched_seqs),
            "exact_seqs": list(self.exact_seqs),
            "attested_count": self.attested_count,
            "recorded_counts": list(self.recorded_counts),
            "served_digest": self.served_digest,
            "findings": list(self.findings),
        }


def _unverifiable(reason: str) -> ReplayVerdict:
    """A verdict about the record itself — nothing was read from the ledger."""
    return ReplayVerdict(
        verdict=VERDICT_UNVERIFIABLE,
        replayable=False,
        reason=reason,
        run_id="",
        record_consistent=False,
        chain_ok=False,
        ledger_enabled=False,
        ledger_present=False,
        rows_examined=0,
        matched_seqs=(),
        exact_seqs=(),
        attested_count=0,
        recorded_counts=(),
        served_digest="",
        findings=(),
    )


def _row_findings(record: RecallAttestation, rows: Sequence[ServedRun]) -> tuple[str, ...]:
    """Every disagreement between *record* and the rows that claim its run id.

    Two checks, and they are not the same check twice:

    * the digest comparison is defence in depth. A matching ``run_id`` already
      implies a matching ``served_digest`` under SHA-256, and the chain walk
      has already proven the row's ids hash to it — so this can only fire if
      one of those two properties has been broken, which is worth saying out
      loud rather than assuming.
    * the count comparison is load-bearing and fires on its own: nothing
      outside the ledger can witness ``result_count``.
    """
    out: list[str] = []
    for row in rows:
        if served_set_digest(row.ids) != record.results_digest:
            out.append(f"row {row.seq}: the recorded ids do not hash to the attested results_digest {record.results_digest[:16]}…")
        if len(row.ids) != record.result_count:
            out.append(
                f"row {row.seq}: the ledger recorded {len(row.ids)} served ids but the attestation "
                f"claims result_count={record.result_count}"
            )
    return tuple(out)


def _silence_reason(*, enabled: bool, present: bool, rows: int) -> str:
    """Name which kind of silence "no matching row" is.

    An absent ledger and a recording-but-incomplete one are different facts,
    and collapsing them into "not found" is how an opted-out workspace gets
    read as a failed verification.
    """
    if not present:
        return (
            "no ledger in this workspace: the served-set ledger records by default "
            "since 5.0.2, so either nothing has been served here yet or the workspace "
            'opted out with {"served_ledger": {"enabled": false}} in mind-mem.json. '
            "Absence of a row is not evidence the run did not happen."
        )
    if not enabled:
        return (
            f"the ledger holds {rows} row(s) but this workspace has opted out of recording "
            "(served_ledger.enabled is false), so it stopped: a run served after the opt-out "
            "leaves no row"
        )
    return (
        f"the ledger is enabled and holds {rows} row(s), none carrying this run id. The ledger "
        "proves nothing about completeness — an append that never happened leaves no gap — so "
        "this is a missing record, not a refuted run"
    )


def replay_check(workspace: str, attestation: Mapping[str, Any]) -> ReplayVerdict:
    """Check one recall attestation against the served-set ledger.

    Read-only and clock-free: the verdict for a given (workspace, attestation)
    is the same on any host on any day, which is what makes it evidence rather
    than a status line.

    Args:
        workspace: Workspace root the recall was served from.
        attestation: The ``attestation`` object off a recall envelope, or an
            equivalent dict. Hostile input is answered, never raised on.

    Returns:
        A :class:`ReplayVerdict`. ``replayable`` is True only for
        :data:`VERDICT_REPLAYED` and :data:`VERDICT_ANSWER_RECORDED`.
    """
    if not isinstance(attestation, Mapping):
        return _unverifiable("the attestation is not a mapping")
    record_dict = dict(attestation)
    if not verify_recall_attestation(record_dict):
        return _unverifiable(
            "the record is not a well-formed, internally consistent RECALL_ATTEST_v2 attestation: "
            "it does not hash to its own bound fields, so nothing it claims can be checked"
        )
    # Guarded by the verification above: ``from_dict`` cannot raise on a record
    # ``verify_recall_attestation`` has already accepted.
    record = RecallAttestation.from_dict(record_dict)
    run = record.query_id

    enabled = ledger_enabled(workspace)
    present = os.path.isfile(ledger_path(workspace))
    chain = verify_served_chain(workspace)
    if not chain.ok:
        return ReplayVerdict(
            verdict=VERDICT_CHAIN_BROKEN,
            replayable=False,
            reason=f"the ledger failed its own chain check, so no row in it is evidence: {chain.reason}",
            run_id=run,
            record_consistent=True,
            chain_ok=False,
            ledger_enabled=enabled,
            ledger_present=present,
            rows_examined=chain.rows_checked,
            matched_seqs=(),
            exact_seqs=(),
            attested_count=record.result_count,
            recorded_counts=(),
            served_digest=record.results_digest,
            findings=(chain.reason,),
        )

    rows = read_served_runs(workspace)
    matches = tuple(row for row in rows if row.run_id == run)

    def _verdict(
        verdict: str,
        *,
        replayable: bool,
        reason: str,
        matched_seqs: tuple[int, ...] = (),
        exact_seqs: tuple[int, ...] = (),
        recorded_counts: tuple[int, ...] = (),
        findings: tuple[str, ...] = (),
    ) -> ReplayVerdict:
        """Fill in the eight fields every post-chain verdict shares."""
        return ReplayVerdict(
            verdict=verdict,
            replayable=replayable,
            reason=reason,
            run_id=run,
            record_consistent=True,
            chain_ok=True,
            ledger_enabled=enabled,
            ledger_present=present,
            rows_examined=len(rows),
            matched_seqs=matched_seqs,
            exact_seqs=exact_seqs,
            attested_count=record.result_count,
            recorded_counts=recorded_counts,
            served_digest=record.results_digest,
            findings=findings,
        )

    if not matches:
        return _verdict(
            VERDICT_NOT_RECORDED,
            replayable=False,
            reason=_silence_reason(enabled=enabled, present=present, rows=len(rows)),
        )

    matched_seqs = tuple(sorted(row.seq for row in matches))
    recorded_counts = tuple(sorted({len(row.ids) for row in matches}))
    findings = _row_findings(record, matches)
    exact_seqs = tuple(
        sorted(row.seq for row in matches if row.index_anchor == record.index_anchor and row.scoring_instant == record.scoring_instant)
    )
    if findings:
        return _verdict(
            VERDICT_MISMATCH,
            replayable=False,
            reason=(f"{len(matches)} ledger row(s) carry this run id and contradict the attestation: " + "; ".join(findings)),
            matched_seqs=matched_seqs,
            exact_seqs=exact_seqs,
            recorded_counts=recorded_counts,
            findings=findings,
        )
    if exact_seqs:
        return _verdict(
            VERDICT_REPLAYED,
            replayable=True,
            reason=(
                f"row(s) {list(exact_seqs)} record this run: same answer, same index_anchor "
                f"({record.index_anchor[:16]}\u2026) and same scoring_instant ({record.scoring_instant})"
            ),
            matched_seqs=matched_seqs,
            exact_seqs=exact_seqs,
            recorded_counts=recorded_counts,
        )
    return _verdict(
        VERDICT_ANSWER_RECORDED,
        replayable=True,
        reason=(
            f"row(s) {list(matched_seqs)} record this exact answer, under a different index_anchor or "
            "scoring_instant. run_id excludes both by design, so this is a second serving of the same "
            "answer and not a disagreement"
        ),
        matched_seqs=matched_seqs,
        recorded_counts=recorded_counts,
    )


def _load_attestation(source: str) -> Mapping[str, Any]:
    """Read an attestation from a file or ``-`` for stdin.

    Accepts either a bare attestation object or a whole recall envelope with
    one under ``attestation``, because those are the two things a caller
    actually has in hand.
    """
    if source == "-":
        text = sys.stdin.read()
    else:
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
    payload: Any = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("input is not a JSON object")
    nested: Any = payload.get("attestation")
    if isinstance(nested, dict):
        return dict(nested)
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m mind_mem.replay_check --workspace <ws> --attestation <file|->``.

    Exit status is the verdict: ``0`` when the ledger corroborates the
    attestation, ``1`` otherwise. A verb whose failure is only visible in its
    JSON is a verb no gate can call.
    """
    parser = argparse.ArgumentParser(
        prog="python -m mind_mem.replay_check",
        description="Check a recall attestation against the served-set ledger. Reads only; stores nothing.",
    )
    parser.add_argument("--workspace", default=".", help="Workspace root (default: the current directory).")
    parser.add_argument(
        "--attestation",
        default="-",
        help="Path to a JSON attestation or recall envelope; '-' reads stdin (default).",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (0 for one line).")
    args = parser.parse_args(argv)
    try:
        attestation = _load_attestation(args.attestation)
    except (OSError, ValueError) as exc:
        verdict = _unverifiable(f"could not read an attestation from {args.attestation!r}: {exc}")
    else:
        verdict = replay_check(args.workspace, attestation)
    print(json.dumps(verdict.to_dict(), indent=args.indent or None, sort_keys=True))
    return 0 if verdict.replayable else 1


if __name__ == "__main__":  # pragma: no cover — exercised as a subprocess in tests
    sys.exit(main())
