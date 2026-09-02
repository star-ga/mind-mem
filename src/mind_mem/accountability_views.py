# Copyright 2026 STARGA, Inc.
"""RA.2 — precision and waste, as **derived views** over what is already stored.

Retrieval quality is measured everywhere. Two questions almost nothing
answers, because answering them needs a store that kept the receipts:

* **precision** — of the blocks we actually served, what fraction were
  later credited with having helped, broken out by the kind of question
  that was asked;
* **waste** — what is in the corpus that no run has served in the window
  we can see.

Both are computed here, on demand, from rows other subsystems already
wrote. Four refusals give this module its shape.

**Nothing is stored.** There is no table, no column, no cache and no
sidecar in this module — deliberately no ``write``/``record``/``append``
function exists, and every database handle it opens is opened
``mode=ro``, so a report over a workspace with no index *creates no
index*. A stored precision number is a per-block score with a persistence
model, and a stored score is exactly what a later ranking learns to read
back. Recomputed every call, or not available. Stated exactly, because
"writes nothing" is a claim and not a mood: no row in any table changes,
and the only files a report can leave behind are SQLite's own ``-wal`` /
``-shm`` sidecars for a database that already existed — reader
bookkeeping the engine does, not a write by this module
(``tests/test_accountability_views.py`` pins both halves).

**Nothing here reaches a ranking.** This module imports the served-set
ledger, which the scoring path may not (``tests/test_recall_attestation_v2.py``
enforces that edge). The direction is one-way by construction: recall
does not import this module, and nothing in it returns a value any
scoring term consumes.

**"Unserved" is a fact about attention, never about worth.** The waste
view classifies every unserved block with
:func:`~mind_mem.retention_class.retention_class` and reports the
``PROTECTED`` ones under their own name, as *not* waste, with the reason
they are protected. A constitutional constraint or a release decision
that nobody has queried this month is doing its job silently; a view that
counted it as dead weight would be the deletion mistake with a metric
attached.

**A window nobody can see past is stated, not implied.** ``retrieval_log``
is pruned at 30 days (``retrieval_graph.log_retrieval``), so serve
evidence read from it means "not served *in the retained window*", never
"never served". The served-set ledger is append-only and is not pruned,
so when it is enabled it widens the window; every view names the sources
that actually contributed to it. An empty source is reported as
*unavailable with a reason*, never as a zero — a zero from a table that
does not exist is the vacuous pass this repo has been bitten by before.

THE JOIN, and the half of it that is still missing
--------------------------------------------------
Two precisions are computed here, and they are not the same measurement.

:func:`precision_by_intent` is **block-level** and needs no run identity:
for each intent, the distinct blocks served under it, and how many of
those carry a credit anywhere. It works on any workspace, today, with no
flag, because ``retrieval_log`` is written on every recall.

:func:`run_precision` is the **run-level** join RA.1 exists for: this run
served these blocks, and this outcome credits that run. Its key is
``served_ledger.run_id``, content-derived from the three values a run's
attestation already publishes — so
:func:`run_id_of_attestation` mints it from an attestation dict without
the caller touching the ledger's internals or minting a second identity.

The remaining gap is one field, and it is named rather than papered over:
the recall envelope does not carry that id, so a client calling
``report_outcome(query_id=…)`` has nothing joinable to pass. Until the
envelope carries it, :func:`run_precision` reports
``available=False`` with the reason, and counts the credit rows it could
not join instead of silently dropping them.

    from mind_mem.accountability_views import accountability_report
    report = accountability_report("/path/to/workspace")
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .admissibility import admit_corpus
from .calibration import _OUTCOME_TO_FEEDBACK
from .observability import get_logger
from .outcome_attribution import OUTCOME_FAILURE, OUTCOME_SUCCESS
from .recall_digests import query_hash
from .retention_class import PROTECTED, RETENTION_CLASSES, protected_reason, retention_class
from .retrieval_graph import graph_db_path
from .served_ledger import ledger_enabled, read_served_runs, run_id

_log = get_logger("accountability_views")

#: Bucket for a run whose intent classifier recorded nothing. Its own
#: value rather than ``""`` so an unclassified run is visibly unclassified
#: in a report instead of collapsing into a blank row.
INTENT_UNKNOWN: Final = "unknown"

#: Serve-evidence sources, by the name they are reported under.
SOURCE_RETRIEVAL_LOG: Final = "retrieval_log"
SOURCE_SERVED_LEDGER: Final = "served_ledger"

#: What the retrieval log can see. ``retrieval_graph.log_retrieval`` deletes
#: rows older than 30 days, so its silence about a block is bounded.
RETRIEVAL_LOG_WINDOW: Final = "retrieval_log: rows pruned at 30 days"

#: What the ledger can see. Append-only, never pruned — but default OFF.
SERVED_LEDGER_WINDOW: Final = "served_ledger: append-only, not pruned"

#: The two credit verdicts, derived from the outcome→feedback projection in
#: :mod:`mind_mem.calibration` rather than re-spelled here, so the feedback
#: vocabulary and the outcome vocabulary cannot drift apart.
_FEEDBACK_CREDITED: Final = _OUTCOME_TO_FEEDBACK[OUTCOME_SUCCESS]
_FEEDBACK_IMPLICATED: Final = _OUTCOME_TO_FEEDBACK[OUTCOME_FAILURE]

#: Ceiling on the block ids a single view names, so a report over a large
#: corpus stays a report. Counts are always exact; only the id lists are cut.
MAX_NAMED_IDS: Final = 200

__all__ = [
    "INTENT_UNKNOWN",
    "MAX_NAMED_IDS",
    "RETRIEVAL_LOG_WINDOW",
    "SERVED_LEDGER_WINDOW",
    "SOURCE_RETRIEVAL_LOG",
    "SOURCE_SERVED_LEDGER",
    "CreditRow",
    "IntentPrecision",
    "ObservedRun",
    "PrecisionView",
    "RunPrecision",
    "WasteView",
    "accountability_report",
    "credit_rows",
    "main",
    "observed_runs",
    "precision_by_intent",
    "run_id_of_attestation",
    "run_precision",
    "waste_view",
]


# ---------------------------------------------------------------------------
# Read-only access. This module opens databases it must never create.
# ---------------------------------------------------------------------------


def _read_only_connect(path: str) -> sqlite3.Connection | None:
    """Open *path* read-only, or ``None`` when there is nothing to open.

    ``mode=ro`` is the enforcement, not the intention: ``sqlite3.connect``
    on a plain path *creates* the file, and ``retrieval_graph._connect``
    additionally ``makedirs`` its parent. Either would turn "show me a
    report" into "write to the store", which is the one thing a derived
    view may not do.

    What ``mode=ro`` does not prevent, and is not claimed to: opening a
    WAL database materialises its ``-wal`` / ``-shm`` sidecars. That is
    the engine's reader bookkeeping over a database that already exists;
    no row moves, and a workspace with no database still gets none,
    because the ``isfile`` guard above returns first.
    """
    if not os.path.isfile(path):
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error as exc:  # pragma: no cover — defensive
        _log.debug("accountability_db_open_failed", path=path, error=str(exc))
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    """Run *sql*, returning ``[]`` when the table is not there yet.

    A missing table means the subsystem that owns it has never run, which
    is a legitimate state and not an error — but it is reported as an
    absent *source*, never as a zero measurement.
    """
    try:
        return list(conn.execute(sql))
    except sqlite3.Error as exc:
        _log.debug("accountability_query_skipped", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# The join key
# ---------------------------------------------------------------------------


def run_id_of_attestation(attestation: Mapping[str, Any]) -> str:
    """The ledger ``run_id`` for the run *attestation* describes.

    ``run_id`` is ``SHA256(MM_RUN_v1\\0 ‖ query_hash ‖ served_digest ‖
    pipeline_hash)`` and every one of those three is already a field on a
    ``RECALL_ATTEST_v2`` record — so a client holding a recall envelope can
    name the run it was served without the ledger, and without a second
    identity being minted anywhere.

    Delegates to :func:`mind_mem.served_ledger.run_id`; this function
    re-spells nothing, it only maps attestation field names onto it.

    Args:
        attestation: The ``attestation`` object from a recall envelope.

    Returns:
        The 64-character run id.

    Raises:
        KeyError: the mapping is missing one of the three bound fields.
        ValueError: a field is not a 64-character lowercase hex digest.
    """
    return run_id(
        query_hash=str(attestation["query_hash"]),
        served_digest=str(attestation["results_digest"]),
        pipeline_hash=str(attestation["config_hash"]),
    )


# ---------------------------------------------------------------------------
# What was served
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedRun:
    """One run we have serve evidence for.

    ``run_id`` is ``""`` for a run seen only through ``retrieval_log``,
    which stores no pipeline hash and therefore cannot name a run — the
    difference between "we know these blocks were served" and "we know
    *which run* served them" is exactly the difference between the two
    precisions this module computes.
    """

    query_hash: str
    intent: str
    ids: tuple[str, ...]
    source: str
    run_id: str = ""


def _observed_from_retrieval_log(conn: sqlite3.Connection) -> tuple[ObservedRun, ...]:
    """Runs read off ``retrieval_log``, keyed by the CANONICAL query digest.

    ``retrieval_log.query_hash`` is a truncated bare SHA-256 of the query
    with no domain tag — a third spelling of "the query", and not the one
    the ledger and the attestation agree on. It is deliberately ignored
    here and the canonical :func:`~mind_mem.recall_digests.query_hash` is
    recomputed from the stored ``query_text``, so a retrieval-log row and a
    ledger row for the same question land on the same key.
    """
    runs: list[ObservedRun] = []
    for row in _rows(conn, "SELECT query_text, mem_ids, intent_type FROM retrieval_log"):
        try:
            ids = tuple(str(i) for i in json.loads(row["mem_ids"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        runs.append(
            ObservedRun(
                query_hash=query_hash(str(row["query_text"] or "")),
                intent=str(row["intent_type"] or "") or INTENT_UNKNOWN,
                ids=ids,
                source=SOURCE_RETRIEVAL_LOG,
            )
        )
    return tuple(runs)


def _observed_from_ledger(workspace: str, intents: Mapping[str, str]) -> tuple[ObservedRun, ...]:
    """Runs read off the served-set ledger, which carries a real ``run_id``.

    The ledger stores no query text and no intent — by design, since a
    stored judgement about an answer is what the ledger refuses to hold —
    so the intent is looked up from *intents*, the retrieval log's
    ``query_hash -> intent`` map. A run whose retrieval-log row has aged
    out of the 30-day window keeps its identity and loses its intent,
    which is the honest outcome and lands in :data:`INTENT_UNKNOWN`.
    """
    return tuple(
        ObservedRun(
            query_hash=row.query_hash,
            intent=intents.get(row.query_hash, INTENT_UNKNOWN),
            ids=tuple(row.ids),
            source=SOURCE_SERVED_LEDGER,
            run_id=row.run_id,
        )
        for row in read_served_runs(workspace)
    )


def observed_runs(workspace: str) -> tuple[ObservedRun, ...]:
    """Every run we have serve evidence for, from both sources.

    Read-only: opens no database that does not already exist and writes
    nothing. Sources are additive — a run present in both appears twice,
    under each source, because the two windows are different and a view
    that silently merged them could not report which one it saw.
    """
    conn = _read_only_connect(graph_db_path(workspace))
    if conn is None:
        from_log: tuple[ObservedRun, ...] = ()
    else:
        try:
            from_log = _observed_from_retrieval_log(conn)
        finally:
            conn.close()
    intents = {run.query_hash: run.intent for run in from_log if run.intent != INTENT_UNKNOWN}
    return from_log + _observed_from_ledger(workspace, intents)


# ---------------------------------------------------------------------------
# What was credited
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreditRow:
    """One stored verdict about a block: it helped, or it did not."""

    query_id: str
    block_id: str
    verdict: str
    source: str


def credit_rows(workspace: str) -> tuple[CreditRow, ...]:
    """Every credit/implication verdict in the calibration store.

    Two tables, one meaning: ``recall_outcome`` records whether *acting* on
    a block worked, ``calibration_feedback`` records whether it was
    *useful*. Both are keyed on a caller-supplied ``query_id``, which is
    what makes the run-level join possible at all — see
    :func:`run_precision` for why it is not yet reachable in practice.
    """
    conn = _read_only_connect(graph_db_path(workspace))
    if conn is None:
        return ()
    try:
        rows = [
            CreditRow(
                query_id=str(row["query_id"] or ""),
                block_id=str(row["block_id"] or ""),
                verdict=str(row["outcome"] or ""),
                source="recall_outcome",
            )
            for row in _rows(conn, "SELECT query_id, block_id, outcome FROM recall_outcome")
        ]
        rows += [
            CreditRow(
                query_id=str(row["query_id"] or ""),
                block_id=str(row["block_id"] or ""),
                verdict=_verdict_of_feedback(str(row["feedback"] or "")),
                source="calibration_feedback",
            )
            for row in _rows(conn, "SELECT query_id, block_id, feedback FROM calibration_feedback")
        ]
    finally:
        conn.close()
    return tuple(rows)


def _verdict_of_feedback(feedback: str) -> str:
    """Map a feedback label onto the outcome vocabulary (one vocabulary, not two)."""
    if feedback == _FEEDBACK_CREDITED:
        return OUTCOME_SUCCESS
    if feedback == _FEEDBACK_IMPLICATED:
        return OUTCOME_FAILURE
    return ""


def _credited_ids(rows: Iterable[CreditRow]) -> frozenset[str]:
    return frozenset(row.block_id for row in rows if row.verdict == OUTCOME_SUCCESS and row.block_id)


def _implicated_ids(rows: Iterable[CreditRow]) -> frozenset[str]:
    return frozenset(row.block_id for row in rows if row.verdict == OUTCOME_FAILURE and row.block_id)


# ---------------------------------------------------------------------------
# View 1 — precision, per intent type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentPrecision:
    """Credited-over-served for one intent type. Derived, never stored.

    ``observations`` counts **serve-evidence records**, not runs: the two
    sources are separate windows over the same history, so a run recorded
    in the retrieval log AND in the ledger contributes one observation to
    each. Naming it ``runs`` would have been a count nobody could act on —
    it moves the moment the ledger is switched on, with no change in what
    the store actually served. ``precision`` is unaffected either way: it
    is computed over DISTINCT blocks.
    """

    intent: str
    observations: int
    served_blocks: int
    credited_blocks: int
    implicated_blocks: int
    precision: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "observations": self.observations,
            "served_blocks": self.served_blocks,
            "credited_blocks": self.credited_blocks,
            "implicated_blocks": self.implicated_blocks,
            "precision": self.precision,
        }


@dataclass(frozen=True)
class PrecisionView:
    """The per-intent precision table, plus what it could not account for."""

    available: bool
    reason: str
    rows: tuple[IntentPrecision, ...]
    observations: int
    credit_rows: int
    credit_rows_on_unserved_blocks: int
    sources: tuple[str, ...]
    window: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "rows": [row.to_dict() for row in self.rows],
            "observations": self.observations,
            "credit_rows": self.credit_rows,
            "credit_rows_on_unserved_blocks": self.credit_rows_on_unserved_blocks,
            "sources": list(self.sources),
            "window": self.window,
        }


def _ratio(numerator: int, denominator: int) -> float:
    """``numerator / denominator``, rounded, and ``0.0`` on an empty denominator.

    Rounded to six places so a report is byte-stable across hosts rather
    than carrying the last bits of a repeating binary fraction.
    """
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def precision_by_intent(workspace: str) -> PrecisionView:
    """Credited-over-served per intent type, recomputed from stored rows.

    Block-level: for each intent, the DISTINCT blocks served under it, and
    how many of those carry a success verdict anywhere in the calibration
    store. This needs no run identity, which is why it is the view that
    works today — ``retrieval_log`` is written on every recall with no
    flag, and every credit row names a block.

    What it does not claim: that the credit was earned on *this* run. A
    block credited after a WHY-question and later served on a WHEN-question
    counts as credited under both. The run-scoped version of the question
    is :func:`run_precision`, and it is honest about being unreachable.

    Returns:
        A :class:`PrecisionView`. ``available`` is ``False`` — with a
        reason — when there is no serve evidence at all, so an empty
        workspace reports "nothing observed", never a precision of zero.
    """
    runs = observed_runs(workspace)
    rows = credit_rows(workspace)
    credited = _credited_ids(rows)
    implicated = _implicated_ids(rows)
    sources = tuple(sorted({run.source for run in runs}))

    if not runs:
        return PrecisionView(
            available=False,
            reason="no serve evidence: retrieval_log holds no rows and the served-set ledger is empty",
            rows=(),
            observations=0,
            credit_rows=len(rows),
            credit_rows_on_unserved_blocks=len(rows),
            sources=(),
            window="",
        )

    served_by_intent: dict[str, set[str]] = {}
    seen_by_intent: dict[str, int] = {}
    for run in runs:
        served_by_intent.setdefault(run.intent, set()).update(i for i in run.ids if i)
        seen_by_intent[run.intent] = seen_by_intent.get(run.intent, 0) + 1

    all_served = {block_id for ids in served_by_intent.values() for block_id in ids}
    table = tuple(
        IntentPrecision(
            intent=intent,
            observations=seen_by_intent[intent],
            served_blocks=len(served),
            credited_blocks=len(served & credited),
            implicated_blocks=len(served & implicated),
            precision=_ratio(len(served & credited), len(served)),
        )
        for intent, served in sorted(served_by_intent.items())
    )
    return PrecisionView(
        available=True,
        reason="",
        rows=table,
        observations=len(runs),
        credit_rows=len(rows),
        credit_rows_on_unserved_blocks=sum(1 for row in rows if row.block_id not in all_served),
        sources=sources,
        window=_window_of(sources),
    )


def _window_of(sources: Sequence[str]) -> str:
    """Name what the sources in play can actually see."""
    parts = []
    if SOURCE_RETRIEVAL_LOG in sources:
        parts.append(RETRIEVAL_LOG_WINDOW)
    if SOURCE_SERVED_LEDGER in sources:
        parts.append(SERVED_LEDGER_WINDOW)
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# View 2 — the run-scoped precision RA.1 exists for
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPrecision:
    """Precision computed on an exact run join, when one is reachable."""

    available: bool
    reason: str
    runs: int
    joined_runs: int
    served: int
    credited: int
    precision: float
    credit_rows_with_unjoinable_query_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "runs": self.runs,
            "joined_runs": self.joined_runs,
            "served": self.served,
            "credited": self.credited,
            "precision": self.precision,
            "credit_rows_with_unjoinable_query_id": self.credit_rows_with_unjoinable_query_id,
        }


def run_precision(workspace: str) -> RunPrecision:
    """Credited-over-served joined on ``run_id`` — this run, these blocks.

    The strong form of the question. A credit row joins when its
    ``query_id`` IS the run's ledger ``run_id``, which is exactly what
    :func:`run_id_of_attestation` computes from a recall envelope's
    attestation.

    ``available`` is ``False``, with the reason, when the ledger holds no
    rows (it is default-OFF) or when no credit row carries a joinable id.
    The second case is the live one and it is a wiring gap, not a data
    gap: the recall envelope does not yet publish the id, so a client has
    nothing joinable to report against. The count of unjoinable credit
    rows is reported either way — a row that could not be joined is
    named, never dropped.
    """
    ledger_runs = tuple(run for run in observed_runs(workspace) if run.run_id)
    rows = credit_rows(workspace)
    by_run = {run.run_id: run for run in ledger_runs}
    joinable = [row for row in rows if row.query_id in by_run]
    unjoinable = len(rows) - len(joinable)

    if not ledger_runs:
        reason = "served-set ledger holds no rows"
        if not ledger_enabled(workspace):
            reason += " (served_ledger.enabled is not true in mind-mem.json)"
        return RunPrecision(False, reason, 0, 0, 0, 0, 0.0, unjoinable)
    if not joinable:
        return RunPrecision(
            available=False,
            reason=(
                "no credit row carries a run_id: the recall envelope does not publish one, "
                "so report_outcome has nothing joinable to receive"
            ),
            runs=len(ledger_runs),
            joined_runs=0,
            served=0,
            credited=0,
            precision=0.0,
            credit_rows_with_unjoinable_query_id=unjoinable,
        )

    joined_run_ids = {row.query_id for row in joinable}
    served = 0
    credited = 0
    for rid in sorted(joined_run_ids):
        ids = tuple(i for i in by_run[rid].ids if i)
        credited_here = {row.block_id for row in joinable if row.query_id == rid and row.verdict == OUTCOME_SUCCESS}
        served += len(ids)
        credited += sum(1 for i in ids if i in credited_here)
    return RunPrecision(
        available=True,
        reason="",
        runs=len(ledger_runs),
        joined_runs=len(joined_run_ids),
        served=served,
        credited=credited,
        precision=_ratio(credited, served),
        credit_rows_with_unjoinable_query_id=unjoinable,
    )


# ---------------------------------------------------------------------------
# View 3 — waste, which is a question and not a verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WasteView:
    """Corpus blocks with no serve evidence in the observed window.

    ``protected_unserved`` is reported separately and by name because
    those blocks are **not** waste under any reading: an active release
    decision and an operator-authored guardrail are load-bearing exactly
    when nobody is querying them.
    """

    corpus_admitted: int
    corpus_withheld: int
    served_at_least_once: int
    unserved: int
    unserved_by_retention_class: Mapping[str, int]
    protected_unserved: tuple[tuple[str, str], ...]
    unserved_ids: tuple[str, ...]
    unserved_ids_truncated: bool
    unserved_ratio: float
    sources: tuple[str, ...]
    window: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_admitted": self.corpus_admitted,
            "corpus_withheld": self.corpus_withheld,
            "served_at_least_once": self.served_at_least_once,
            "unserved": self.unserved,
            "unserved_by_retention_class": dict(self.unserved_by_retention_class),
            "protected_unserved": [{"block_id": bid, "reason": why} for bid, why in self.protected_unserved],
            "unserved_ids": list(self.unserved_ids),
            "unserved_ids_truncated": self.unserved_ids_truncated,
            "unserved_ratio": self.unserved_ratio,
            "sources": list(self.sources),
            "window": self.window,
        }


def waste_view(workspace: str) -> WasteView:
    """Which admitted blocks no observed run served, split by retention class.

    Two rails on the read, both load-bearing:

    * the corpus is filtered through
      :func:`~mind_mem.admissibility.admit_corpus` — the shared gate, called
      and never re-implemented — so a quarantined or unrecognised-status
      block cannot be *named* by this view. Withheld content is reported
      as a count and nothing else, which keeps the number honest without
      turning a report into a read surface for content the store is
      withholding.
    * an unserved block is a **question**, and the answer is not "delete
      it". ``PROTECTED`` blocks are listed under their own key with the
      reason they are protected; the rest are unserved ``GOVERNED``
      content whose removal, if anyone ever wanted it, still goes through
      a proposal.

    Returns:
        A :class:`WasteView`. ``sources`` and ``window`` name what the
        silence is relative to; with no source at all, ``unserved`` is the
        whole admitted corpus and ``sources`` is empty — which is the
        report saying "nothing has been observed", not "nothing is used".
    """
    from .storage import get_block_store

    try:
        blocks = get_block_store(workspace).get_all()
    except Exception as exc:  # pragma: no cover — a missing corpus is an empty one
        _log.debug("accountability_corpus_read_failed", error=str(exc))
        blocks = []
    admitted = admit_corpus(blocks)
    admitted_ids = [str(block.get("_id") or "") for block in admitted]

    runs = observed_runs(workspace)
    sources = tuple(sorted({run.source for run in runs}))
    served = {block_id for run in runs for block_id in run.ids if block_id}

    unserved_blocks = [block for block, bid in zip(admitted, admitted_ids) if bid and bid not in served]
    by_class: dict[str, int] = {name: 0 for name in RETENTION_CLASSES}
    protected: list[tuple[str, str]] = []
    plain_unserved: list[str] = []
    for block in unserved_blocks:
        block_id = str(block.get("_id") or "")
        klass = retention_class(block)
        by_class[klass] = by_class.get(klass, 0) + 1
        if klass == PROTECTED:
            protected.append((block_id, protected_reason(block)))
        else:
            plain_unserved.append(block_id)

    return WasteView(
        corpus_admitted=len(admitted),
        corpus_withheld=len(blocks) - len(admitted),
        served_at_least_once=sum(1 for bid in admitted_ids if bid in served),
        unserved=len(unserved_blocks),
        unserved_by_retention_class=by_class,
        protected_unserved=tuple(sorted(protected))[:MAX_NAMED_IDS],
        unserved_ids=tuple(sorted(plain_unserved))[:MAX_NAMED_IDS],
        unserved_ids_truncated=len(plain_unserved) > MAX_NAMED_IDS,
        unserved_ratio=_ratio(len(unserved_blocks), len(admitted)),
        sources=sources,
        window=_window_of(sources),
    )


# ---------------------------------------------------------------------------
# The report + its entry point
# ---------------------------------------------------------------------------


def accountability_report(workspace: str) -> dict[str, Any]:
    """Both views over *workspace*, recomputed. Writes nothing, ever."""
    return {
        "schema": "MM_ACCOUNTABILITY_v1",
        "workspace": os.path.abspath(workspace),
        "precision_by_intent": precision_by_intent(workspace).to_dict(),
        "run_precision": run_precision(workspace).to_dict(),
        "waste": waste_view(workspace).to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m mind_mem.accountability_views --workspace <path>``.

    Prints the report as JSON on stdout and returns 0. A read-only verb:
    it opens no database it did not find, and writes only to stdout.
    """
    parser = argparse.ArgumentParser(
        prog="python -m mind_mem.accountability_views",
        description="RA.2 — precision and waste as derived views. Reads only; stores nothing.",
    )
    parser.add_argument("--workspace", default=".", help="Workspace root (default: the current directory).")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (0 for one line).")
    args = parser.parse_args(argv)
    report = accountability_report(args.workspace)
    print(json.dumps(report, indent=args.indent or None, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised as a subprocess in tests
    sys.exit(main())
