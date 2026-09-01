# Copyright 2026 STARGA, Inc.
"""Self-improving retrieval: signal harvest + governed weight promotion.

The roadmap calls for LoRA fine-tuning of a local embedding model and a
graceful weight swap. Actually *running* the trainer needs PyTorch +
Qwen3-Embedding + ms-marco-MiniLM, which this codebase won't bundle. What
mind-mem owns is the two halves either side of that gradient step, and
5.1.0 wires both:

**Harvest** — :func:`run_harvest_job` drains the live
:mod:`~mind_mem.interaction_signals` ledger into
:func:`build_training_tuples` and appends the result to an append-only
training queue, advancing a persisted cursor so a periodic run consumes
each signal once. Its caller is the ``dream_cycle`` maintenance pass
(``v4.online_training``, default OFF), which the daemon already
schedules; ``index_stats`` reports the resulting counters beside
``interaction_signals``.

**Promotion** — the registry half now lives in
:mod:`~mind_mem.model_gate`. This module previously shipped a second,
process-local, untested registry of active / candidate / rollback weight
slots. :class:`WeightRegistry` survives as a thin facade over the
persisted ledger there, so the promotion rule, the revert log and the
load-gate coupling have exactly one implementation.

Two properties worth stating plainly:

**The harvest is admission-filtered.** A signal's ``previous_results``
are block ids, and a training tuple made from them is a durable, exported
artifact naming blocks the user saw. Those ids go through
:func:`~mind_mem.admissibility.admit_corpus` before they become positives
or negatives, so a quarantined block cannot be taught as a retrieval
target by the same corpus recall is still withholding it from.
:func:`harvest_tuples` takes the admitted set as a REQUIRED argument —
a caller cannot forget to filter, only refuse to.

**Nothing here reads a clock on a scored path.** ``build_training_tuples``
and ``harvest_tuples`` are pure. The one timestamp in the module — the
``harvested_at`` stamp on a queue row — is injected, defaulting to the
gate's own UTC helper.

Every piece exposes a dict-friendly ``stats()`` / report so callers can
surface training state through the standard ``index_stats`` envelope.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .model_gate import (
    EVENT_REVERTED,
    MIN_IMPROVEMENT_DEFAULT,
    PromotionDecision,
    WeightRef,
    _now_iso,
    active_weights,
    candidate_weights,
    load_promotion_ledger,
    promote_weights,
    promotion_stats,
    register_candidate,
    revert_weights,
    rollback_weights,
    set_active_weights,
)

_DEFAULT_BUFFER_CAP: int = 100_000

#: v4 feature flag gating the whole harvest surface. Default OFF: with the
#: flag unset nothing here is imported, no file is created and no counter
#: appears in ``index_stats``.
ONLINE_TRAINING_FLAG = "online_training"

#: Workspace-relative home for the harvest artifacts. Deliberately NOT under
#: a corpus directory: a training tuple is derived telemetry, not a block,
#: and must never be parsed as one.
TRAINING_DIRNAME = os.path.join("memory", "training")
QUEUE_FILENAME = "training_tuples.jsonl"
CURSOR_FILENAME = "harvest_cursor.json"

#: Cap on signals consumed by one harvest run, so a periodic pass over a
#: ledger that grew unattended for a month stays bounded.
DEFAULT_HARVEST_LIMIT: int = 2_000


# ---------------------------------------------------------------------------
# Signal harvest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingTuple:
    query: str
    positive_ids: tuple[str, ...]
    negative_ids: tuple[str, ...]
    signal_type: str
    weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "positive_ids": list(self.positive_ids),
            "negative_ids": list(self.negative_ids),
            "signal_type": self.signal_type,
            "weight": self.weight,
        }


def build_training_tuples(signals: Iterable[Mapping[str, Any]]) -> list[TrainingTuple]:
    """Convert interaction signals into training tuples.

    Mapping: a RE_QUERY / REFINEMENT signal promotes the re-asked
    query's target set as positives and the prior query's results
    that didn't re-appear as negatives. CORRECTION signals reverse
    the roles — the previous results were wrong.

    Pure: no clock, no randomness, no IO. Same signals in, same tuples out,
    in the same order.
    """
    out: list[TrainingTuple] = []
    for sig in signals:
        if not isinstance(sig, Mapping):
            continue
        sig_type = str(sig.get("signal_type", ""))
        prev = tuple(str(x) for x in sig.get("previous_results", []) if x)
        new_query = str(sig.get("new_query", ""))
        if not new_query:
            continue
        if sig_type == "correction":
            out.append(
                TrainingTuple(
                    query=new_query,
                    positive_ids=(),  # user didn't approve prior results
                    negative_ids=prev,
                    signal_type=sig_type,
                    weight=1.25,  # explicit correction is the strongest signal
                )
            )
        elif sig_type in {"re_query", "refinement"}:
            out.append(
                TrainingTuple(
                    query=new_query,
                    positive_ids=prev,  # user re-asked, prior set approximates target
                    negative_ids=(),
                    signal_type=sig_type,
                    weight=0.75 if sig_type == "refinement" else 1.0,
                )
            )
    return out


def _restrict(ids: tuple[str, ...], admitted: frozenset[str]) -> tuple[str, ...]:
    return tuple(i for i in ids if i in admitted)


def harvest_tuples(
    signals: Iterable[Mapping[str, Any]],
    *,
    admitted_ids: frozenset[str],
) -> tuple[list[TrainingTuple], dict[str, int]]:
    """Training tuples for *signals*, restricted to admitted block ids.

    ``admitted_ids`` is required, not defaulted: this is the leg that turns
    a withheld block id into a training target, and a default would make
    forgetting to filter the quiet path. Callers get the set from
    :func:`~mind_mem.admissibility.admit_corpus` over the workspace corpus.

    A tuple left with neither a positive nor a negative after the
    restriction carries no supervision and is dropped rather than written —
    an empty tuple in the queue is noise a trainer would have to re-filter.

    Pure. Returns ``(tuples, counters)``.
    """
    built = build_training_tuples(signals)
    kept: list[TrainingTuple] = []
    withheld = 0
    dropped = 0
    by_type: dict[str, int] = {}
    for t in built:
        pos = _restrict(t.positive_ids, admitted_ids)
        neg = _restrict(t.negative_ids, admitted_ids)
        withheld += (len(t.positive_ids) - len(pos)) + (len(t.negative_ids) - len(neg))
        if not pos and not neg:
            dropped += 1
            continue
        kept.append(
            TrainingTuple(
                query=t.query,
                positive_ids=pos,
                negative_ids=neg,
                signal_type=t.signal_type,
                weight=t.weight,
            )
        )
        by_type[t.signal_type] = by_type.get(t.signal_type, 0) + 1
    counters = {
        "tuples_built": len(built),
        "tuples_kept": len(kept),
        "tuples_dropped": dropped,
        "ids_withheld": withheld,
    }
    return kept, {**counters, **{f"kept_{k}": v for k, v in sorted(by_type.items())}}


# ---------------------------------------------------------------------------
# Harvest job — the periodic drain
# ---------------------------------------------------------------------------


def training_dir(workspace: str) -> str:
    return os.path.join(os.path.abspath(workspace), TRAINING_DIRNAME)


def queue_path(workspace: str) -> str:
    return os.path.join(training_dir(workspace), QUEUE_FILENAME)


def cursor_path(workspace: str) -> str:
    return os.path.join(training_dir(workspace), CURSOR_FILENAME)


def _signal_ledger_path(workspace: str) -> str:
    # Same path ``mcp.tools._helpers._signal_store_path`` builds. Duplicated
    # rather than imported so this module keeps no dependency on the MCP tier.
    return os.path.join(os.path.abspath(workspace), "memory", "interaction_signals.jsonl")


def is_online_training_enabled() -> bool:
    """Quiet probe for ``v4.online_training``.

    ``is_enabled_quiet``, never ``is_enabled``: the loud resolver logs
    ``v4_config_unreadable`` on a malformed config, and a probe on an
    OFF-by-default path that logs makes the flag-off build observably
    different from the build that never had the feature.
    """
    try:
        from .v4.feature_flags import is_enabled_quiet

        return is_enabled_quiet(ONLINE_TRAINING_FLAG)
    except Exception:
        return False


def read_cursor(workspace: str) -> dict[str, Any]:
    """``{"consumed": int, "last_signal_id": str}``; empty-shaped on absence."""
    path = cursor_path(workspace)
    default = {"consumed": 0, "last_signal_id": ""}
    if not os.path.isfile(path):
        return default
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        return default
    if not isinstance(data, dict):
        return default
    try:
        consumed = int(data.get("consumed", 0))
    except (TypeError, ValueError):
        consumed = 0
    return {
        "consumed": max(0, consumed),
        "last_signal_id": str(data.get("last_signal_id", "")),
    }


def _write_cursor(workspace: str, cursor: Mapping[str, Any]) -> None:
    """Atomically replace the cursor — write-temp + rename.

    A half-written cursor would either re-harvest or skip signals on the
    next pass; both are silent corruption of the training set.
    """
    path = cursor_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".cursor.", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(dict(cursor), fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def workspace_admitted_ids(workspace: str) -> frozenset[str]:
    """Block ids the corpus currently admits.

    ``iter_blocks(active_only=False)`` then ``admit_corpus`` — deliberately
    NOT ``iter_active_blocks``. Selecting on a store's notion of "active" is
    not the admission decision: the release set in ``decisions/DECISIONS.md``
    readmits ids a status column alone would withhold, and a status column
    read from an index cache goes stale in the fail-OPEN direction. The
    admission function is the one authority, so it is the one thing called.
    """
    from .admissibility import admit_corpus
    from .storage import iter_blocks

    blocks = iter_blocks(workspace, active_only=False)
    return frozenset(str(b["_id"]) for b in admit_corpus(blocks) if b.get("_id"))


def run_harvest_job(
    workspace: str,
    *,
    limit: int = DEFAULT_HARVEST_LIMIT,
    now: str | None = None,
    admitted_ids: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Drain new interaction signals into the training queue.

    The job shape the daemon wants: idempotent, bounded, resumable, and a
    no-op when there is nothing new. Returns a report dict.

    With ``v4.online_training`` off this returns ``{"enabled": False}``
    having touched no file and emitted no log line.

    Cursor resynchronisation: the signal ledger is append-only, so a cursor
    that has run past the end of it — or whose ``last_signal_id`` no longer
    matches the record at that offset — describes a ledger that was
    truncated or replaced. Rather than skip whatever now sits at that
    offset, the run restarts from zero and says so in ``resynced``. Duplicate
    tuples in the queue are cheap; silently unharvested corrections are not.
    """
    if not is_online_training_enabled():
        return {"enabled": False}

    from .interaction_signals import SignalStore

    ws = os.path.abspath(workspace)
    stamp = now or _now_iso()
    store = SignalStore(_signal_ledger_path(ws))
    signals = store.all_signals()

    cursor = read_cursor(ws)
    consumed = cursor["consumed"]
    resynced = False
    if consumed > len(signals) or (consumed > 0 and signals[consumed - 1].signal_id != cursor["last_signal_id"]):
        consumed = 0
        resynced = True

    batch = signals[consumed : consumed + max(0, int(limit))]
    admitted = workspace_admitted_ids(ws) if admitted_ids is None else admitted_ids
    tuples, counters = harvest_tuples((s.to_dict() for s in batch), admitted_ids=admitted)

    if tuples:
        path = queue_path(ws)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for t in tuples:
                fh.write(json.dumps({**t.as_dict(), "harvested_at": stamp}, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    after = consumed + len(batch)
    if batch or resynced:
        _write_cursor(
            ws,
            {
                "consumed": after,
                "last_signal_id": signals[after - 1].signal_id if after else "",
                "updated_at": stamp,
            },
        )

    return {
        "enabled": True,
        "signals_total": len(signals),
        "signals_new": len(batch),
        "consumed_before": cursor["consumed"],
        "consumed_after": after,
        "resynced": resynced,
        "corpus_admitted": len(admitted),
        "signal_stats": store.stats().as_dict(),
        "queue_path": queue_path(ws),
        **counters,
    }


def harvest_stats(workspace: str) -> dict[str, Any]:
    """Read-only harvest state for ``index_stats``. Never writes, never drains."""
    ws = os.path.abspath(workspace)
    cursor = read_cursor(ws)
    path = queue_path(ws)
    queued = 0
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            queued = sum(1 for line in fh if line.strip())
    total = 0
    ledger = _signal_ledger_path(ws)
    if os.path.isfile(ledger):
        with open(ledger, "r", encoding="utf-8", errors="replace") as fh:
            total = sum(1 for line in fh if line.strip())
    return {
        "consumed": cursor["consumed"],
        "signals_total": total,
        "signals_pending": max(0, total - cursor["consumed"]),
        "queued_tuples": queued,
    }


# ---------------------------------------------------------------------------
# Weight registry — a facade over the persisted ledger in ``model_gate``
# ---------------------------------------------------------------------------


class WeightRegistry:
    """Active + candidate + rollback weight refs, persisted.

    A facade. Every method delegates to :mod:`~mind_mem.model_gate`, which
    owns the promotion rule, the load-gate coupling and the event log. This
    class exists so the historical API keeps resolving — it holds no state
    of its own, which is the point: the state it used to hold died with the
    process, and a promotion decision that does not outlive the daemon is
    not an audit trail.

    ``ledger_path`` overrides the ledger file for this instance (tests, or
    an operator keeping per-deployment ledgers). ``None`` resolves through
    ``MIND_MEM_PROMOTION_LEDGER`` / the gate registry's directory.
    """

    def __init__(self, *, ledger_path: str | None = None) -> None:
        self._ledger_path = ledger_path

    # ----- slots -----------------------------------------------------------

    def set_active(self, ref: WeightRef) -> None:
        set_active_weights(ref, ledger_path=self._ledger_path)

    def set_candidate(self, ref: WeightRef) -> None:
        register_candidate(ref, ledger_path=self._ledger_path)

    def active(self, model_id: str) -> Optional[WeightRef]:
        return active_weights(model_id, ledger_path=self._ledger_path)

    def candidate(self, model_id: str) -> Optional[WeightRef]:
        return candidate_weights(model_id, ledger_path=self._ledger_path)

    def rollback(self, model_id: str) -> Optional[WeightRef]:
        return rollback_weights(model_id, ledger_path=self._ledger_path)

    # ----- decisions -------------------------------------------------------

    def promote(
        self,
        model_id: str,
        *,
        new_mrr: float,
        min_improvement: float = MIN_IMPROVEMENT_DEFAULT,
        verify_load_gate: bool = True,
        now: str | None = None,
    ) -> tuple[bool, str]:
        """Promote the candidate → active when it beats the baseline.

        Returns ``(ok, reason)`` as it always did; the full decision (and
        its persisted event) is available via :func:`promote_candidate` or
        :func:`~mind_mem.model_gate.promotion_events`.
        """
        decision = promote_weights(
            model_id,
            candidate_mrr=new_mrr,
            min_improvement=min_improvement,
            verify_load_gate=verify_load_gate,
            now=now,
            ledger_path=self._ledger_path,
        )
        return decision.ok, (decision.detail or decision.reason)

    def revert(self, model_id: str, reason: str, *, now: str | None = None) -> bool:
        return revert_weights(model_id, reason=reason, now=now, ledger_path=self._ledger_path).ok

    def stats(self) -> dict[str, Any]:
        """Ledger snapshot. Key names preserved from the in-memory version."""
        snapshot = promotion_stats(self._ledger_path)
        events = load_promotion_ledger(self._ledger_path).get("events", [])
        return {
            "active": snapshot["active"],
            "candidate": {
                m: ref.as_dict()
                for m in snapshot["candidate_pending"]
                if (ref := candidate_weights(m, ledger_path=self._ledger_path)) is not None
            },
            "rollback_available": snapshot["rollback_available"],
            "revert_events": sum(1 for e in events if e.get("event") == EVENT_REVERTED),
            "events": snapshot["events"],
            "events_by_kind": snapshot["events_by_kind"],
        }


def promote_candidate(
    registry: WeightRegistry,
    *,
    model_id: str,
    candidate_mrr: float,
    baseline_mrr: float,
    min_improvement: float = MIN_IMPROVEMENT_DEFAULT,
    verify_load_gate: bool = True,
) -> dict:
    """Apply the promotion rule + return a decision dict.

    ``baseline_mrr`` is the caller's own measurement and is echoed back for
    the record; the *authoritative* baseline is the ledger's active
    ``base_mrr``, which is what the rule actually compares against. When the
    two disagree the ledger wins — an operator quoting a stale baseline
    must not be able to talk a regression through the gate.
    """
    decision: PromotionDecision = promote_weights(
        model_id,
        candidate_mrr=candidate_mrr,
        min_improvement=min_improvement,
        verify_load_gate=verify_load_gate,
        ledger_path=registry._ledger_path,
    )
    return {
        "promoted": decision.ok,
        "reason": decision.detail or decision.reason,
        "reason_code": decision.reason,
        "candidate_mrr": candidate_mrr,
        "baseline_mrr": baseline_mrr,
        "ledger_baseline_mrr": decision.baseline_mrr,
        "improvement": candidate_mrr - baseline_mrr,
        "min_improvement": min_improvement,
    }


# ---------------------------------------------------------------------------
# Training loop stub
# ---------------------------------------------------------------------------


TrainStepFn = Callable[[list[TrainingTuple]], Mapping[str, Any]]


class TrainingLoop:
    """Thread-safe coordinator for async online training.

    The actual gradient step is the caller's concern — they supply a
    ``train_step(tuples) -> stats`` callable. The loop queues tuples,
    tracks steps completed, and exposes a stats surface that the MCP
    layer can surface through ``index_stats``.
    """

    def __init__(
        self,
        train_step: TrainStepFn,
        *,
        batch_size: int = 32,
        buffer_cap: int = _DEFAULT_BUFFER_CAP,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if buffer_cap < batch_size:
            raise ValueError("buffer_cap must be >= batch_size")
        self._fn = train_step
        self._batch_size = int(batch_size)
        self._buffer_cap = int(buffer_cap)
        self._buffer: "deque[TrainingTuple]" = deque(maxlen=self._buffer_cap)
        self._lock = threading.RLock()
        self._steps_run = 0
        self._errors = 0
        self._overflow_dropped = 0

    def submit(self, tuples: Iterable[TrainingTuple]) -> int:
        with self._lock:
            for t in tuples:
                if not isinstance(t, TrainingTuple):
                    continue
                if len(self._buffer) == self._buffer_cap:
                    self._overflow_dropped += 1
                self._buffer.append(t)
        return self.try_flush()

    def try_flush(self) -> int:
        """Run as many batches as the buffer permits. Returns step count."""
        flushed = 0
        while True:
            with self._lock:
                if len(self._buffer) < self._batch_size:
                    break
                batch = [self._buffer.popleft() for _ in range(self._batch_size)]
            try:
                self._fn(batch)
                with self._lock:
                    self._steps_run += 1
                    flushed += 1
            except Exception:
                # A caller-supplied gradient step is third-party code; a
                # raise from it must not take the harvest pass down with it.
                with self._lock:
                    self._errors += 1
        return flushed

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "buffered": len(self._buffer),
                "buffer_cap": self._buffer_cap,
                "overflow_dropped": self._overflow_dropped,
                "steps_run": self._steps_run,
                "errors": self._errors,
                "batch_size": self._batch_size,
            }


# ---------------------------------------------------------------------------
# CLI — the daemon/cron job entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m mind_mem.online_trainer <workspace>``.

    Matches the ``[sys.executable, "-m", "mind_mem.<module>", workspace]``
    shape ``cron_runner.JOB_DEFS`` invokes, so registering the harvest as a
    cron job is a one-line table entry. The live caller today is the
    ``dream_cycle`` maintenance pass.
    """
    import argparse

    parser = argparse.ArgumentParser(description="mind-mem online-training signal harvest")
    parser.add_argument("workspace", help="path to the mind-mem workspace")
    parser.add_argument("--limit", type=int, default=DEFAULT_HARVEST_LIMIT)
    args = parser.parse_args(argv)

    report = run_harvest_job(args.workspace, limit=args.limit)
    print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201 — CLI entry point
    return 0 if report.get("enabled") else 1


__all__ = [
    "ONLINE_TRAINING_FLAG",
    "PromotionDecision",
    "TrainingTuple",
    "TrainingLoop",
    "TrainStepFn",
    "WeightRef",
    "WeightRegistry",
    "build_training_tuples",
    "cursor_path",
    "harvest_stats",
    "harvest_tuples",
    "is_online_training_enabled",
    "main",
    "promote_candidate",
    "queue_path",
    "read_cursor",
    "run_harvest_job",
    "workspace_admitted_ids",
]


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    raise SystemExit(main())
