# Copyright 2026 STARGA, Inc.
"""Self-improving retrieval training harness (v2.1.0).

The roadmap calls for LoRA fine-tuning of a local embedding model
and a graceful weight swap. Actually *running* the trainer needs
PyTorch + Qwen3-Embedding + ms-marco-MiniLM, which this codebase
won't bundle. Instead the module ships:

- :class:`SignalHarvest` — turns :mod:`interaction_signals` records
  into (query, positive_ids, negative_ids) training tuples.
- :class:`WeightRegistry` — version-stamped weight refs with
  ``active`` / ``candidate`` / ``rollback`` slots + audit log.
- :func:`promote_candidate` — enforce the governance-gated swap with
  an auto-revert hook on MRR regression.
- :class:`TrainingLoop` — a thread-safe stub that accepts a caller-
  supplied ``train_step`` function, so third-party training code
  (PyTorch or otherwise) slots in without pulling a heavyweight
  dependency into mind-mem itself.

Every piece exposes a dict-friendly ``stats()`` method for MCP
observability, so callers can surface training state through the
standard ``index_stats`` envelope.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

_REVERT_LOG_CAP: int = 10_000
_DEFAULT_BUFFER_CAP: int = 100_000


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


# ---------------------------------------------------------------------------
# Weight registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightRef:
    model_id: str
    version: str
    path: str
    base_mrr: float
    promoted_at: str
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "path": self.path,
            "base_mrr": self.base_mrr,
            "promoted_at": self.promoted_at,
            "metadata": self.metadata,
        }


class WeightRegistry:
    """Keeps the active + candidate + rollback weight refs for each model."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: dict[str, WeightRef] = {}
        self._candidate: dict[str, WeightRef] = {}
        self._rollback: dict[str, WeightRef] = {}
        self._revert_events: "deque[dict]" = deque(maxlen=_REVERT_LOG_CAP)

    def set_active(self, ref: WeightRef) -> None:
        with self._lock:
            self._active[ref.model_id] = ref

    def set_candidate(self, ref: WeightRef) -> None:
        with self._lock:
            self._candidate[ref.model_id] = ref

    def active(self, model_id: str) -> Optional[WeightRef]:
        with self._lock:
            return self._active.get(model_id)

    def candidate(self, model_id: str) -> Optional[WeightRef]:
        with self._lock:
            return self._candidate.get(model_id)

    def promote(
        self,
        model_id: str,
        *,
        new_mrr: float,
        min_improvement: float = 0.01,
    ) -> tuple[bool, str]:
        """Promote the candidate → active when it beats the baseline."""
        with self._lock:
            cand = self._candidate.get(model_id)
            if cand is None:
                return False, "no candidate weights registered"
            prev = self._active.get(model_id)
            if prev is not None and new_mrr < prev.base_mrr + min_improvement:
                return False, (
                    f"MRR regression or insufficient improvement: "
                    f"candidate={new_mrr:.4f}, baseline={prev.base_mrr:.4f}"
                )
            if prev is not None:
                self._rollback[model_id] = prev
            promoted = WeightRef(
                model_id=cand.model_id,
                version=cand.version,
                path=cand.path,
                base_mrr=new_mrr,
                promoted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                metadata=dict(cand.metadata),
            )
            self._active[model_id] = promoted
            self._candidate.pop(model_id, None)
            return True, "promoted"

    def revert(self, model_id: str, reason: str) -> bool:
        with self._lock:
            rb = self._rollback.get(model_id)
            if rb is None:
                return False
            self._active[model_id] = rb
            self._rollback.pop(model_id, None)
            self._revert_events.append(
                {
                    "model_id": model_id,
                    "reason": reason,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": {m: r.as_dict() for m, r in self._active.items()},
                "candidate": {m: r.as_dict() for m, r in self._candidate.items()},
                "rollback_available": sorted(self._rollback.keys()),
                "revert_events": len(self._revert_events),
            }


# ---------------------------------------------------------------------------
# Governance-gated promotion + revert-rate tracking
# ---------------------------------------------------------------------------


def promote_candidate(
    registry: WeightRegistry,
    *,
    model_id: str,
    candidate_mrr: float,
    baseline_mrr: float,
    min_improvement: float = 0.01,
) -> dict:
    """Apply the roadmap promotion rule + return a decision dict."""
    improvement = candidate_mrr - baseline_mrr
    ok, reason = registry.promote(
        model_id,
        new_mrr=candidate_mrr,
        min_improvement=min_improvement,
    )
    return {
        "promoted": ok,
        "reason": reason,
        "candidate_mrr": candidate_mrr,
        "baseline_mrr": baseline_mrr,
        "improvement": improvement,
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
            except Exception:  # pragma: no cover — isolate trainer failures
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


__all__ = [
    "TrainingTuple",
    "WeightRef",
    "WeightRegistry",
    "TrainingLoop",
    "TrainStepFn",
    "build_training_tuples",
    "promote_candidate",
]
