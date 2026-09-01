"""Per-feature retrieval attribution (v3.3.0 architect audit item #7).

Every v3.3.0 retrieval feature (query decomposition, graph expand,
entity prefetch, session boost, temporal decay, truth score, CE
rerank, reranker ensemble, query expansion) fires conditionally. When
the LoCoMo benchmark returns a score, the operator needs to know
which features actually contributed — without it, a +5-point lift
is indistinguishable from noise.

:class:`RetrievalTrace` is a thread-local accumulator: each helper
records ``(feature_name, latency_ms, added_count, top_score_delta)``
into the active trace; the caller reads the trace at the end of
``search()`` and emits it alongside the results.

Zero-cost when disabled: if no trace is active, the record calls
no-op. Opt-in via ``retrieval.trace_attribution`` in
``mind-mem.json``.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .observability import get_logger

_log = get_logger("retrieval_trace")


@dataclass
class FeatureStep:
    """One feature's contribution to a single recall call."""

    feature: str
    latency_ms: float
    added_count: int = 0
    top_score_delta: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    """Per-request accumulator of every feature that fired."""

    query: str
    steps: list[FeatureStep] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)

    def record(self, step: FeatureStep) -> None:
        self.steps.append(step)

    def total_latency_ms(self) -> float:
        return round((time.monotonic() - self.started_at) * 1000.0, 3)

    def summary(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "total_latency_ms": self.total_latency_ms(),
            "steps": [
                {
                    "feature": s.feature,
                    "latency_ms": round(s.latency_ms, 3),
                    "added_count": s.added_count,
                    "top_score_delta": round(s.top_score_delta, 4),
                    **({"metadata": s.metadata} if s.metadata else {}),
                }
                for s in self.steps
            ],
        }


_active: contextvars.ContextVar[RetrievalTrace | None] = contextvars.ContextVar("mind_mem_retrieval_trace", default=None)


@contextmanager
def trace(query: str) -> Iterator[RetrievalTrace]:
    """Activate an attribution trace for the current context.

    Nested traces are supported: the inner trace overrides the outer
    for the duration of the block, the outer resumes on exit.
    """
    t = RetrievalTrace(query=query)
    token = _active.set(t)
    try:
        yield t
    finally:
        _active.reset(token)


def current_trace() -> RetrievalTrace | None:
    """Return the active trace for this context, or None."""
    return _active.get()


@contextmanager
def step(feature: str, **metadata: Any) -> Iterator[dict[str, Any]]:
    """Timer context-manager that records a feature step on exit.

    Yields a mutable dict the caller can populate with
    ``added_count`` / ``top_score_delta`` before the block exits:

    .. code-block:: python

        with step("graph_expand") as rec:
            out = graph_expand(seeds, corpus, ...)
            rec["added_count"] = len(out) - len(seeds)
            rec["top_score_delta"] = (out[0].score - seeds[0].score)
    """
    t = _active.get()
    start = time.monotonic()
    rec: dict[str, Any] = {"added_count": 0, "top_score_delta": 0.0}
    try:
        yield rec
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if t is not None:
            t.record(
                FeatureStep(
                    feature=feature,
                    latency_ms=elapsed_ms,
                    added_count=int(rec.get("added_count", 0) or 0),
                    top_score_delta=float(rec.get("top_score_delta", 0.0) or 0.0),
                    metadata={**metadata, **{k: v for k, v in rec.items() if k not in {"added_count", "top_score_delta"}}},
                )
            )
        # Log every step — cheap structured-log attribution even when
        # no trace is active.
        _log.debug(
            "retrieval_step",
            feature=feature,
            latency_ms=round(elapsed_ms, 3),
            added_count=int(rec.get("added_count", 0) or 0),
            top_score_delta=round(float(rec.get("top_score_delta", 0.0) or 0.0), 4),
        )


def is_trace_enabled(config: dict[str, Any] | None) -> bool:
    if not config or not isinstance(config, dict):
        return False
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return False
    return bool(retrieval.get("trace_attribution", False))


__all__ = [
    "FeatureStep",
    "RetrievalTrace",
    "current_trace",
    "is_trace_enabled",
    "step",
    "trace",
]
