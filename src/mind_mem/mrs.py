# Copyright 2026 STARGA, Inc.
"""Model Reliability Score (MRS) framework (v2.6.0).

Aggregates per-model SLIs (latency percentiles, error rate, quality
drift, token throughput, cost per query) into a composite
reliability score in [0, 100]. Operators configure weights + alert
thresholds in a YAML-like dict; a run produces a report, optionally
flagging any SLI that crossed its threshold.

Pure stdlib. Alert delivery (email, Slack, PagerDuty) is intentionally
out of scope — callers wire whatever transport they already have.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class SLI:
    """A single Service Level Indicator reading."""

    name: str
    value: float
    unit: str = ""
    threshold: Optional[float] = None  # violation when value > threshold
    weight: float = 1.0


@dataclass(frozen=True)
class MRSReport:
    """Rolled-up MRS report for a single target (model, endpoint, backend)."""

    target: str
    score: float  # 0..100
    slis: list[SLI]
    violations: list[str]
    computed_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "score": round(self.score, 2),
            "slis": [
                {
                    "name": s.name,
                    "value": s.value,
                    "unit": s.unit,
                    "threshold": s.threshold,
                    "weight": s.weight,
                }
                for s in self.slis
            ],
            "violations": list(self.violations),
            "computed_at": self.computed_at,
        }


def percentile(values: Iterable[float], p: float) -> float:
    """Approximate percentile without numpy. p in [0, 100]."""
    arr = sorted(values)
    if not arr:
        return 0.0
    if len(arr) == 1:
        return arr[0]
    k = (len(arr) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    if f == c:
        return arr[f]
    return arr[f] + (arr[c] - arr[f]) * (k - f)


def latency_slis(latencies_ms: Iterable[float]) -> list[SLI]:
    arr = list(latencies_ms)
    if not arr:
        return []
    return [
        SLI(name="p50_ms", value=percentile(arr, 50), unit="ms", threshold=100.0),
        SLI(name="p95_ms", value=percentile(arr, 95), unit="ms", threshold=500.0),
        SLI(name="p99_ms", value=percentile(arr, 99), unit="ms", threshold=1500.0),
    ]


def error_rate_sli(error_count: int, total: int, threshold: float = 0.01) -> SLI:
    rate = (error_count / total) if total > 0 else 0.0
    return SLI(name="error_rate", value=rate, unit="fraction", threshold=threshold)


def cost_sli(cost_per_query: float, threshold: float = 0.10) -> SLI:
    return SLI(name="cost_per_query", value=cost_per_query, unit="USD", threshold=threshold)


def throughput_sli(tokens_per_second: float, min_acceptable: float = 10.0) -> SLI:
    # Represent as deficit against a floor so the same "greater = violation"
    # rule applies consistently across SLIs.
    return SLI(
        name="throughput_deficit",
        value=max(0.0, min_acceptable - tokens_per_second),
        unit="tokens/s below floor",
        threshold=0.0,
    )


def retrieval_slis(
    *,
    relevance_decay: float,
    contradiction_density: float,
    staleness_ratio: float,
) -> list[SLI]:
    """Memory-retrieval-specific SLIs from the roadmap."""
    return [
        SLI(
            name="relevance_decay",
            value=relevance_decay,
            unit="fraction/day",
            threshold=0.05,
        ),
        SLI(
            name="contradiction_density",
            value=contradiction_density,
            unit="per 100 blocks",
            threshold=0.5,
        ),
        SLI(
            name="staleness_ratio",
            value=staleness_ratio,
            unit="fraction",
            threshold=0.2,
        ),
    ]


def compute_mrs(target: str, slis: Iterable[SLI], *, computed_at: str = "") -> MRSReport:
    """Aggregate SLIs into a 0..100 composite MRS score.

    Each SLI contributes ``weight * (1 - penalty)`` where penalty
    scales linearly from 0 (well under threshold) to 1 (double the
    threshold or worse). Targets without a threshold contribute full
    weight (no penalty possible).
    """
    slis_list = list(slis)
    total_weight = sum(max(0.0, s.weight) for s in slis_list) or 1.0
    score_accum = 0.0
    violations: list[str] = []
    for s in slis_list:
        w = max(0.0, s.weight)
        if w == 0:
            continue
        if s.threshold is None:
            score_accum += w
            continue
        if s.threshold <= 0:
            # Deficit-style SLIs where any positive value is a hit.
            penalty = min(1.0, s.value / max(1e-9, s.threshold + 1.0))
        else:
            penalty = min(1.0, max(0.0, (s.value - s.threshold) / s.threshold))
        if s.value > s.threshold:
            violations.append(s.name)
        score_accum += w * (1.0 - penalty)
    score = 100.0 * (score_accum / total_weight)
    return MRSReport(
        target=target,
        score=max(0.0, min(100.0, score)),
        slis=slis_list,
        violations=violations,
        computed_at=computed_at,
    )


def parse_slo_spec(spec: Mapping[str, Any]) -> list[SLI]:
    """Turn a roadmap-style YAML-ish SLO spec into :class:`SLI` inputs.

    Expected shape::

        {"slis": [
            {"name": "p99_ms", "threshold": 1500, "weight": 1.0},
            ...
        ]}

    Current values are left at 0; callers fill them in before
    :func:`compute_mrs`. This lets an SLO file define WHAT to measure
    independently of runtime readings.
    """
    out: list[SLI] = []
    for entry in spec.get("slis", []):
        if not isinstance(entry, Mapping):
            continue
        out.append(
            SLI(
                name=str(entry.get("name", "unnamed")),
                value=float(entry.get("value", 0.0)),
                unit=str(entry.get("unit", "")),
                threshold=(
                    float(entry["threshold"]) if "threshold" in entry else None
                ),
                weight=float(entry.get("weight", 1.0)),
            )
        )
    return out


__all__ = [
    "SLI",
    "MRSReport",
    "percentile",
    "latency_slis",
    "error_rate_sli",
    "cost_sli",
    "throughput_sli",
    "retrieval_slis",
    "compute_mrs",
    "parse_slo_spec",
]
