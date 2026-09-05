#!/usr/bin/env python3
"""Recompute every headline metric from RAW per-unit rows, and nothing else.

This module is the hinge of the repro package. A benchmark runner writes raw
rows; this module turns raw rows into the published numbers; the verifier
(``benchmarks/repro_verify.py``) runs it again over the *committed* rows and
demands the same answer. Because both sides call the same function, the
committed ``metrics.json`` is not a claim about a run that happened once — it
is a value anyone can regenerate from the artifact in front of them.

Two rules keep that honest:

* **Never trust a summary field in a row.** A NIAH row carries ``found``, but
  ``niah_metrics`` re-derives the hit from the retrieved ids/excerpts using the
  same decision rule the benchmark used, and reports any row whose stored
  verdict disagrees with its own evidence. A tampered result list therefore
  fails, not just a tampered aggregate.
* **No clock, no filesystem, no config.** Everything here is a pure function of
  the rows, so a third party with only the NDJSON can reach the same numbers.

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

#: Package schema tag. Bump when the row or manifest layout changes shape.
SCHEMA = "mind-mem/repro-package@1"

#: A unit that neither returned nor was killed is still a unit; these are the
#: statuses ``benchmarks/hard_timeout`` can report.
STATUS_OK = "ok"


def decision_fingerprint(pairs: list[tuple[str, list[str]]]) -> str:
    """A single hash over what was *retrieved*, with the timing left out.

    Latency does not reproduce across boxes, so a whole-file diff cannot answer
    "did this rerun make the same decisions?". This does: it covers each unit id
    and the ids it returned, in rank order, and nothing else. Two runs that
    agree here retrieved identically even where they timed differently.
    """
    payload = json.dumps(sorted(pairs), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _percentiles(values: list[float]) -> dict[str, float]:
    """Nearest-rank percentiles — deterministic, no interpolation, no numpy."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)

    def at(p: float) -> float:
        # Nearest-rank: ceil(p * n), 1-indexed, clamped.
        idx = max(1, min(n, int(-(-p * n // 1))))
        return round(ordered[idx - 1], 3)

    return {
        "n": n,
        "p50_ms": at(0.50),
        "p90_ms": at(0.90),
        "p95_ms": at(0.95),
        "p99_ms": at(0.99),
        "max_ms": round(ordered[-1], 3),
        "mean_ms": round(sum(ordered) / n, 3),
    }


# ---------------------------------------------------------------------------
# NIAH
# ---------------------------------------------------------------------------


def niah_hit(row: dict[str, Any]) -> bool:
    """Re-apply the benchmark's own hit rule to one row's raw evidence.

    Mirrors ``tests/test_niah.py::_check_needle_found``: the needle counts as
    found when its block id is in the top-K, or when some returned excerpt
    contains every expected keyword. Recomputed here rather than read from
    ``row['found']`` so that editing the retrieved list changes the metric.
    """
    needle_id = str(row.get("needle_id", ""))
    keywords = [str(k).lower() for k in row.get("expected_keywords", [])]
    for hit in row.get("retrieved", []):
        if str(hit.get("id", "")) == needle_id:
            return True
        excerpt = str(hit.get("excerpt", "")).lower()
        if keywords and all(kw in excerpt for kw in keywords):
            return True
    return False


def niah_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass rate plus the per-size / per-depth breakdown, from raw rows only."""
    by_size: dict[str, list[int]] = {}
    by_depth: dict[str, list[int]] = {}
    passed = 0
    disagreements: list[str] = []
    killed = 0
    latencies: list[float] = []

    for row in rows:
        recomputed = niah_hit(row)
        stored = bool(row.get("found", False))
        if recomputed != stored:
            disagreements.append(str(row.get("unit_id", "?")))
        status = str(row.get("unit_status", STATUS_OK))
        if status != STATUS_OK:
            killed += 1
        else:
            latencies.append(float(row.get("latency_ms", 0.0)))
        passed += int(recomputed)
        size = str(row.get("haystack_size", "?"))
        depth = str(row.get("depth_pct", "?"))
        by_size.setdefault(size, [0, 0])
        by_depth.setdefault(depth, [0, 0])
        by_size[size][0] += int(recomputed)
        by_size[size][1] += 1
        by_depth[depth][0] += int(recomputed)
        by_depth[depth][1] += 1

    total = len(rows)
    return {
        "benchmark": "NIAH",
        "headline": {
            "metric": "top_k_hit_rate",
            "passed": passed,
            "total": total,
            "as_text": f"{passed}/{total}",
            "value": round(passed / total, 6) if total else 0.0,
        },
        "breakdown": {
            "by_haystack_size": {
                k: {"passed": v[0], "total": v[1]} for k, v in sorted(by_size.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)
            },
            "by_depth_pct": {
                k: {"passed": v[0], "total": v[1]} for k, v in sorted(by_depth.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)
            },
        },
        "integrity": {
            "rows": total,
            "killed_or_crashed": killed,
            "rows_whose_stored_verdict_disagrees_with_their_evidence": sorted(disagreements),
        },
        "determinism": {
            "decision_fingerprint": decision_fingerprint(
                [(str(r.get("unit_id", "?")), [str(h.get("id", "?")) for h in r.get("retrieved", [])]) for r in rows]
            ),
            "basis": "sha256 over (unit_id, retrieved ids in rank order) for every row; excludes timing, so it is comparable across boxes",
        },
        # Timing is measured, not deterministic: it is recomputable from these
        # same rows, but it does NOT reproduce across boxes or runs. Labelled
        # so nobody reads a latency diff as a correctness diff.
        "timing_not_run_to_run_stable": _percentiles(latencies),
    }


# ---------------------------------------------------------------------------
# LongMemEval
# ---------------------------------------------------------------------------


def longmemeval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate LongMemEval rows with the shipped scorer, from raw rows only.

    Reuses ``mind_mem.bench.eval_scorer`` rather than re-deriving the metric:
    a second implementation that drifts is a second number, which is the
    failure this whole package exists to prevent.
    """
    from mind_mem.bench.eval_scorer import QuestionScore, aggregate

    scores: list[QuestionScore] = []
    killed = 0
    for row in rows:
        if str(row.get("unit_status", STATUS_OK)) != STATUS_OK:
            killed += 1
        scores.append(
            QuestionScore(
                question_id=str(row["question_id"]),
                question_type=str(row["question_type"]),
                n_gold=int(row["n_gold"]),
                n_retrieved=int(row["n_retrieved"]),
                latency_ms=float(row["latency_ms"]),
                first_gold_rank=row["first_gold_rank"],
                reciprocal_rank=float(row["reciprocal_rank"]),
                precision_at_k={int(k): float(v) for k, v in row["precision_at_k"].items()},
                recall_at_k={int(k): float(v) for k, v in row["recall_at_k"].items()},
                recall_any_at_k={int(k): int(v) for k, v in row["recall_any_at_k"].items()},
                recall_all_at_k={int(k): int(v) for k, v in row["recall_all_at_k"].items()},
                hit=bool(row["hit"]),
            )
        )
    agg = aggregate(scores)
    return {
        "benchmark": "LongMemEval-S",
        "headline": {
            "metric": "recall_any@5",
            "value": agg.get("overall", {}).get("recall_any@5"),
            "as_text": str(agg.get("overall", {}).get("recall_any@5")),
        },
        "overall": agg.get("overall", {"n": 0}),
        "by_type": agg.get("by_type", {}),
        "integrity": {"rows": len(rows), "killed_or_crashed": killed},
    }


#: benchmark name -> raw rows -> metrics. The verifier dispatches on the
#: ``benchmark`` field of the manifest; an unknown name is a hard failure,
#: never a skip, because a package nobody can recompute is not evidence.
METRIC_FNS: dict[str, Callable[[list[dict[str, Any]]], dict[str, Any]]] = {
    "NIAH": niah_metrics,
    "LongMemEval-S": longmemeval_metrics,
}
