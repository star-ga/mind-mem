#!/usr/bin/env python3
"""Dual-protocol retrieval scoring for the eval harness.

LongMemEval retrieves at session granularity but disagrees with the
community on the *metric*: agentmemory-style reports use ``recall_any@k``
(≥1 gold session in the top-k), while official LongMemEval headlines the
stricter ``recall_all@k`` (every gold session in the top-k). We compute
**both**, at every k, per question — so a scorecard can never quietly pick
the flattering one. See ``benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


@dataclass
class QuestionScore:
    question_id: str
    question_type: str
    n_gold: int
    n_retrieved: int
    latency_ms: float
    first_gold_rank: int | None
    reciprocal_rank: float
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    recall_any_at_k: dict[int, int] = field(default_factory=dict)
    recall_all_at_k: dict[int, int] = field(default_factory=dict)
    hit: bool = False

    def to_ndjson_row(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "n_gold": self.n_gold,
            "n_retrieved": self.n_retrieved,
            "latency_ms": round(self.latency_ms, 3),
            "first_gold_rank": self.first_gold_rank,
            "reciprocal_rank": round(self.reciprocal_rank, 6),
            "hit": self.hit,
            "precision_at_k": {str(k): round(v, 6) for k, v in self.precision_at_k.items()},
            "recall_at_k": {str(k): round(v, 6) for k, v in self.recall_at_k.items()},
            "recall_any_at_k": {str(k): v for k, v in self.recall_any_at_k.items()},
            "recall_all_at_k": {str(k): v for k, v in self.recall_all_at_k.items()},
        }


def score_question(
    question_id: str,
    question_type: str,
    retrieved_doc_ids: list[str],
    gold_doc_ids: set[str],
    latency_ms: float,
    k_values: tuple[int, ...] = K_VALUES,
) -> QuestionScore:
    """Score one question under both recall protocols at every k."""
    n_gold = len(gold_doc_ids)
    first_gold_rank: int | None = None
    for rank, did in enumerate(retrieved_doc_ids, 1):
        if did in gold_doc_ids:
            first_gold_rank = rank
            break
    rr = (1.0 / first_gold_rank) if first_gold_rank else 0.0

    qs = QuestionScore(
        question_id=question_id,
        question_type=question_type,
        n_gold=n_gold,
        n_retrieved=len(retrieved_doc_ids),
        latency_ms=latency_ms,
        first_gold_rank=first_gold_rank,
        reciprocal_rank=rr,
        hit=bool(first_gold_rank is not None),
    )
    for k in k_values:
        topk = retrieved_doc_ids[:k]
        inter = sum(1 for d in topk if d in gold_doc_ids)
        qs.precision_at_k[k] = inter / k if k else 0.0
        qs.recall_at_k[k] = inter / n_gold if n_gold else 0.0
        topk_set = set(topk)
        qs.recall_any_at_k[k] = 1 if (gold_doc_ids & topk_set) else 0
        qs.recall_all_at_k[k] = 1 if (n_gold and gold_doc_ids.issubset(topk_set)) else 0
    return qs


def aggregate(scores: list[QuestionScore], k_values: tuple[int, ...] = K_VALUES) -> dict[str, Any]:
    """Mean each metric overall and per question_type."""
    if not scores:
        return {"n": 0}

    def _mean(items: list[QuestionScore]) -> dict[str, Any]:
        n = len(items)
        out: dict[str, Any] = {"n": n}
        for k in k_values:
            out[f"precision@{k}"] = round(sum(s.precision_at_k[k] for s in items) / n, 4)
            out[f"recall@{k}"] = round(sum(s.recall_at_k[k] for s in items) / n, 4)
            out[f"recall_any@{k}"] = round(sum(s.recall_any_at_k[k] for s in items) / n, 4)
            out[f"recall_all@{k}"] = round(sum(s.recall_all_at_k[k] for s in items) / n, 4)
        out["mrr"] = round(sum(s.reciprocal_rank for s in items) / n, 4)
        out["hit_rate"] = round(sum(1 for s in items if s.hit) / n, 4)
        out["mean_latency_ms"] = round(sum(s.latency_ms for s in items) / n, 3)
        return out

    by_type: dict[str, list[QuestionScore]] = {}
    for s in scores:
        by_type.setdefault(s.question_type, []).append(s)

    return {
        "n": len(scores),
        "overall": _mean(scores),
        "by_type": {t: _mean(v) for t, v in sorted(by_type.items())},
    }
