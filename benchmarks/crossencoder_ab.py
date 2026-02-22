#!/usr/bin/env python3
"""Cross-Encoder A/B Test — retrieval-level comparison.

Compares retrieval quality with and without cross-encoder reranking on
LoCoMo conv-0.  Measures:
  - Reciprocal Rank (MRR) of gold-answer keywords in retrieved context
  - Rank displacement per question (how many positions the best hit moves)
  - Per-category breakdown

Also loads the existing v1.1.1 LLM-as-judge baseline to correlate
retrieval changes with judge-score outcomes.

Usage:
    python3 benchmarks/crossencoder_ab.py
    python3 benchmarks/crossencoder_ab.py --blend-weight 0.4 --top-k 18
    python3 benchmarks/crossencoder_ab.py --output benchmarks/ce_ab_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_HERE, "..", "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.append(_HERE)

def _gold_keywords(answer: str) -> list[str]:
    """Extract lowercased keywords from a gold answer for hit matching."""
    import re

    # Remove common filler
    text = str(answer).lower().strip()
    # Tokenize on non-alpha
    tokens = re.findall(r"[a-z]{2,}", text)
    # Remove very common words
    stops = {
        "the", "is", "was", "are", "were", "and", "or", "but", "for",
        "not", "this", "that", "with", "from", "has", "had", "have",
        "she", "he", "they", "her", "his", "its", "it", "an", "of",
        "in", "to", "on", "at", "by", "as", "be", "no", "yes", "did",
        "does", "do", "about", "would", "could", "should", "will",
        "been", "being", "also", "just", "than", "more", "some",
    }
    return [t for t in tokens if t not in stops]


def _best_hit_rank(retrieved: list[dict], gold_kw: list[str]) -> int | None:
    """Return 1-based rank of the first hit containing >=50% of gold keywords.

    Returns None if no hit matches.
    """
    if not gold_kw:
        return None
    threshold = max(1, len(gold_kw) // 2)
    for rank, r in enumerate(retrieved, 1):
        text = (r.get("excerpt", "") or r.get("Statement", "")).lower()
        matches = sum(1 for kw in gold_kw if kw in text)
        if matches >= threshold:
            return rank
    return None


def _reciprocal_rank(rank: int | None) -> float:
    """Reciprocal rank (0 if not found)."""
    if rank is None:
        return 0.0
    return 1.0 / rank


def run_ab_test(
    conv_index: int = 0,
    top_k: int = 18,
    blend_weight: float = 0.6,
    ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict:
    """Run cross-encoder A/B on a single LoCoMo conversation.

    Returns a dict with per-question and aggregate metrics.
    """
    from cross_encoder_reranker import CrossEncoderReranker  # noqa: E402
    from locomo_harness import (  # noqa: E402
        CATEGORY_NAMES,
        build_workspace,
        download_dataset,
    )
    from recall import recall  # noqa: E402

    dataset = download_dataset()
    sample = dataset[conv_index]
    del dataset

    sample_id = sample.get("sample_id", conv_index)
    qa_pairs = sample.get("qa", [])

    print(f"[ce-ab] conv={conv_index} sample={sample_id} qa={len(qa_pairs)}")
    print(f"[ce-ab] cross-encoder model: {ce_model}")
    print(f"[ce-ab] blend_weight={blend_weight} top_k={top_k}")

    # Build workspace
    conv_tmp = tempfile.mkdtemp(prefix=f"ceab_{sample_id}_")
    try:
        workspace = build_workspace(sample, conv_tmp)

        # Initialize cross-encoder once
        ce = CrossEncoderReranker(model=ce_model)

        results = []
        t0 = time.time()

        for qi, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            cat_raw = qa.get("category", 0)
            category = CATEGORY_NAMES.get(cat_raw, f"cat-{cat_raw}")
            is_adversarial = (cat_raw == 5) or (str(cat_raw).lower() == "adversarial")
            gold_answer = qa.get("adversarial_answer", qa.get("answer", "")) if is_adversarial else qa.get("answer", "")

            if not question:
                continue

            gold_kw = _gold_keywords(str(gold_answer))

            # --- Run A: BM25-only (baseline) ---
            bm25_results = recall(workspace, question, limit=top_k, active_only=False)

            # --- Run B: BM25 + cross-encoder rerank ---
            # Feed wider pool to CE, then take top_k
            wide_pool = recall(workspace, question, limit=top_k * 3, active_only=False)
            for r in wide_pool:
                if "content" not in r:
                    r["content"] = r.get("excerpt", "")
            ce_results = ce.rerank(
                question, wide_pool, top_k=top_k, blend_weight=blend_weight,
            )

            # Measure retrieval quality
            rank_a = _best_hit_rank(bm25_results, gold_kw)
            rank_b = _best_hit_rank(ce_results, gold_kw)

            rr_a = _reciprocal_rank(rank_a)
            rr_b = _reciprocal_rank(rank_b)

            record = {
                "question": question,
                "category": category,
                "gold_answer": str(gold_answer),
                "gold_keywords": gold_kw[:10],
                "rank_bm25": rank_a,
                "rank_ce": rank_b,
                "rr_bm25": round(rr_a, 4),
                "rr_ce": round(rr_b, 4),
                "rr_delta": round(rr_b - rr_a, 4),
                "found_bm25": rank_a is not None,
                "found_ce": rank_b is not None,
            }
            results.append(record)

            if (qi + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"[ce-ab] {qi+1}/{len(qa_pairs)} ({elapsed:.1f}s)")

        elapsed = time.time() - t0
        print(f"[ce-ab] done: {len(results)} questions in {elapsed:.1f}s")

    finally:
        shutil.rmtree(conv_tmp, ignore_errors=True)

    # --- Aggregate metrics ---
    metrics = _aggregate(results)
    metrics["config"] = {
        "conv_index": conv_index,
        "sample_id": sample_id,
        "top_k": top_k,
        "blend_weight": blend_weight,
        "ce_model": ce_model,
        "elapsed_seconds": round(elapsed, 1),
        "num_questions": len(results),
    }
    metrics["per_question"] = results
    return metrics


def _aggregate(results: list[dict]) -> dict:
    """Compute aggregate metrics from per-question results."""
    from collections import defaultdict

    cats = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r)
    cats["overall"] = results

    metrics = {}
    for cat, items in sorted(cats.items()):
        n = len(items)
        mrr_bm25 = sum(r["rr_bm25"] for r in items) / n if n else 0
        mrr_ce = sum(r["rr_ce"] for r in items) / n if n else 0
        found_bm25 = sum(1 for r in items if r["found_bm25"])
        found_ce = sum(1 for r in items if r["found_ce"])

        # Count improvements / regressions
        improved = sum(1 for r in items if r["rr_delta"] > 0)
        regressed = sum(1 for r in items if r["rr_delta"] < 0)
        unchanged = n - improved - regressed

        metrics[cat] = {
            "count": n,
            "mrr_bm25": round(mrr_bm25, 4),
            "mrr_ce": round(mrr_ce, 4),
            "mrr_delta": round(mrr_ce - mrr_bm25, 4),
            "hit_rate_bm25": round(found_bm25 / n * 100, 1) if n else 0,
            "hit_rate_ce": round(found_ce / n * 100, 1) if n else 0,
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
        }

    return {"metrics": metrics}


def _print_table(data: dict) -> None:
    """Print formatted comparison table."""
    metrics = data["metrics"]
    config = data.get("config", {})

    print()
    print("=" * 90)
    print("Cross-Encoder A/B Test — Retrieval Quality (LoCoMo)")
    print("=" * 90)
    print(f"  Model:        {config.get('ce_model', '?')}")
    print(f"  Blend weight: {config.get('blend_weight', '?')}")
    print(f"  Top-k:        {config.get('top_k', '?')}")
    print(f"  Questions:    {config.get('num_questions', '?')}")
    print(f"  Elapsed:      {config.get('elapsed_seconds', '?')}s")
    print()

    header = (
        f"{'Category':<16} {'N':>5}  "
        f"{'MRR(BM25)':>10} {'MRR(CE)':>10} {'Delta':>8}  "
        f"{'Hit%(BM25)':>10} {'Hit%(CE)':>10}  "
        f"{'Up':>4} {'Down':>4} {'Same':>4}"
    )
    print(header)
    print("-" * 90)

    # Print overall first, then per-category
    for cat in ["overall"] + [c for c in sorted(metrics) if c != "overall"]:
        m = metrics[cat]
        delta_str = f"{m['mrr_delta']:+.4f}"
        label = cat.upper() if cat == "overall" else f"  {cat}"
        print(
            f"{label:<16} {m['count']:>5}  "
            f"{m['mrr_bm25']:>10.4f} {m['mrr_ce']:>10.4f} {delta_str:>8}  "
            f"{m['hit_rate_bm25']:>9.1f}% {m['hit_rate_ce']:>9.1f}%  "
            f"{m['improved']:>4} {m['regressed']:>4} {m['unchanged']:>4}"
        )
        if cat == "overall":
            print("-" * 90)

    print("=" * 90)

    # Verdict
    overall = metrics.get("overall", {})
    delta = overall.get("mrr_delta", 0)
    if delta > 0.01:
        verdict = f"Cross-encoder IMPROVES retrieval by {delta:+.4f} MRR"
    elif delta < -0.01:
        verdict = f"Cross-encoder DEGRADES retrieval by {delta:+.4f} MRR"
    else:
        verdict = f"Cross-encoder has NEGLIGIBLE impact ({delta:+.4f} MRR)"
    print(f"\nVerdict: {verdict}")
    print()


def _correlate_with_judge(data: dict, judge_path: str) -> None:
    """Correlate retrieval changes with existing judge scores."""
    if not os.path.isfile(judge_path):
        print(f"[ce-ab] Judge baseline not found at {judge_path}, skipping correlation")
        return

    with open(judge_path) as f:
        judge_data = json.load(f)

    judge_by_q = {}
    for r in judge_data.get("per_question", []):
        judge_by_q[r["question"]] = r.get("judge_score", 0)

    improved_scores = []
    regressed_scores = []
    unchanged_scores = []

    for r in data.get("per_question", []):
        q = r["question"]
        js = judge_by_q.get(q)
        if js is None:
            continue
        if r["rr_delta"] > 0:
            improved_scores.append(js)
        elif r["rr_delta"] < 0:
            regressed_scores.append(js)
        else:
            unchanged_scores.append(js)

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0

    print("Correlation with LLM-as-Judge Scores (v1.1.1 baseline):")
    print(f"  Improved retrieval  (n={len(improved_scores):>3}): avg judge={_avg(improved_scores):.1f}")
    print(f"  Regressed retrieval (n={len(regressed_scores):>3}): avg judge={_avg(regressed_scores):.1f}")
    print(f"  Unchanged retrieval (n={len(unchanged_scores):>3}): avg judge={_avg(unchanged_scores):.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Encoder A/B Test for mind-mem retrieval"
    )
    parser.add_argument(
        "--conv", type=int, default=0,
        help="Conversation index (default: 0)",
    )
    parser.add_argument(
        "--top-k", type=int, default=18,
        help="Number of results to retrieve (default: 18)",
    )
    parser.add_argument(
        "--blend-weight", type=float, default=0.6,
        help="CE blend weight: 0=pure BM25, 1=pure CE (default: 0.6)",
    )
    parser.add_argument(
        "--ce-model", type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write JSON results to this file",
    )
    args = parser.parse_args()

    data = run_ab_test(
        conv_index=args.conv,
        top_k=args.top_k,
        blend_weight=args.blend_weight,
        ce_model=args.ce_model,
    )

    _print_table(data)

    # Correlate with judge baseline
    judge_path = os.path.join(_HERE, "locomo_v1.1.1_mistral_large_conv0.json")
    _correlate_with_judge(data, judge_path)

    # Save results
    out_path = args.output or os.path.join(
        _HERE, f"crossencoder_ab_conv{args.conv}.json"
    )
    # Don't include per_question in saved file to keep it compact
    save_data = {k: v for k, v in data.items() if k != "per_question"}
    save_data["per_question_count"] = len(data.get("per_question", []))
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
