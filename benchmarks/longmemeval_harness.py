#!/usr/bin/env python3
"""LongMemEval Benchmark Harness for mind-mem recall engine.

Evaluates mind-mem BM25 recall against the LongMemEval benchmark (ICLR 2025).
Downloads the dataset from HuggingFace, converts chat sessions into mind-mem
block format, runs retrieval queries, and reports Recall@K and MRR metrics
with per-question-type breakdowns.

Usage:
    python3 benchmarks/longmemeval_harness.py
    python3 benchmarks/longmemeval_harness.py --dry-run
    python3 benchmarks/longmemeval_harness.py --subset longmemeval_m
    python3 benchmarks/longmemeval_harness.py --data-path /path/to/local.json
    python3 benchmarks/longmemeval_harness.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import urllib.error

# Add mind-mem scripts to path for recall imports
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from recall import recall  # noqa: E402


# HuggingFace dataset URLs (cleaned release)
HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
SUBSET_URLS = {
    "longmemeval_s": f"{HF_BASE}/longmemeval_s_cleaned.json",
    "longmemeval_m": f"{HF_BASE}/longmemeval_m_cleaned.json",
}

# Recall@K values to evaluate
K_VALUES = [1, 3, 5, 10]


def download_dataset(subset: str, cache_dir: str) -> str:
    """Download LongMemEval JSON from HuggingFace. Returns path to local file."""
    if subset not in SUBSET_URLS:
        raise ValueError(f"Unknown subset '{subset}'. Choose from: {list(SUBSET_URLS.keys())}")

    url = SUBSET_URLS[subset]
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"{subset}.json")

    if os.path.isfile(local_path):
        print(f"  Using cached dataset: {local_path}")
        return local_path

    print(f"  Downloading {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mind-mem-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        with open(local_path, "wb") as f:
            f.write(data)
        size_mb = len(data) / (1024 * 1024)
        print(f"  Downloaded {size_mb:.1f} MB -> {local_path}")
    except urllib.error.URLError as e:
        print(f"  ERROR: Failed to download dataset: {e}", file=sys.stderr)
        print("  Use --data-path to provide a local copy.", file=sys.stderr)
        sys.exit(1)

    return local_path


def load_dataset(path: str) -> list[dict]:
    """Load and validate the LongMemEval JSON dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Some HF datasets wrap in {"data": [...]} or similar
        for key in ("data", "instances", "questions"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    print(f"  Loaded {len(data)} questions from dataset")
    return data


def session_to_block_text(session: list[dict], session_idx: int, date: str) -> str:
    """Convert a single LongMemEval session (list of turns) into a mind-mem block.

    Block format:
        [SESSION-{idx}]
        Statement: {all turns concatenated}
        Date: {date}
        Status: active
    """
    parts = []
    for turn in session:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        parts.append(f"[{role}]: {content}")

    statement = "\n".join(parts)
    block = f"[SESSION-{session_idx}]\nStatement: {statement}\nDate: {date}\nStatus: active\n"
    return block


def build_workspace(question: dict, tmpdir: str) -> str:
    """Create a temporary mind-mem workspace from a LongMemEval question's haystack.

    Writes all haystack sessions as blocks into decisions/DECISIONS.md
    (one of the CORPUS_FILES that recall() scans).

    Returns the workspace path.
    """
    workspace = os.path.join(tmpdir, f"ws_{question.get('question_id', 'unknown')}")
    decisions_dir = os.path.join(workspace, "decisions")
    os.makedirs(decisions_dir, exist_ok=True)

    sessions = question.get("haystack_sessions", [])
    session_ids = question.get("haystack_session_ids", list(range(len(sessions))))
    dates = question.get("haystack_dates", [])

    blocks = []
    for i, session in enumerate(sessions):
        sid = session_ids[i] if i < len(session_ids) else i
        date = dates[i] if i < len(dates) else "2024-01-01"
        blocks.append(session_to_block_text(session, sid, date))

    content = "\n---\n\n".join(blocks)
    decisions_path = os.path.join(decisions_dir, "DECISIONS.md")
    with open(decisions_path, "w", encoding="utf-8") as f:
        f.write(content)

    return workspace


def evaluate_question(question: dict, tmpdir: str, max_k: int = 10) -> dict | None:
    """Run recall for a single question and compute retrieval metrics.

    Returns a dict with:
        question_id, question_type, recall@K values, reciprocal_rank,
        retrieved_ids, answer_session_ids
    Or None if the question should be skipped (abstention).
    """
    qid = question.get("question_id", "")

    # Skip abstention questions
    if qid.endswith("_abs"):
        return None

    query = question.get("question", "")
    answer_ids = set(question.get("answer_session_ids", []))
    qtype = question.get("question_type", "unknown")

    if not query or not answer_ids:
        return None

    # Build workspace and run recall
    workspace = build_workspace(question, tmpdir)
    results = recall(workspace, query, limit=max_k, active_only=False)

    # Extract session IDs from retrieved results
    # Block IDs are "SESSION-{session_id}" where session_id is a string
    # like "sharegpt_yywfIrx_0" or "answer_280352e9"
    retrieved_ids = []
    for r in results:
        bid = r.get("_id", "")
        if bid.startswith("SESSION-"):
            sid = bid[len("SESSION-"):]
            retrieved_ids.append(sid)

    # Compute Recall@K for each K
    recall_at_k = {}
    for k in K_VALUES:
        top_k = set(retrieved_ids[:k])
        hit = 1 if answer_ids & top_k else 0
        recall_at_k[f"recall@{k}"] = hit

    # Compute reciprocal rank (1/rank of first correct hit, 0 if not found)
    rr = 0.0
    for rank, sid in enumerate(retrieved_ids, 1):
        if sid in answer_ids:
            rr = 1.0 / rank
            break

    # Clean up workspace
    shutil.rmtree(workspace, ignore_errors=True)

    return {
        "question_id": qid,
        "question_type": qtype,
        **recall_at_k,
        "reciprocal_rank": rr,
        "retrieved_ids": retrieved_ids,
        "answer_session_ids": list(answer_ids),
    }


def aggregate_results(per_question: list[dict]) -> dict:
    """Compute aggregate metrics overall and per question type."""
    if not per_question:
        return {"overall": {}, "by_type": {}, "n": 0}

    # Group by type
    by_type: dict[str, list[dict]] = {}
    for q in per_question:
        qtype = q["question_type"]
        by_type.setdefault(qtype, []).append(q)

    def compute_metrics(items: list[dict]) -> dict:
        n = len(items)
        if n == 0:
            return {"n": 0}
        metrics = {"n": n}
        for k in K_VALUES:
            key = f"recall@{k}"
            metrics[key] = sum(q[key] for q in items) / n
        metrics["mrr"] = sum(q["reciprocal_rank"] for q in items) / n
        return metrics

    overall = compute_metrics(per_question)
    type_metrics = {t: compute_metrics(items) for t, items in sorted(by_type.items())}

    return {
        "overall": overall,
        "by_type": type_metrics,
        "n": len(per_question),
    }


def print_results_table(agg: dict):
    """Print a formatted results table."""
    print("\n" + "=" * 80)
    print("LongMemEval Benchmark Results (mind-mem BM25 Recall)")
    print("=" * 80)

    header = f"{'Category':<35} {'N':>5}  {'R@1':>6}  {'R@3':>6}  {'R@5':>6}  {'R@10':>6}  {'MRR':>6}"
    print(header)
    print("-" * 80)

    # Overall
    o = agg["overall"]
    if o.get("n", 0) > 0:
        print(
            f"{'OVERALL':<35} {o['n']:>5}  "
            f"{o.get('recall@1', 0):>6.3f}  "
            f"{o.get('recall@3', 0):>6.3f}  "
            f"{o.get('recall@5', 0):>6.3f}  "
            f"{o.get('recall@10', 0):>6.3f}  "
            f"{o.get('mrr', 0):>6.3f}"
        )
    print("-" * 80)

    # Per type
    for qtype, m in agg.get("by_type", {}).items():
        if m.get("n", 0) > 0:
            print(
                f"{qtype:<35} {m['n']:>5}  "
                f"{m.get('recall@1', 0):>6.3f}  "
                f"{m.get('recall@3', 0):>6.3f}  "
                f"{m.get('recall@5', 0):>6.3f}  "
                f"{m.get('recall@10', 0):>6.3f}  "
                f"{m.get('mrr', 0):>6.3f}"
            )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="LongMemEval Benchmark Harness for mind-mem recall engine"
    )
    parser.add_argument(
        "--subset",
        default="longmemeval_s",
        choices=list(SUBSET_URLS.keys()),
        help="Dataset subset to evaluate (default: longmemeval_s)",
    )
    parser.add_argument(
        "--data-path",
        help="Path to a local LongMemEval JSON file (skips download)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only evaluate first 10 questions",
    )
    parser.add_argument(
        "--output",
        help="Write JSON results to this file",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"),
        help="Directory to cache downloaded datasets",
    )
    args = parser.parse_args()

    print(f"LongMemEval Harness — subset={args.subset}, dry_run={args.dry_run}")
    print()

    # Step 1: Load dataset
    print("[1/3] Loading dataset...")
    if args.data_path:
        data_path = args.data_path
        print(f"  Using local file: {data_path}")
    else:
        data_path = download_dataset(args.subset, args.cache_dir)
    dataset = load_dataset(data_path)

    if args.dry_run:
        dataset = dataset[:10]
        print(f"  Dry-run: using first {len(dataset)} questions")

    # Step 2: Evaluate
    print(f"\n[2/3] Evaluating {len(dataset)} questions...")
    per_question = []
    skipped = 0
    t0 = time.time()

    for i, question in enumerate(dataset):
        result = evaluate_question(question, tempfile.gettempdir())
        if result is None:
            skipped += 1
            continue
        per_question.append(result)

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i + 1}/{len(dataset)}] {rate:.1f} q/s, evaluated={len(per_question)}, skipped={skipped}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(per_question)} evaluated, {skipped} skipped)")

    # Step 3: Aggregate and report
    print("\n[3/3] Computing metrics...")
    agg = aggregate_results(per_question)
    print_results_table(agg)

    # Save full results
    full_results = {
        "subset": args.subset,
        "dry_run": args.dry_run,
        "elapsed_seconds": round(elapsed, 2),
        "evaluated": len(per_question),
        "skipped": skipped,
        "aggregate": agg,
        "per_question": per_question,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nFull results written to {args.output}")

    # Also print compact JSON summary to stdout
    summary = {k: v for k, v in full_results.items() if k != "per_question"}
    print(f"\nJSON summary:\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
