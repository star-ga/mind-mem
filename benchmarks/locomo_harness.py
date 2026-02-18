#!/usr/bin/env python3
"""LoCoMo Benchmark Harness for mind-mem Recall Engine.

Evaluates mind-mem recall against the LoCoMo long-term conversational memory
benchmark from Snap Research. Downloads the dataset, converts conversations
to mind-mem workspace format, runs QA evaluation, and reports metrics.

Usage:
    python3 benchmarks/locomo_harness.py
    python3 benchmarks/locomo_harness.py --dry-run
    python3 benchmarks/locomo_harness.py --limit-k 5 --output results.json

Reference: https://github.com/snap-research/locomo
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

# Add scripts/ to path so we can import recall and block_parser
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
sys.path.insert(0, _SCRIPTS_DIR)

from recall import recall  # noqa: E402

# LoCoMo QA category mapping (from ACL 2024 paper)
CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}

# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------

LOCOMO_DATASET_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_FILE = os.path.join(CACHE_DIR, "locomo10.json")


def download_dataset(force: bool = False) -> list[dict]:
    """Download LoCoMo dataset or load from cache. Returns list of samples."""
    if not force and os.path.isfile(CACHE_FILE):
        print(f"[dataset] Loading cached dataset from {CACHE_FILE}")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[dataset] Downloading LoCoMo dataset from GitHub...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        req = urllib.request.Request(
            LOCOMO_DATASET_URL,
            headers={"User-Agent": "mind-mem-benchmark/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"[dataset] Downloaded {len(data)} conversations, cached to {CACHE_FILE}")
        return data
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"[dataset] ERROR: Failed to download dataset: {e}", file=sys.stderr)
        if os.path.isfile(CACHE_FILE):
            print("[dataset] Falling back to cached copy", file=sys.stderr)
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Workspace builder — convert LoCoMo conversation to mind-mem format
# ---------------------------------------------------------------------------

def _parse_sessions(conversation: dict) -> list[tuple[int, str, list[dict]]]:
    """Extract sessions from LoCoMo conversation dict.

    LoCoMo format: keys like 'session_1', 'session_1_date_time', 'speaker_a', etc.
    Each session is a list of turns: [{"speaker": ..., "dia_id": "D1:3", "text": ...}]

    Returns list of (session_num, date_str, turns).
    """
    import re
    sessions = []
    for key in sorted(conversation.keys()):
        m = re.match(r"^session_(\d+)$", key)
        if not m:
            continue
        num = int(m.group(1))
        turns = conversation[key]
        date_key = f"session_{num}_date_time"
        date_str = conversation.get(date_key, "unknown")
        sessions.append((num, date_str, turns))
    return sessions


def build_workspace(sample: dict, base_dir: str) -> str:
    """Convert a LoCoMo conversation sample into a mind-mem workspace directory.

    Creates the workspace at base_dir/sample_id/ with decisions/DECISIONS.md
    containing all dialog turns as blocks. Each turn becomes a block with its
    dia_id (e.g., D1:3) embedded so we can match against QA evidence.

    Also runs the entity extractor to produce atomic fact cards alongside
    the raw conversation blocks. Fact cards are short, precisely attributed
    statements that BM25 can match more accurately than full conversation turns.

    Returns the workspace path.
    """
    sample_id = sample.get("sample_id", "unknown")
    workspace = os.path.join(base_dir, str(sample_id))
    decisions_dir = os.path.join(workspace, "decisions")
    os.makedirs(decisions_dir, exist_ok=True)

    lines = ["# LoCoMo Conversation Memory", ""]

    conversation = sample.get("conversation", {})
    sessions = _parse_sessions(conversation)

    # Speaker name lookup
    speaker_a = conversation.get("speaker_a", "speaker_a")
    speaker_b = conversation.get("speaker_b", "speaker_b")
    speaker_names = {"speaker_a": speaker_a, "speaker_b": speaker_b}

    for session_num, date_str, turns in sessions:
        for turn in turns:
            dia_id = turn.get("dia_id", "")
            raw_speaker = turn.get("speaker", "unknown")
            resolved_speaker = speaker_names.get(raw_speaker, raw_speaker)
            text = turn.get("text", "")

            # Use dia_id as block ID. LoCoMo uses "D1:3" format.
            # Block parser needs [A-Z]+-... format, so prefix with DIA-.
            # D1:3 -> DIA-D1-3 (matches [A-Z]+-[^\]]+ pattern)
            block_id = f"DIA-{dia_id.replace(':', '-')}"

            lines.append(f"[{block_id}]")
            lines.append(f"Statement: [{resolved_speaker}] {text}")
            lines.append(f"Date: {date_str}")
            lines.append(f"Status: active")
            lines.append(f"DiaID: {dia_id}")
            lines.append(f"Tags: session-{session_num}, {resolved_speaker}")
            lines.append("")

    # --- Entity extraction: produce atomic fact cards ---
    try:
        from extractor import extract_from_conversation, format_as_blocks

        fact_counter = 1
        for session_num, date_str, turns in sessions:
            cards = extract_from_conversation(
                turns, speaker_a, speaker_b, date_str,
            )
            if cards:
                block_text = format_as_blocks(
                    cards, id_prefix="FACT", counter_start=fact_counter,
                )
                lines.append(block_text)
                fact_counter += len(cards)
    except ImportError:
        pass  # extractor not available, skip

    decisions_path = os.path.join(decisions_dir, "DECISIONS.md")
    with open(decisions_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return workspace


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(
    sample: dict,
    workspace: str,
    max_k: int = 10,
) -> list[dict]:
    """Run QA evaluation for one LoCoMo sample.

    For each QA pair, calls recall() and checks if evidence dialog IDs
    appear in the results. Returns per-question result dicts.
    """
    qa_pairs = sample.get("qa", [])
    results = []

    for qa in qa_pairs:
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        cat_raw = qa.get("category", 0)
        category = CATEGORY_NAMES.get(cat_raw, f"cat-{cat_raw}")
        evidence_ids = set(qa.get("evidence", []))

        if not question:
            continue

        # Call recall engine
        retrieved = recall(workspace, question, limit=max_k, active_only=False)

        # Extract dia_ids from retrieved blocks.
        # Block IDs are like "DIA-D1-3" (from "D1:3").
        # Evidence IDs are like "D1:3".
        # Check DiaID field first, then reconstruct from block ID.
        retrieved_dialog_ids = []
        for r in retrieved:
            bid = r.get("_id", "")
            dia_id = r.get("DiaID", "")
            if not dia_id and bid.startswith("DIA-"):
                # DIA-D1-3 -> D1-3 -> D1:3
                raw = bid[4:]  # strip "DIA-"
                # Find the second part after first digit group: D1-3 -> D1:3
                import re as _re
                m = _re.match(r"(D\d+)-(\d+)", raw)
                if m:
                    dia_id = f"{m.group(1)}:{m.group(2)}"
            retrieved_dialog_ids.append(dia_id)

        # Compute hit positions for evidence IDs
        hit_rank = None  # 1-based rank of first evidence hit
        hits_at_k = {1: False, 3: False, 5: False, 10: False}

        for rank, did in enumerate(retrieved_dialog_ids, 1):
            if did in evidence_ids:
                if hit_rank is None:
                    hit_rank = rank
                for k in hits_at_k:
                    if rank <= k:
                        hits_at_k[k] = True

        reciprocal_rank = (1.0 / hit_rank) if hit_rank else 0.0

        results.append({
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": list(evidence_ids),
            "retrieved_ids": retrieved_dialog_ids[:max_k],
            "hit_rank": hit_rank,
            "reciprocal_rank": reciprocal_rank,
            "recall_at": {str(k): v for k, v in hits_at_k.items()},
        })

    return results


def aggregate_metrics(all_results: list[dict]) -> dict:
    """Compute aggregate metrics from per-question results.

    Returns overall and per-category Recall@K and MRR.
    """
    if not all_results:
        return {"error": "no results"}

    # Group by category
    by_category = {}
    for r in all_results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    def compute_group_metrics(group: list[dict]) -> dict:
        n = len(group)
        if n == 0:
            return {}
        mrr = sum(r["reciprocal_rank"] for r in group) / n
        recall_at = {}
        for k in ["1", "3", "5", "10"]:
            hits = sum(1 for r in group if r["recall_at"].get(k, False))
            recall_at[k] = round(hits / n, 4)
        return {
            "count": n,
            "mrr": round(mrr, 4),
            "recall_at": recall_at,
        }

    metrics = {
        "overall": compute_group_metrics(all_results),
        "by_category": {},
    }
    for cat, group in sorted(by_category.items()):
        metrics["by_category"][cat] = compute_group_metrics(group)

    return metrics


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_results_table(metrics: dict) -> None:
    """Print a formatted results table."""
    print()
    print("=" * 72)
    print("LoCoMo Benchmark Results — mind-mem Recall Engine")
    print("=" * 72)

    header = f"{'Category':<20} {'Count':>6} {'MRR':>8} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}"
    print(header)
    print("-" * 72)

    overall = metrics.get("overall", {})
    ra = overall.get("recall_at", {})
    print(
        f"{'OVERALL':<20} {overall.get('count', 0):>6} "
        f"{overall.get('mrr', 0):>8.4f} "
        f"{ra.get('1', 0):>8.4f} "
        f"{ra.get('3', 0):>8.4f} "
        f"{ra.get('5', 0):>8.4f} "
        f"{ra.get('10', 0):>8.4f}"
    )
    print("-" * 72)

    for cat, cat_metrics in sorted(metrics.get("by_category", {}).items()):
        ra = cat_metrics.get("recall_at", {})
        print(
            f"{cat:<20} {cat_metrics.get('count', 0):>6} "
            f"{cat_metrics.get('mrr', 0):>8.4f} "
            f"{ra.get('1', 0):>8.4f} "
            f"{ra.get('3', 0):>8.4f} "
            f"{ra.get('5', 0):>8.4f} "
            f"{ra.get('10', 0):>8.4f}"
        )

    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoCoMo Benchmark Harness for mind-mem Recall Engine"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test with only the first conversation",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Maximum K for Recall@K evaluation (default: 10)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write JSON results to this file",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download dataset even if cached",
    )
    parser.add_argument(
        "--conv-ids",
        type=str,
        default=None,
        help="Comma-separated conversation indices to run (e.g. '4,7,8')",
    )
    args = parser.parse_args()

    # 1. Load dataset
    dataset = download_dataset(force=args.force_download)
    if args.conv_ids:
        conv_indices = [int(x.strip()) for x in args.conv_ids.split(",")]
        dataset = [dataset[i] for i in conv_indices if i < len(dataset)]
        print(f"[harness] Running conversations: {conv_indices} ({len(dataset)} total)")
    elif args.dry_run:
        dataset = dataset[:1]
        print(f"[harness] Dry-run mode: using 1 of {len(dataset)} conversations")
    else:
        print(f"[harness] Evaluating {len(dataset)} conversations")

    # 2. Run evaluation in a temp directory
    tmp_base = tempfile.mkdtemp(prefix="locomo_bench_")
    all_results = []
    sample_summaries = []

    try:
        t0 = time.time()

        for i, sample in enumerate(dataset):
            sample_id = sample.get("sample_id", i)
            conversation = sample.get("conversation", {})
            sessions = _parse_sessions(conversation)
            n_sessions = len(sessions)
            n_turns = sum(len(turns) for _, _, turns in sessions)
            n_qa = len(sample.get("qa", []))

            print(
                f"[harness] [{i + 1}/{len(dataset)}] sample={sample_id} "
                f"sessions={n_sessions} turns={n_turns} qa_pairs={n_qa}"
            )

            # Build workspace
            workspace = build_workspace(sample, tmp_base)

            # Evaluate
            sample_results = evaluate_sample(sample, workspace, max_k=args.max_k)
            all_results.extend(sample_results)

            # Per-sample summary
            sample_metrics = aggregate_metrics(sample_results)
            sample_summaries.append({
                "sample_id": sample_id,
                "sessions": n_sessions,
                "turns": n_turns,
                "qa_pairs": n_qa,
                "metrics": sample_metrics,
            })

            # Mini progress
            sm = sample_metrics.get("overall", {})
            print(
                f"         MRR={sm.get('mrr', 0):.4f} "
                f"R@1={sm.get('recall_at', {}).get('1', 0):.4f} "
                f"R@5={sm.get('recall_at', {}).get('5', 0):.4f}"
            )

        elapsed = time.time() - t0

        # 3. Aggregate
        overall_metrics = aggregate_metrics(all_results)

        # 4. Display
        print_results_table(overall_metrics)
        print(f"Total questions: {len(all_results)}")
        print(f"Elapsed time: {elapsed:.1f}s")

        # 5. Write JSON output
        output_data = {
            "benchmark": "locomo",
            "engine": "mind-mem-recall-bm25",
            "max_k": args.max_k,
            "dry_run": args.dry_run,
            "num_conversations": len(dataset),
            "num_questions": len(all_results),
            "elapsed_seconds": round(elapsed, 2),
            "overall": overall_metrics,
            "per_sample": sample_summaries,
        }

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            # Default output path
            default_out = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "locomo_results.json",
            )
            with open(default_out, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results written to {default_out}")

    finally:
        # Cleanup temp workspaces
        shutil.rmtree(tmp_base, ignore_errors=True)


if __name__ == "__main__":
    main()
