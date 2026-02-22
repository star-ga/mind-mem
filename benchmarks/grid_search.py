#!/usr/bin/env python3
"""BM25F Field Weight Grid Search for mind-mem Recall Engine.

Tests different BM25F field weight combinations against the LoCoMo
retrieval benchmark, recording R@1, R@5, R@10, and MRR for each
combination. Outputs a sorted comparison table and saves results
to benchmarks/grid_search_results.json.

Usage:
    python3 benchmarks/grid_search.py
    python3 benchmarks/grid_search.py --dry-run
    python3 benchmarks/grid_search.py --conv-ids 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import tempfile
import time

# Add scripts/ and benchmarks/ to path
_BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_BENCHMARKS_DIR, "..", "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _BENCHMARKS_DIR)

import _recall_constants  # noqa: E402
from locomo_harness import (  # noqa: E402
    aggregate_metrics,
    build_workspace,
    download_dataset,
    evaluate_sample,
)

# Primary fields to tune (highest-impact on retrieval quality).
# Each is varied at 3 levels: baseline * 0.5, baseline, baseline * 1.5.
PRIMARY_FIELDS = ["Statement", "Title", "Name", "Summary", "Context"]

# Multiplier steps relative to baseline: 50%, 100%, 150%
MULTIPLIER_STEPS = [0.5, 1.0, 1.5]


def generate_grid(
    baseline: dict[str, float],
    fields: list[str] | None = None,
    steps: list[float] | None = None,
) -> list[dict[str, float]]:
    """Generate weight combinations by varying each primary field independently.

    Instead of a full cartesian product (3^5 = 243 combos), uses a
    one-at-a-time design: for each field, test each multiplier step while
    keeping all other fields at baseline. This gives (fields * steps) + 1
    combinations including baseline.

    Args:
        baseline: Current field weight dict.
        fields: Fields to vary (defaults to PRIMARY_FIELDS).
        steps: Multiplier steps (defaults to MULTIPLIER_STEPS).

    Returns:
        List of weight dicts to evaluate.
    """
    if fields is None:
        fields = PRIMARY_FIELDS
    if steps is None:
        steps = MULTIPLIER_STEPS

    combos = []

    # Always include baseline first
    combos.append(dict(baseline))

    for field in fields:
        if field not in baseline:
            continue
        base_val = baseline[field]
        for mult in steps:
            if mult == 1.0:
                continue  # skip baseline duplicate
            variant = dict(baseline)
            variant[field] = round(base_val * mult, 2)
            combos.append(variant)

    return combos


def generate_full_grid(
    baseline: dict[str, float],
    fields: list[str] | None = None,
    steps: list[float] | None = None,
) -> list[dict[str, float]]:
    """Generate full cartesian product grid (expensive).

    For 5 fields * 3 steps = 243 combinations.

    Args:
        baseline: Current field weight dict.
        fields: Fields to vary (defaults to PRIMARY_FIELDS).
        steps: Multiplier steps (defaults to MULTIPLIER_STEPS).

    Returns:
        List of weight dicts to evaluate.
    """
    if fields is None:
        fields = PRIMARY_FIELDS
    if steps is None:
        steps = MULTIPLIER_STEPS

    # Build per-field value lists
    field_values = {}
    for field in fields:
        if field not in baseline:
            continue
        base_val = baseline[field]
        field_values[field] = [round(base_val * m, 2) for m in steps]

    ordered_fields = [f for f in fields if f in field_values]
    value_lists = [field_values[f] for f in ordered_fields]

    combos = []
    for values in itertools.product(*value_lists):
        variant = dict(baseline)
        for field, val in zip(ordered_fields, values):
            variant[field] = val
        combos.append(variant)

    return combos


def _weights_label(weights: dict[str, float], baseline: dict[str, float]) -> str:
    """Human-readable label showing which fields differ from baseline."""
    diffs = []
    for field in PRIMARY_FIELDS:
        if field in weights and field in baseline:
            if weights[field] != baseline[field]:
                diffs.append(f"{field}={weights[field]}")
    return ", ".join(diffs) if diffs else "baseline"


def run_grid_search(
    dataset: list[dict],
    combos: list[dict[str, float]],
    baseline: dict[str, float],
    max_k: int = 10,
) -> list[dict]:
    """Run the grid search across all weight combinations.

    For each combo, patches _recall_constants.FIELD_WEIGHTS, builds
    workspaces, evaluates, and collects metrics.

    Args:
        dataset: LoCoMo dataset samples.
        combos: List of weight dicts to evaluate.
        baseline: Original baseline weights (for labeling).
        max_k: Recall@K cutoff.

    Returns:
        List of result dicts sorted by R@10 descending.
    """
    results = []

    for ci, weights in enumerate(combos):
        label = _weights_label(weights, baseline)
        print(f"\n[grid] [{ci + 1}/{len(combos)}] Testing: {label}")

        # Patch the module-level FIELD_WEIGHTS used by recall()
        original_weights = dict(_recall_constants.FIELD_WEIGHTS)
        _recall_constants.FIELD_WEIGHTS.clear()
        _recall_constants.FIELD_WEIGHTS.update(weights)

        tmp_base = tempfile.mkdtemp(prefix="grid_search_")
        all_qa_results = []

        try:
            t0 = time.time()

            for si, sample in enumerate(dataset):
                workspace = build_workspace(sample, tmp_base)
                sample_results = evaluate_sample(sample, workspace, max_k=max_k)
                all_qa_results.extend(sample_results)

            elapsed = time.time() - t0
            metrics = aggregate_metrics(all_qa_results)
            overall = metrics.get("overall", {})
            ra = overall.get("recall_at", {})

            result = {
                "label": label,
                "weights": {f: weights[f] for f in PRIMARY_FIELDS if f in weights},
                "r_at_1": ra.get("1", 0.0),
                "r_at_5": ra.get("5", 0.0),
                "r_at_10": ra.get("10", 0.0),
                "mrr": overall.get("mrr", 0.0),
                "count": overall.get("count", 0),
                "elapsed_s": round(elapsed, 2),
                "by_category": metrics.get("by_category", {}),
            }
            results.append(result)

            print(
                f"         R@1={result['r_at_1']:.4f} "
                f"R@5={result['r_at_5']:.4f} "
                f"R@10={result['r_at_10']:.4f} "
                f"MRR={result['mrr']:.4f} "
                f"({elapsed:.1f}s)"
            )

        finally:
            # Restore original weights
            _recall_constants.FIELD_WEIGHTS.clear()
            _recall_constants.FIELD_WEIGHTS.update(original_weights)
            shutil.rmtree(tmp_base, ignore_errors=True)

    # Sort by R@10 descending, then MRR descending
    results.sort(key=lambda r: (r["r_at_10"], r["mrr"]), reverse=True)
    return results


def print_comparison_table(results: list[dict], baseline_label: str = "baseline") -> None:
    """Print a sorted comparison table of grid search results."""
    print()
    print("=" * 90)
    print("BM25F Weight Grid Search Results")
    print("=" * 90)

    header = (
        f"{'Rank':>4} {'Label':<35} {'R@1':>8} {'R@5':>8} "
        f"{'R@10':>8} {'MRR':>8} {'Time':>7}"
    )
    print(header)
    print("-" * 90)

    baseline_r10 = None
    for i, r in enumerate(results):
        if r["label"] == baseline_label:
            baseline_r10 = r["r_at_10"]
            break

    for i, r in enumerate(results):
        label = r["label"]
        if len(label) > 33:
            label = label[:30] + "..."

        delta = ""
        if baseline_r10 is not None and r["label"] != baseline_label:
            diff = (r["r_at_10"] - baseline_r10) * 100
            delta = f" ({diff:+.1f}pp)" if diff != 0 else ""

        marker = " *" if r["label"] == baseline_label else ""

        print(
            f"{i + 1:>4} {label:<35} {r['r_at_1']:>8.4f} {r['r_at_5']:>8.4f} "
            f"{r['r_at_10']:>8.4f} {r['mrr']:>8.4f} {r['elapsed_s']:>6.1f}s"
            f"{delta}{marker}"
        )

    print("=" * 90)
    if baseline_r10 is not None:
        print(f"  * = baseline (R@10={baseline_r10:.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="BM25F Field Weight Grid Search for mind-mem"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use only the first conversation",
    )
    parser.add_argument(
        "--conv-ids",
        type=str,
        default=None,
        help="Comma-separated conversation indices (e.g. '0,1,2')",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Recall@K cutoff (default: 10)",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Use full cartesian product instead of one-at-a-time",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON path (default: benchmarks/grid_search_results.json)",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = download_dataset()
    if args.conv_ids:
        indices = [int(x.strip()) for x in args.conv_ids.split(",")]
        dataset = [dataset[i] for i in indices if i < len(dataset)]
        print(f"[grid] Using conversations: {indices} ({len(dataset)} total)")
    elif args.dry_run:
        dataset = dataset[:1]
        print("[grid] Dry-run: using 1 conversation")
    else:
        print(f"[grid] Using all {len(dataset)} conversations")

    # Current baseline weights
    baseline = dict(_recall_constants.FIELD_WEIGHTS)

    # Generate grid
    if args.full_grid:
        combos = generate_full_grid(baseline)
        print(f"[grid] Full cartesian grid: {len(combos)} combinations")
    else:
        combos = generate_grid(baseline)
        print(f"[grid] One-at-a-time grid: {len(combos)} combinations")

    # Run
    results = run_grid_search(dataset, combos, baseline, max_k=args.max_k)

    # Display
    print_comparison_table(results)

    # Check if best combo improves R@10 by >= 1pp over baseline
    baseline_result = next((r for r in results if r["label"] == "baseline"), None)
    best_result = results[0] if results else None

    if baseline_result and best_result and best_result["label"] != "baseline":
        improvement = best_result["r_at_10"] - baseline_result["r_at_10"]
        if improvement >= 0.01:
            print(
                f"[grid] Best combo improves R@10 by {improvement * 100:.1f}pp "
                f"(>= 1pp threshold). Consider updating defaults."
            )
            print(f"[grid] Best weights: {best_result['weights']}")
        else:
            print(
                f"[grid] Best improvement is {improvement * 100:.1f}pp "
                f"(< 1pp threshold). Keeping current defaults."
            )

    # Save results
    output_path = args.output or os.path.join(_BENCHMARKS_DIR, "grid_search_results.json")
    output_data = {
        "benchmark": "locomo-grid-search",
        "engine": "mind-mem-recall-bm25f",
        "max_k": args.max_k,
        "num_conversations": len(dataset),
        "num_combinations": len(combos),
        "baseline_weights": {f: baseline[f] for f in PRIMARY_FIELDS if f in baseline},
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[grid] Results saved to {output_path}")


if __name__ == "__main__":
    main()
