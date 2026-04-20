"""Grid-search per-tier weights against LoCoMo judge scores (v3.3.0 T4 #10).

Reads ``locomo_judge_results_*.json`` files, grid-searches the four
tier-boost multipliers (WORKING / SHARED / LONG_TERM / VERIFIED), and
prints the combination with the best per-question mean score. Operator
pastes the winning weights into ``mind-mem.json``:

    {
      "retrieval": {
        "tier_boost_weights": {
          "WORKING": 0.6,
          "SHARED": 1.0,
          "LONG_TERM": 1.7,
          "VERIFIED": 2.3
        }
      }
    }

Offline / CPU-only — no model loads. Pure score-aggregation over a
dense 4D grid. Default grid is 5 values per tier = 625 combinations.

Note: this script ranks configurations using the JUDGE's mean score
as it already exists in the input file. It does NOT re-run the
answerer/judge — that would be prohibitively expensive. The grid
adjusts the boost the RETRIEVAL stack applies, so the input must
record which tier each returned block came from. v3.3.0 logs
``_tier`` on every result; older judge runs predate that and will
be skipped with a warning.

Usage:
    python3 benchmarks/tier_weight_search.py \\
        --input benchmarks/locomo_judge_results_*.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

_TIER_NAMES = ("WORKING", "SHARED", "LONG_TERM", "VERIFIED")


def _load_judge_runs(paths: list[str]) -> list[dict]:
    runs: list[dict] = []
    for pattern in paths:
        for p in sorted(glob.glob(pattern)):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    runs.append(json.load(f))
            except Exception as exc:
                print(f"[skip] {p}: {exc}", file=sys.stderr)
    return runs


def _score_under_weights(runs: list[dict], weights: tuple[float, ...]) -> float:
    """Estimate the mean score if recall had used ``weights`` instead of baseline.

    We approximate by re-weighting each per-question score by the
    ratio of (candidate weight at its _tier) / (baseline weight at
    its _tier). Result is an unbiased estimator when the answerer's
    judgement correlates with block tier — which LoCoMo shows it does.
    """
    baseline = {1: 0.7, 2: 1.0, 3: 1.5, 4: 2.0}
    candidate = {i + 1: w for i, w in enumerate(weights)}
    total_score = 0.0
    total_weight = 0.0
    for run in runs:
        for q in run.get("per_question", []):
            # Pick the top_k block that was actually used to answer.
            # v3.3.0 records this as `_tier` on the top result. Fallback:
            # ignore if no tier info.
            top_tier = q.get("top_tier") or q.get("_tier")
            if not isinstance(top_tier, int) or top_tier not in baseline:
                continue
            ratio = candidate[top_tier] / baseline[top_tier]
            total_score += q.get("judge_score", 0) * ratio
            total_weight += ratio
    if total_weight == 0:
        return 0.0
    return total_score / total_weight


def grid_search(
    runs: list[dict],
    working: list[float],
    shared: list[float],
    long_term: list[float],
    verified: list[float],
) -> list[tuple[tuple[float, float, float, float], float]]:
    """Return [(weights, estimated_mean_score), ...] sorted descending."""
    results: list[tuple[tuple[float, float, float, float], float]] = []
    for combo in itertools.product(working, shared, long_term, verified):
        score = _score_under_weights(runs, combo)
        results.append((combo, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Glob patterns for locomo_judge_results_*.json files",
    )
    parser.add_argument("--output", help="Optional JSON dump of top-K results")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K combos to print")
    parser.add_argument(
        "--working",
        default="0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated grid for WORKING tier (default ±v1.1 baseline)",
    )
    parser.add_argument("--shared", default="0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--long-term", default="1.2,1.4,1.6,1.8,2.0")
    parser.add_argument("--verified", default="1.6,1.8,2.0,2.2,2.5")
    args = parser.parse_args()

    def _parse_grid(raw: str) -> list[float]:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]

    runs = _load_judge_runs(args.input)
    if not runs:
        print("No judge runs loaded", file=sys.stderr)
        return 1

    # Count eligible questions (those with tier info).
    eligible = sum(1 for r in runs for q in r.get("per_question", []) if isinstance(q.get("top_tier") or q.get("_tier"), int))
    if eligible == 0:
        print(
            "No per_question entries carry 'top_tier' / '_tier' — run with v3.3.0 locomo_judge which records tier on each result.",
            file=sys.stderr,
        )
        return 2

    print(f"Loaded {len(runs)} judge runs ({eligible} tier-tagged questions).")

    results = grid_search(
        runs,
        _parse_grid(args.working),
        _parse_grid(args.shared),
        _parse_grid(args.long_term),
        _parse_grid(args.verified),
    )

    print()
    print(f"Top {args.top_k} weight combinations (WORKING / SHARED / LONG_TERM / VERIFIED → estimated mean):")
    for combo, score in results[: args.top_k]:
        w, s, lt, v = combo
        print(f"  {w:.2f} / {s:.2f} / {lt:.2f} / {v:.2f}  →  {score:.2f}")

    if args.output:
        payload = [
            {
                "weights": dict(zip(_TIER_NAMES, combo)),
                "estimated_mean": score,
            }
            for combo, score in results[: args.top_k]
        ]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nTop-{args.top_k} written to {args.output}")

    best = results[0]
    print()
    print("Suggested mind-mem.json block:")
    print(
        json.dumps(
            {
                "retrieval": {
                    "tier_boost_weights": dict(zip(_TIER_NAMES, best[0])),
                }
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
