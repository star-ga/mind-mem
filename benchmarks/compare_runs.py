#!/usr/bin/env python3
"""Compare two LoCoMo benchmark runs side-by-side.

Usage:
    python3 benchmarks/compare_runs.py run_a.json run_b.json
    python3 benchmarks/compare_runs.py run_a.json.conv0.jsonl run_b.json.conv0.jsonl

Outputs a table comparing overall and per-category metrics, highlighting
improvements and regressions. Supports both final JSON and per-conv JSONL.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(path: str) -> list[dict]:
    """Load per-question results from JSON or JSONL."""
    p = Path(path)
    if p.suffix == ".jsonl" or ".jsonl" in p.name:
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(p) as f:
            data = json.load(f)
        return data.get("per_question", [])


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from per-question results."""
    if not results:
        return {"count": 0, "mean": 0, "acc50": 0, "acc75": 0}

    scores = [r.get("judge_score", 0) for r in results]
    n = len(scores)
    return {
        "count": n,
        "mean": sum(scores) / n,
        "acc50": sum(1 for s in scores if s >= 50) / n * 100,
        "acc75": sum(1 for s in scores if s >= 75) / n * 100,
    }


def delta_str(a: float, b: float, fmt: str = ".1f") -> str:
    """Format delta with + sign and color hint."""
    d = b - a
    sign = "+" if d > 0 else ""
    return f"{sign}{d:{fmt}}"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]
    label_a = Path(path_a).stem
    label_b = Path(path_b).stem

    results_a = load_results(path_a)
    results_b = load_results(path_b)

    # Filter out API errors
    valid_a = [r for r in results_a if "HTTP Error" not in str(r.get("generated_answer", ""))]
    valid_b = [r for r in results_b if "HTTP Error" not in str(r.get("generated_answer", ""))]

    # Overall
    m_a = compute_metrics(valid_a)
    m_b = compute_metrics(valid_b)

    # Per category
    cats_a = defaultdict(list)
    cats_b = defaultdict(list)
    for r in valid_a:
        cats_a[r.get("category", "unknown")].append(r)
    for r in valid_b:
        cats_b[r.get("category", "unknown")].append(r)

    all_cats = sorted(set(list(cats_a.keys()) + list(cats_b.keys())))

    # Print table
    w = max(len(label_a), len(label_b), 12)
    print()
    print(f"{'':20s} {'Run A':>{w}s}  {'Run B':>{w}s}  {'Delta':>8s}")
    print(f"{'':20s} {label_a[:w]:>{w}s}  {label_b[:w]:>{w}s}")
    print("-" * (20 + w * 2 + 16))

    def row(label: str, a: float, b: float, fmt: str = ".1f"):
        d = delta_str(a, b, fmt)
        marker = " ***" if abs(b - a) >= 3 else ""
        print(f"{label:20s} {a:{w}{fmt}}  {b:{w}{fmt}}  {d:>8s}{marker}")

    row("Overall mean", m_a["mean"], m_b["mean"])
    row("Accuracy @50", m_a["acc50"], m_b["acc50"])
    row("Accuracy @75", m_a["acc75"], m_b["acc75"])
    print(f"{'Questions':20s} {m_a['count']:>{w}d}  {m_b['count']:>{w}d}")
    print("-" * (20 + w * 2 + 16))

    for cat in all_cats:
        ca = compute_metrics(cats_a.get(cat, []))
        cb = compute_metrics(cats_b.get(cat, []))
        row(f"  {cat}", ca["mean"], cb["mean"])

    print()
    print("*** = delta >= 3 points (notable)")


if __name__ == "__main__":
    main()
