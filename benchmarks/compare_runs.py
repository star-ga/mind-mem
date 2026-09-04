#!/usr/bin/env python3
"""Compare two benchmark runs -- unpaired for LoCoMo, paired for ranking.

Two modes, because two questions:

``paired`` (the ranking gate)
    Runs the paired scorecard over two per-question NDJSON artifacts: exact
    McNemar on recall_any@k and recall_all@k, a paired sign test on MRR, and a
    seeded bootstrap CI on each mean difference. Defaults its baseline to the
    committed 2026-09-03 rep-1 artifact, so "did this move ranking?" always has
    one fixed thing to be measured against.

        python3 benchmarks/compare_runs.py paired NEW.ndjson
        python3 benchmarks/compare_runs.py paired NEW.ndjson --baseline OLD.ndjson
        python3 benchmarks/compare_runs.py paired B.ndjson --baseline A.ndjson --require-identical

    ``--require-identical`` is the no-diff assertion: it exits non-zero unless
    every tested metric moved on **zero** questions in **both** directions. A
    change claiming "latency only, no ranking movement" runs this and gets a
    measurement instead of an argument. (For the same claim in-process, at
    served ``(id, score)`` granularity, use ``benchmarks/ranking_identity.py``.)

``locomo`` (the original, still the default for two positional paths)
    Side-by-side judge-score table over two LoCoMo runs. Unpaired and
    descriptive; it reports means, not evidence, and says so.

        python3 benchmarks/compare_runs.py run_a.json run_b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.paired_scorecard import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PINNED_BASELINE,
    MetricComparison,
    PairingError,
    Scorecard,
    build_scorecard,
)

# --------------------------------------------------------------------------
# locomo mode -- unchanged behaviour, unpaired judge scores
# --------------------------------------------------------------------------


def load_results(path: str) -> list[dict]:
    """Load per-question results from JSON or JSONL."""
    p = Path(path)
    if p.suffix == ".jsonl" or ".jsonl" in p.name:
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(p, encoding="utf-8") as f:
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


def run_locomo(path_a: str, path_b: str) -> int:
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
    print("*** = delta >= 3 points (notable); unpaired means, not evidence of a difference")
    return 0


# --------------------------------------------------------------------------
# paired mode -- the ranking gate
# --------------------------------------------------------------------------


def _elide(label: str, width: int) -> str:
    """Shorten a label from the MIDDLE, keeping both ends.

    Run artifacts are named by a shared date-and-suite prefix and differ only
    in their tail (``-mind_mem-rep1`` vs ``-bm25_baseline-rep1``). Trimming
    the tail renders the two arms of a comparison as the same string, which
    is a worse table than a wide one: the reader cannot tell which row is
    which. Keeping both ends keeps them distinguishable.
    """
    if len(label) <= width:
        return label
    keep = width - 1
    head = keep // 2
    return label[:head] + "~" + label[len(label) - (keep - head) :]


def render_comparison(c: MetricComparison, baseline_label: str, candidate_label: str) -> str:
    """One metric block. Both discordant directions, never a net."""
    ci = c.ci
    width = 28
    base = _elide(baseline_label, width)
    cand = _elide(candidate_label, width)
    return "\n".join(
        [
            f"{c.label}  [{c.test}]",
            f"  {base:<{width}s} {c.baseline_mean:.4f}",
            f"  {cand:<{width}s} {c.candidate_mean:.4f}",
            f"  discordant                   {c.n_discordant}  "
            f"(candidate-only {c.candidate_only} / baseline-only {c.baseline_only}; concordant {c.concordant})",
            f"  p-value                      {float(c.p_value):.4f}  (alpha {float(c.alpha):g}, "
            f"min discordant for significance {c.min_discordant_for_significance})",
            f"  mean diff                    {ci.mean_diff:+.4f}  "
            f"[{ci.low:+.4f}, {ci.high:+.4f}] {ci.confidence:.0%} bootstrap CI, "
            f"seed {ci.seed}, {ci.resamples} resamples",
            f"  verdict                      {c.verdict} -- {c.note}",
        ]
    )


def render_scorecard(card: Scorecard) -> str:
    header = [
        "",
        "PAIRED SCORECARD",
        f"  baseline    {card.baseline_path}",
        f"  candidate   {card.candidate_path}",
        f"  paired on   {card.n_pairs} question(s) at k={card.k}",
    ]
    if card.dropped_non_ok:
        header.append(f"  dropped     {len(card.dropped_non_ok)} question(s) whose unit did not complete")
    body = [render_comparison(c, card.baseline_label, card.candidate_label) for c in card.comparisons]
    moved = card.moved()
    if card.identical:
        tail = "IDENTICAL: zero discordant questions on every tested metric, in both directions."
    else:
        tail = "MOVED: " + ", ".join(f"{c.label} ({c.candidate_only} up / {c.baseline_only} down)" for c in moved)
    return "\n".join(header) + "\n\n" + "\n\n".join(body) + "\n\n" + tail


def run_paired(args: argparse.Namespace) -> int:
    try:
        card = build_scorecard(
            args.baseline,
            args.candidate,
            k=args.k,
            seed=args.seed,
            resamples=args.resamples,
            drop_non_ok=args.drop_non_ok,
        )
    except PairingError as exc:
        print(f"cannot pair the runs: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        # A mistyped path is the likeliest way to invoke this wrongly, and a
        # traceback buries which of the two artifacts was not there.
        print(f"cannot read an artifact: {exc}", file=sys.stderr)
        return 2
    print(render_scorecard(card))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(card.as_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {args.json}")
    if args.require_identical and not card.identical:
        print("\nFAIL --require-identical: the served ranking moved (see MOVED above)", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compare_runs.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode")

    paired = sub.add_parser("paired", help="paired scorecard over two per-question NDJSON runs")
    paired.add_argument("candidate", help="per-question NDJSON of the run under test")
    paired.add_argument(
        "--baseline", default=str(PINNED_BASELINE), help="per-question NDJSON to compare against (default: the pinned artifact)"
    )
    paired.add_argument("--k", type=int, default=5, help="cut-off for the recall metrics (default 5)")
    paired.add_argument("--seed", type=int, default=BOOTSTRAP_SEED, help=f"bootstrap seed (default {BOOTSTRAP_SEED}, committed)")
    paired.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES, help=f"bootstrap resamples (default {BOOTSTRAP_RESAMPLES})")
    paired.add_argument("--drop-non-ok", action="store_true", help="exclude questions whose unit did not complete, and record which")
    paired.add_argument("--require-identical", action="store_true", help="exit 1 unless zero questions moved on every metric")
    paired.add_argument("--json", help="also write the scorecard as JSON to this path")

    locomo = sub.add_parser("locomo", help="unpaired LoCoMo judge-score table (the original behaviour)")
    locomo.add_argument("run_a")
    locomo.add_argument("run_b")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward compatibility: two bare paths still mean the LoCoMo table.
    if len(argv) == 2 and argv[0] not in {"paired", "locomo"} and not argv[0].startswith("-"):
        return run_locomo(argv[0], argv[1])
    args = build_parser().parse_args(argv)
    if args.mode == "paired":
        return run_paired(args)
    if args.mode == "locomo":
        return run_locomo(args.run_a, args.run_b)
    build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
