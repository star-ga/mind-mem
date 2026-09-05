# Copyright 2026 STARGA, Inc.
"""Read the ablation battery back and render the table, with its uncertainty.

One row per pre-committed mask in :data:`benchmarks.ablation_mask.MASKS`, each
compared against **two** references and never against only the flattering one:

* the unmasked ``control`` run — what does removing this modifier do to the
  product's own ordering, and
* the committed zero-dep ``bm25_baseline`` artifact — does removing it close
  any of the gap that started this.

Both comparisons go through :mod:`benchmarks.paired_scorecard` (exact McNemar
on the binary metrics, exact paired sign test on MRR, seeded bootstrap CI),
so the discordant counts are reported in **both** directions and a 11-vs-11
tie can never be rendered as "no change".

The column that keeps this honest is ``detectable``. A mask whose ranking is
identical to the control's reports zero discordant questions — and so does a
battery too thin to move anything. The two are told apart by the same tests
run on a pair that IS known to differ (control vs baseline): if that pair
shows a significant delta over the same n, the battery had the power to see
one, and a zero really is a null.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.ablation_mask import MASKS  # noqa: E402
from benchmarks.paired_scorecard import build_scorecard, load_run  # noqa: E402

#: The two question types the recency hypothesis lives in.
SUSPECT_TYPES = ("single-session-preference", "temporal-reasoning")

METRICS = ("recall_any_at_k", "recall_all_at_k", "reciprocal_rank")


def headline(rows: Sequence[dict[str, Any]], k: int = 5) -> dict[str, float]:
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "recall_any@k": sum(int(r["recall_any_at_k"][str(k)]) for r in rows) / n,
        "recall_all@k": sum(int(r["recall_all_at_k"][str(k)]) for r in rows) / n,
        "mrr": sum(float(r["reciprocal_rank"]) for r in rows) / n,
    }


def by_type(rows: Sequence[dict[str, Any]], qtype: str, k: int = 5) -> dict[str, float]:
    return headline([r for r in rows if r.get("question_type") == qtype], k=k)


def _cmp(baseline: str, candidate: str, k: int) -> dict[str, Any]:
    card = build_scorecard(baseline, candidate, k=k)
    out: dict[str, Any] = {"n_pairs": card.n_pairs, "identical": card.identical}
    for c in card.comparisons:
        out[c.metric] = {
            "baseline_mean": round(c.baseline_mean, 6),
            "candidate_mean": round(c.candidate_mean, 6),
            "candidate_only": c.candidate_only,
            "baseline_only": c.baseline_only,
            "n_discordant": c.n_discordant,
            "p_value": float(c.p_value),
            "verdict": c.verdict,
            "min_discordant_for_significance": c.min_discordant_for_significance,
            "ci": [round(c.ci.low, 6), round(c.ci.high, 6)],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the LongMemEval-S ablation table")
    ap.add_argument("--dir", default="benchmarks/.cache/ablation")
    ap.add_argument("--floor", default="docs/benchmarks/2026-09-03-longmemeval-s-full-bm25_baseline-rep1.ndjson")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None, help="write the JSON result here")
    a = ap.parse_args(argv)

    control_path = os.path.join(a.dir, "lme-control.ndjson")
    result: dict[str, Any] = {
        "k": a.k,
        "control_path": control_path,
        "floor_path": a.floor,
        "masks": {},
        "power_check": {},
    }

    # Power check FIRST: the pair that is known to differ, over the same n and
    # the same tests. Without it a table of zeros is unreadable.
    result["power_check"]["control_vs_floor"] = _cmp(a.floor, control_path, a.k)

    floor_rows = load_run(a.floor)
    result["floor_headline"] = headline(floor_rows, a.k)
    result["floor_by_type"] = {t: by_type(floor_rows, t, a.k) for t in SUSPECT_TYPES}

    for mask in MASKS:
        path = os.path.join(a.dir, f"lme-{mask}.ndjson")
        if not os.path.isfile(path):
            result["masks"][mask] = {"missing": path}
            continue
        rows = load_run(path)
        entry: dict[str, Any] = {
            "stages_disabled": list(MASKS[mask]),
            "path": path,
            "headline": headline(rows, a.k),
            "by_type": {t: by_type(rows, t, a.k) for t in SUSPECT_TYPES},
        }
        if mask != "control":
            entry["vs_control"] = _cmp(control_path, path, a.k)
            entry["vs_floor"] = _cmp(a.floor, path, a.k)
        result["masks"][mask] = entry

    text = json.dumps(result, indent=2, sort_keys=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
