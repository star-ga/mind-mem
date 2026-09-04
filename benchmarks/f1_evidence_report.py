# Copyright 2026 STARGA, Inc.
"""Turn two score-contract probe artifacts into the report the release cites.

:mod:`benchmarks.f1_score_contract_probe` captures one tree's served lists.
This module compares two of them through :mod:`benchmarks.ranking_identity`
and writes a single JSON artifact answering three questions the 5.0.2 entry
makes claims about:

1. **Does the product default move?** Every case whose key starts with a
   :data:`~benchmarks.f1_score_contract_probe.DEFAULT_PATH_PREFIXES` prefix
   goes through :func:`~benchmarks.ranking_identity.assert_battery_unchanged`.
   Failure raises; there is no "mostly identical".

2. **Where it does move, which cases and how?** Every other case is diffed and
   named. Silence about the ones that moved would make claim 1 a selection.

3. **How broken was the column F1 replaced?** The count of served lists that
   were not non-increasing in ``score``, over the historical 45-case battery
   the CHANGELOG quotes, computed from the artifact rather than recalled.

What is compared, and what deliberately is not
----------------------------------------------
The fingerprint is over the served **ids in rank order**. It cannot be over
``(id, score)``: F1 *is* a change to the score column, so a ``(id, score)``
comparison is guaranteed to differ on every hit and would answer a question
nobody asked. The score column's change is recorded as a separate digest pair
-- shown to differ, which is the fix landing -- while the ordering claim is
carried by the id fingerprint, where a reordering has nowhere to hide.

Two ways this comparison could pass without meaning anything are closed:

* **Both arms are the same tree.** Each artifact records the resolved
  ``hybrid_recall.py`` path and its SHA-256; equal digests are refused.
* **The comparison inspected nothing.** ``ranking_identity`` refuses an empty
  battery, a battery whose keys changed, and lists thinner than
  ``min_results``.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.f1_score_contract_probe import DEFAULT_PATH_PREFIXES, LEGACY_PREFIXES  # noqa: E402
from benchmarks.ranking_identity import (  # noqa: E402
    Fingerprint,
    assert_battery_unchanged,
    compare_rankings,
    fingerprint_digest,
    ranking_fingerprint,
)

#: Neighbouring scores closer than this are equal. ``score`` is rounded to six
#: places upstream, so a bare ``>`` would call rounding a violation.
MONOTONIC_TOLERANCE = 1e-9

#: Fewest hits a default-path list must carry for its identity to be evidence.
MIN_RESULTS = 2


class SameTreeTwice(Exception):
    """Both artifacts came from one tree, so the comparison cannot fail."""


def _order_fingerprint(rows: Sequence[Sequence[Any]]) -> Fingerprint:
    """The served ids in rank order, as a ``ranking_identity`` fingerprint.

    Rank is carried in the score slot so the encoding stays the canonical one
    -- ids and positions, length-prefixed, no bespoke second spelling of "a
    served answer".
    """
    return ranking_fingerprint(
        [{"_id": str(row[0]), "rank": position} for position, row in enumerate(rows)],
        id_field="_id",
        score_field="rank",
    )


def _score_fingerprint(rows: Sequence[Sequence[Any]]) -> Fingerprint:
    """The served ``(id, score)`` list -- the column F1 intentionally changes."""
    return ranking_fingerprint([{"_id": str(row[0]), "score": float(row[1])} for row in rows], id_field="_id", score_field="score")


def not_non_increasing(rows: Sequence[Sequence[Any]]) -> bool:
    """Does this served list rise at any adjacent pair?"""
    return any(float(rows[i + 1][1]) - float(rows[i][1]) > MONOTONIC_TOLERANCE for i in range(len(rows) - 1))


def monotonicity(cases: Mapping[str, Sequence[Sequence[Any]]], prefixes: Sequence[str]) -> dict[str, Any]:
    """Count the lists whose ``score`` column did not order them.

    ``orderable`` is reported next to the violation count on purpose: a
    one-hit list cannot violate, so "24 of 45" without it hides that the
    denominator able to fail was 27.
    """
    selected = {k: v for k, v in cases.items() if any(k.startswith(p) for p in prefixes)}
    orderable = {k: v for k, v in selected.items() if len(v) >= 2}
    violating = sorted(k for k, v in orderable.items() if not_non_increasing(v))
    return {
        "prefixes": list(prefixes),
        "lists": len(selected),
        "orderable_lists": len(orderable),
        "not_non_increasing": len(violating),
        "violating_cases": violating,
    }


def build_report(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two probe artifacts; raise unless the default path held."""
    before_prov = before["provenance"]
    after_prov = after["provenance"]
    if before_prov["hybrid_recall_sha256"] == after_prov["hybrid_recall_sha256"]:
        raise SameTreeTwice(
            f"both artifacts hash to {before_prov['hybrid_recall_sha256'][:16]}; a before/after pair from one tree cannot detect a change"
        )

    before_cases: Mapping[str, Sequence[Sequence[Any]]] = before["cases"]
    after_cases: Mapping[str, Sequence[Sequence[Any]]] = after["cases"]

    default_keys = sorted(k for k in before_cases if any(k.startswith(p) for p in DEFAULT_PATH_PREFIXES))
    default_before = {k: _order_fingerprint(before_cases[k]) for k in default_keys}
    default_after = {k: _order_fingerprint(after_cases[k]) for k in default_keys}
    # Raises RankingMoved / VacuousComparison. Nothing below runs on failure.
    assert_battery_unchanged(default_before, default_after, label="default fused path", min_results=MIN_RESULTS)

    moved: list[dict[str, Any]] = []
    for key in sorted(before_cases):
        diff = compare_rankings(_order_fingerprint(before_cases[key]), _order_fingerprint(after_cases[key]))
        if diff.moved:
            moved.append(
                {
                    "case": key,
                    "before_ids": [str(row[0]) for row in before_cases[key]],
                    "after_ids": [str(row[0]) for row in after_cases[key]],
                    "first_divergence_rank": None if diff.first_divergence is None else diff.first_divergence + 1,
                }
            )

    score_changed = sorted(k for k in before_cases if _score_fingerprint(before_cases[k]) != _score_fingerprint(after_cases[k]))

    return {
        "before": before_prov,
        "after": after_prov,
        "cases": len(before_cases),
        "default_path": {
            "prefixes": list(DEFAULT_PATH_PREFIXES),
            "cases": len(default_keys),
            "hits": sum(len(before_cases[k]) for k in default_keys),
            "min_results": MIN_RESULTS,
            "identical": True,
            "digest": fingerprint_digest(tuple(pair for key in default_keys for pair in default_before[key])),
        },
        "served_order_moved": {"cases": len(moved), "detail": moved},
        "score_column_changed": {
            "cases": len(score_changed),
            "note": (
                "F1 rewrites the score column by design; a case here without a "
                "served_order_moved entry kept its order and changed only the number attached to it."
            ),
        },
        "monotonicity_before": monotonicity(before_cases, LEGACY_PREFIXES),
        "monotonicity_after": monotonicity(after_cases, LEGACY_PREFIXES),
        "reproduce": [
            "python3.12 benchmarks/f1_score_contract_probe.py AFTER.json",
            "F1_PROBE_SRC=<tree>/src python3.12 benchmarks/f1_score_contract_probe.py BEFORE.json",
            "python3.12 benchmarks/f1_evidence_report.py BEFORE.json AFTER.json REPORT.json",
        ],
    }


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: f1_evidence_report.py BEFORE.json AFTER.json REPORT.json", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as handle:
        before = json.load(handle)
    with open(argv[2], encoding="utf-8") as handle:
        after = json.load(handle)
    report = build_report(before, after)
    with open(argv[3], "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"wrote {argv[3]}: default path identical over {report['default_path']['cases']} case(s) / "
        f"{report['default_path']['hits']} hit(s); {report['served_order_moved']['cases']} of {report['cases']} case(s) moved; "
        f"pre-fix non-monotone {report['monotonicity_before']['not_non_increasing']} of "
        f"{report['monotonicity_before']['lists']} ({report['monotonicity_before']['orderable_lists']} orderable), "
        f"post-fix {report['monotonicity_after']['not_non_increasing']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
