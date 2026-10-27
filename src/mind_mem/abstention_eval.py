# Copyright 2026 STARGA, Inc.
"""The falsification condition for computed abstention, as something to run.

The roadmap entry that proposed :mod:`mind_mem.recall_abstention` named its own
kill switch: *run recall over a labelled set where the right answer is known to
be absent; if the derived statistic does not separate those queries from ones
where the answer is present, there is no signal here and no amount of
threshold tuning will create one.* Nobody has measured the score distributions
a given workspace actually produces, so whether the statistic means anything
there is an empirical question -- this module asks it.

Give it labelled queries (``{"query": "...", "present": true|false}``, one JSON
object per line) and it serves each one through the same door ``mm recall``
uses, computes the set-level separation for every result list, and reports:

* the **AUC** of separation as a present-vs-absent classifier (0.5 is chance);
* the distribution of the statistic per class, and how many queries came back
  empty (an empty absent query is a correct, trivial abstention; an empty
  present query is a miss);
* what the configured bands would do -- how often they would ``act`` on an
  absent answer (the costly error for a hallucination guardrail) and how often
  they would stop on a present one;
* a **verdict**: ``separates`` (AUC >= 0.80), ``weak`` (>= 0.60), ``no_signal``
  (below that: close computed abstention as declined for this workspace) or
  ``insufficient`` (fewer than 5 labelled queries in a class).

Honesty about what it measures
------------------------------
The statistic is exactly as informative as the ranker's scores. On a fixture in
which the right block ranks first the AUC is high; on the unindexed full-scan
fallback over the same fixture it was at chance, because that ranker did not
put the right block first -- the statistic was faithfully reporting that no
clear winner had been found. A ``no_signal`` verdict therefore says "this
serving path does not produce separable scores for these queries", which is
worth knowing before any caller trusts an ``act``.

Determinism and side effects
----------------------------
No randomness and no clock beyond the recall's own scoring instant (pass
``scoring_instant`` to pin it). A measurement is not a serve: every query runs
inside :func:`mind_mem.recall.serving_scope`, so the evaluation appends no
candidate set nobody was handed to the served-set ledger.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .recall_abstention import BAND_ACT, BAND_DO_NOT_ACT, AbstentionPolicy, assess, resolve_policy, validate_intended_use

__all__ = ["auc", "evaluate", "load_labelled"]

#: Labelled queries required per class before a verdict is offered.
MIN_PER_CLASS = 5
#: AUC at or above which the statistic is said to separate.
SEPARATES_AUC = 0.80
#: AUC below which there is no signal (between the two: ``weak``).
NO_SIGNAL_AUC = 0.60

_MAX_LABELLED = 100_000
_MAX_QUERY_LEN = 8192

Serve = Callable[[str], Sequence[Mapping[str, Any]]]


def auc(positive: Iterable[float], negative: Iterable[float]) -> float | None:
    """P(a positive outscores a negative), ties counting half; ``None`` if a class is empty.

    The Mann-Whitney U statistic over midranks: O(n log n), exact, symmetric
    (``auc(p, n) + auc(n, p) == 1``).
    """
    pos = [float(v) for v in positive]
    neg = [float(v) for v in negative]
    if not pos or not neg:
        return None
    combined = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum = 0.0
    i = 0
    n = len(combined)
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        midrank = (i + j) / 2 + 1
        rank_sum += midrank * sum(1 for k in range(i, j + 1) if combined[k][1])
        i = j + 1
    u = rank_sum - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def load_labelled(path: str) -> list[dict[str, Any]]:
    """Read ``{"query": str, "present": bool}`` JSONL. Raises :class:`ValueError`."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read labelled queries: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if len(rows) >= _MAX_LABELLED:
            raise ValueError(f"more than {_MAX_LABELLED} labelled queries")
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"line {number}: not valid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {number}: expected an object with 'query' and 'present'")
        query = row.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > _MAX_QUERY_LEN:
            raise ValueError(f"line {number}: 'query' must be a non-empty string of at most {_MAX_QUERY_LEN} characters")
        if not isinstance(row.get("present"), bool):
            raise ValueError(f"line {number}: 'present' must be true or false")
        rows.append({"query": query, "present": row["present"]})
    return rows


def _nearest_rank(sorted_values: Sequence[float], q: float) -> float:
    index = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": round(sum(ordered) / len(ordered), 4),
        "median": round(_nearest_rank(ordered, 0.5), 4),
        "p10": round(_nearest_rank(ordered, 0.1), 4),
        "p90": round(_nearest_rank(ordered, 0.9), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def _library_serve(workspace: str, limit: int, scoring_instant: str | None) -> Serve:
    """Serve through the door ``mm recall`` uses, without attesting the serve."""
    from .recall import recall, serving_scope

    def serve(query: str) -> Sequence[Mapping[str, Any]]:
        with serving_scope():
            return list(recall(workspace, query, limit=limit, scoring_instant=scoring_instant))

    return serve


def _policy_for(workspace: str) -> AbstentionPolicy:
    try:
        from ._recall_core import _get_config

        config = _get_config(workspace)
        section = config.get("recall") if isinstance(config, dict) else None
    except Exception:  # noqa: BLE001 -- an unreadable config reads as the defaults
        section = None
    return resolve_policy(section if isinstance(section, dict) else None)


def _verdict(present: int, absent: int, value: float | None) -> tuple[str, str]:
    if present < MIN_PER_CLASS or absent < MIN_PER_CLASS or value is None:
        return "insufficient", f"need at least {MIN_PER_CLASS} labelled queries in each class (present={present}, absent={absent})"
    if value >= SEPARATES_AUC:
        return "separates", f"AUC {value:.4f}: the statistic separates present from absent queries on this workspace"
    if value >= NO_SIGNAL_AUC:
        return "weak", f"AUC {value:.4f}: weak separation -- tune the bands on this labelled set before trusting 'act'"
    return (
        "no_signal",
        f"AUC {value:.4f}: the statistic does not separate present from absent answers on this workspace; "
        "close computed abstention as declined here -- no threshold tuning will create the signal",
    )


def evaluate(
    workspace: str,
    labelled: Iterable[Mapping[str, Any]],
    *,
    limit: int = 10,
    intended_use: str | None = None,
    policy: AbstentionPolicy | None = None,
    scoring_instant: str | None = None,
    serve: Serve | None = None,
) -> dict[str, Any]:
    """Run the falsification experiment over *labelled* queries. See the module docstring.

    *serve* replaces the default serving door (a callable ``query -> hits``);
    it exists so a caller can measure a different path, and so the experiment
    itself can be shown to come out negative.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
        raise ValueError("limit must be an integer in [1, 200]")
    use = validate_intended_use(intended_use)
    active = policy if policy is not None else _policy_for(workspace)
    door = serve if serve is not None else _library_serve(workspace, limit, scoring_instant)

    separations: dict[bool, list[float]] = {True: [], False: []}
    empty = {True: 0, False: 0}
    acted = {True: 0, False: 0}
    stopped = {True: 0, False: 0}
    resolved_use = use or active.default_use
    for row in labelled:
        is_present = bool(row["present"])
        verdict = assess(door(str(row["query"])), intended_use=use, policy=active)
        separations[is_present].append(float(verdict["separation"]))
        empty[is_present] += 1 if verdict["candidates"] == 0 else 0
        acted[is_present] += 1 if verdict["band"] == BAND_ACT else 0
        stopped[is_present] += 1 if verdict["band"] == BAND_DO_NOT_ACT else 0

    present_n, absent_n = len(separations[True]), len(separations[False])
    value = auc(separations[True], separations[False])
    verdict_name, detail = _verdict(present_n, absent_n, value)
    return {
        "queries": present_n + absent_n,
        "present": present_n,
        "absent": absent_n,
        "intended_use": resolved_use,
        "auc": None if value is None else round(value, 4),
        "separation": {"present": _summary(separations[True]), "absent": _summary(separations[False])},
        "empty": {"present": empty[True], "absent": empty[False]},
        "at_thresholds": {
            "act_rate_present": _rate(acted[True], present_n),
            "act_rate_absent": _rate(acted[False], absent_n),
            "abstain_rate_present": _rate(stopped[True], present_n),
            "abstain_rate_absent": _rate(stopped[False], absent_n),
        },
        "verdict": verdict_name,
        "detail": detail,
        "workspace": os.path.basename(os.path.abspath(workspace)),
    }
