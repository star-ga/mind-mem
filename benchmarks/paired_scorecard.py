# Copyright 2026 STARGA, Inc.
"""The paired scorecard: a ranking delta with its uncertainty attached.

Every retrieval fix this project ships can move ranking, and it has already
shipped a silent recall regression once. So no ranking change lands on an
unpaired comparison of two headline percentages. Two runs over the *same* 470
questions are a paired design, and the paired tests are the ones with the
power to see a 22-question disagreement inside a 470-question tie.

What is computed, and why each

* **recall_any@k and recall_all@k** are binary per question, so the test is
  McNemar's, exact. Concordant questions carry no information; only the
  questions where the arms disagreed do. The two discordant counts are
  reported **separately and always** -- "22 discordant" hides whether that is
  4-vs-18 (significant) or 11-vs-11 (nothing).
* **MRR** is continuous, so McNemar does not apply to it directly. The paired
  **sign test** does: it counts which arm won each question and asks the same
  exact binomial about that split, which is why it shares
  :func:`~mind_mem.bench.ab_stats.mcnemar_exact` rather than reimplementing
  the arithmetic. It is deliberately blind to margin -- a rank-2-to-rank-1 win
  counts the same as rank-10-to-rank-1 -- which is what the bootstrap adds.
* **A bootstrap 95% CI on the mean paired difference**, resampling *questions*
  (never rows independently -- that would break the pairing that makes the
  whole design work). The sign test says whether a direction is real; the
  interval says how big it could be. The seed is a committed constant and
  travels in the output, because a bootstrap whose seed is not recorded is
  not reproducible and its interval is decoration.

Nothing here reads a clock, opens a network socket, or draws an unseeded
random number.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mind_mem.bench.ab_stats import DEFAULT_ALPHA, mcnemar_exact, smallest_significant_discordant  # noqa: E402

#: The before-artifact. The 2026-09-03 full LongMemEval-S run, rep 1, 470
#: questions, committed. A future run is compared against THIS unless the
#: caller names another baseline, so "did it move?" has one fixed answer to
#: move away from rather than whatever was lying around.
PINNED_BASELINE = Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "2026-09-03-longmemeval-s-full-mind_mem-rep1.ndjson"

#: Committed bootstrap seed. Changing it changes every published interval, so
#: it is a constant in the source and is echoed into every scorecard.
BOOTSTRAP_SEED = 20260903

#: Resamples per bootstrap. 10,000 puts the Monte-Carlo error on a percentile
#: bound well below the third decimal at these sample sizes.
BOOTSTRAP_RESAMPLES = 10_000

#: Two-sided coverage of the reported interval.
CONFIDENCE = 0.95

#: Guard against a mis-pointed path pulling an unbounded file into memory.
MAX_ROWS = 200_000

#: The binary metrics the scorecard tests, by NDJSON field name.
BINARY_METRICS = ("recall_any_at_k", "recall_all_at_k")

#: The continuous metric the scorecard sign-tests.
CONTINUOUS_METRIC = "reciprocal_rank"


class PairingError(Exception):
    """The two runs cannot be paired, so no paired test is valid.

    Different question sets, a duplicated question id, or a unit that did not
    complete. Every one of these has a tempting silent repair -- intersect the
    ids, keep the last duplicate, treat an errored unit as a miss -- and every
    one of them would quietly answer a different question than the one asked.
    """


def load_run(path: str | Path) -> list[dict[str, Any]]:
    """Read one per-question NDJSON artifact.

    Raises:
        PairingError: The file is empty, over :data:`MAX_ROWS`, or has a line
            that is not a JSON object.
    """
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if len(rows) >= MAX_ROWS:
                raise PairingError(f"{path}: more than {MAX_ROWS} rows")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PairingError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise PairingError(f"{path}:{lineno}: expected a JSON object, got {type(row).__name__}")
            rows.append(row)
    if not rows:
        raise PairingError(f"{path}: no rows; an empty run cannot be compared")
    return rows


def index_by_question(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Mapping[str, Any]]:
    """Key rows by ``question_id``, refusing duplicates.

    A duplicate id is not a harmless repeat: whichever copy wins decides the
    metric, so silently keeping one would make the comparison depend on file
    order.
    """
    out: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise PairingError(f"{label}: row {position} has no usable question_id ({qid!r})")
        if qid in out:
            raise PairingError(f"{label}: question_id {qid!r} appears more than once")
        out[qid] = row
    return out


@dataclass(frozen=True)
class PairedRuns:
    """Two runs, aligned question by question."""

    baseline_label: str
    candidate_label: str
    question_ids: tuple[str, ...]
    baseline: Mapping[str, Mapping[str, Any]]
    candidate: Mapping[str, Mapping[str, Any]]
    dropped_non_ok: tuple[str, ...] = ()

    @property
    def n_pairs(self) -> int:
        return len(self.question_ids)


def pair_runs(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    drop_non_ok: bool = False,
) -> PairedRuns:
    """Align two runs on ``question_id``, or refuse.

    The id sets must match exactly. Intersecting them would silently answer a
    question about whichever subset happened to survive both runs, which is a
    different -- and flattering -- question.

    Args:
        drop_non_ok: Exclude questions whose ``unit_status`` is not ``"ok"`` in
            either arm, and record which. Off by default: an errored unit's
            metrics are not a measurement, and dropping it must be a decision
            somebody made, not one the tool made for them.
    """
    base = index_by_question(baseline_rows, label=baseline_label)
    cand = index_by_question(candidate_rows, label=candidate_label)
    if set(base) != set(cand):
        only_base = sorted(set(base) - set(cand))
        only_cand = sorted(set(cand) - set(base))
        raise PairingError(
            f"the runs cover different questions: {len(only_base)} only in {baseline_label} "
            f"(e.g. {only_base[:3]}), {len(only_cand)} only in {candidate_label} (e.g. {only_cand[:3]})"
        )
    ordered = tuple(base)
    non_ok = tuple(qid for qid in ordered if base[qid].get("unit_status") != "ok" or cand[qid].get("unit_status") != "ok")
    if non_ok and not drop_non_ok:
        raise PairingError(
            f"{len(non_ok)} question(s) did not complete in one or both arms (e.g. {list(non_ok[:3])}); "
            "an errored unit's metrics are not a measurement -- pass drop_non_ok to exclude them on the record"
        )
    excluded = set(non_ok)
    kept = tuple(qid for qid in ordered if qid not in excluded)
    if not kept:
        raise PairingError("no completed question is present in both arms")
    return PairedRuns(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        question_ids=kept,
        baseline=base,
        candidate=cand,
        dropped_non_ok=non_ok,
    )


def binary_value(row: Mapping[str, Any], metric: str, k: int) -> int:
    """Read one 0/1 per-question metric at ``k``, refusing anything else."""
    table = row.get(metric)
    if not isinstance(table, Mapping):
        raise PairingError(f"question {row.get('question_id')!r}: {metric} is not a per-k table")
    if str(k) not in table:
        raise PairingError(f"question {row.get('question_id')!r}: {metric} has no k={k} (has {sorted(table)})")
    value = table[str(k)]
    if value not in (0, 1, 0.0, 1.0, True, False):
        raise PairingError(f"question {row.get('question_id')!r}: {metric}@{k} is {value!r}, not a 0/1 outcome")
    return int(value)


def continuous_value(row: Mapping[str, Any], metric: str) -> float:
    """Read one continuous per-question metric, refusing a non-number."""
    if metric not in row:
        raise PairingError(f"question {row.get('question_id')!r}: no {metric}")
    value = row[metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairingError(f"question {row.get('question_id')!r}: {metric} is {value!r}, not a number")
    return float(value)


def _draw_index(rng: random.Random, n: int) -> int:
    """Uniform index in ``[0, n)`` from a documented primitive.

    Built on ``getrandbits`` with rejection rather than ``choices`` or
    ``randrange`` so the draw sequence depends only on the Mersenne Twister
    and this function -- verified identical on CPython 3.12 and 3.14. A
    committed seed that reproduces on one interpreter and not the next is not
    a committed seed.
    """
    if n <= 0:
        raise ValueError("cannot draw from an empty sample")
    if n == 1:
        return 0
    bits = (n - 1).bit_length()
    while True:
        value = rng.getrandbits(bits)
        if value < n:
            return value


@dataclass(frozen=True)
class BootstrapCI:
    """A percentile interval on the mean paired difference, and its seed."""

    mean_diff: float
    low: float
    high: float
    seed: int
    resamples: int
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_diff": round(self.mean_diff, 6),
            "ci_low": round(self.low, 6),
            "ci_high": round(self.high, 6),
            "ci_excludes_zero": self.low > 0.0 or self.high < 0.0,
            "bootstrap_seed": self.seed,
            "bootstrap_resamples": self.resamples,
            "confidence": self.confidence,
        }


def bootstrap_mean_diff_ci(
    diffs: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
) -> BootstrapCI:
    """Percentile bootstrap CI on the mean of ``diffs``, resampling questions.

    ``diffs`` is one paired difference per question -- candidate minus
    baseline -- so resampling it with replacement resamples *questions* and
    keeps each arm's two observations together. Resampling the arms
    independently would destroy the pairing and widen the interval into
    meaninglessness.
    """
    n = len(diffs)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[_draw_index(rng, n)]
        means.append(total / n)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = min(resamples - 1, max(0, int(tail * resamples)))
    high_index = min(resamples - 1, max(0, int((1.0 - tail) * resamples) - 1))
    return BootstrapCI(
        mean_diff=sum(diffs) / n,
        low=means[low_index],
        high=means[high_index],
        seed=seed,
        resamples=resamples,
        confidence=confidence,
    )


def _verdict(candidate_only: int, baseline_only: int, p_value: Fraction, alpha: Fraction, floor: int) -> tuple[str, str]:
    """Name the outcome, including when there is nothing to name."""
    total = candidate_only + baseline_only
    if total == 0:
        return "no_evidence", "The arms agreed on every question; a paired test has nothing to work with."
    if total < floor:
        return (
            "underpowered",
            f"{total} discordant question(s): below {floor}, no split can reach p<={float(alpha):g}, "
            "so this difference cannot be evidence at any effect size.",
        )
    split = f"{candidate_only} candidate-only / {baseline_only} baseline-only"
    if p_value > alpha:
        return "not_significant", f"{total} discordant ({split}); p={float(p_value):.4f} does not clear {float(alpha):g}."
    direction = "candidate_better" if candidate_only > baseline_only else "baseline_better"
    return direction, f"{total} discordant ({split}); p={float(p_value):.4f}."


@dataclass(frozen=True)
class MetricComparison:
    """One metric, tested paired, with both discordant directions named."""

    metric: str
    k: int | None
    test: str
    n_pairs: int
    baseline_mean: float
    candidate_mean: float
    candidate_only: int
    baseline_only: int
    concordant: int
    p_value: Fraction
    alpha: Fraction
    min_discordant_for_significance: int
    verdict: str
    note: str
    ci: BootstrapCI

    @property
    def n_discordant(self) -> int:
        return self.candidate_only + self.baseline_only

    @property
    def label(self) -> str:
        return self.metric if self.k is None else f"{self.metric}@{self.k}"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "metric": self.metric,
            "k": self.k,
            "label": self.label,
            "test": self.test,
            "n_pairs": self.n_pairs,
            "baseline_mean": round(self.baseline_mean, 6),
            "candidate_mean": round(self.candidate_mean, 6),
            # Both directions, always. A single "net" number would let
            # 4-vs-18 and 11-vs-11 print the same headline.
            "candidate_only": self.candidate_only,
            "baseline_only": self.baseline_only,
            "n_discordant": self.n_discordant,
            "concordant": self.concordant,
            "p_value": round(float(self.p_value), 6),
            "alpha": round(float(self.alpha), 6),
            "min_discordant_for_significance": self.min_discordant_for_significance,
            "verdict": self.verdict,
            "note": self.note,
        }
        out.update(self.ci.as_dict())
        return out


def _compare(
    *,
    metric: str,
    k: int | None,
    test: str,
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    alpha: Fraction,
    seed: int,
    resamples: int,
    confidence: float,
) -> MetricComparison:
    """The shared body of both tests: split, exact binomial, bootstrap.

    McNemar on 0/1 outcomes and the paired sign test on a continuous metric
    are the *same* exact binomial over the questions where the arms disagreed;
    only what counts as a disagreement differs. Writing it once is why
    ``mcnemar_exact`` is imported rather than reimplemented.
    """
    n = len(baseline_values)
    candidate_only = sum(1 for b, c in zip(baseline_values, candidate_values) if c > b)
    baseline_only = sum(1 for b, c in zip(baseline_values, candidate_values) if b > c)
    p_value = mcnemar_exact(candidate_only, baseline_only)
    floor = smallest_significant_discordant(alpha)
    verdict, note = _verdict(candidate_only, baseline_only, p_value, alpha, floor)
    diffs = [c - b for b, c in zip(baseline_values, candidate_values)]
    return MetricComparison(
        metric=metric,
        k=k,
        test=test,
        n_pairs=n,
        baseline_mean=sum(baseline_values) / n,
        candidate_mean=sum(candidate_values) / n,
        candidate_only=candidate_only,
        baseline_only=baseline_only,
        concordant=n - candidate_only - baseline_only,
        p_value=p_value,
        alpha=alpha,
        min_discordant_for_significance=floor,
        verdict=verdict,
        note=note,
        ci=bootstrap_mean_diff_ci(diffs, seed=seed, resamples=resamples, confidence=confidence),
    )


def compare_binary(
    runs: PairedRuns,
    metric: str,
    k: int,
    *,
    alpha: Fraction = DEFAULT_ALPHA,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
) -> MetricComparison:
    """Exact McNemar over a 0/1 per-question metric at ``k``."""
    baseline_values = [float(binary_value(runs.baseline[q], metric, k)) for q in runs.question_ids]
    candidate_values = [float(binary_value(runs.candidate[q], metric, k)) for q in runs.question_ids]
    return _compare(
        metric=metric,
        k=k,
        test="mcnemar_exact_two_sided",
        baseline_values=baseline_values,
        candidate_values=candidate_values,
        alpha=alpha,
        seed=seed,
        resamples=resamples,
        confidence=confidence,
    )


def compare_continuous(
    runs: PairedRuns,
    metric: str = CONTINUOUS_METRIC,
    *,
    alpha: Fraction = DEFAULT_ALPHA,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
) -> MetricComparison:
    """Paired sign test over a continuous per-question metric."""
    baseline_values = [continuous_value(runs.baseline[q], metric) for q in runs.question_ids]
    candidate_values = [continuous_value(runs.candidate[q], metric) for q in runs.question_ids]
    return _compare(
        metric=metric,
        k=None,
        test="paired_sign_test_exact_two_sided",
        baseline_values=baseline_values,
        candidate_values=candidate_values,
        alpha=alpha,
        seed=seed,
        resamples=resamples,
        confidence=confidence,
    )


@dataclass(frozen=True)
class Scorecard:
    """Every paired test over one comparison, plus how to reproduce it."""

    baseline_label: str
    candidate_label: str
    baseline_path: str
    candidate_path: str
    n_pairs: int
    k: int
    dropped_non_ok: tuple[str, ...]
    comparisons: tuple[MetricComparison, ...] = field(default_factory=tuple)

    @property
    def identical(self) -> bool:
        """True iff no tested metric moved on any question.

        This is the "off path is byte-identical" assertion at run-artifact
        level: zero discordant questions in EITHER direction on every metric.
        A net of zero is not enough -- 11-vs-11 nets to zero and is 22 moved
        questions.

        Its scope is the tested metrics at the tested ``k``, which is what an
        artifact records -- so a reshuffle purely among non-gold documents
        below the cut-off can be invisible here. That is not a hole to be
        argued away but the reason the byte-identity gate also exists one
        layer down: ``benchmarks/ranking_identity.py`` compares the served
        ``(id, score)`` list itself, where no reordering can hide.
        """
        return all(c.n_discordant == 0 for c in self.comparisons)

    def moved(self) -> tuple[MetricComparison, ...]:
        return tuple(c for c in self.comparisons if c.n_discordant > 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "baseline_path": self.baseline_path,
            "candidate_path": self.candidate_path,
            "n_pairs": self.n_pairs,
            "k": self.k,
            "dropped_non_ok": list(self.dropped_non_ok),
            "identical": self.identical,
            "comparisons": [c.as_dict() for c in self.comparisons],
        }


def build_scorecard(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    k: int = 5,
    binary_metrics: Iterable[str] = BINARY_METRICS,
    continuous_metric: str = CONTINUOUS_METRIC,
    alpha: Fraction = DEFAULT_ALPHA,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
    drop_non_ok: bool = False,
) -> Scorecard:
    """Load two NDJSON runs, pair them, and run every test."""
    baseline_path = Path(baseline_path)
    candidate_path = Path(candidate_path)
    runs = pair_runs(
        load_run(baseline_path),
        load_run(candidate_path),
        baseline_label=baseline_path.stem,
        candidate_label=candidate_path.stem,
        drop_non_ok=drop_non_ok,
    )
    comparisons = [
        compare_binary(runs, metric, k, alpha=alpha, seed=seed, resamples=resamples, confidence=confidence) for metric in binary_metrics
    ]
    comparisons.append(compare_continuous(runs, continuous_metric, alpha=alpha, seed=seed, resamples=resamples, confidence=confidence))
    return Scorecard(
        baseline_label=runs.baseline_label,
        candidate_label=runs.candidate_label,
        baseline_path=str(baseline_path),
        candidate_path=str(candidate_path),
        n_pairs=runs.n_pairs,
        k=k,
        dropped_non_ok=runs.dropped_non_ok,
        comparisons=tuple(comparisons),
    )


__all__ = [
    "BINARY_METRICS",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONFIDENCE",
    "CONTINUOUS_METRIC",
    "MAX_ROWS",
    "PINNED_BASELINE",
    "BootstrapCI",
    "MetricComparison",
    "PairedRuns",
    "PairingError",
    "Scorecard",
    "binary_value",
    "bootstrap_mean_diff_ci",
    "build_scorecard",
    "compare_binary",
    "compare_continuous",
    "continuous_value",
    "index_by_question",
    "load_run",
    "pair_runs",
]
