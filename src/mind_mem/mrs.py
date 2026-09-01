# Copyright 2026 STARGA, Inc.
"""Model Reliability Score (MRS) framework (v2.6.0).

Aggregates per-model SLIs (latency percentiles, error rate, quality
drift, token throughput, cost per query) into a composite
reliability score in [0, 100]. Operators configure weights + alert
thresholds in a YAML-like dict; a run produces a report, optionally
flagging any SLI that crossed its threshold.

Pure stdlib. Alert delivery (email, Slack, PagerDuty) is intentionally
out of scope — callers wire whatever transport they already have.

Wiring (5.0.1)
--------------
The scoring core above is a pure function of the readings it is handed.
Everything that turns a *workspace* into readings lives in the
"workspace collection" section at the bottom of this module, and is
reached from exactly one place today: the ``mrs`` block of the
``memory_health`` MCP tool, behind ``mrs.enabled`` in ``mind-mem.json``
(**default OFF** — with the flag off nothing here is imported, no log
line is emitted and no metric moves, so the dashboard is byte-identical
to 5.0.0). Breaches are routed to :mod:`mind_mem.alerting` from that
same call site.

Two properties this module keeps, and that ``tests/test_mrs_wiring.py``
pins:

* **No clock, no randomness.** ``computed_at`` is *injected*, never
  read — a report is a pure function of its readings. That is what lets
  the score be compared across runs at all.
* **Corpus counts go through admission.** :func:`collect_corpus_counts`
  reads blocks, so it calls ``admit_corpus`` like every other
  block-reading leg. The population an SLI about *retrieval* should
  describe is the servable corpus, so this is the correct denominator as
  well as the required gate — a quarantined block is not a retrieval
  problem, it is quarantine working.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Sequence

from .feature_gate import FieldSpec, strict_number


@dataclass(frozen=True)
class SLI:
    """A single Service Level Indicator reading."""

    name: str
    value: float
    unit: str = ""
    threshold: Optional[float] = None  # violation when value > threshold
    weight: float = 1.0


@dataclass(frozen=True)
class MRSReport:
    """Rolled-up MRS report for a single target (model, endpoint, backend)."""

    target: str
    score: float  # 0..100
    slis: list[SLI]
    violations: list[str]
    computed_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "score": round(self.score, 2),
            "slis": [
                {
                    "name": s.name,
                    "value": s.value,
                    "unit": s.unit,
                    "threshold": s.threshold,
                    "weight": s.weight,
                }
                for s in self.slis
            ],
            "violations": list(self.violations),
            "computed_at": self.computed_at,
        }


def percentile(values: Iterable[float], p: float) -> float:
    """Approximate percentile without numpy. p in [0, 100]."""
    arr = sorted(values)
    if not arr:
        return 0.0
    if len(arr) == 1:
        return arr[0]
    k = (len(arr) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    if f == c:
        return arr[f]
    return arr[f] + (arr[c] - arr[f]) * (k - f)


def latency_slis(latencies_ms: Iterable[float]) -> list[SLI]:
    arr = list(latencies_ms)
    if not arr:
        return []
    return [
        SLI(name="p50_ms", value=percentile(arr, 50), unit="ms", threshold=100.0),
        SLI(name="p95_ms", value=percentile(arr, 95), unit="ms", threshold=500.0),
        SLI(name="p99_ms", value=percentile(arr, 99), unit="ms", threshold=1500.0),
    ]


def error_rate_sli(error_count: int, total: int, threshold: float = 0.01) -> SLI:
    rate = (error_count / total) if total > 0 else 0.0
    return SLI(name="error_rate", value=rate, unit="fraction", threshold=threshold)


def cost_sli(cost_per_query: float, threshold: float = 0.10) -> SLI:
    return SLI(name="cost_per_query", value=cost_per_query, unit="USD", threshold=threshold)


def throughput_sli(tokens_per_second: float, min_acceptable: float = 10.0) -> SLI:
    # Represent as deficit against a floor so the same "greater = violation"
    # rule applies consistently across SLIs.
    return SLI(
        name="throughput_deficit",
        value=max(0.0, min_acceptable - tokens_per_second),
        unit="tokens/s below floor",
        threshold=0.0,
    )


def retrieval_slis(
    *,
    relevance_decay: float,
    contradiction_density: float,
    staleness_ratio: float,
) -> list[SLI]:
    """Memory-retrieval-specific SLIs from the roadmap."""
    return [
        SLI(
            name="relevance_decay",
            value=relevance_decay,
            unit="fraction/day",
            threshold=0.05,
        ),
        SLI(
            name="contradiction_density",
            value=contradiction_density,
            unit="per 100 blocks",
            threshold=0.5,
        ),
        SLI(
            name="staleness_ratio",
            value=staleness_ratio,
            unit="fraction",
            threshold=0.2,
        ),
    ]


def compute_mrs(target: str, slis: Iterable[SLI], *, computed_at: str = "") -> MRSReport:
    """Aggregate SLIs into a 0..100 composite MRS score.

    Each SLI contributes ``weight * (1 - penalty)`` where penalty
    scales linearly from 0 (well under threshold) to 1 (double the
    threshold or worse). Targets without a threshold contribute full
    weight (no penalty possible).
    """
    slis_list = list(slis)
    total_weight = sum(max(0.0, s.weight) for s in slis_list) or 1.0
    score_accum = 0.0
    violations: list[str] = []
    for s in slis_list:
        w = max(0.0, s.weight)
        if w == 0:
            continue
        if s.threshold is None:
            score_accum += w
            continue
        if s.threshold <= 0:
            # Deficit-style SLIs where any positive value is a hit.
            penalty = min(1.0, s.value / max(1e-9, s.threshold + 1.0))
        else:
            penalty = min(1.0, max(0.0, (s.value - s.threshold) / s.threshold))
        if s.value > s.threshold:
            violations.append(s.name)
        score_accum += w * (1.0 - penalty)
    score = 100.0 * (score_accum / total_weight)
    return MRSReport(
        target=target,
        score=max(0.0, min(100.0, score)),
        slis=slis_list,
        violations=violations,
        computed_at=computed_at,
    )


def parse_slo_spec(spec: Mapping[str, Any]) -> list[SLI]:
    """Turn a roadmap-style YAML-ish SLO spec into :class:`SLI` inputs.

    Expected shape::

        {"slis": [
            {"name": "p99_ms", "threshold": 1500, "weight": 1.0},
            ...
        ]}

    Current values are left at 0; callers fill them in before
    :func:`compute_mrs`. This lets an SLO file define WHAT to measure
    independently of runtime readings.
    """
    out: list[SLI] = []
    for entry in spec.get("slis", []):
        if not isinstance(entry, Mapping):
            continue
        out.append(
            SLI(
                name=str(entry.get("name", "unnamed")),
                value=float(entry.get("value", 0.0)),
                unit=str(entry.get("unit", "")),
                threshold=(float(entry["threshold"]) if "threshold" in entry else None),
                weight=float(entry.get("weight", 1.0)),
            )
        )
    return out


def merge_slo_spec(measured: Iterable[SLI], spec: Iterable[SLI]) -> list[SLI]:
    """Overlay operator-declared thresholds/weights onto *measured* readings.

    :func:`parse_slo_spec` answers "what should we measure and where is
    the line"; the collectors answer "what did we read". This joins them
    on ``name``, keeping the measured *value* and taking the spec's
    ``threshold`` / ``weight``.

    A spec entry that omits ``threshold`` leaves the measured SLI's own
    threshold in place. ``parse_slo_spec`` renders an omitted threshold
    as ``None``, and ``None`` means "no threshold at all" to
    :func:`compute_mrs` — so copying it over would let
    ``{"name": "p99_ms", "weight": 2}`` *silently disable* the p99
    violation the operator was trying to weight up.

    A spec entry naming an SLI nobody measured is ignored: a threshold
    cannot conjure a reading, and inventing a 0.0 one would report a
    green SLI for something never observed.
    """
    by_name = {s.name: s for s in spec}
    out: list[SLI] = []
    for reading in measured:
        override = by_name.get(reading.name)
        if override is None:
            out.append(reading)
            continue
        out.append(
            replace(
                reading,
                threshold=(override.threshold if override.threshold is not None else reading.threshold),
                weight=override.weight,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Workspace collection — the only part of this module that does IO.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusCounts:
    """The drift / contradiction / staleness readings for one workspace.

    ``servable_blocks`` is the denominator every ratio below uses, and it
    counts the corpus *after* admission — see the module docstring.
    """

    servable_blocks: int = 0
    stale_blocks: int = 0
    unresolved_contradictions: int = 0
    drift_items: int = 0


def corpus_retrieval_slis(counts: CorpusCounts, *, observation_days: Optional[float] = 1.0) -> list[SLI]:
    """Turn :class:`CorpusCounts` into the three retrieval SLIs.

    * ``staleness_ratio`` — servable blocks carrying a staleness flag.
    * ``contradiction_density`` — unresolved contradictions per 100
      servable blocks.
    * ``relevance_decay`` — drift items per servable block per day of the
      observation window. ``observation_days`` is *injected* because
      deriving it would mean reading a clock; the default of 1.0 reports
      the accumulated density as-is rather than inventing a window. A
      zero, negative or absent window falls back to 1.0 rather than
      raising or reporting an infinite rate.

    An empty corpus scores clean (all zeros) rather than dividing by
    zero: nothing servable is not a retrieval-reliability failure.
    """
    denominator = max(0, int(counts.servable_blocks))
    if denominator == 0:
        return retrieval_slis(relevance_decay=0.0, contradiction_density=0.0, staleness_ratio=0.0)
    days = observation_days if observation_days and observation_days > 0 else 1.0
    return retrieval_slis(
        relevance_decay=max(0, counts.drift_items) / denominator / days,
        contradiction_density=max(0, counts.unresolved_contradictions) / denominator * 100.0,
        staleness_ratio=max(0, counts.stale_blocks) / denominator,
    )


#: Corpus file whose blocks are counted as unresolved contradictions.
#: Mirrors ``memory_health``'s ``unresolved_contradictions`` field, which
#: counts every block in this file — so the two numbers stay comparable
#: instead of quietly answering different questions.
CONTRADICTIONS_FILE = "intelligence/CONTRADICTIONS.md"

#: Corpus file whose blocks are counted as drift items, mirroring
#: ``memory_health``'s ``drift_items``.
DRIFT_FILE = "intelligence/DRIFT.md"

#: Key this module stamps on a parsed block so an admitted block can be
#: traced back to the file it came from. Private to the collector; it is
#: never written to disk and never leaves :func:`collect_corpus_counts`.
_SOURCE_KEY = "_mrs_source"


def collect_corpus_counts(workspace: str) -> CorpusCounts:
    """Read *workspace*'s drift, contradiction and staleness detectors.

    **Admission.** Every block read here goes through ``admit_corpus``
    before it is counted, and the staleness flags are then intersected
    with the ids that survived — a flag on a quarantined block is not a
    retrieval SLI. The whole corpus is admitted in one call because the
    release set is derived from the decision blocks *in the list*, so
    admitting file-by-file would withhold legitimately released blocks.

    Never raises: a missing corpus directory, an unparseable file or an
    absent causal-graph DB degrades that one reading to zero rather than
    failing the dashboard that called it.

    The heavy imports are function-local on purpose — the scoring core
    above stays importable without dragging in the block parser, the
    corpus registry or sqlite.
    """
    import os

    from .admissibility import admit_corpus
    from .block_parser import parse_file
    from .corpus_registry import CORPUS_DIRS

    root = os.path.abspath(workspace)
    parsed: list[dict] = []
    for subdir in CORPUS_DIRS:
        dir_path = os.path.join(root, subdir)
        if not os.path.isdir(dir_path):
            continue
        for filename in sorted(os.listdir(dir_path)):
            # ``*_ARCHIVE.md`` is excluded for the same reason
            # ``memory_health`` excludes it: an archived block is out of
            # the serving population, so counting it would inflate the
            # denominator every ratio here divides by.
            if not filename.endswith(".md") or filename.endswith("_ARCHIVE.md"):
                continue
            try:
                blocks = parse_file(os.path.join(dir_path, filename))
            except (OSError, ValueError):
                continue
            source = f"{subdir}/{filename}"
            for block in blocks:
                block[_SOURCE_KEY] = source
            parsed.extend(blocks)

    admitted = admit_corpus(parsed)
    admitted_ids = {str(block.get("_id") or "") for block in admitted}
    admitted_ids.discard("")

    return CorpusCounts(
        servable_blocks=len(admitted),
        stale_blocks=_stale_flag_count(root, admitted_ids),
        unresolved_contradictions=sum(1 for b in admitted if b.get(_SOURCE_KEY) == CONTRADICTIONS_FILE),
        drift_items=sum(1 for b in admitted if b.get(_SOURCE_KEY) == DRIFT_FILE),
    )


def _stale_flag_count(workspace: str, admitted_ids: set[str]) -> int:
    """Staleness flags whose block survived admission. Never raises."""
    import sqlite3

    try:
        from .causal_graph import CausalGraph

        flags = CausalGraph(workspace).get_stale_blocks()
    except (ImportError, sqlite3.Error, OSError, ValueError):
        return 0
    return sum(1 for flag in flags if str(flag.get("block_id") or "") in admitted_ids)


def workspace_mrs_report(
    workspace: str,
    *,
    latency_ms: Iterable[float] = (),
    error_count: int = 0,
    request_count: int = 0,
    counts: Optional[CorpusCounts] = None,
    observation_days: float = 1.0,
    slo_spec: Optional[Mapping[str, Any]] = None,
    target: str = "retrieval",
    computed_at: str = "",
) -> MRSReport:
    """The composite MRS for *workspace*: latency + errors + corpus health.

    *latency_ms* and the error counters are **injected** — this module
    does not decide where a latency series lives, and reading one would
    make the report depend on process state it cannot see. *counts* is
    injected too when a caller has already collected them; otherwise
    :func:`collect_corpus_counts` reads them.

    The error-rate SLI is omitted entirely when ``request_count`` is 0:
    zero errors out of zero requests is not a 0% error rate, it is no
    reading, and scoring it as clean would report reliability for a
    surface nobody called.
    """
    slis: list[SLI] = list(latency_slis(latency_ms))
    if request_count > 0:
        slis.append(error_rate_sli(error_count, request_count))
    if counts is None:
        counts = collect_corpus_counts(workspace)
    slis.extend(corpus_retrieval_slis(counts, observation_days=observation_days))
    if slo_spec:
        slis = merge_slo_spec(slis, parse_slo_spec(slo_spec))
    return compute_mrs(target, slis, computed_at=computed_at)


def route_mrs_alerts(
    report: MRSReport,
    *,
    router: Any,
    alert_below: float = 100.0,
    severity: str = "warning",
) -> list[bool]:
    """Fire *report* at *router* when it breached, else do nothing.

    The payload carries the target, the score, the violated SLI names and
    the numeric readings — **aggregates only, never block content**. An
    alert sink writes to a log line or a third-party webhook, which is
    exactly the surface where corpus text must not appear.

    Returns the router's per-sink results, or ``[]`` when there was
    nothing to report.
    """
    if not report.violations and report.score >= alert_below:
        return []
    return list(
        router.fire(
            severity=severity,
            event="mrs_degraded",
            payload={
                "target": report.target,
                "score": round(report.score, 2),
                "violations": list(report.violations),
                "readings": {s.name: s.value for s in report.slis},
            },
        )
    )


# ---------------------------------------------------------------------------
# Config gate — ``mrs`` in mind-mem.json. Default OFF.
# ---------------------------------------------------------------------------


def _strict_bool(value: Any) -> bool:
    """Coerce to ``bool`` only when the raw value already *is* one.

    The missing member of :mod:`mind_mem.feature_gate`'s ``strict_int`` /
    ``strict_number`` family, and it matters more than they do: plain
    ``bool`` would read ``"alert": "false"`` as **True**, which is the
    one direction a switch must never fail in.
    """
    if not isinstance(value, bool):
        raise TypeError(f"expected bool, got {type(value).__name__}")
    return value


#: Knob bounds for the ``mrs`` config section. The coercers are the ones
#: :mod:`mind_mem.feature_gate` already defines, so a malformed value
#: falls back to the default here exactly as it does for every migrated
#: retrieval gate — one validation contract, not a second one.
_MRS_FIELDS: dict[str, FieldSpec] = {
    "target": FieldSpec(default="retrieval", coerce=str),
    "latency_metric": FieldSpec(default="mcp_tool_duration_ms", coerce=str),
    "observation_days": FieldSpec(default=1.0, coerce=strict_number, validate=lambda v: v > 0),
    "alert": FieldSpec(default=True, coerce=_strict_bool),
    "alert_severity": FieldSpec(
        default="warning",
        coerce=str,
        validate=lambda v: v in ("debug", "info", "warning", "critical"),
    ),
    "alert_below": FieldSpec(default=100.0, coerce=strict_number, validate=lambda v: 0.0 <= v <= 100.0),
}


def _mrs_section(config: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not isinstance(config, Mapping):
        return None
    section = config.get("mrs")
    return section if isinstance(section, Mapping) else None


def is_mrs_enabled(config: Optional[Mapping[str, Any]]) -> bool:
    """True only when ``mrs.enabled`` is a literal ``true``. Default OFF.

    Two deliberate strictnesses, both in the fail-closed direction, and
    both because MRS reads the whole corpus and can fire alerts:

    * **No ``auto_enable``.** It turns on when an operator says so, never
      because a heuristic guessed.
    * **No truthiness.** The migrated retrieval gates read
      ``section.get("enabled", False)``, which turns ON for the *string*
      ``"false"``. That answer is preserved verbatim for them because
      changing a migrated gate's answer is a behaviour change; this gate
      is new, owes nothing to that history, and a switch must not fail
      open on a typo.
    """
    section = _mrs_section(config)
    if section is None:
        return False
    try:
        return _strict_bool(section.get("enabled", False))
    except TypeError:
        return False


def resolve_mrs_config(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Resolve the ``mrs`` knobs, with every malformed value defaulted."""
    section = _mrs_section(config) or {}
    resolved: dict[str, Any] = {name: spec.resolve(section.get(name)) for name, spec in _MRS_FIELDS.items()}
    slo = section.get("slo")
    resolved["slo"] = dict(slo) if isinstance(slo, Mapping) else {}
    latency = section.get("latency_ms")
    resolved["latency_ms"] = (
        [float(v) for v in latency if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if isinstance(latency, Sequence) and not isinstance(latency, (str, bytes))
        else None
    )
    return resolved


__all__ = [
    "SLI",
    "CONTRADICTIONS_FILE",
    "DRIFT_FILE",
    "CorpusCounts",
    "MRSReport",
    "collect_corpus_counts",
    "compute_mrs",
    "corpus_retrieval_slis",
    "cost_sli",
    "error_rate_sli",
    "is_mrs_enabled",
    "latency_slis",
    "merge_slo_spec",
    "parse_slo_spec",
    "percentile",
    "resolve_mrs_config",
    "retrieval_slis",
    "route_mrs_alerts",
    "throughput_sli",
    "workspace_mrs_report",
]
