"""Per-actor memory trust scores (roadmap Group D) — opt-in recall re-rank.

A corpus that accepts writes from many actors needs a way to say *how
reliable has this writer been so far*. This module does **not** invent a
new scoring subsystem: it aggregates signals that already exist in
mind-mem into one per-actor reliability value in ``[0.01, 0.99]``.

Signals aggregated (all pre-existing):

======================  =========================================================
source                  contribution
======================  =========================================================
``truth_score``         mean probabilistic truth of the actor's blocks
``calibration``         mean human-feedback weight of the actor's blocks
                        (``[0.5, 1.5]`` → normalised to ``[0, 1]``)
contradiction history   fraction of the actor's blocks NOT left in a
                        contradicted state (``superseded`` / ``rejected`` /
                        ``deprecated``), or an explicit contradicted-id set
rollback history        fraction of the actor's audit-chain writes that were
                        NOT rolled back (``audit_chain`` ``rollback`` ops)
======================  =========================================================

Actor identity comes from :mod:`block_provenance` (``ActorId``).

**Determinism.** :func:`compute_actor_trust` is a pure function of
:class:`ActorSignals`: fixed inputs → fixed output, no clock, no I/O, no
iteration-order dependence. The only time-dependent input is the
per-block ``truth_score`` itself, which the caller supplies (the recall
pipeline annotates it upstream) or which is computed with an explicit
half-life.

**Zero regression.** Everything here is opt-in and default-OFF:

.. code-block:: json

    {"retrieval": {"trust_scores": {"enabled": true, "rerank": true}}}

With ``enabled`` false the recall pipeline returns the exact same list
object it was given — no added fields, no reordering.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .block_provenance import extract_provenance
from .calibration import MAX_CALIBRATION_WEIGHT, MIN_CALIBRATION_WEIGHT
from .observability import get_logger
from .trust_signals import load_calibration_weights, load_rollback_history
from .truth_score import _DEFAULT_AGE_HALF_LIFE_DAYS, truth_score

_log = get_logger("trust_scores")

# --- constants --------------------------------------------------------------

#: Trust assigned to an actor with no evidence at all (and the shrinkage prior).
NEUTRAL_TRUST = 0.5
MIN_TRUST = 0.01
MAX_TRUST = 0.99

#: Additive field written onto recall hits when the feature is enabled.
TRUST_FIELD = "actor_trust"
#: Additive field written by the re-ranker only (flag ON).
TRUST_SCORE_FIELD = "trust_adjusted_score"

#: Component weights. Sum is exactly 1.0; absent components fall back to
#: NEUTRAL_TRUST so the weights never need renormalising.
COMPONENT_WEIGHTS: Mapping[str, float] = {
    "truth": 0.45,
    "calibration": 0.25,
    "contradiction": 0.20,
    "rollback": 0.10,
}

#: Pseudo-observations pulling a thin-evidence actor back to NEUTRAL_TRUST.
PRIOR_STRENGTH = 3.0

#: Rounding for every emitted score — keeps output stable across platforms.
ROUND_DP = 4

#: How hard the opt-in re-rank may demote a fully-untrusted actor.
DEFAULT_RERANK_WEIGHT = 0.35

#: Statuses that mean "governance already ruled against this block".
CONTRADICTED_STATUSES: frozenset[str] = frozenset({"superseded", "rejected", "deprecated"})

#: Result score fields the re-ranker will use as its base, in priority order.
_BASE_SCORE_FIELDS: tuple[str, ...] = ("rrf_score", "score", "bm25_score")

# --- signals ----------------------------------------------------------------


@dataclass(frozen=True)
class ActorSignals:
    """Immutable bundle of the evidence collected for one actor.

    Args:
        block_truth: Per-block ``truth_score`` values in ``[0, 1]``.
        calibration_weights: Per-block calibration weights in
            ``[MIN_CALIBRATION_WEIGHT, MAX_CALIBRATION_WEIGHT]``.
        contradicted_blocks: How many of the actor's blocks are in a
            contradicted state.
        total_blocks: How many blocks the actor wrote (``>= contradicted_blocks``).
        rollbacks: Audit-chain ``rollback`` operations attributed to the actor.
        total_writes: All audit-chain operations by the actor (``>= rollbacks``).

    Raises:
        ValueError: any count is negative, a ratio numerator exceeds its
            denominator, or a score falls outside its declared range.
    """

    block_truth: tuple[float, ...] = ()
    calibration_weights: tuple[float, ...] = ()
    contradicted_blocks: int = 0
    total_blocks: int = 0
    rollbacks: int = 0
    total_writes: int = 0

    def __post_init__(self) -> None:
        for name, seq, lo, hi in (
            ("block_truth", self.block_truth, 0.0, 1.0),
            ("calibration_weights", self.calibration_weights, MIN_CALIBRATION_WEIGHT, MAX_CALIBRATION_WEIGHT),
        ):
            if not isinstance(seq, tuple):
                raise ValueError(f"{name} must be a tuple, got {type(seq).__name__}")
            for value in seq:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"{name} entries must be numeric, got {type(value).__name__}")
                if not (lo <= float(value) <= hi):
                    raise ValueError(f"{name} entry {value!r} outside [{lo}, {hi}]")
        for name, count in (
            ("contradicted_blocks", self.contradicted_blocks),
            ("total_blocks", self.total_blocks),
            ("rollbacks", self.rollbacks),
            ("total_writes", self.total_writes),
        ):
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{name} must be a non-negative int, got {count!r}")
        if self.contradicted_blocks > self.total_blocks:
            raise ValueError(f"contradicted_blocks ({self.contradicted_blocks}) > total_blocks ({self.total_blocks})")
        if self.rollbacks > self.total_writes:
            raise ValueError(f"rollbacks ({self.rollbacks}) > total_writes ({self.total_writes})")

    @property
    def evidence_count(self) -> int:
        """Observations backing this actor — drives shrinkage to neutral."""
        blocks = max(len(self.block_truth), len(self.calibration_weights), self.total_blocks)
        return blocks + self.total_writes


@dataclass(frozen=True)
class ActorTrust:
    """Computed trust for one actor, with its component breakdown."""

    actor_id: str
    trust: float
    evidence_count: int
    truth: float
    calibration: float
    contradiction: float
    rollback: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "trust": self.trust,
            "evidence_count": self.evidence_count,
            "components": {
                "truth": self.truth,
                "calibration": self.calibration,
                "contradiction": self.contradiction,
                "rollback": self.rollback,
            },
        }


@dataclass(frozen=True)
class TrustConfig:
    """Resolved ``retrieval.trust_scores`` settings (all default-OFF)."""

    enabled: bool = False
    rerank: bool = False
    rerank_weight: float = DEFAULT_RERANK_WEIGHT
    use_calibration: bool = True
    use_rollback_history: bool = True
    age_half_life_days: float = _DEFAULT_AGE_HALF_LIFE_DAYS


# --- pure math --------------------------------------------------------------


def _mean(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return float(sum(float(v) for v in values) / len(values))


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


def compute_actor_trust(actor_id: str, signals: ActorSignals) -> ActorTrust:
    """Aggregate *signals* into one trust value in ``[MIN_TRUST, MAX_TRUST]``.

    Pure and deterministic: no clock, no I/O, no randomness. Each missing
    component defaults to :data:`NEUTRAL_TRUST`, then the weighted blend is
    shrunk toward neutral by :data:`PRIOR_STRENGTH` pseudo-observations so a
    brand-new actor is neither trusted nor punished.

    Raises:
        TypeError: *signals* is not an :class:`ActorSignals`.
    """
    if not isinstance(signals, ActorSignals):
        raise TypeError(f"signals must be ActorSignals, got {type(signals).__name__}")

    truth_mean = _mean(signals.block_truth)
    truth_c = NEUTRAL_TRUST if truth_mean is None else _clamp(truth_mean, 0.0, 1.0)

    cal_mean = _mean(signals.calibration_weights)
    if cal_mean is None:
        cal_c = NEUTRAL_TRUST
    else:
        span = MAX_CALIBRATION_WEIGHT - MIN_CALIBRATION_WEIGHT
        cal_c = _clamp((cal_mean - MIN_CALIBRATION_WEIGHT) / span, 0.0, 1.0) if span > 0 else NEUTRAL_TRUST

    if signals.total_blocks > 0:
        contra_c = 1.0 - (signals.contradicted_blocks / signals.total_blocks)
    else:
        contra_c = NEUTRAL_TRUST

    if signals.total_writes > 0:
        roll_c = 1.0 - (signals.rollbacks / signals.total_writes)
    else:
        roll_c = NEUTRAL_TRUST

    raw = (
        COMPONENT_WEIGHTS["truth"] * truth_c
        + COMPONENT_WEIGHTS["calibration"] * cal_c
        + COMPONENT_WEIGHTS["contradiction"] * contra_c
        + COMPONENT_WEIGHTS["rollback"] * roll_c
    )

    evidence = signals.evidence_count
    shrunk = (raw * evidence + NEUTRAL_TRUST * PRIOR_STRENGTH) / (evidence + PRIOR_STRENGTH)

    return ActorTrust(
        actor_id=actor_id,
        trust=round(_clamp(shrunk, MIN_TRUST, MAX_TRUST), ROUND_DP),
        evidence_count=evidence,
        truth=round(truth_c, ROUND_DP),
        calibration=round(cal_c, ROUND_DP),
        contradiction=round(contra_c, ROUND_DP),
        rollback=round(roll_c, ROUND_DP),
    )


# --- aggregation over blocks ------------------------------------------------


def _actor_of(block: Mapping[str, Any]) -> str:
    return extract_provenance(dict(block)).get("actor_id", "")


def _block_truth(block: Mapping[str, Any], age_half_life_days: float) -> float:
    """Reuse an upstream ``truth_score`` annotation, else compute one."""
    existing = block.get("truth_score")
    if isinstance(existing, (int, float)) and not isinstance(existing, bool):
        return _clamp(float(existing), 0.0, 1.0)
    return truth_score(dict(block), age_half_life_days=age_half_life_days)


def aggregate_actor_signals(
    blocks: Iterable[Mapping[str, Any]],
    *,
    calibration_weights: Mapping[str, float] | None = None,
    contradicted_ids: frozenset[str] | set[str] | None = None,
    rollback_counts: Mapping[str, int] | None = None,
    write_counts: Mapping[str, int] | None = None,
    age_half_life_days: float = _DEFAULT_AGE_HALF_LIFE_DAYS,
) -> dict[str, ActorSignals]:
    """Group *blocks* by ``ActorId`` and fold the existing signals per actor.

    Args:
        blocks: Block dicts (recall hits or parsed corpus blocks).
        calibration_weights: ``block_id -> weight`` from
            :meth:`calibration.CalibrationManager.get_block_weights`.
        contradicted_ids: Explicit contradicted block ids. When ``None``,
            :data:`CONTRADICTED_STATUSES` on the block is used instead.
        rollback_counts: ``actor_id -> rollback op count``.
        write_counts: ``actor_id -> total audit op count``.
        age_half_life_days: Half-life used only when a block carries no
            upstream ``truth_score`` annotation.

    Returns:
        ``actor_id -> ActorSignals``. Blocks with no ``ActorId`` are skipped.
    """
    truth_by_actor: dict[str, list[float]] = {}
    cal_by_actor: dict[str, list[float]] = {}
    totals: dict[str, int] = {}
    contradicted: dict[str, int] = {}

    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        actor = _actor_of(block)
        if not actor:
            continue
        totals[actor] = totals.get(actor, 0) + 1
        truth_by_actor.setdefault(actor, []).append(_block_truth(block, age_half_life_days))

        bid = str(block.get("_id") or "")
        if calibration_weights and bid in calibration_weights:
            weight = calibration_weights[bid]
            if isinstance(weight, (int, float)) and not isinstance(weight, bool):
                cal_by_actor.setdefault(actor, []).append(_clamp(float(weight), MIN_CALIBRATION_WEIGHT, MAX_CALIBRATION_WEIGHT))

        if contradicted_ids is not None:
            is_bad = bid in contradicted_ids
        else:
            is_bad = str(block.get("Status") or "").strip().lower() in CONTRADICTED_STATUSES
        if is_bad:
            contradicted[actor] = contradicted.get(actor, 0) + 1

    out: dict[str, ActorSignals] = {}
    for actor in sorted(totals):
        writes = int((write_counts or {}).get(actor, 0))
        rolls = min(int((rollback_counts or {}).get(actor, 0)), writes)
        out[actor] = ActorSignals(
            block_truth=tuple(truth_by_actor.get(actor, ())),
            calibration_weights=tuple(cal_by_actor.get(actor, ())),
            contradicted_blocks=contradicted.get(actor, 0),
            total_blocks=totals[actor],
            rollbacks=rolls,
            total_writes=writes,
        )
    return out


def compute_trust_map(signals_by_actor: Mapping[str, ActorSignals]) -> dict[str, ActorTrust]:
    """Map every actor's signals to its :class:`ActorTrust` (deterministic)."""
    return {actor: compute_actor_trust(actor, signals_by_actor[actor]) for actor in sorted(signals_by_actor)}


# --- recall surface ---------------------------------------------------------


def annotate_trust(results: list[dict], trust_map: Mapping[str, ActorTrust], *, default_trust: float = NEUTRAL_TRUST) -> list[dict]:
    """Return NEW result dicts carrying the additive :data:`TRUST_FIELD`.

    Copy-on-write — the caller's dicts are never mutated. Hits without an
    ``ActorId`` (or an unknown actor) get *default_trust*.
    """
    annotated: list[dict] = []
    for r in results:
        actor = _actor_of(r)
        entry = trust_map.get(actor) if actor else None
        out = dict(r)
        out[TRUST_FIELD] = entry.trust if entry is not None else round(default_trust, ROUND_DP)
        annotated.append(out)
    return annotated


def _base_score(result: Mapping[str, Any], index: int) -> float:
    for key in _BASE_SCORE_FIELDS:
        value = result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    # No score field on the hit — fall back to a rank-derived proxy so the
    # existing order is preserved when every trust value is equal.
    return 1.0 / (1.0 + index)


def rerank_by_trust(results: list[dict], *, weight: float = DEFAULT_RERANK_WEIGHT) -> list[dict]:
    """Stable-sort *results* by ``base_score × (1 − weight·(1 − trust))``.

    A fully-trusted actor keeps its score; a zero-trust actor loses
    *weight* of it. Ties preserve the incoming order (stable sort), so an
    all-equal-trust result set is returned in its original order.

    Raises:
        ValueError: *weight* is outside ``[0, 1]``.
    """
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not (0.0 <= float(weight) <= 1.0):
        raise ValueError(f"weight must be a float in [0, 1], got {weight!r}")

    scored: list[dict] = []
    for index, r in enumerate(results):
        trust = r.get(TRUST_FIELD, NEUTRAL_TRUST)
        trust_value = _clamp(float(trust), 0.0, 1.0) if isinstance(trust, (int, float)) and not isinstance(trust, bool) else NEUTRAL_TRUST
        multiplier = 1.0 - float(weight) * (1.0 - trust_value)
        out = dict(r)
        out[TRUST_SCORE_FIELD] = round(_base_score(r, index) * multiplier, 6)
        scored.append(out)

    return sorted(scored, key=lambda r: float(r[TRUST_SCORE_FIELD]), reverse=True)


# --- config + orchestration -------------------------------------------------


def resolve_trust_config(config: Mapping[str, Any] | None) -> TrustConfig:
    """Read ``retrieval.trust_scores`` defensively; unknown shapes → defaults."""
    if not isinstance(config, Mapping):
        return TrustConfig()
    retrieval = config.get("retrieval")
    if not isinstance(retrieval, Mapping):
        return TrustConfig()
    section = retrieval.get("trust_scores")
    if not isinstance(section, Mapping):
        return TrustConfig()

    raw_weight = section.get("rerank_weight", DEFAULT_RERANK_WEIGHT)
    try:
        rerank_weight = _clamp(float(raw_weight), 0.0, 1.0)
    except (TypeError, ValueError):
        rerank_weight = DEFAULT_RERANK_WEIGHT

    raw_hl = section.get("age_half_life_days", _DEFAULT_AGE_HALF_LIFE_DAYS)
    try:
        half_life = float(raw_hl)
    except (TypeError, ValueError):
        half_life = _DEFAULT_AGE_HALF_LIFE_DAYS

    return TrustConfig(
        enabled=bool(section.get("enabled", False)),
        rerank=bool(section.get("rerank", False)),
        rerank_weight=rerank_weight,
        use_calibration=bool(section.get("use_calibration", True)),
        use_rollback_history=bool(section.get("use_rollback_history", True)),
        age_half_life_days=half_life,
    )


def is_trust_scores_enabled(config: Mapping[str, Any] | None) -> bool:
    """True only when ``retrieval.trust_scores.enabled`` is explicitly set."""
    return resolve_trust_config(config).enabled


def apply_trust_scores(
    results: list[dict],
    *,
    config: Mapping[str, Any] | None = None,
    workspace: str | None = None,
    calibration_weights: Mapping[str, float] | None = None,
    rollback_counts: Mapping[str, int] | None = None,
    write_counts: Mapping[str, int] | None = None,
    contradicted_ids: frozenset[str] | set[str] | None = None,
) -> list[dict]:
    """Annotate (and optionally re-rank) recall hits with per-actor trust.

    Returns the **same list object** untouched when
    ``retrieval.trust_scores.enabled`` is false — the zero-regression
    contract. Injected signal maps take precedence over workspace lookups
    so tests never touch disk.
    """
    if not results:
        return results
    cfg = resolve_trust_config(config)
    if not cfg.enabled:
        return results

    block_ids = [str(r.get("_id") or "") for r in results if r.get("_id")]
    if calibration_weights is None and cfg.use_calibration and workspace:
        calibration_weights = load_calibration_weights(workspace, block_ids)
    if (rollback_counts is None or write_counts is None) and cfg.use_rollback_history and workspace:
        loaded_rollbacks, loaded_writes = load_rollback_history(workspace)
        rollback_counts = loaded_rollbacks if rollback_counts is None else rollback_counts
        write_counts = loaded_writes if write_counts is None else write_counts

    signals = aggregate_actor_signals(
        results,
        calibration_weights=calibration_weights,
        contradicted_ids=contradicted_ids,
        rollback_counts=rollback_counts,
        write_counts=write_counts,
        age_half_life_days=cfg.age_half_life_days,
    )
    trust_map = compute_trust_map(signals)
    annotated = annotate_trust(results, trust_map)
    _log.info("trust_scores_annotated", hits=len(annotated), actors=len(trust_map), rerank=cfg.rerank)
    if not cfg.rerank:
        return annotated
    return rerank_by_trust(annotated, weight=cfg.rerank_weight)


# deferred: trust is aggregated only over the blocks present in the current
# result set (plus workspace-wide calibration/rollback history). A corpus-wide
# per-actor rollup would be sharper but needs an indexed actor->block table —
# upgrade path: read block_meta.actor_id (block_metadata.py already stores it)
# and cache the rollup per reindex generation.

__all__ = [
    "ActorSignals",
    "ActorTrust",
    "TrustConfig",
    "COMPONENT_WEIGHTS",
    "CONTRADICTED_STATUSES",
    "DEFAULT_RERANK_WEIGHT",
    "NEUTRAL_TRUST",
    "PRIOR_STRENGTH",
    "TRUST_FIELD",
    "TRUST_SCORE_FIELD",
    "aggregate_actor_signals",
    "annotate_trust",
    "apply_trust_scores",
    "compute_actor_trust",
    "compute_trust_map",
    "is_trust_scores_enabled",
    "load_calibration_weights",
    "load_rollback_history",
    "rerank_by_trust",
    "resolve_trust_config",
]
