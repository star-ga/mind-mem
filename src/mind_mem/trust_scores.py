"""Standalone trust surface — a thin façade over the validity gate.

Trust is **not** a separate scoring subsystem. "How much do we trust where
this block came from" is the fifth component of the recall validity gate
(:mod:`validity_gate` -> :mod:`provenance_class`), ordered

    operator > agent-verified > agent-inferred > external-ingest

and this module does exactly one thing: expose that single component on
recall hits under the stable ``actor_trust`` field, with an opt-in re-rank.
Every number it emits comes from :func:`validity_gate.provenance_component`
— one composite path, not two.

What this module deliberately does **not** do is score actors. A per-actor
reputation aggregated from that actor's history (mean truth, calibration
mean, contradiction share, rollback share, shrinkage prior) is a learned,
mutable, corpus-order-dependent quantity: two machines holding different
slices of the same corpus would rank the same hit differently. That breaks
the determinism wedge, so it is gone. What survives is per-block,
table-driven, and reproducible: the class of the writer's *role*, plus
explicit per-block verification evidence.

**Zero regression.** Everything here is opt-in and default-OFF:

.. code-block:: json

    {"retrieval": {"trust_scores": {"enabled": true, "rerank": true}}}

With ``enabled`` false :func:`apply_trust_scores` returns the exact same
list object it was given — no added fields, no reordering, no I/O.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from .observability import get_logger
from .provenance_class import PROVENANCE_ORDER, PROVENANCE_WEIGHTS, UNKNOWN
from .trust_signals import load_calibration_weights, load_rollback_history
from .validity_gate import confirmed_block_ids, provenance_component

_log = get_logger("trust_scores")

# --- constants --------------------------------------------------------------

#: Trust of a hit carrying no provenance at all. The gate's governing rule is
#: *absence of a signal is neutral; only affirmative evidence debits*, so an
#: unclassifiable hit is never demoted for being unclassifiable.
NEUTRAL_TRUST = PROVENANCE_WEIGHTS[UNKNOWN]

#: Additive field written onto recall hits when the feature is enabled.
TRUST_FIELD = "actor_trust"
#: Additive field written by the re-ranker only (flag ON).
TRUST_SCORE_FIELD = "trust_adjusted_score"

#: Rounding for every emitted score — keeps output stable across platforms.
ROUND_DP = 4

#: How hard the opt-in re-rank may demote fully-untrusted provenance.
DEFAULT_RERANK_WEIGHT = 0.35

#: Result score fields the re-ranker will use as its base, in priority order.
_BASE_SCORE_FIELDS: tuple[str, ...] = ("rrf_score", "score", "bm25_score")


@dataclass(frozen=True)
class TrustConfig:
    """Resolved ``retrieval.trust_scores`` settings (all default-OFF)."""

    enabled: bool = False
    rerank: bool = False
    rerank_weight: float = DEFAULT_RERANK_WEIGHT
    use_calibration: bool = True


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


# --- recall surface ---------------------------------------------------------


def annotate_trust(
    results: list[dict],
    *,
    confirmed_ids: frozenset[str] = frozenset(),
) -> list[dict]:
    """Return NEW result dicts carrying the additive :data:`TRUST_FIELD`.

    Copy-on-write — the caller's dicts are never mutated. The value is the
    gate's provenance component for that hit, so it is identical to
    ``hit["validity"]["provenance"]`` when the gate runs with its
    provenance component enabled.
    """
    annotated: list[dict] = []
    for result in results:
        out = dict(result)
        out[TRUST_FIELD] = round(provenance_component(result, confirmed_ids), ROUND_DP)
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

    Operator-class provenance keeps its score; external-ingest loses
    ``0.75·weight`` of it. Ties preserve the incoming order (stable sort),
    so an all-equal-provenance result set is returned in its original order.

    Raises:
        ValueError: *weight* is outside ``[0, 1]``.
    """
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not (0.0 <= float(weight) <= 1.0):
        raise ValueError(f"weight must be a float in [0, 1], got {weight!r}")

    scored: list[dict] = []
    for index, result in enumerate(results):
        raw = result.get(TRUST_FIELD, NEUTRAL_TRUST)
        trust = _clamp(float(raw), 0.0, 1.0) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else NEUTRAL_TRUST
        multiplier = 1.0 - float(weight) * (1.0 - trust)
        out = dict(result)
        out[TRUST_SCORE_FIELD] = round(_base_score(result, index) * multiplier, 6)
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

    return TrustConfig(
        enabled=bool(section.get("enabled", False)),
        rerank=bool(section.get("rerank", False)),
        rerank_weight=rerank_weight,
        use_calibration=bool(section.get("use_calibration", True)),
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
    scoring_instant: date | None = None,
) -> list[dict]:
    """Annotate (and optionally re-rank) recall hits with provenance trust.

    Delegates the whole computation to the validity gate's fifth component.
    Returns the **same list object** untouched when
    ``retrieval.trust_scores.enabled`` is false — the zero-regression
    contract. An injected *calibration_weights* map takes precedence over
    the workspace lookup so tests never touch disk.
    """
    if not results:
        return results
    cfg = resolve_trust_config(config)
    if not cfg.enabled:
        return results

    if calibration_weights is None and cfg.use_calibration and workspace:
        block_ids = [str(r.get("_id") or "") for r in results if r.get("_id")]
        calibration_weights = load_calibration_weights(workspace, block_ids, scoring_instant=scoring_instant)
    confirmed_ids = confirmed_block_ids(calibration_weights)

    annotated = annotate_trust(results, confirmed_ids=confirmed_ids)
    _log.info("trust_scores_annotated", hits=len(annotated), rerank=cfg.rerank)
    if not cfg.rerank:
        return annotated
    return rerank_by_trust(annotated, weight=cfg.rerank_weight)


__all__ = [
    "DEFAULT_RERANK_WEIGHT",
    "NEUTRAL_TRUST",
    "PROVENANCE_ORDER",
    "PROVENANCE_WEIGHTS",
    "TRUST_FIELD",
    "TRUST_SCORE_FIELD",
    "TrustConfig",
    "annotate_trust",
    "apply_trust_scores",
    "confirmed_block_ids",
    "is_trust_scores_enabled",
    "load_calibration_weights",
    "load_rollback_history",
    "rerank_by_trust",
    "resolve_trust_config",
]
