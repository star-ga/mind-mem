"""v4 surprise-weighted retrieval term (Group A: cognition / model layer).

Surprise is a deterministic retrieval-time signal: how *semantically far*
a candidate block is from the rolling recall context. It is **not** a
gradient — no model training is involved. The distance is a plain
cosine-distance computation against a centroid of recent recall results.

Two consumers depend on this module:

    1. The :data:`KernelKind.SURPRISE_WEIGHTED` strategy (in
       :mod:`mind_mem.v4.cognitive_kernel`) ranks candidates by
       descending surprise instead of by RRF score. Useful when an
       agent needs to catch its own wrong assumptions.

    2. The :mod:`mind_mem.v4.tier_memory` promotion path: a WARM block
       whose read produces surprise above
       :attr:`TierConfig.promote_threshold` bumps back to HOT instead
       of aging toward COLD.

Surprise is a number in ``[0.0, 1.0]``:

    0.0    candidate is identical to context  (no surprise)
    0.5    orthogonal to context              (mild surprise)
    1.0    pointing the opposite direction    (maximum surprise)

The math is plain cosine distance — ``1 - cos_sim(candidate, context)`` —
clamped into the unit interval so downstream thresholds compare
apples-to-apples. Vectors are :class:`list[float]`; we don't pull
``numpy`` here so the v4 surface stays importable on a fresh install
without optional embeddings extras.

This module ships the **scoring math + a stub centroid loader**. The
real centroid wiring (read last-K hits from the recall log, pull
their embeddings from ``recall_vector``) lands when the v4 recall
path is wired through the cognitive kernel — for now, callers pass
the centroid in directly so this module is unit-testable in isolation.

Feature-flag gated under ``v4.surprise_retrieval``. v3.x callers see
no behaviour change.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .feature_flags import flag_config, require_enabled

__all__ = [
    "FLAG",
    "compute_surprise",
    "centroid",
    "should_promote_on_surprise",
    "surprise_threshold",
    "DEFAULT_PROMOTE_THRESHOLD",
]


#: Feature-flag key in ``mind-mem.json: v4: {...}``.
FLAG: str = "surprise_retrieval"

#: Default promotion threshold. Mirrors :attr:`TierConfig.promote_threshold`
#: so the two modules agree by default; callers can override via
#: ``mind-mem.json: v4: surprise_retrieval: promote_threshold: 0.65``.
DEFAULT_PROMOTE_THRESHOLD: float = 0.65


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def compute_surprise(
    candidate: Sequence[float],
    context: Sequence[float] | None,
) -> float:
    """Return surprise of ``candidate`` against ``context`` in ``[0, 1]``.

    Empty / mismatched / zero-norm vectors collapse to **mild surprise
    (0.5)** rather than raising — the goal is a usable ranking signal,
    not a strict numeric contract. The mild-surprise default keeps a
    candidate from being either trivially promoted or trivially demoted
    when the math is undefined.

    No flag check here: the math is pure and cheap. Calling
    :func:`should_promote_on_surprise` is what's gated, because that's
    the function that *acts* on the score.
    """
    if context is None or not candidate or not context:
        return 0.5
    if len(candidate) != len(context):
        return 0.5

    dot = 0.0
    norm_c = 0.0
    norm_x = 0.0
    for a, b in zip(candidate, context):
        a_f = float(a)
        b_f = float(b)
        dot += a_f * b_f
        norm_c += a_f * a_f
        norm_x += b_f * b_f
    if norm_c == 0.0 or norm_x == 0.0:
        return 0.5

    cos_sim = dot / (math.sqrt(norm_c) * math.sqrt(norm_x))
    # Clamp cos_sim into [-1, 1] so float drift doesn't push surprise
    # outside [0, 1]. cos_sim of -1 (opposite direction) maps to
    # surprise 1.0; cos_sim of 1 (identical) maps to surprise 0.0.
    cos_sim = max(-1.0, min(1.0, cos_sim))
    surprise = (1.0 - cos_sim) * 0.5
    # Numerical safety: if the float math drifts a tick past 1.0 or
    # below 0.0, clamp.
    return max(0.0, min(1.0, surprise))


def centroid(vectors: Iterable[Sequence[float]]) -> list[float] | None:
    """Compute the unweighted mean centroid of an iterable of vectors.

    Returns ``None`` for an empty iterable, mismatched lengths, or
    when every input is empty. Round-trip: the centroid of a single
    vector ``v`` is ``v`` itself. The centroid of two vectors is the
    component-wise mean.

    Designed so callers can wire it directly to the recall-log read
    side once that lands — pull the last K embeddings, hand them to
    :func:`centroid`, then to :func:`compute_surprise`.
    """
    accum: list[float] | None = None
    n = 0
    for v in vectors:
        if not v:
            continue
        if accum is None:
            accum = [float(x) for x in v]
            n = 1
            continue
        if len(v) != len(accum):
            # Inconsistent dimension — bail out rather than mix shapes.
            return None
        for i, x in enumerate(v):
            accum[i] += float(x)
        n += 1
    if accum is None or n == 0:
        return None
    return [x / n for x in accum]


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def should_promote_on_surprise(
    surprise: float,
    *,
    threshold: float | None = None,
) -> bool:
    """Decide whether a WARM block should bump back to HOT.

    Returns ``True`` iff ``surprise >= threshold``. The threshold
    defaults to :data:`DEFAULT_PROMOTE_THRESHOLD` (0.65) and can be
    overridden either per-call (``threshold=...``) or via
    ``mind-mem.json``'s ``promote_threshold`` knob under the
    ``surprise_retrieval`` block.

    Raises :class:`FeatureDisabledError` when the flag is OFF — this
    function *acts* on the score (decides a tier promotion), so it's
    gated; the underlying :func:`compute_surprise` math is not.
    """
    require_enabled(FLAG)
    if threshold is None:
        threshold = surprise_threshold()
    return float(surprise) >= float(threshold)


def surprise_threshold() -> float:
    """Read the configured promotion threshold.

    Falls back to :data:`DEFAULT_PROMOTE_THRESHOLD` if the flag block
    is missing the key or stores a non-numeric value. Range-clamped to
    ``[0.0, 1.0]`` so a typo can't accidentally promote everything.
    """
    raw = flag_config(FLAG)
    if not isinstance(raw, dict):
        return DEFAULT_PROMOTE_THRESHOLD
    v = raw.get("promote_threshold", DEFAULT_PROMOTE_THRESHOLD)
    try:
        out = float(v)
    except (TypeError, ValueError):
        return DEFAULT_PROMOTE_THRESHOLD
    return max(0.0, min(1.0, out))
