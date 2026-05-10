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
from enum import Enum

from .feature_flags import flag_config, require_enabled

__all__ = [
    "FLAG",
    "FallbackPolicy",
    "EmbeddingFailureError",
    "compute_surprise",
    "centroid",
    "should_promote_on_surprise",
    "surprise_threshold",
    "DEFAULT_PROMOTE_THRESHOLD",
    "DEFAULT_FALLBACK_POLICY",
]


#: Feature-flag key in ``mind-mem.json: v4: {...}``.
FLAG: str = "surprise_retrieval"

#: Default promotion threshold. Mirrors :attr:`TierConfig.promote_threshold`
#: so the two modules agree by default; callers can override via
#: ``mind-mem.json: v4: surprise_retrieval: promote_threshold: 0.65``.
DEFAULT_PROMOTE_THRESHOLD: float = 0.65


class FallbackPolicy(str, Enum):
    """How :func:`compute_surprise` handles an unusable embedding.

    Round-4 audit (Mistral 9.9→10) flagged that the implicit
    ``return 0.5`` on bad embeddings buries failures: callers can't
    distinguish "honest mild surprise" from "embedding broken,
    falling back to neutral". This enum makes the choice explicit.

    NEUTRAL  → return 0.5 (default — preserves prior behaviour).
    PROMOTE  → return 1.0 (treat as max surprise; biases tier
               promotion when the embedder is flaky).
    DEMOTE   → return 0.0 (treat as no surprise; biases tier demotion
               so a broken embedder doesn't block COLD aging).
    RAISE    → raise :class:`EmbeddingFailureError` so the caller can
               retry the embedder before scoring.
    """

    NEUTRAL = "neutral"
    PROMOTE = "promote"
    DEMOTE = "demote"
    RAISE = "raise"


#: Default fallback. NEUTRAL preserves prior behaviour for callers that
#: never set the policy. Switch to ``PROMOTE`` to fail-open on tier
#: promotion or ``RAISE`` to surface embedder bugs in CI.
DEFAULT_FALLBACK_POLICY: FallbackPolicy = FallbackPolicy.NEUTRAL


class EmbeddingFailureError(RuntimeError):
    """Raised by :func:`compute_surprise` under ``FallbackPolicy.RAISE``
    when one of the input vectors is missing, mismatched, or zero-norm.

    Carries the failure ``reason`` as a short tag so callers can
    branch on it (``"missing"``, ``"length_mismatch"``,
    ``"zero_norm"``).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"surprise embedding unusable: {reason}")
        self.reason = reason


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def compute_surprise(
    candidate: Sequence[float],
    context: Sequence[float] | None,
    *,
    fallback_policy: FallbackPolicy | str | None = None,
) -> float:
    """Return surprise of ``candidate`` against ``context`` in ``[0, 1]``.

    When the inputs are unusable (empty, dimension-mismatched, or
    zero-norm), the ``fallback_policy`` decides what to return:

        NEUTRAL (default)   → 0.5 — mild surprise, the historical
                              behaviour preserved for callers that
                              never opt in.
        PROMOTE             → 1.0 — bias toward tier promotion so a
                              flaky embedder doesn't strand WARM blocks.
        DEMOTE              → 0.0 — bias toward COLD aging so a flaky
                              embedder doesn't block decay.
        RAISE               → raises :class:`EmbeddingFailureError`
                              with a short reason tag.

    No flag check here: the math is pure and cheap. Calling
    :func:`should_promote_on_surprise` is what's gated, because that's
    the function that *acts* on the score.
    """
    policy = _coerce_fallback(fallback_policy)
    if context is None or not candidate or not context:
        return _apply_fallback(policy, "missing")
    if len(candidate) != len(context):
        return _apply_fallback(policy, "length_mismatch")

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
        return _apply_fallback(policy, "zero_norm")

    cos_sim = dot / (math.sqrt(norm_c) * math.sqrt(norm_x))
    # Clamp cos_sim into [-1, 1] so float drift doesn't push surprise
    # outside [0, 1]. cos_sim of -1 (opposite direction) maps to
    # surprise 1.0; cos_sim of 1 (identical) maps to surprise 0.0.
    cos_sim = max(-1.0, min(1.0, cos_sim))
    surprise = (1.0 - cos_sim) * 0.5
    # Numerical safety: if the float math drifts a tick past 1.0 or
    # below 0.0, clamp.
    return max(0.0, min(1.0, surprise))


def _coerce_fallback(policy: FallbackPolicy | str | None) -> FallbackPolicy:
    if policy is None:
        return _configured_fallback()
    if isinstance(policy, FallbackPolicy):
        return policy
    try:
        return FallbackPolicy(str(policy).lower())
    except ValueError:
        return DEFAULT_FALLBACK_POLICY


def _apply_fallback(policy: FallbackPolicy, reason: str) -> float:
    if policy is FallbackPolicy.PROMOTE:
        return 1.0
    if policy is FallbackPolicy.DEMOTE:
        return 0.0
    if policy is FallbackPolicy.RAISE:
        raise EmbeddingFailureError(reason)
    return 0.5


def _configured_fallback() -> FallbackPolicy:
    raw = flag_config(FLAG)
    if not isinstance(raw, dict):
        return DEFAULT_FALLBACK_POLICY
    name = raw.get("fallback_policy")
    if name is None:
        return DEFAULT_FALLBACK_POLICY
    try:
        return FallbackPolicy(str(name).lower())
    except ValueError:
        return DEFAULT_FALLBACK_POLICY


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
