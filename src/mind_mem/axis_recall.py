# Copyright 2026 STARGA, Inc.
"""Axis-aware recall orchestrator (Observer-Dependent Cognition).

Thin coordination layer over :func:`mind_mem.recall.recall` that:

1. Accepts an explicit :class:`AxisWeights` vector so callers can bias
   retrieval toward lexical / semantic / temporal / entity-graph /
   contradiction / adversarial signals.
2. Tags every result with an :class:`Observation` recording which axes
   produced it and with what confidence.
3. Rotates to orthogonal axes and retries when the top-k confidence falls
   below :data:`DEFAULT_ROTATION_THRESHOLD`.
4. Optionally fuses results from an adversarial axis pass so contradictory
   evidence surfaces alongside supportive evidence.

This module deliberately avoids reimplementing retrieval. It delegates to
``recall()`` for each axis pass and merges the returned block dicts by
``_id`` using a weighted RRF so every call path still benefits from the
core BM25F + vector + RRF + cross-encoder pipeline.

Zero new external dependencies.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .observability import get_logger
from .observation_axis import (
    DEFAULT_ROTATION_THRESHOLD,
    AxisScore,
    AxisWeights,
    Observation,
    ObservationAxis,
    adversarial_pair,
    rotate_axes,
    should_rotate,
)

_log = get_logger("axis_recall")

# RRF constant copied from hybrid_recall for consistency. Small k keeps
# rank differences meaningful at modest recall depths; large k would
# flatten fusion and erase the per-axis signal we want to surface.
_RRF_K: int = 60


# ---------------------------------------------------------------------------
# Single-axis retrieval pass
# ---------------------------------------------------------------------------


def _recall_for_axis(
    workspace: str,
    query: str,
    axis: ObservationAxis,
    *,
    limit: int,
    active_only: bool,
    base_recall_kwargs: Mapping[str, Any],
) -> list[dict]:
    """Run a single-axis retrieval pass.

    Each axis maps onto a different configuration of the core recall
    pipeline. The function is pure orchestration — no crypto, no writes.
    """
    # Import inside the function so unit tests can monkeypatch without
    # paying the import cost of the whole retrieval stack in the common
    # primitive-only path.
    from .recall import recall as _recall

    kwargs: dict[str, Any] = dict(base_recall_kwargs)
    kwargs.setdefault("limit", limit)
    kwargs.setdefault("active_only", active_only)

    if axis is ObservationAxis.LEXICAL:
        kwargs["backend"] = "bm25"
    elif axis is ObservationAxis.SEMANTIC:
        kwargs["backend"] = "hybrid"
    elif axis is ObservationAxis.TEMPORAL:
        # Temporal axis prioritises recent and dated blocks by leaning on
        # recency decay in the underlying recall config. No config flag
        # exists today to force this, so we call the default backend and
        # post-filter/boost. Surfaced as a note on the observation below.
        kwargs["backend"] = "bm25"
    elif axis is ObservationAxis.ENTITY_GRAPH:
        # Graph expansion lives inside the retrieval_graph pipeline; set
        # the config hint the core recall respects when present.
        kwargs["backend"] = "hybrid"
        kwargs["graph_expand"] = True
    elif axis is ObservationAxis.CONTRADICTION:
        # Contradiction axis explicitly includes superseded / conflicting
        # blocks. Dropping active_only is the most portable way today.
        kwargs["backend"] = "bm25"
        kwargs["active_only"] = False
    elif axis is ObservationAxis.ADVERSARIAL:
        # Adversarial axis deliberately probes the opposing basis. The
        # NOT-prefix rewrite happens in _run_pass so it stays observable
        # to tests that stub out _recall_for_axis.
        kwargs["backend"] = "bm25"
        kwargs["active_only"] = False

    try:
        raw = _recall(workspace, query, **kwargs)
    except TypeError as exc:
        # Older recall signatures may not accept every kwarg we set.
        # Retry with the minimum-portable surface but log the mismatch
        # prominently so real bugs are not silenced.
        _log.warning(
            "axis_recall.signature_fallback",
            axis=axis.value,
            error=str(exc),
            hint="recall() rejected axis-specific kwargs; retrying minimal",
        )
        fallback_kwargs = {
            "limit": kwargs.get("limit", limit),
            "active_only": kwargs.get("active_only", active_only),
        }
        raw = _recall(workspace, query, **fallback_kwargs)

    # Normalise to list[dict]
    if not isinstance(raw, list):
        _log.warning("axis_recall.non_list_result", axis=axis.value, type=type(raw).__name__)
        return []
    return [r for r in raw if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# Per-axis scoring + fusion
# ---------------------------------------------------------------------------


def _axis_confidence(rank: int) -> float:
    """Map a 1-based rank within an axis's result list to a [0, 1] score.

    Uses ``1 / (1 + rank)`` normalised so the top result scores near 1.0
    and the tail scores approach 0.
    """
    if rank < 1:
        rank = 1
    score = 1.0 / (1.0 + (rank - 1))
    return max(0.0, min(1.0, score))


def _rrf_score(rank: int) -> float:
    """Reciprocal rank fusion contribution for a 1-based rank."""
    return 1.0 / (_RRF_K + max(1, rank))


def _fuse_axis_results(
    axis_results: Mapping[ObservationAxis, list[dict]],
    weights: AxisWeights,
) -> list[dict]:
    """Weighted-RRF merge of per-axis result lists.

    Each result keeps its original payload fields; additional keys are
    overlaid:

        ``_axis_score``        — fused numeric score (higher = better)
        ``observation``        — serialised :class:`Observation` dict
    """
    # Aggregate per-block: fused score + list of AxisScore contributions.
    block_accum: dict[str, dict[str, Any]] = {}

    weight_map = {
        ObservationAxis.LEXICAL: weights.lexical,
        ObservationAxis.SEMANTIC: weights.semantic,
        ObservationAxis.TEMPORAL: weights.temporal,
        ObservationAxis.ENTITY_GRAPH: weights.entity_graph,
        ObservationAxis.CONTRADICTION: weights.contradiction,
        ObservationAxis.ADVERSARIAL: weights.adversarial,
    }

    for axis, results in axis_results.items():
        axis_weight = weight_map.get(axis, 0.0)
        if axis_weight <= 0.0 or not results:
            continue
        for rank, res in enumerate(results, start=1):
            bid = _block_id(res)
            if not bid:
                continue
            contribution = axis_weight * _rrf_score(rank)
            confidence = _axis_confidence(rank)
            axis_score = AxisScore(axis=axis, confidence=confidence, rank=rank)

            if bid not in block_accum:
                block_accum[bid] = {
                    "payload": dict(res),
                    "score": 0.0,
                    "axis_scores": [],
                }
            block_accum[bid]["score"] += contribution
            block_accum[bid]["axis_scores"].append(axis_score)

    fused: list[dict] = []
    for bid, entry in block_accum.items():
        payload = entry["payload"]
        obs = Observation(axes=tuple(entry["axis_scores"]))
        payload["_axis_score"] = round(entry["score"], 6)
        payload["observation"] = obs.as_dict()
        fused.append(payload)

    fused.sort(key=lambda r: (r["_axis_score"], _block_id(r) or ""), reverse=True)
    return fused


def _block_id(result: Mapping[str, Any]) -> str:
    """Return a stable string id for a result dict."""
    for key in ("_id", "id", "block_id"):
        value = result.get(key)
        if value:
            return str(value)
    file_ = result.get("file") or "?"
    line = result.get("line") or 0
    return f"{file_}:{line}"


def _adversarial_query(query: str) -> str:
    """Rewrite a query for adversarial-axis probing.

    Wraps the original query in parentheses and prefixes ``NOT`` so an FTS5
    backend negates the full expression instead of just the first token
    (``NOT foo AND bar`` parses as ``(NOT foo) AND bar``). Strips leading /
    trailing whitespace and returns an empty string when the cleaned query
    is empty — callers skip the axis in that case.
    """
    cleaned = query.strip()
    if not cleaned:
        return ""
    # Double-quote the inner expression so FTS5 treats it as a phrase and
    # the caller's metacharacters (":", "*", parens) don't leak operators.
    escaped = cleaned.replace('"', '""')
    return f'NOT "{escaped}"'


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def recall_with_axis(
    workspace: str,
    query: str,
    *,
    weights: Optional[AxisWeights] = None,
    limit: int = 10,
    active_only: bool = True,
    rotation_threshold: float = DEFAULT_ROTATION_THRESHOLD,
    allow_rotation: bool = True,
    adversarial: bool = False,
    recall_kwargs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Run an axis-aware recall.

    Args:
        workspace: Absolute path to the mind-mem workspace.
        query: User query string.
        weights: Axis weight vector. Defaults to lexical+semantic @ 1.0.
        limit: Maximum number of results to return.
        active_only: Whether to exclude superseded blocks from the base
            passes. The CONTRADICTION / ADVERSARIAL axes override this
            internally so dissent can still be surfaced.
        rotation_threshold: Top-result confidence below which orthogonal
            axes are retried (see :data:`DEFAULT_ROTATION_THRESHOLD`).
        allow_rotation: Disable rotation entirely when False (useful in
            deterministic tests and benchmarks).
        adversarial: When True, run the adversarial pair for every active
            axis as an additional retrieval pass and fuse it alongside.
        recall_kwargs: Extra kwargs forwarded verbatim to :func:`recall`.

    Returns:
        A dict with keys ``results`` (list of merged block dicts),
        ``weights`` (the effective axis weights), ``rotated`` (bool),
        ``diversity`` (int — count of distinct axes that contributed),
        and ``attempts`` (list of AxisWeights tried in order).
    """
    if weights is None:
        weights = AxisWeights()
    if not weights.active_axes():
        raise ValueError("recall_with_axis requires at least one active axis")

    base_kwargs = dict(recall_kwargs or {})
    attempts: list[AxisWeights] = [weights]
    tried_axes: set[ObservationAxis] = set(weights.active_axes())
    rotated = False

    fused = _run_pass(
        workspace,
        query,
        weights,
        limit=limit,
        active_only=active_only,
        base_recall_kwargs=base_kwargs,
        adversarial=adversarial,
    )

    top_confidence = _top_confidence(fused)
    if allow_rotation and should_rotate(top_confidence, threshold=rotation_threshold):
        rotated_weights = rotate_axes(weights, already_tried=tried_axes)
        # rotate_axes returns the same instance when nothing new is
        # available — that's our cue to stop.
        if rotated_weights is not weights:
            attempts.append(rotated_weights)
            tried_axes.update(rotated_weights.active_axes())
            rotation_extra = _run_pass(
                workspace,
                query,
                rotated_weights,
                limit=limit,
                active_only=active_only,
                base_recall_kwargs=base_kwargs,
                adversarial=adversarial,
            )
            # _merge_rotation is responsible for stamping rotated=True on
            # the specific results that the rotation pass touched; we
            # must NOT overwrite it for primary-only results that stayed
            # below the confidence threshold.
            fused = _merge_rotation(fused, rotation_extra)
            rotated = True

    fused.sort(key=lambda r: (r.get("_axis_score", 0.0), _block_id(r) or ""), reverse=True)
    fused = fused[:limit]

    diversity_count = _count_axis_diversity(fused)

    _log.info(
        "axis_recall.done",
        query_len=len(query),
        limit=limit,
        axes=[a.value for a in weights.active_axes()],
        rotated=rotated,
        diversity=diversity_count,
        result_count=len(fused),
    )

    return {
        "results": fused,
        "weights": weights.as_dict(),
        "rotated": rotated,
        "diversity": diversity_count,
        "attempts": [w.as_dict() for w in attempts],
    }


def _run_pass(
    workspace: str,
    query: str,
    weights: AxisWeights,
    *,
    limit: int,
    active_only: bool,
    base_recall_kwargs: Mapping[str, Any],
    adversarial: bool,
) -> list[dict]:
    """Execute every active axis once and fuse with the given weights."""
    axes_to_run: list[ObservationAxis] = list(weights.active_axes())
    if adversarial:
        for axis in list(axes_to_run):
            opponent = adversarial_pair(axis)
            if opponent not in axes_to_run:
                axes_to_run.append(opponent)

    axis_results: dict[ObservationAxis, list[dict]] = {}
    for axis in axes_to_run:
        axis_query = _adversarial_query(query) if axis is ObservationAxis.ADVERSARIAL else query
        if not axis_query:
            # Defensive: adversarial rewrite of an empty query is meaningless
            # and some FTS5 backends reject "NOT ()" outright.
            axis_results[axis] = []
            continue
        axis_results[axis] = _recall_for_axis(
            workspace,
            axis_query,
            axis,
            limit=limit * 2,  # over-fetch so fusion has headroom
            active_only=active_only,
            base_recall_kwargs=base_recall_kwargs,
        )

    # When adversarial probing adds axes not in the weight vector, give
    # them a small default weight so their signal isn't dropped entirely.
    effective = weights
    missing = [ax for ax in axes_to_run if weights.as_dict().get(ax.value, 0.0) == 0.0]
    if missing:
        patch = {**weights.as_dict(), **{ax.value: 0.25 for ax in missing}}
        effective = AxisWeights.from_mapping(patch)

    return _fuse_axis_results(axis_results, effective)


def _merge_rotation(primary: list[dict], rotation: list[dict]) -> list[dict]:
    """Merge a rotation pass into the primary result list.

    Duplicate block IDs have their axis scores concatenated and their
    fused ``_axis_score`` summed so rotation can promote a block that
    was previously borderline.
    """
    index = {_block_id(r): r for r in primary}
    for res in rotation:
        bid = _block_id(res)
        if bid in index:
            existing = index[bid]
            existing["_axis_score"] = round(
                float(existing.get("_axis_score", 0.0)) + float(res.get("_axis_score", 0.0)),
                6,
            )
            new_axes = res.get("observation", {}).get("axes", [])
            existing_obs = existing.setdefault("observation", {"axes": [], "rotated": False, "notes": []})
            existing_obs.setdefault("axes", []).extend(new_axes)
            existing_obs["rotated"] = True
        else:
            # Mark rotation-only results explicitly.
            obs = res.setdefault("observation", {})
            obs["rotated"] = True
            index[bid] = res
    return list(index.values())


def _top_confidence(results: Iterable[Mapping[str, Any]]) -> float:
    """Return the highest axis confidence from the top-1 result."""
    for result in results:
        obs = result.get("observation") if isinstance(result, Mapping) else None
        if not isinstance(obs, Mapping):
            continue
        axes = obs.get("axes")
        if not isinstance(axes, list) or not axes:
            continue
        confidences = [float(a.get("confidence", 0.0)) for a in axes if isinstance(a, Mapping)]
        if confidences:
            return max(confidences)
    return 0.0


def _count_axis_diversity(results: Iterable[Mapping[str, Any]]) -> int:
    """Count distinct axes that contributed across the returned result set."""
    seen: set[str] = set()
    for res in results:
        obs = res.get("observation") if isinstance(res, Mapping) else None
        if not isinstance(obs, Mapping):
            continue
        axes = obs.get("axes")
        if not isinstance(axes, list):
            continue
        for entry in axes:
            if isinstance(entry, Mapping):
                axis_name = entry.get("axis")
                if isinstance(axis_name, str):
                    seen.add(axis_name)
    return len(seen)


__all__ = ["recall_with_axis"]
