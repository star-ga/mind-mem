"""Score decomposition record for explainable recall (v3.11.0, Pattern 1).

When a caller passes ``explain=True`` to ``recall`` or ``hybrid_search``,
every hit in the response gains a top-level ``_explain`` field whose shape
is defined by :class:`ScoreExplain`.

Design constraints:
- Frozen dataclass: immutable after construction, dict-serializable.
- Honest gate: fields that are not computed by the current pipeline are
  returned as ``None`` rather than synthesized values.  A field is only
  non-None when the value is already present in the hit dict — no
  re-computation, no approximation.
- ``_explain.final`` IS the ``score`` field used to order results — it is
  read from it, not recomputed, so the two cannot disagree. What
  :func:`attach_explain` asserts at runtime is the claim that can actually
  be false: the ranked hits it was handed are non-increasing in ``score``.
  (Before v5.0.2 the runtime check compared ``rrf_score or score`` against
  ``rrf_score or score`` — a field against itself — and could not fail.)
- When ``explain=False`` (the default), this module is never imported by
  hot-path code.  The envelope shape is byte-identical to v3.10.x.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["ScoreExplain", "attach_explain"]


@dataclass(frozen=True)
class ScoreExplain:
    """Per-hit score decomposition.

    Attributes:
        bm25: Raw BM25F score from the retrieval stage.  Present on every
            hit that went through the BM25 pipeline.
        vector: Raw cosine similarity from the vector backend, when the
            hybrid path ran and the backend exposed this value.  ``None``
            when vector search was not used or did not surface the value.
        rrf_rank: 1-based rank of this hit within the RRF fusion list, when
            the hybrid (BM25+vector) path ran.  ``None`` on BM25-only paths.
        governance_boost: Multiplicative factor applied by the governance
            engine.  ``0.0`` when no governance boost was applied (the
            current default — this field is reserved for future use).
        intent_match: The query intent type classified by the intent router
            (e.g. ``"factual"``, ``"temporal"``, ``"multi-hop"``).  Empty
            string when the router was not invoked.
        staleness_penalty: Subtractive or multiplicative penalty applied for
            stale blocks.  ``0.0`` when no penalty was applied (reserved for
            block-lineage staleness propagation in v3.11.0 Pattern 3).
        final: The score value used to sort this hit — read directly from
            the hit's ``score`` field, which the one-score contract makes
            the sort key at every stage exit.
    """

    bm25: float
    vector: float | None
    rrf_rank: int | None
    governance_boost: float
    intent_match: str
    staleness_penalty: float
    final: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


def attach_explain(
    results: list[dict[str, Any]],
    *,
    intent_match: str = "",
    workspace: str | None = None,
) -> list[dict[str, Any]]:
    """Inject ``_explain`` into every hit dict in-place.

    Each result dict is mutated by adding an ``_explain`` key.  The caller
    is responsible for ensuring ``results`` is already in final sorted order
    before calling this function, because ``rrf_rank`` is derived from
    position.

    Args:
        results: List of hit dicts, sorted descending by ``score``.
        intent_match: Query-level intent type string.
        workspace: When provided, persisted lineage staleness penalties
            from ``block_staleness`` are looked up per hit and surfaced
            in ``_explain.staleness_penalty``. Omit (``None``) to keep
            the v3.11.0 default of ``0.0``.

    Returns:
        The same list, with ``_explain`` injected on every element.
    """

    penalties: dict[str, float] = {}
    if workspace:
        ids = [str(h["_id"]) for h in results if h.get("_id")]
        if ids:
            from mind_mem.lineage_staleness import list_staleness_scores

            penalties = list_staleness_scores(workspace, ids)

    finals: list[tuple[int, float]] = []

    for rank_0, hit in enumerate(results):
        # ONE SCORE CONTRACT: ``score`` is the sort key at every stage exit,
        # so ``final`` reads it and nothing else. It used to read a STALE
        # ``rrf_score`` -- stale because every stage after fusion (rerank,
        # boost, decay) rewrites ``score`` and leaves ``rrf_score`` at the
        # value fusion produced, so a reranked hit reported the pre-rerank
        # number as its final.
        final_score = _as_float(hit.get("score"))

        # ``bm25`` is the BM25 leg's RAW retrieval score. On the fused path
        # that value is ``leg_scores["bm25"]``; reading ``score`` there
        # reported the FUSED number under a field documented as raw BM25F,
        # and after this change would report it under a field whose own
        # docstring says it is the retrieval-stage score.
        leg_scores = hit.get("leg_scores")
        if not isinstance(leg_scores, dict):
            leg_scores = {}
        if "bm25" in leg_scores:
            bm25_raw = _as_float(leg_scores.get("bm25"))
        elif "bm25_score" in hit:
            bm25_raw = _as_float(hit.get("bm25_score"))
        else:
            # BM25-only path: ``score`` IS the BM25F score, unfused.
            bm25_raw = final_score

        # Honest gate: non-None only when the value is present in the hit.
        # The fused path now carries it (``leg_scores``); every other path
        # still reports None rather than a synthesized number.
        vector: float | None = None
        if "vector" in leg_scores:
            vector = _as_float(leg_scores.get("vector"))

        rrf_rank: int | None = None
        if hit.get("fusion") == "rrf" or "rrf_score" in hit:
            rrf_rank = rank_0 + 1

        block_id = hit.get("_id", "")
        staleness_penalty = penalties.get(str(block_id), 0.0)

        explain = ScoreExplain(
            bm25=bm25_raw,
            vector=vector,
            rrf_rank=rrf_rank,
            governance_boost=0.0,
            intent_match=intent_match,
            staleness_penalty=staleness_penalty,
            final=round(final_score, 6),
        )

        # Retrieval-ranked hits only. The expansion stages (graph walk, KG
        # fusion, entity prefetch) APPEND their blocks after the ranked list
        # by documented design -- an appended 1-hop neighbour of the top hit
        # can legitimately outscore the last ranked hit -- so they are not
        # part of the ordering claim and are excluded rather than allowed to
        # fire a false alarm.
        if not _is_appended(hit):
            finals.append((rank_0, explain.final))

        hit["_explain"] = explain.to_dict()

    _assert_non_increasing(finals, results)

    return results


def _as_float(value: Any) -> float:
    """Coerce a score field to float, treating None/garbage as 0.0."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


#: Fields stamped by the post-ranking expansion stages. A hit carrying one
#: was APPENDED to the ranked list, not ranked into it.
_APPENDED_MARKERS = ("_graph_hop", "_kg_hop", "_prefetch")


def _is_appended(hit: dict[str, Any]) -> bool:
    return any(marker in hit for marker in _APPENDED_MARKERS)


def _assert_non_increasing(finals: list[tuple[int, float]], results: list[dict[str, Any]]) -> None:
    """Raise unless the ranked hits are non-increasing in ``score``.

    This is the invariant that replaced a tautology. The old check compared
    ``float(hit.get("rrf_score") or hit.get("score", 0.0))`` against a value
    computed by that same expression one line earlier, so it could not fail
    and never did -- it was a field compared to itself. What is actually
    worth asserting is the ONE SCORE CONTRACT: whatever stage ran last, the
    list it handed back is ordered by ``score``. A stage that reorders on
    one field and leaves another as the sort key trips this here, in the
    response path, rather than showing up as quietly worse recall.

    Tolerance is 1e-9 -- ``final`` is rounded to 6 places, so equal-scoring
    neighbours must not read as a violation.
    """
    for (prev_rank, prev_final), (rank, final) in zip(finals, finals[1:]):
        if final - prev_final > 1e-9:
            prev_id = results[prev_rank].get("_id", "?")
            this_id = results[rank].get("_id", "?")
            raise RuntimeError(
                f"recall results are not non-increasing in score: "
                f"rank {rank} ({this_id}, {final}) outranks "
                f"rank {prev_rank} ({prev_id}, {prev_final})"
            )
