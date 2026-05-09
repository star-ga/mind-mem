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
- ``_explain.final`` MUST equal the ``score`` field used to order results.
  The helper :func:`attach_explain` asserts this invariant at runtime.
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
        final: The score value used to sort this hit — must equal the hit's
            ``score`` field.
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

    for rank_0, hit in enumerate(results):
        bm25_raw = float(hit.get("score", 0.0))
        rrf_rank: int | None = None
        vector: float | None = None

        # The hybrid path injects ``rrf_score`` and ``fusion == "rrf"``
        if hit.get("fusion") == "rrf" or "rrf_score" in hit:
            rrf_rank = rank_0 + 1
            final_score = float(hit.get("rrf_score") or hit.get("score", 0.0))
        else:
            final_score = bm25_raw

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

        # Math-consistency invariant: final must equal the sort key.
        # We compare against ``rrf_score`` when present (the RRF path uses
        # that as the sort key), otherwise ``score``. Hard-raise instead
        # of ``assert`` so the invariant survives ``python -O``.
        sort_key = float(hit.get("rrf_score") or hit.get("score", 0.0))
        if abs(explain.final - sort_key) >= 1e-9:
            raise RuntimeError(
                f"_explain.final ({explain.final}) != sort key ({sort_key}) "
                f"for hit {hit.get('_id', '?')}"
            )

        hit["_explain"] = explain.to_dict()

    return results
