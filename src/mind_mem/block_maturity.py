"""Block maturity metric — consolidation gate (Group H, v4.0.x).

A *maturity score* is a 0.0–1.0 signal that indicates how "consolidated"
a memory block is.  The higher the score, the more the block has been
corroborated, reviewed, and time-tested.  The metric is used as a
**consolidation gate**: operators can surface only blocks whose maturity
meets a minimum threshold, which governs what graduates from
``ephemeral → durable`` in the lifecycle.

**Design goals**

- Optional and additive — no existing recall path is affected when
  ``min_maturity`` is not set.
- Stateless per block — the score is derived from fields already present
  in the block dict (no extra DB round-trip required for the filter).
- Extensible — the ``maturity_score`` function accepts an optional
  ``incoming_edge_count`` integer so callers that have already queried
  the lineage graph can pass in corroboration evidence without this
  module pulling in ``block_lineage``.

**Score components** (all additive, final value clamped to [0.0, 1.0]):

1. *Status contribution* (0.3 weight) — ``active`` blocks score full
   weight; ``wip`` blocks score half; ``deprecated``/``archived`` score
   zero.
2. *Lifecycle contribution* (0.2 weight) — ``durable`` blocks score
   full weight; ``generated`` blocks score half (auto-generated content
   is less mature by default); ``ephemeral`` scores zero.
3. *Explicit Maturity override* (takes precedence if present) — when the
   block frontmatter includes a ``Maturity: <float>`` field in [0, 1],
   that value is returned directly, bypassing components 1–2 and 4.
   Useful for manual curation.
4. *Incoming-edge corroboration* (0.5 weight, optional) — the fraction
   of the maximum expected incoming edges (``MATURITY_EDGE_SATURATION``,
   default 5) that have been recorded.  Each incoming ``supports`` or
   ``cites`` edge on the block boosts maturity; the contribution is
   capped when the edge count reaches the saturation value.
   Defaults to 0.0 when ``incoming_edge_count`` is ``None``.

**Usage**

    from mind_mem.block_maturity import maturity_score, apply_min_maturity_filter

    score = maturity_score(block_dict)
    filtered = apply_min_maturity_filter(hits, min_maturity=0.4)

The filter is wired into :func:`mind_mem._recall_core.recall` via the
``min_maturity`` keyword argument (default ``None`` = disabled).
"""

from __future__ import annotations

__all__ = [
    "MATURITY_EDGE_SATURATION",
    "MATURITY_LIFECYCLE_WEIGHT",
    "MATURITY_EDGE_WEIGHT",
    "MATURITY_STATUS_WEIGHT",
    "apply_min_maturity_filter",
    "maturity_score",
]

# ---------------------------------------------------------------------------
# Tuneable weights / constants
# ---------------------------------------------------------------------------

#: Weight of the *status* component in the composite score.
MATURITY_STATUS_WEIGHT: float = 0.3

#: Weight of the *lifecycle* component in the composite score.
MATURITY_LIFECYCLE_WEIGHT: float = 0.2

#: Weight of the *incoming-edge corroboration* component.
MATURITY_EDGE_WEIGHT: float = 0.5

#: Number of incoming edges at which the edge component saturates at 1.0.
MATURITY_EDGE_SATURATION: int = 5


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def _status_component(block: dict) -> float:
    """Return a [0.0, 1.0] contribution based on the block Status field."""
    raw = block.get("Status") or block.get("status") or ""
    s = str(raw).strip().lower()
    if s == "active":
        return 1.0
    if s in ("wip", "in-progress", "in_progress"):
        return 0.5
    # deprecated, archived, rejected, unknown → 0
    return 0.0


def _lifecycle_component(block: dict) -> float:
    """Return a [0.0, 1.0] contribution based on the block Lifecycle field.

    Blocks without a Lifecycle field default to ``"durable"`` (the existing
    implicit assumption), matching the behaviour of the lifecycle recall filter.
    """
    raw = block.get("Lifecycle") or block.get("lifecycle") or "durable"
    lc = str(raw).strip().lower()
    if lc == "durable":
        return 1.0
    if lc == "generated":
        return 0.5
    # ephemeral → 0 (session-scoped hints are not consolidated knowledge)
    return 0.0


def _edge_component(incoming_edge_count: int | None) -> float:
    """Return a [0.0, 1.0] edge-corroboration contribution.

    When ``incoming_edge_count`` is ``None`` the caller did not supply
    lineage information; the component defaults to ``0.0`` (conservative).
    """
    if incoming_edge_count is None or incoming_edge_count <= 0:
        return 0.0
    return min(1.0, incoming_edge_count / MATURITY_EDGE_SATURATION)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def maturity_score(
    block: dict,
    *,
    incoming_edge_count: int | None = None,
) -> float:
    """Compute a maturity score in [0.0, 1.0] for *block*.

    Args:
        block: A block dict as returned by the recall pipeline (or parsed
            by ``block_parser``).  Recognised keys: ``Maturity``,
            ``Status``, ``Lifecycle``.
        incoming_edge_count: Optional number of incoming ``supports`` /
            ``cites`` lineage edges for this block.  When ``None`` the
            edge component contributes ``0.0``.

    Returns:
        A float in ``[0.0, 1.0]``.  Higher = more consolidated.
    """
    # --- Explicit override: frontmatter Maturity field ---
    raw_override = block.get("Maturity") or block.get("maturity")
    if raw_override is not None:
        try:
            v = float(raw_override)
            return max(0.0, min(1.0, v))
        except (ValueError, TypeError):
            pass  # fall through to computed score

    # --- Composite: status + lifecycle + edge corroboration ---
    s = _status_component(block) * MATURITY_STATUS_WEIGHT
    lc = _lifecycle_component(block) * MATURITY_LIFECYCLE_WEIGHT
    e = _edge_component(incoming_edge_count) * MATURITY_EDGE_WEIGHT
    return min(1.0, s + lc + e)


def apply_min_maturity_filter(
    hits: list[dict],
    min_maturity: float,
) -> list[dict]:
    """Return only hits whose maturity score meets *min_maturity*.

    This is a **post-rank filter**: it preserves the relative ordering of
    the hits that pass the threshold; only hits below the threshold are
    dropped.

    A ``Maturity`` field present in the block dict is used directly (i.e.
    a block that has already been curated with an explicit maturity value
    will be filtered against that value, not a computed approximation).

    Edge-corroboration is *not* considered here (the filter runs after
    the recall pipeline, which does not carry lineage edge counts).  If
    operators need edge-aware maturity filtering, they should compute
    :func:`maturity_score` with ``incoming_edge_count`` and then call
    this helper with the result injected into the ``Maturity`` field.

    Args:
        hits: Scored recall results (list of dicts).
        min_maturity: Minimum maturity threshold in [0.0, 1.0].  Hits
            whose computed score is strictly below this value are removed.

    Returns:
        Subset of *hits* passing the threshold, in original order.
    """
    threshold = float(min_maturity)
    result = []
    for h in hits:
        score = maturity_score(h)
        if score >= threshold:
            result.append(h)
    return result
