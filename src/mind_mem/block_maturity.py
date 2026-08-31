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
- Extensible — the ``maturity_score`` function accepts optional
  ``incoming_edge_count`` and ``distinct_project_count`` integers so
  callers that have already queried the lineage graph can pass in
  corroboration evidence without this module pulling in
  ``block_lineage``.  That dependency direction is a hard constraint,
  not a preference: it is what keeps the score stateless.

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
5. *Corroboration breadth* (0.15 weight, optional) — the fraction of
   ``MATURITY_PROJECT_SATURATION`` (default 3) reached by the number of
   **distinct originating projects** among those incoming edges.
   Defaults to 0.0 when ``distinct_project_count`` is ``None``.

**Why breadth is not just more edges.**  The edge component counts *how
many* incoming edges a block has, never *where they came from*.  Five
edges asserted inside a single repository during one campaign score
exactly the same as five arriving from five independent contexts, though
the first is closer to one observation restated than to five
observations.  Breadth measures independence, which is why it saturates
so much faster than the edge count does: one project to two is the whole
finding, four to five is noise.

**Two weight profiles, and why.**  Weights are *rebalanced*, never
appended — a fifth weight stacked on top of four that already sum to 1.0
would make the final clamp load-bearing and silently compress the
existing components.  So breadth's weight comes out of the edge
component's rather than being added to it, and which profile applies is
decided by whether the caller supplied breadth at all:

===================================  ======  =========  ======  =======
profile                              status  lifecycle  edge    breadth
===================================  ======  =========  ======  =======
``distinct_project_count is None``   0.3     0.2        0.5     --
breadth supplied                     0.3     0.2        0.35    0.15
===================================  ======  =========  ======  =======

Both rows sum to exactly 1.0.  The first row is the pre-breadth
behaviour bit for bit, so a caller that does not know the breadth of a
block's corroboration is scored exactly as it was before this component
existed — it is not quietly penalised for holding data the graph could
not give it.  A caller that *does* supply breadth opts into the
rebalanced profile for that call, and nothing else moves.

**Usage**

    from mind_mem.block_maturity import maturity_score, apply_min_maturity_filter

    score = maturity_score(block_dict)
    filtered = apply_min_maturity_filter(hits, min_maturity=0.4)

    # Breadth-aware: the caller queried the lineage graph itself.
    score = maturity_score(block_dict, incoming_edge_count=5, distinct_project_count=3)

The filter is wired into :func:`mind_mem._recall_core.recall` via the
``min_maturity`` keyword argument (default ``None`` = disabled).
:func:`apply_min_maturity_filter` is deliberately breadth-blind — see
its docstring.
"""

from __future__ import annotations

__all__ = [
    "MATURITY_EDGE_SATURATION",
    "MATURITY_LIFECYCLE_WEIGHT",
    "MATURITY_EDGE_WEIGHT",
    "MATURITY_EDGE_WEIGHT_WITH_BREADTH",
    "MATURITY_PROJECT_SATURATION",
    "MATURITY_PROJECT_WEIGHT",
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

#: Weight of the *incoming-edge corroboration* component when the caller
#: supplies no corroboration breadth.
MATURITY_EDGE_WEIGHT: float = 0.5

#: Number of incoming edges at which the edge component saturates at 1.0.
MATURITY_EDGE_SATURATION: int = 5

#: Weight of the *incoming-edge corroboration* component when the caller
#: *does* supply breadth.  Taken out of :data:`MATURITY_EDGE_WEIGHT`, not
#: added alongside it: breadth refines the same corroboration signal, so
#: the two together must still weigh exactly what the edge count weighed
#: alone, leaving the composite sum at 1.0 and the final clamp inert.
MATURITY_EDGE_WEIGHT_WITH_BREADTH: float = 0.35

#: Weight of the *corroboration breadth* component (unused unless the
#: caller supplies ``distinct_project_count``).
MATURITY_PROJECT_WEIGHT: float = 0.15

#: Number of distinct originating projects at which breadth saturates at
#: 1.0.  Deliberately far below :data:`MATURITY_EDGE_SATURATION`: the
#: signal is independence, and independence saturates fast.
MATURITY_PROJECT_SATURATION: int = 3


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


def _project_component(distinct_project_count: int | None) -> float:
    """Return a [0.0, 1.0] corroboration-breadth contribution.

    When ``distinct_project_count`` is ``None`` the caller did not supply
    provenance breadth; the component defaults to ``0.0`` (conservative),
    exactly as the edge component does.
    """
    if distinct_project_count is None or distinct_project_count <= 0:
        return 0.0
    return min(1.0, distinct_project_count / MATURITY_PROJECT_SATURATION)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def maturity_score(
    block: dict,
    *,
    incoming_edge_count: int | None = None,
    distinct_project_count: int | None = None,
) -> float:
    """Compute a maturity score in [0.0, 1.0] for *block*.

    Args:
        block: A block dict as returned by the recall pipeline (or parsed
            by ``block_parser``).  Recognised keys: ``Maturity``,
            ``Status``, ``Lifecycle``.
        incoming_edge_count: Optional number of incoming ``supports`` /
            ``cites`` lineage edges for this block.  When ``None`` the
            edge component contributes ``0.0``.
        distinct_project_count: Optional number of **distinct originating
            projects** among those incoming edges — see
            :func:`mind_mem.block_lineage.distinct_project_counts`, which
            the caller queries; this module never does.  When ``None``
            the breadth component contributes ``0.0`` *and* the edge
            component keeps its full pre-breadth weight, so the score is
            identical to what it was before breadth existed.  Supplying
            it selects the rebalanced profile documented in the module
            docstring.

    Returns:
        A float in ``[0.0, 1.0]``.  Higher = more consolidated.
    """
    # --- Explicit override: frontmatter Maturity field ---
    # Use a sentinel to distinguish "key absent" from "key present but falsy
    # (e.g. Maturity=0 / Maturity=0.0)".  A plain `or` would silently skip
    # a legitimate zero value because 0 / 0.0 are falsy in Python.
    _MISSING = object()
    raw_override = block.get("Maturity", _MISSING)
    if raw_override is _MISSING:
        raw_override = block.get("maturity", _MISSING)
    if raw_override is not _MISSING:
        try:
            v = float(raw_override)
            return max(0.0, min(1.0, v))
        except (ValueError, TypeError):
            pass  # fall through to computed score

    # --- Composite: status + lifecycle + corroboration (count, breadth) ---
    # The corroboration budget is fixed at MATURITY_EDGE_WEIGHT and is split
    # between count and breadth only when the caller knows the breadth; a
    # caller that does not spends the whole budget on the count, which is
    # what every pre-breadth call site does and why their scores do not move.
    if distinct_project_count is None:
        edge_weight = MATURITY_EDGE_WEIGHT
        project_weight = 0.0
    else:
        edge_weight = MATURITY_EDGE_WEIGHT_WITH_BREADTH
        project_weight = MATURITY_PROJECT_WEIGHT

    s = _status_component(block) * MATURITY_STATUS_WEIGHT
    lc = _lifecycle_component(block) * MATURITY_LIFECYCLE_WEIGHT
    e = _edge_component(incoming_edge_count) * edge_weight
    p = _project_component(distinct_project_count) * project_weight
    return min(1.0, s + lc + e + p)


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

    Edge-corroboration and corroboration breadth are *not* considered
    here (the filter runs after the recall pipeline, which carries
    neither lineage edge counts nor edge provenance).  Both stay out on
    purpose: widening a post-rank filter to fetch lineage would turn it
    into a second query path.  If operators need corroboration-aware
    maturity filtering, they should compute :func:`maturity_score` with
    ``incoming_edge_count`` (and ``distinct_project_count``) and then
    call this helper with the result injected into the ``Maturity``
    field — the same escape hatch covers both.

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
