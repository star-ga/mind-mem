"""Multi-hop graph traversal for recall (v3.3.0 Tier 1 #2).

Given an initial set of retrieved blocks, walk the cross-reference
graph up to ``max_hops`` steps and append newly-discovered blocks
with a decayed score. Fuses multi-hop evidence into the same result
set that the BM25/hybrid pipeline produces.

The graph is built from the existing ``build_xref_graph`` helper in
``_recall_scoring`` — no new ingestion required. Block IDs mentioned
anywhere in a block's text field become edges; ``supersedes``,
``supersededBy``, ``relates_to`` fields are treated as explicit edges.

Opt-in via:

    {
      "retrieval": {
        "multi_hop": {
          "enabled": false,
          "auto_enable": true,
          "max_hops": 2,
          "decay": 0.5,
          "max_neighbors_per_hop": 5,
          "uncertainty": {
            "enabled": false,
            "hop_confidence": 0.9,
            "min_confidence": 0.1
          }
        }
      }
    }

Auto-enabled for multi-hop query types (per ``detect_query_type``)
unless ``auto_enable`` is explicitly false.

``retrieval.multi_hop.uncertainty`` (default OFF) additionally routes
every candidate neighbour through
:class:`~mind_mem.uncertainty_propagation.UncertaintyPropagator`. Without
it a hop-3 block is appended with no confidence at all, so a consumer
reads it exactly as trustworthy as a hop-1 block. With it each appended
block carries ``_hop_confidence`` — the parent-chained, decayed
confidence — and branches that fall below ``min_confidence`` are pruned
mid-walk instead of being returned as if they were direct hits.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

from .feature_gate import FeatureGate, FieldSpec, multi_hop_detector, strict_int, strict_number
from .observability import get_logger

_log = get_logger("graph_recall")


_DEFAULT_MAX_TOTAL_ADDED = 50

# Uncertainty-propagation defaults (only consulted when the
# ``retrieval.multi_hop.uncertainty`` sub-block turns the surface on).
#: Raw confidence credited to a neighbour discovered across one xref edge.
#: The block-xref graph carries no per-edge weights, so every edge is worth
#: the same before parent-chaining and decay are applied.
_DEFAULT_HOP_CONFIDENCE = 0.9
#: Adjusted confidence below which a branch stops being walked at all.
_DEFAULT_MIN_CONFIDENCE = 0.1
#: Per-hop decay for weighted causal chains (the propagator's own default).
_DEFAULT_CHAIN_DECAY = 0.85

#: Stand-in id for a candidate's already-adjusted parent when a two-element
#: chain is handed to ``propagate()``. Contains a NUL byte, so it can never
#: collide with a real block id (which must match ``^[A-Z]+-...``).
_PARENT_SENTINEL = "\x00uncertainty-parent"


def _seed_confidence(result: dict) -> float:
    """Confidence a seed enters the walk with, clamped to ``[0.0, 1.0]``.

    A seed is a direct retrieval hit — hop 0 — so it is fully trusted at
    1.0 unless an earlier stage already stamped ``_hop_confidence`` on it
    (chained expansions: entity prefetch feeding graph expansion). Only
    that private field is honoured; a block's own ``confidence`` field
    means edge/extraction confidence elsewhere in the codebase and must
    not be silently reinterpreted as retrieval confidence.
    """
    raw = result.get("_hop_confidence")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 1.0
    return max(0.0, min(1.0, float(raw)))


def graph_expand(
    results: list[dict],
    all_blocks: list[dict],
    *,
    max_hops: int = 2,
    decay: float = 0.5,
    max_neighbors_per_hop: int = 5,
    max_total_added: int = _DEFAULT_MAX_TOTAL_ADDED,
    score_field: str = "score",
    uncertainty: bool = False,
    hop_confidence: float = _DEFAULT_HOP_CONFIDENCE,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> list[dict]:
    """Walk the block cross-reference graph from each seed.

    Args:
        results: Initial ranked recall results. Each must carry
            ``_id`` and a numeric ``score_field``.
        all_blocks: Full block corpus (needed to build the xref graph
            and resolve neighbour-ID → block-dict). Not walked by ID;
            the function reads ``build_xref_graph`` once.
        max_hops: Maximum graph distance from seed blocks.
        decay: Multiplicative score decay per hop (``score_at_hop_n =
            seed_score * decay ** n``).
        max_neighbors_per_hop: Cap on how many new blocks are added
            per seed per hop — guards against pathological fan-out.
        score_field: Field used for seed scores + new-block scores.
        uncertainty: Opt-in (default OFF). When True every candidate
            neighbour's confidence is computed by
            :meth:`UncertaintyPropagator.propagate` from its *immediate*
            parent's already-adjusted confidence, and a branch whose
            adjusted confidence falls under ``min_confidence`` is pruned
            by :meth:`UncertaintyPropagator.should_truncate` — the node
            is neither appended nor walked further. OFF, this function
            behaves exactly as it did before the flag existed: no extra
            field, no pruning, byte-identical output.
        hop_confidence: Raw confidence credited to crossing one xref
            edge, before parent-chaining and decay. Ignored when
            ``uncertainty`` is False.
        min_confidence: Prune threshold for adjusted confidence.
            Ignored when ``uncertainty`` is False.

    Returns:
        ``results`` with appended graph-walked blocks (new entries
        carry ``_graph_hop`` and ``_graph_parent`` fields, plus
        ``_hop_confidence`` when ``uncertainty`` is on). Order
        preserved: original results first, newly-added blocks after,
        sorted by descending score within the appended block.
    """
    if not results or not all_blocks or max_hops <= 0:
        return results

    from ._recall_scoring import build_xref_graph
    from .admissibility import admit_corpus

    # Only admissible blocks are resolvable. This walk appends raw corpus
    # blocks straight into the result list, so a withheld block reachable
    # here is a withheld block served — filtering the corpus is what makes
    # it unreachable rather than merely unwanted.
    all_blocks = admit_corpus(all_blocks)

    id_to_block: dict[str, dict] = {str(b.get("_id")): b for b in all_blocks if b.get("_id")}
    if not id_to_block:
        return results

    graph = build_xref_graph(all_blocks)
    seen_ids: set[str] = {str(r.get("_id")) for r in results if r.get("_id")}

    # Opt-in multi-hop uncertainty. The propagator shares ``decay`` with the
    # score walk so a single knob governs both how fast a hop loses rank and
    # how fast it loses trust. Constructed once; it holds no state between
    # calls and reads no clock, so the walk stays a pure function of
    # (results, corpus, params).
    propagator = None
    if uncertainty:
        from .uncertainty_propagation import HopResult, UncertaintyPropagator

        propagator = UncertaintyPropagator(decay_factor=decay)

    appended: list[dict] = []
    # BFS frontier: (block_id, ORIGINAL seed_score, hop, parent_id).
    # The "seed_score" stays constant across hops — decay is computed
    # relative to the original seed so a 2-hop node gets decay**2, not
    # decay**2 * decay (which would compound on the already-decayed score).
    # deque.popleft() is O(1); a plain list's pop(0) is O(N) and
    # degrades on wide seed sets — review by python-reviewer (2026-04-20).
    # The 5th element is the node's own adjusted confidence, threaded so a
    # child discounts against its IMMEDIATE parent. ``_graph_parent`` keeps
    # naming the original seed — that field is a provenance label consumers
    # already read, and re-pointing it would be a silent contract change.
    frontier: deque[tuple[str, float, int, str, float]] = deque()
    for r in results:
        bid = str(r.get("_id") or "")
        if not bid:
            continue
        base_score = float(r.get(score_field, 0.0) or 0.0)
        frontier.append((bid, base_score, 0, bid, _seed_confidence(r)))

    while frontier:
        if len(appended) >= max_total_added:
            # Security guard — unbounded graph (adversarial xref chains)
            # could otherwise grow the result set arbitrarily. Cap across
            # ALL hops, not just per-hop (security review 2026-04-20).
            _log.info("graph_expand_total_cap_hit", cap=max_total_added)
            break
        bid, seed_score, hop, parent, parent_conf = frontier.popleft()
        if hop >= max_hops:
            continue
        neighbors: Iterable[str] = graph.get(bid, set())
        added_this_hop = 0
        for nid in neighbors:
            if nid in seen_ids:
                continue
            if len(appended) >= max_total_added:
                break
            block = id_to_block.get(nid)
            if block is None:  # pragma: no cover — unreachable by construction
                # Not counted as an unresolved id, because it cannot be one:
                # ``build_xref_graph`` only emits edges *between* ids present
                # in the list it was given, and that is the same filtered
                # list ``id_to_block`` indexes. A cross-reference to a
                # withheld or absent block never becomes an edge. The guard
                # stays as a structural assertion, not a live path.
                continue
            next_hop = hop + 1
            child_conf = parent_conf
            if propagator is not None:
                # A two-element chain — the parent collapsed to a root
                # carrying its already-adjusted confidence, then this
                # candidate — is exactly the recursion ``propagate`` runs
                # over a full chain, so the propagator (not this loop) owns
                # the arithmetic while the walk stays O(1) per neighbour.
                adjusted = propagator.propagate(
                    [
                        HopResult(
                            block_id=_PARENT_SENTINEL,
                            content="",
                            confidence=parent_conf,
                            hop_depth=hop,
                            parent_hop_id=None,
                        ),
                        HopResult(
                            block_id=nid,
                            content="",
                            confidence=hop_confidence,
                            hop_depth=next_hop,
                            parent_hop_id=_PARENT_SENTINEL,
                        ),
                    ]
                )[1]
                if propagator.should_truncate(adjusted, min_confidence):
                    # Prune the whole branch: not appended, not enqueued,
                    # and deliberately NOT marked seen — a shorter, more
                    # confident route to the same block may still reach it.
                    _log.debug(
                        "graph_expand_branch_truncated",
                        block_id=nid,
                        hop=next_hop,
                        confidence=adjusted.confidence,
                        min_confidence=min_confidence,
                    )
                    continue
                child_conf = adjusted.confidence
            decayed = seed_score * (decay**next_hop)
            new = dict(block)
            new[score_field] = decayed
            new["_graph_hop"] = next_hop
            new["_graph_parent"] = parent
            if propagator is not None:
                new["_hop_confidence"] = child_conf
            appended.append(new)
            seen_ids.add(nid)
            # Pass the ORIGINAL seed_score so deeper hops compute
            # decay ** hop against the starting score, not the decayed.
            frontier.append((nid, seed_score, next_hop, parent, child_conf))
            added_this_hop += 1
            if added_this_hop >= max_neighbors_per_hop:
                break

    if appended:
        _log.info(
            "graph_expanded",
            seeds=len(results),
            added=len(appended),
            max_hops=max_hops,
        )
        # Keep appended list sorted so the first graph-walked neighbour
        # (highest decayed score) appears before weaker ones.
        appended.sort(key=lambda b: b.get(score_field, 0.0), reverse=True)
    return list(results) + appended


#: The ``retrieval.multi_hop`` gate, declared once.
#:
#: This module used to carry its own copy of the enable/resolve pair every
#: v3.3.0 retrieval feature shipped. The copies had drifted (bounds checked
#: in one, skipped in another), so the shared resolver in
#: :mod:`mind_mem.feature_gate` now owns the mechanism and this declaration
#: owns the policy. ``implicit_section=True`` keeps the historical reading of
#: a config that has a ``retrieval`` block but no ``multi_hop`` key: the old
#: ``retrieval.get("multi_hop", {})`` treated that as an empty section, so
#: auto-enable still fired on a multi-hop query. The ceiling of 3 on
#: ``max_hops`` (config-driven-DoS guard, security review 2026-04-20) rides
#: in the coercion so it survives the move.
GRAPH_EXPAND_GATE = FeatureGate(
    name="multi_hop",
    fields={
        "max_hops": FieldSpec(
            default=2,
            coerce=lambda v: min(3, strict_int(v)),
            validate=lambda v: v > 0,
        ),
        "decay": FieldSpec(
            default=0.5,
            coerce=strict_number,
            validate=lambda v: 0 < v <= 1,
        ),
        "max_neighbors_per_hop": FieldSpec(
            default=5,
            coerce=strict_int,
            validate=lambda v: v > 0,
        ),
    },
    auto_detector=multi_hop_detector,
    implicit_section=True,
)


def is_graph_expand_enabled(
    config: dict[str, Any] | None,
    query: str | None = None,
) -> bool:
    """Resolve whether graph expansion should fire for this call.

    Priority:
      1. ``retrieval.multi_hop.enabled: true`` — always on.
      2. ``retrieval.multi_hop.auto_enable: false`` — always off.
      3. Auto-enable when the query classifies as multi-hop.

    Delegates to :data:`GRAPH_EXPAND_GATE`; the priority order above is
    the gate's, not a second implementation of it.
    """
    return GRAPH_EXPAND_GATE.is_enabled(config, query=query)


def _uncertainty_block(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return the ``retrieval.multi_hop.uncertainty`` sub-block, or ``{}``."""
    if not config or not isinstance(config, dict):
        return {}
    retrieval = config.get("retrieval")
    if not isinstance(retrieval, dict):
        return {}
    mh = retrieval.get("multi_hop")
    if not isinstance(mh, dict):
        return {}
    unc = mh.get("uncertainty")
    return unc if isinstance(unc, dict) else {}


def is_uncertainty_enabled(config: dict[str, Any] | None) -> bool:
    """Whether multi-hop uncertainty propagation is switched on.

    Fail-closed: anything other than a literal ``true`` under
    ``retrieval.multi_hop.uncertainty.enabled`` leaves the surface off,
    so a typo cannot turn it on. This is the single gate shared by the
    recall walk (:func:`graph_expand`) and the ``traverse_graph`` MCP
    envelope — one flag, one meaning.
    """
    return _uncertainty_block(config).get("enabled") is True


def _confidence_param(block: dict[str, Any], key: str, default: float) -> float:
    """Read a ``[0.0, 1.0]`` confidence knob, falling back on anything else."""
    raw = block.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return default
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        return default
    return value


def resolve_uncertainty_params(config: dict[str, Any] | None) -> dict[str, float]:
    """Resolve ``hop_confidence`` / ``min_confidence`` from config.

    Callers should gate on :func:`is_uncertainty_enabled` first; this only
    reads the tuning knobs and never decides whether the surface is on.
    """
    block = _uncertainty_block(config)
    return {
        "hop_confidence": _confidence_param(block, "hop_confidence", _DEFAULT_HOP_CONFIDENCE),
        "min_confidence": _confidence_param(block, "min_confidence", _DEFAULT_MIN_CONFIDENCE),
    }


def resolve_chain_decay(config: dict[str, Any] | None) -> float:
    """Per-edge trust decay for the causal-chain envelope.

    Kept apart from ``retrieval.multi_hop.decay`` on purpose. That knob is
    the recall walk's *rank* decay, and :func:`graph_expand` reuses it so a
    hop loses rank and trust in lockstep over a weightless xref graph. A
    causal chain is different: its edges carry real weights, so the decay
    here is only the residual per-hop discount and defaults to the
    propagator's own 0.85 rather than the much harsher recall default.

    Returned as a plain float (never folded into
    :func:`resolve_graph_config`, whose result is splatted straight into
    ``graph_expand`` and must not grow keys that function does not accept).
    """
    return _confidence_param(_uncertainty_block(config), "chain_decay", _DEFAULT_CHAIN_DECAY)


def resolve_graph_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Extract graph-expansion parameters from config with defaults.

    The three walk knobs come from :data:`GRAPH_EXPAND_GATE`. The
    uncertainty keys stay hand-read here because they live one level
    deeper (``retrieval.multi_hop.uncertainty``) than the one section a
    gate addresses, and they are emitted **only** when that surface is
    enabled — so the flag-off return value is exactly the three-key dict
    it has always been and ``graph_expand`` is called with its historical
    argument set.
    """
    out: dict[str, Any] = GRAPH_EXPAND_GATE.resolve(config)
    if is_uncertainty_enabled(config):
        out["uncertainty"] = True
        out.update(resolve_uncertainty_params(config))
    return out


__all__ = [
    "GRAPH_EXPAND_GATE",
    "graph_expand",
    "is_graph_expand_enabled",
    "is_uncertainty_enabled",
    "resolve_chain_decay",
    "resolve_graph_config",
    "resolve_uncertainty_params",
]
