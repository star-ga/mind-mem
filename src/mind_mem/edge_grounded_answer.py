# Copyright 2026 STARGA, Inc.
"""Edge-grounded answering — cite the edge, and say what is missing.

The evidence-chain side of this product can already prove what a recall
*served*. The construction side could not: a governed edge existed, was
approved, carried a receipt and a schema stamp — and nothing could turn a
question into an answer that pointed at those edges. ``graph_query``
returns rows; rows are not an answer, and an answer with no citation is
indistinguishable from an invention.

This module is the missing half, in three pieces:

1. :func:`build_context` walks the k-hop subgraph (k=2 by default) around
   a seed entity and serialises it as triples, each one carrying the id
   of the edge it came from, that edge's ``source_block_id``, its origin
   marker and its schema stamp. Read-only by construction — it goes
   through :meth:`KnowledgeGraph.edges_of`, which resolves through
   ``lookup`` and therefore cannot mint an entity as a side effect of
   being asked a question.

2. Every omission is *named*. A seed the registry has never heard of, a
   predicate with no edges, a traversal that hit the hop or triple cap,
   edges withheld by their validity window, edges with no schema stamp,
   and citations whose provenance block is not in the corpus all come
   back as :class:`Gap` entries. "The graph does not contain this" is an
   answer, and the failure mode of every graph-answer system is
   presenting a thin subgraph as a complete one.

3. :func:`answer` runs a generator that may see **only** the serialised
   triples, then checks every ``[[E-…]]`` it emitted against the ids
   actually served. A citation to an edge that was not in the context is
   a fabrication: it is reported, and the answer is marked not grounded.
   The generator is injected, so the whole path proves out in CI with
   zero model calls; with no generator bound the answer *is* the cited
   triple list, which is the honest thing to return rather than prose
   nobody produced.

Replay: pass ``as_of`` and the validity filter is evaluated against that
timestamp instead of the wall clock, so an answer can be reproduced
exactly. Ordering is deterministic everywhere (confidence desc, then
lexicographic), so two runs over one graph serialise byte-identically.

Stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional, Sequence

from .graph_schema import version_of as schema_version_of
from .knowledge_graph import Corroboration, Edge, KnowledgeGraph, Predicate, _parse_iso8601, edge_id

#: How a claim cites its supporting edge. Matches :func:`edge_id`'s output
#: (``E-`` + 16 hex) so a citation cannot name anything the graph could
#: not have produced.
CITATION_RE = re.compile(r"\[\[(E-[0-9a-f]{16})\]\]")

#: Default hop radius. The roadmap's k=2: far enough to relate two
#: entities through a shared neighbour, near enough that the serialised
#: context stays reviewable.
DEFAULT_HOPS = 2

#: Hard ceiling on serialised triples. A cap that is hit is reported as a
#: gap rather than silently truncating the evidence.
DEFAULT_MAX_TRIPLES = 128

# Gap kinds — a closed vocabulary so a consumer can branch on them.
GAP_UNKNOWN_ENTITY = "unknown_entity"
GAP_NO_EDGES = "no_edges"
GAP_PREDICATE_ABSENT = "predicate_absent"
GAP_HOP_LIMIT = "hop_limit"
GAP_TRIPLE_CAP = "triple_cap"
GAP_EXPIRED_WITHHELD = "expired_withheld"
GAP_UNVERSIONED_EDGE = "unversioned_edge"
GAP_PROVENANCE_MISSING = "provenance_missing"
GAP_FABRICATED_CITATION = "fabricated_citation"
GAP_UNCITED_ANSWER = "uncited_answer"
GAP_SINGLE_SOURCE_CLAIM = "single_source_claim"


@dataclass(frozen=True)
class Gap:
    """One thing the served subgraph does **not** establish."""

    kind: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


@dataclass(frozen=True)
class GroundedTriple:
    """One served edge, with everything a claim needs to cite it."""

    subject: str
    predicate: str
    object: str
    hop: int
    edge_id: str
    source_block_id: str
    confidence: float
    origin: Optional[str]
    schema_version: Optional[str]
    valid_from: Optional[str]
    valid_until: Optional[str]
    sources: int = 1
    corroborated_confidence: float = 0.0
    corroborating_blocks: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "hop": self.hop,
            "edge_id": self.edge_id,
            "source_block_id": self.source_block_id,
            "confidence": round(self.confidence, 6),
            "origin": self.origin,
            "schema_version": self.schema_version,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "sources": self.sources,
            "corroborated_confidence": self.corroborated_confidence,
            "corroborating_blocks": list(self.corroborating_blocks),
        }

    def as_line(self) -> str:
        """One tab-separated line: the unit a generator is allowed to use.

        Carries the corroboration count, because a generator that cannot
        see that one claim rests on three independent blocks and another
        on one will weight them the same.
        """
        return (
            f"{self.subject}\t{self.predicate}\t{self.object}\t[[{self.edge_id}]]"
            f"\tsrc={self.source_block_id}\tsources={self.sources}"
            f"\tconf={self.corroborated_confidence:g}"
        )


@dataclass(frozen=True)
class EdgeGroundedContext:
    """The k-hop subgraph a claim may be made from, plus its holes."""

    seed: str
    seed_entity_id: Optional[str]
    hops: int
    triples: tuple[GroundedTriple, ...]
    gaps: tuple[Gap, ...]

    @property
    def ranked_triples(self) -> tuple[GroundedTriple, ...]:
        """The served triples ordered by evidential weight, best first.

        ``triples`` keeps traversal (hop) order, which is the *shape* of
        the subgraph; this is the *weighting* — corroborated confidence
        desc, then source count desc, then lexicographic so the order is
        total and reproducible. A consumer that treats every edge as equal
        now has to do so on purpose.
        """
        return tuple(
            sorted(
                self.triples,
                key=lambda t: (
                    -t.corroborated_confidence,
                    -t.sources,
                    t.subject,
                    t.predicate,
                    t.object,
                    t.edge_id,
                ),
            )
        )

    @property
    def edge_ids(self) -> frozenset[str]:
        """Exactly the ids a claim is allowed to cite."""
        return frozenset(t.edge_id for t in self.triples)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "seed_entity_id": self.seed_entity_id,
            "hops": self.hops,
            "triples": [t.as_dict() for t in self.triples],
            "gaps": [g.as_dict() for g in self.gaps],
        }

    def serialize(self) -> str:
        """Deterministic text handed to a generator — and nothing else.

        The gap list is part of it on purpose: a generator that cannot see
        what is missing will fill the hole itself.
        """
        header = f"# edge-grounded context: seed={self.seed_entity_id or self.seed} hops={self.hops} triples={len(self.triples)}"
        lines = [header, "# answer ONLY from the triples below; cite each claim as [[E-...]]", "# triples"]
        if self.triples:
            lines.extend(t.as_line() for t in self.triples)
        else:
            lines.append("(none)")
        lines.append("# gaps — the graph does NOT establish these")
        if self.gaps:
            lines.extend(f"- {g.kind}: {g.detail}" for g in self.gaps)
        else:
            lines.append("(none)")
        return "\n".join(lines)


@dataclass(frozen=True)
class GroundedAnswer:
    """An answer plus the proof that it stayed inside the served edges."""

    context: EdgeGroundedContext
    text: str
    citations: tuple[str, ...]
    fabricated_citations: tuple[str, ...]
    grounded: bool
    generator: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": list(self.citations),
            "fabricated_citations": list(self.fabricated_citations),
            "grounded": self.grounded,
            "generator": self.generator,
            "context": self.context.as_dict(),
        }


def _edge_identity(edge: Edge) -> str:
    """The receipt id of a stored edge, recomputed from its own tuple."""
    return edge_id(edge.subject, edge.predicate, edge.object, edge.source_block_id)


def _is_live(edge: Edge, *, as_of: Optional[datetime]) -> bool:
    """Whether *edge* is inside its validity window at *as_of*.

    ``as_of=None`` means "now". A malformed ``valid_until`` counts as
    expired, matching ``_query_edges``: corrupt data must not be able to
    keep a stale claim alive.
    """
    if edge.valid_until is None:
        return True
    moment = as_of if as_of is not None else datetime.now(timezone.utc)
    try:
        return _parse_iso8601(edge.valid_until) >= moment
    except ValueError:
        return False


def build_context(
    kg: KnowledgeGraph,
    seed: str,
    *,
    hops: int = DEFAULT_HOPS,
    predicates: Optional[Sequence[str]] = None,
    direction: str = "both",
    max_triples: int = DEFAULT_MAX_TRIPLES,
    include_expired: bool = False,
    as_of: Optional[str] = None,
    known_block_ids: Optional[Iterable[str]] = None,
) -> EdgeGroundedContext:
    """Serialise the k-hop subgraph around *seed* as citable triples.

    Args:
        kg: The graph to read. Never written.
        seed: Any surface form of the starting entity; resolved through
            the registry's read-only ``lookup``.
        hops: Radius, clamped to ``[1, 8]`` (the traversal cap the graph
            layer already enforces).
        predicates: Optional predicate filter. A named predicate that
            yields nothing becomes a :data:`GAP_PREDICATE_ABSENT` gap
            rather than vanishing.
        direction: ``"outgoing"`` / ``"incoming"`` / ``"both"``.
        max_triples: Cap on served triples; hitting it is reported.
        include_expired: Serve edges past ``valid_until`` too.
        as_of: ISO-8601 instant the validity filter is evaluated against.
            Supplying it makes the answer replayable; omitting it reads
            the wall clock, exactly as every other graph read does.
        known_block_ids: When given, any cited ``source_block_id`` outside
            this set is reported as :data:`GAP_PROVENANCE_MISSING` — a
            citation whose document is gone is not a citation.

    Returns:
        An :class:`EdgeGroundedContext`. Deterministic for a fixed graph,
        seed and ``as_of``.
    """
    hops = max(1, min(int(hops), 8))
    max_triples = max(1, int(max_triples))
    moment = _parse_iso8601(as_of) if as_of else None
    wanted_predicates: list[str] = []
    if predicates:
        wanted_predicates = sorted({Predicate.from_str(p).value for p in predicates})

    gaps: list[Gap] = []
    seed_id = kg.entities.lookup(seed)
    if seed_id is None:
        gaps.append(Gap(GAP_UNKNOWN_ENTITY, f"the registry has no entity for {seed!r}"))
        return EdgeGroundedContext(seed=seed, seed_entity_id=None, hops=hops, triples=(), gaps=tuple(gaps))

    # One pass over the edges table, before the walk: corroboration is a
    # property of the CLAIM (subject, predicate, object) across every
    # block that asserts it, so it cannot be read off the single row the
    # traversal happens to reach.
    corroboration = kg.corroboration_index()
    triples: list[GroundedTriple] = []
    served: set[str] = set()
    seen_nodes: set[str] = {seed_id}
    frontier: list[str] = [seed_id]
    withheld_expired = 0
    capped = False
    unexplored = False

    for hop in range(1, hops + 1):
        next_frontier: list[str] = []
        for node in frontier:
            if capped:
                unexplored = True
                break
            # Fetched with the clock OFF (``include_expired=True``) so the
            # validity decision is made here against ``as_of`` and the
            # count of what was withheld is knowable.
            candidates = kg.edges_of(node, direction=direction, include_expired=True)
            for edge in candidates:
                if wanted_predicates and edge.predicate.value not in wanted_predicates:
                    continue
                if not include_expired and not _is_live(edge, as_of=moment):
                    withheld_expired += 1
                    continue
                eid = _edge_identity(edge)
                if eid in served:
                    continue
                if len(triples) >= max_triples:
                    capped = True
                    unexplored = True
                    break
                served.add(eid)
                claim_key = (edge.subject, edge.predicate.value, edge.object)
                claim = corroboration.get(claim_key, Corroboration.from_edges(claim_key, [edge]))
                triples.append(
                    GroundedTriple(
                        subject=edge.subject,
                        predicate=edge.predicate.value,
                        object=edge.object,
                        hop=hop,
                        edge_id=eid,
                        source_block_id=edge.source_block_id,
                        confidence=edge.confidence,
                        origin=(edge.metadata or {}).get("origin"),
                        schema_version=schema_version_of(edge.metadata),
                        valid_from=edge.valid_from,
                        valid_until=edge.valid_until,
                        sources=claim.sources,
                        corroborated_confidence=claim.corroborated_confidence,
                        corroborating_blocks=claim.source_block_ids,
                    )
                )
                for endpoint in (edge.subject, edge.object):
                    if endpoint not in seen_nodes:
                        seen_nodes.add(endpoint)
                        next_frontier.append(endpoint)
        if capped:
            break
        frontier = sorted(next_frontier)
        if not frontier:
            break
    else:
        # The loop ran to the hop limit with nodes still queued: the
        # subgraph is truncated by radius, and saying so is the point.
        if frontier:
            unexplored = True

    if not triples:
        gaps.append(Gap(GAP_NO_EDGES, f"no edges within {hops} hop(s) of {seed_id!r}"))
    for predicate in wanted_predicates:
        if not any(t.predicate == predicate for t in triples):
            gaps.append(Gap(GAP_PREDICATE_ABSENT, f"no {predicate!r} edge in the served subgraph"))
    if capped:
        gaps.append(Gap(GAP_TRIPLE_CAP, f"subgraph truncated at max_triples={max_triples}"))
    if unexplored and not capped:
        gaps.append(Gap(GAP_HOP_LIMIT, f"entities beyond hop {hops} were not expanded"))
    if withheld_expired:
        gaps.append(
            Gap(
                GAP_EXPIRED_WITHHELD,
                f"{withheld_expired} edge(s) outside their validity window were not served",
            )
        )
    single_source = sorted({t.edge_id for t in triples if t.sources < 2})
    if single_source:
        gaps.append(
            Gap(
                GAP_SINGLE_SOURCE_CLAIM,
                f"{len(single_source)} served claim(s) rest on a single source block: {', '.join(single_source[:5])}",
            )
        )
    unversioned = sorted({t.edge_id for t in triples if t.schema_version is None})
    if unversioned:
        gaps.append(
            Gap(
                GAP_UNVERSIONED_EDGE,
                f"{len(unversioned)} served edge(s) carry no schema stamp: {', '.join(unversioned[:5])}",
            )
        )
    if known_block_ids is not None:
        known = {str(b).strip() for b in known_block_ids}
        missing = sorted({t.source_block_id for t in triples if t.source_block_id not in known})
        for block_id in missing:
            gaps.append(Gap(GAP_PROVENANCE_MISSING, f"cited source block {block_id} is not in the corpus"))

    return EdgeGroundedContext(
        seed=seed,
        seed_entity_id=seed_id,
        hops=hops,
        triples=tuple(triples),
        gaps=tuple(gaps),
    )


def _default_render(context: EdgeGroundedContext) -> str:
    """The answer when no generator is bound: the cited triples themselves.

    Not a placeholder — it is the only claim that is certainly supported.
    Returning invented prose here, or an empty string, would both be worse
    than handing back exactly what the graph establishes.
    """
    if not context.triples:
        return f"The graph contains no edges within {context.hops} hop(s) of {context.seed!r}."
    lines = [f"{len(context.triples)} grounded triple(s) about {context.seed_entity_id or context.seed}:"]
    lines.extend(
        f"- {t.subject} {t.predicate} {t.object} [[{t.edge_id}]] "
        f"(source {t.source_block_id}, {t.sources} corroborating block(s), "
        f"confidence {t.corroborated_confidence:g})"
        for t in context.ranked_triples
    )
    return "\n".join(lines)


def answer(
    kg: KnowledgeGraph,
    seed: str,
    *,
    generate_fn: Optional[Callable[[str], str]] = None,
    context: Optional[EdgeGroundedContext] = None,
    **context_kwargs: Any,
) -> GroundedAnswer:
    """Answer about *seed* using only the served edges, and prove it.

    ``generate_fn`` receives the serialised context **and nothing else**;
    whatever it returns is checked citation by citation. Any ``[[E-…]]``
    naming an edge that was not served is a fabrication: it is listed in
    ``fabricated_citations``, added to the context gaps, and
    ``grounded`` is ``False``. An answer that cites nothing at all while
    the graph did serve edges is likewise not grounded — an uncited claim
    is exactly what this mode exists to make impossible to mistake for a
    sourced one.

    With no generator the answer is :func:`_default_render` — the cited
    triples — and it is grounded by construction.
    """
    ctx = context if context is not None else build_context(kg, seed, **context_kwargs)
    if generate_fn is None:
        text = _default_render(ctx)
        cited = tuple(sorted(set(CITATION_RE.findall(text))))
        return GroundedAnswer(
            context=ctx,
            text=text,
            citations=cited,
            fabricated_citations=(),
            grounded=True,
            generator="none",
        )

    text = generate_fn(ctx.serialize())
    if not isinstance(text, str):
        raise TypeError("generate_fn must return a string")
    cited_all = sorted(set(CITATION_RE.findall(text)))
    allowed = ctx.edge_ids
    fabricated = tuple(c for c in cited_all if c not in allowed)
    cited = tuple(c for c in cited_all if c in allowed)
    extra_gaps: list[Gap] = []
    for bogus in fabricated:
        extra_gaps.append(Gap(GAP_FABRICATED_CITATION, f"{bogus} was cited but never served"))
    if not cited_all and ctx.triples and text.strip():
        extra_gaps.append(Gap(GAP_UNCITED_ANSWER, "the answer cites no served edge"))
    final_ctx = (
        ctx
        if not extra_gaps
        else EdgeGroundedContext(
            seed=ctx.seed,
            seed_entity_id=ctx.seed_entity_id,
            hops=ctx.hops,
            triples=ctx.triples,
            gaps=ctx.gaps + tuple(extra_gaps),
        )
    )
    return GroundedAnswer(
        context=final_ctx,
        text=text,
        citations=cited,
        fabricated_citations=fabricated,
        grounded=not extra_gaps,
        generator="injected",
    )


def corpus_block_ids(workspace: str) -> frozenset[str]:
    """Block ids the workspace corpus actually holds.

    Fed to ``build_context(known_block_ids=...)`` so a citation whose
    provenance document has been deleted is reported instead of quietly
    standing. Goes through the graph-ingest corpus loader, which is the
    same view the extraction side reads.
    """
    from .graph_ingest import _load_corpus

    return frozenset(str(b.get("_id") or "").strip() for b in _load_corpus(workspace) if str(b.get("_id") or "").strip())


__all__ = [
    "CITATION_RE",
    "DEFAULT_HOPS",
    "DEFAULT_MAX_TRIPLES",
    "Gap",
    "GroundedTriple",
    "EdgeGroundedContext",
    "GroundedAnswer",
    "build_context",
    "answer",
    "corpus_block_ids",
    "GAP_UNKNOWN_ENTITY",
    "GAP_NO_EDGES",
    "GAP_PREDICATE_ABSENT",
    "GAP_HOP_LIMIT",
    "GAP_TRIPLE_CAP",
    "GAP_EXPIRED_WITHHELD",
    "GAP_UNVERSIONED_EDGE",
    "GAP_PROVENANCE_MISSING",
    "GAP_FABRICATED_CITATION",
    "GAP_UNCITED_ANSWER",
    "GAP_SINGLE_SOURCE_CLAIM",
]
