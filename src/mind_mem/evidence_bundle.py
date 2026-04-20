"""Structured evidence bundle for answerer co-design (v3.3.0 Tier 3 #7).

Rather than handing an answerer LLM 18 raw Markdown blocks and asking
it to extract facts, we pre-digest the retrieval result into a typed
bundle the model can reason over directly:

    {
      "query": "...",
      "facts": [{"claim": "...", "source_id": "D-...", "confidence": 0.8}, ...],
      "relations": [{"subject": "X", "predicate": "supersedes", "object": "Y"}, ...],
      "timeline": [{"date": "2026-04-20", "event": "...", "source_id": "D-..."}],
      "entities": [{"id": "PER-001", "name": "Alice", "type": "person"}],
      "source_blocks": [ ... original blocks for fallback/audit ... ]
    }

The bundle is a strict, typed shape — downstream answerers can scan
``facts`` + ``timeline`` in O(N) without re-parsing Markdown every
time. Gated behind ``recall(format="bundle")`` so existing callers
(``format="blocks"``, the default) see no change.

No LLM required — extraction is rule-based:
- Facts: block ``Statement:`` / ``Fact:`` fields.
- Relations: ``Supersedes`` / ``SupersededBy`` / ``Dependencies`` /
  ``Relates_to`` fields, plus block-ID references in text.
- Timeline: blocks with a ``Date`` or ``Created`` field.
- Entities: entity-prefix blocks (PER-/PRJ-/TOOL-/INC-) already in
  results.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .observability import get_logger

_log = get_logger("evidence_bundle")


# Block-ID regex — duplicated here rather than imported to keep
# evidence_bundle zero-coupling to _recall_constants. If the canonical
# regex evolves, this module drops to "best-effort" gracefully.
_BLOCK_ID_RE = re.compile(r"\b(?:D|T|PER|PRJ|TOOL|INC|FACT|C|SIG|P|DIA)-[\w-]+\b")

_RELATION_FIELDS: tuple[tuple[str, str], ...] = (
    ("Supersedes", "supersedes"),
    ("SupersededBy", "superseded_by"),
    ("Dependencies", "depends_on"),
    ("Relates_to", "relates_to"),
    ("Cites", "cites"),
    ("TestedBy", "tested_by"),
    ("DerivedFrom", "derived_from"),
)

_ENTITY_PREFIXES: tuple[str, ...] = ("PER", "PRJ", "TOOL", "INC")


@dataclass
class Fact:
    claim: str
    source_id: str
    confidence: float = 1.0
    field_name: str | None = None


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str


@dataclass
class TimelineEvent:
    date: str
    event: str
    source_id: str


@dataclass
class EntityRef:
    id: str
    name: str
    type: str


@dataclass
class EvidenceBundle:
    """Structured retrieval output for answerer co-design.

    The ``source_blocks`` list preserves the raw result set so callers
    that need an audit trail or the original Markdown content can still
    reach it. The structured fields are best-effort rules — if a field
    isn't populated it's because the source block didn't contain it,
    not because extraction failed.
    """

    query: str
    facts: list[Fact] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    entities: list[EntityRef] = field(default_factory=list)
    source_blocks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flat JSON-friendly shape for REST / MCP transport."""
        return {
            "query": self.query,
            "facts": [asdict(f) for f in self.facts],
            "relations": [asdict(r) for r in self.relations],
            "timeline": [asdict(t) for t in self.timeline],
            "entities": [asdict(e) for e in self.entities],
            "source_blocks": list(self.source_blocks),
        }


def _extract_facts(block: dict) -> list[Fact]:
    """Pull the primary claim(s) out of a block."""
    bid = str(block.get("_id", ""))
    if not bid:
        return []
    facts: list[Fact] = []
    # Priority: Statement > Fact > Claim > Summary.
    for field_name in ("Statement", "Fact", "Claim", "Summary"):
        val = block.get(field_name)
        if isinstance(val, str) and val.strip():
            facts.append(
                Fact(
                    claim=val.strip(),
                    source_id=bid,
                    field_name=field_name,
                    confidence=_confidence_for(block),
                )
            )
            break  # one primary fact per block
    return facts


def _confidence_for(block: dict) -> float:
    """Best-effort confidence estimate for a block.

    Maps block ``Status`` + ``Tier`` into a [0, 1] float. Tiered blocks
    (VERIFIED / LONG_TERM) rank higher; superseded/active/draft gate
    the ceiling.
    """
    status = str(block.get("Status", "")).lower()
    tier = str(block.get("Tier", block.get("_tier", ""))).upper()
    # Base by status.
    base = {"active": 0.8, "verified": 0.95, "superseded": 0.3, "draft": 0.5}.get(status, 0.7)
    # Tier bump.
    tier_bump = {"VERIFIED": 0.15, "LONG_TERM": 0.1, "SHARED": 0.05, "WORKING": 0.0}.get(tier, 0.0)
    return round(min(1.0, base + tier_bump), 3)


def _extract_relations(block: dict) -> list[Relation]:
    """Pull typed relations from structured fields + block-ID mentions."""
    bid = str(block.get("_id", ""))
    if not bid:
        return []
    relations: list[Relation] = []
    for src_field, predicate in _RELATION_FIELDS:
        val = block.get(src_field)
        if val is None:
            continue
        targets: list[str] = []
        if isinstance(val, list):
            targets.extend(str(v) for v in val)
        elif isinstance(val, str):
            # Split on comma / whitespace; pick only tokens matching a
            # block-ID shape so free-text doesn't pollute relations.
            targets.extend(_BLOCK_ID_RE.findall(val))
        for target in targets:
            if target and target != bid:
                relations.append(Relation(subject=bid, predicate=predicate, object=target))
    return relations


def _extract_timeline(block: dict) -> list[TimelineEvent]:
    """One timeline entry per block with a parseable date."""
    bid = str(block.get("_id", ""))
    if not bid:
        return []
    date = block.get("Date") or block.get("Created") or ""
    if not date or not isinstance(date, str):
        return []
    # Accept ISO (YYYY-MM-DD) prefixes; anything else is dropped.
    if not re.match(r"^\d{4}-\d{2}-\d{2}", str(date)):
        return []
    event = str(block.get("Event") or block.get("Statement") or block.get("Summary") or "").strip()
    if not event:
        return []
    return [TimelineEvent(date=str(date)[:10], event=event, source_id=bid)]


def _extract_entities(block: dict) -> list[EntityRef]:
    bid = str(block.get("_id", ""))
    if not bid:
        return []
    prefix = bid.split("-", 1)[0]
    if prefix not in _ENTITY_PREFIXES:
        return []
    name = str(block.get("Name") or block.get("Statement") or bid).strip()
    typ = str(block.get("Type") or prefix).strip()
    return [EntityRef(id=bid, name=name, type=typ)]


def build_bundle(
    query: str,
    results: list[dict],
    *,
    include_source_blocks: bool = True,
) -> EvidenceBundle:
    """Assemble an :class:`EvidenceBundle` from recall results.

    Args:
        query: Original recall query — echoed into the bundle so the
            answerer has the prompt at hand.
        results: Ranked recall results (already deduped / limited).
        include_source_blocks: When False, drop the raw blocks from
            the bundle — useful when transport size matters and the
            caller will audit via a separate path.
    """
    bundle = EvidenceBundle(query=query)
    seen_relations: set[tuple[str, str, str]] = set()
    seen_entities: set[str] = set()
    for block in results:
        bundle.facts.extend(_extract_facts(block))
        for rel in _extract_relations(block):
            key = (rel.subject, rel.predicate, rel.object)
            if key in seen_relations:
                continue
            seen_relations.add(key)
            bundle.relations.append(rel)
        bundle.timeline.extend(_extract_timeline(block))
        for ent in _extract_entities(block):
            if ent.id in seen_entities:
                continue
            seen_entities.add(ent.id)
            bundle.entities.append(ent)
    # Timeline sorts by date (ascending).
    bundle.timeline.sort(key=lambda t: t.date)
    if include_source_blocks:
        bundle.source_blocks = list(results)
    _log.info(
        "evidence_bundle_built",
        facts=len(bundle.facts),
        relations=len(bundle.relations),
        timeline=len(bundle.timeline),
        entities=len(bundle.entities),
    )
    return bundle


__all__ = [
    "Fact",
    "Relation",
    "TimelineEvent",
    "EntityRef",
    "EvidenceBundle",
    "build_bundle",
]
