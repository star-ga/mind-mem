"""Propose typed KG edges from a block's text. Proposals only, never writes.

ROADMAP: "wire lightweight entity/relation extraction ... into propose_update, so
writing a block PROPOSES typed KG edges. Extracted edges land as proposals, never
auto-committed -- same approval gate as blocks, honoring the Group H wedge
guardrail (source-of-truth graph never self-modifies)."

Two halves already existed and nothing joined them: ``block_parser._ENTITY_ID_RE``
finds canonical ids, and ``knowledge_graph.propose_edge`` is the HITL-gated
staging surface. Writing a block therefore proposed no edges at all.

THE GUARDRAIL IS THE FEATURE, not a caveat on it. Every function here RETURNS
candidates and writes nothing -- no graph call, no proposal call, no file. The
caller (the governed door) decides whether to stage them, so review stays where
review belongs. A test asserts this on the import graph rather than trusting this
paragraph, because the distinction is exactly what a refactor loses.

PREDICATES COME FROM THE EXISTING CLOSED ENUM. An extractor free to invent a
predicate string would put untyped edges into a typed graph -- the same open-set
failure ``upsert_slots`` closed for slots, and for the same reason: two spellings
of one relation never collide, so the graph quietly grows two vocabularies.

RESIDUAL LIMITS, stated rather than implied:

* Relations are recognised by SURFACE PHRASE. "refines X" is a claim about X;
  "unlike the approach in X" is not, and this module cannot tell the difference
  when the phrase is absent. It under-proposes by design -- a missed proposal
  costs a human noticing, a wrong one costs a human un-noticing.
* Named-entity recognition here is a capitalised-run heuristic, not a model. It
  will miss lowercase names and can catch a capitalised sentence opener followed
  by a proper noun. Both failure modes are why these are PROPOSALS.
* No negation handling. "This does not supersede X" proposes ``supersedes``.
  Recorded because a reviewer must know the extractor cannot read a "not".

Pure: no clock, no I/O, no randomness. A proposal set must be replayable, or a
reviewer cannot tell whether re-running produced the same claims.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

#: Hard ceiling on candidates from one block. An unbounded extractor turns one
#: pathological block into a review flood, and a flood is how a HITL gate stops
#: being read at all -- which would make the guardrail worse than useless.
MAX_CANDIDATES = 12

#: Canonical ids, mirroring block_parser._ENTITY_ID_RE. Duplicated deliberately:
#: importing block_parser would pull a parser (and its I/O) into a pure module.
#: A test pins that canonical ids are still found, so drift shows up as a failure.
_CANONICAL_ID = re.compile(
    r"\b(D-\d{8}-\d{3}|T-\d{8}-\d{3}|INC-\d{8}-[a-z0-9-]+"
    r"|PRJ-[a-z0-9-]+|PER-[a-z0-9-]+|TOOL-[a-z0-9-]+"
    r"|C-\d{8}-\d{3}|DREF-\d{8}-\d{3}|I-\d{8}-\d{3})\b"
)

#: A capitalised multi-word run: the "named entity beyond a canonical id" half.
#: TWO words minimum, on purpose -- a single capitalised word is usually a
#: sentence opener, and admitting those would flood the graph with "The", "This"
#: and every proper-noun-shaped false positive.
_NAMED_RUN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

#: Sentence-opening words that begin a capitalised run without naming anything.
_OPENERS = frozenset({"The", "This", "That", "These", "Those", "It", "We", "A", "An"})

#: Surface phrase -> predicate, and the predicate MUST be a member of the closed
#: knowledge_graph.Predicate enum (asserted by a test). Ordered longest-first at
#: match time so "depends on" is not shadowed by a shorter pattern.
_RELATION_PHRASES: dict[str, str] = {
    "supersedes": "supersedes",
    "supersede": "supersedes",
    "refines": "refines",
    "refine": "refines",
    "depends on": "depends_on",
    "depend on": "depends_on",
    "contradicts": "contradicts",
    "contradict": "contradicts",
    "part of": "part_of",
    "member of": "member_of",
    "derived from": "derived_from",
    "justified by": "justified_by",
    "supports": "supports",
    "authored by": "authored_by",
}


def extract_named_entities(text: object) -> list[str]:
    """Canonical ids plus multi-word capitalised names, deduplicated and sorted.

    Sorted so the result is replayable: a reviewer comparing two runs must see the
    same order, or "the extractor changed its mind" is indistinguishable from
    "dict iteration differed".
    """
    s = str(text or "")
    found = set(_CANONICAL_ID.findall(s))
    for run in _NAMED_RUN.findall(s):
        first = run.split()[0]
        if first in _OPENERS:
            # Drop the opener and keep the rest only if a multi-word name remains.
            rest = run.split()[1:]
            if len(rest) >= 2:
                found.add(" ".join(rest))
            continue
        found.add(run)
    return sorted(found)


def _predicate_for(text: str, target: str) -> Optional[str]:
    """The predicate asserted about *target* in *text*, or None.

    Looks for a relation phrase immediately before the target. Requiring adjacency
    is what keeps "refines X" apart from "unlike the approach in X, this refines
    Y" -- a phrase anywhere in the sentence would attach the wrong object.
    """
    idx = text.find(target)
    if idx < 0:
        return None
    window = text[max(0, idx - 40) : idx].lower()
    best: Optional[tuple[int, str]] = None
    for phrase, predicate in _RELATION_PHRASES.items():
        pos = window.rfind(phrase)
        if pos < 0:
            continue
        # Nearest phrase wins; longest breaks a tie so "depends on" beats a
        # substring of itself.
        key = (pos, phrase)
        if best is None or key > (best[0], best[1]):
            best = (pos, phrase)
    return _RELATION_PHRASES[best[1]] if best else None


def candidate_edges(block: Optional[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Typed edge CANDIDATES implied by *block*'s text. Writes nothing.

    Each candidate is plain data -- ``{subject, predicate, object, evidence}`` --
    so a caller can show it to a human before anything is staged. ``evidence`` is
    the phrase that produced the claim, because a reviewer approving an edge needs
    to see why it was proposed, not just what.

    A block never proposes an edge to itself: a self-edge is never information and
    would be a confusing thing to ask a human to approve.
    """
    if not block:
        return []
    subject = str(block.get("_id") or block.get("id") or "").strip()
    text = str(block.get("Statement") or "")
    if not subject or not text:
        return []

    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for target in extract_named_entities(text):
        if target == subject:
            continue
        predicate = _predicate_for(text, target)
        if predicate is None:
            continue
        key = (predicate, target)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": target,
                "evidence": text[:200],
            }
        )
        if len(out) >= MAX_CANDIDATES:
            break
    return out


__all__ = ["MAX_CANDIDATES", "candidate_edges", "extract_named_entities"]
