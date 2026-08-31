"""Provenance class — the FIFTH deterministic component of the validity gate.

Trust is not a separate subsystem. *How a block got into the corpus* is one
more piece of affirmative evidence about its validity, so it is scored the
same way the other four criteria are scored (:mod:`validity_gate`): a pure
function of the block plus pre-fetched stored state, in ``[0.0, 1.0]``,
folded into the composite ``V``.

Four ordered classes, strictly decreasing::

    operator  >  agent-verified  >  agent-inferred  >  external-ingest
      1.00          0.75               0.50               0.25

Classification is table-driven and first-match-wins over the
:mod:`block_provenance` fields (``ActorRole`` / ``ToolId`` /
``ContentSource``) plus the importer's ``Source`` token:

    1. ``ActorRole`` in :data:`OPERATOR_ROLES`                  -> operator
    2. ``ActorRole`` / ``ToolId`` / ``Source`` marks an ingest,
       or ``ContentSource: external``                           -> external-ingest
    3. agent provenance **and** affirmative verification        -> agent-verified
    4. agent provenance                                         -> agent-inferred
    5. no provenance at all                                     -> unknown

Rule 2 outranks rule 3 on purpose: affirmative evidence that content came
from outside the governed store dominates any verification marker travelling
with that same content.

**The T-001 content tag demotes only.** ``ContentSource: external`` joins
rule 2 because it is affirmative evidence of *lower* trust. ``agent`` and
``user`` deliberately change nothing: reading ``user`` as ``operator``
would hand a free trust promotion to whoever wrote the bytes — the same
self-declaration weakness noted at the bottom of this module, but with no
compensating value, since the actor fields already answer "who wrote it".
So the tag can move a block down the ladder and never up it.

**Absence is neutral.** A block with no provenance fields scores
:data:`UNKNOWN` = ``1.0`` — the gate's governing rule is that only
affirmative evidence of invalidity debits, so a legacy corpus that predates
provenance fields is never demoted by this component.

**Verification evidence is per-block and human-sourced**, never learned: an
explicit ``Verified`` / ``VerifiedBy`` / ``Verification`` marker, a verifier
role, or a block whose recorded human calibration weight reached
:data:`CALIBRATION_CONFIRM_MIN`. There is deliberately **no per-actor
learned or anomaly scoring anywhere in this module** — a per-actor
reputation that drifts with history would make the same input score
differently on two machines, which is exactly what the determinism wedge
forbids.

Deterministic by construction: no clock, no randomness, no I/O, no
iteration-order dependence. Same block + same confirmed-id set -> same class
-> same float, on every platform.

Copyright (c) STARGA, Inc.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .block_provenance import CONTENT_SOURCE_EXTERNAL, extract_provenance, read_content_source

# --- the ordered class vocabulary -------------------------------------------

#: Written by an accountable human (or a human-held role).
OPERATOR = "operator"
#: Agent-authored and carrying affirmative verification evidence.
AGENT_VERIFIED = "agent-verified"
#: Agent-authored with no verification evidence.
AGENT_INFERRED = "agent-inferred"
#: Pulled in from a system outside the governed store.
EXTERNAL_INGEST = "external-ingest"
#: No provenance fields at all — scored neutral, never demoted.
UNKNOWN = "unknown"

#: The four real classes, strictly most- to least-trusted. ``UNKNOWN`` is not
#: a rank: it is the absence of a signal.
PROVENANCE_ORDER: tuple[str, ...] = (OPERATOR, AGENT_VERIFIED, AGENT_INFERRED, EXTERNAL_INGEST)

#: Component weight per class. Evenly spaced so the ordering is legible in
#: the composite; ``UNKNOWN`` is neutral (1.0), not top-of-rank.
PROVENANCE_WEIGHTS: Mapping[str, float] = {
    OPERATOR: 1.0,
    AGENT_VERIFIED: 0.75,
    AGENT_INFERRED: 0.5,
    EXTERNAL_INGEST: 0.25,
    UNKNOWN: 1.0,
}

# --- classification tables (all lower-cased at comparison time) -------------

#: Roles a human is accountable for.
OPERATOR_ROLES: frozenset[str] = frozenset({"operator", "human", "user", "owner", "maintainer", "reviewer", "principal"})

#: Roles that mean "this content came from outside the governed store".
#: ``importer`` is what ``importers.engine.build_import_block`` writes.
EXTERNAL_ROLES: frozenset[str] = frozenset({"importer", "import", "ingest", "ingestor", "crawler", "external", "feed", "scraper", "sync"})

#: Roles whose whole job is to confirm, so their writes are verified.
VERIFIED_ROLES: frozenset[str] = frozenset({"verifier", "validator", "auditor"})

#: ``ToolId`` / ``Source`` prefixes stamped by an ingest pipeline.
EXTERNAL_TOKEN_PREFIXES: tuple[str, ...] = ("imported:", "import:", "ingest:", "external:")

#: Block fields that carry an explicit verification marker.
VERIFICATION_FIELDS: tuple[str, ...] = ("Verified", "VerifiedBy", "Verification")

#: Marker values that count as affirmative verification.
VERIFIED_VALUES: frozenset[str] = frozenset({"true", "yes", "1", "verified", "confirmed", "approved"})

#: Recorded human calibration weight (``calibration.py``, range 0.5-1.5) at or
#: above which a block counts as human-confirmed. 1.25 is the midpoint of the
#: positive half, so a single lukewarm vote does not promote a class.
CALIBRATION_CONFIRM_MIN = 1.25

__all__ = [
    "AGENT_INFERRED",
    "AGENT_VERIFIED",
    "CALIBRATION_CONFIRM_MIN",
    "EXTERNAL_INGEST",
    "EXTERNAL_ROLES",
    "EXTERNAL_TOKEN_PREFIXES",
    "OPERATOR",
    "OPERATOR_ROLES",
    "PROVENANCE_ORDER",
    "PROVENANCE_WEIGHTS",
    "UNKNOWN",
    "VERIFICATION_FIELDS",
    "VERIFIED_ROLES",
    "VERIFIED_VALUES",
    "classify_provenance",
    "confirmed_block_ids",
    "provenance_component",
    "provenance_weight",
]


def _norm(value: Any) -> str:
    """Lower-cased, whitespace-stripped ``str`` view of *value* (``""`` for None)."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _is_external_token(token: str) -> bool:
    """True when an ingest pipeline stamped this ``ToolId`` / ``Source``."""
    return any(token.startswith(prefix) for prefix in EXTERNAL_TOKEN_PREFIXES)


def _has_verification_marker(block: Mapping[str, Any]) -> bool:
    """True when the block itself carries affirmative verification evidence.

    ``Verified: true``-style markers only. A field present but falsy (or
    blank) is *not* evidence — absence never promotes.
    """
    for field in VERIFICATION_FIELDS:
        raw = block.get(field)
        if raw is None:
            continue
        if isinstance(raw, bool):
            if raw:
                return True
            continue
        value = _norm(raw)
        if not value:
            continue
        if field == "VerifiedBy":
            return True
        if value in VERIFIED_VALUES:
            return True
    return False


def classify_provenance(
    block: Mapping[str, Any],
    *,
    confirmed_ids: frozenset[str] = frozenset(),
) -> str:
    """Return the provenance class of *block* — one of the module constants.

    Pure and first-match-wins over the ordered rules documented at module
    level. Never raises on odd corpus values (they are coerced via ``str``),
    only on a wrong argument type at the boundary.

    Args:
        block: A block dict or recall hit.
        confirmed_ids: Block ids recorded as human-confirmed (see
            :func:`confirmed_block_ids`). Promotes an *agent* block to
            ``agent-verified``; never promotes an unknown or external one.

    Raises:
        TypeError: *block* is not a mapping, or *confirmed_ids* is not a set.
    """
    if not isinstance(block, Mapping):
        raise TypeError(f"block must be a mapping, got {type(block).__name__}")
    if not isinstance(confirmed_ids, (set, frozenset)):
        raise TypeError(f"confirmed_ids must be a set, got {type(confirmed_ids).__name__}")

    provenance = extract_provenance(dict(block))
    role = _norm(provenance.get("actor_role"))
    tool = _norm(provenance.get("tool_id"))
    actor = _norm(provenance.get("actor_id"))
    source = _norm(block.get("Source"))
    # Fail-closed reader: an out-of-vocabulary tag yields None, so a crafted
    # value can only ever reach "no signal", never a class of its choosing.
    content_source = read_content_source(block)

    if role in OPERATOR_ROLES:
        return OPERATOR
    if role in EXTERNAL_ROLES or content_source == CONTENT_SOURCE_EXTERNAL or _is_external_token(tool) or _is_external_token(source):
        return EXTERNAL_INGEST
    if not (role or tool or actor):
        return UNKNOWN
    if role in VERIFIED_ROLES or _has_verification_marker(block) or _norm(block.get("_id")) in {_norm(cid) for cid in confirmed_ids}:
        return AGENT_VERIFIED
    return AGENT_INFERRED


def provenance_weight(provenance_class: str) -> float:
    """Weight of a class name; an unrecognised name is neutral (``1.0``)."""
    return PROVENANCE_WEIGHTS.get(provenance_class, PROVENANCE_WEIGHTS[UNKNOWN])


def provenance_component(
    block: Mapping[str, Any],
    confirmed_ids: frozenset[str] = frozenset(),
) -> float:
    """The fifth validity component for *block* — THE single implementation.

    Used by :func:`mind_mem.validity_gate.validity_components` (composite V)
    and, through it, by the standalone ``trust_scores`` surface. There is no
    second copy of this math anywhere.
    """
    return provenance_weight(classify_provenance(block, confirmed_ids=confirmed_ids))


def confirmed_block_ids(calibration_weights: Mapping[str, float] | None) -> frozenset[str]:
    """Ids whose recorded human calibration weight reached the confirm bar.

    Pure: reads only the map handed to it (loaded once per recall call by
    :mod:`trust_signals`), so the scoring path stays I/O-free.
    """
    if not calibration_weights:
        return frozenset()
    confirmed = {
        str(block_id)
        for block_id, weight in calibration_weights.items()
        if isinstance(weight, (int, float)) and not isinstance(weight, bool) and float(weight) >= CALIBRATION_CONFIRM_MIN
    }
    return frozenset(confirmed)


# deferred: a block's provenance class is read from the fields on the block
# itself, so a hand-edited corpus can claim ``ActorRole: operator`` without an
# authenticated write — upgrade path: cross-check ActorId against the
# audit_chain entry that introduced the block id (chain.jsonl already records
# agent per operation) once the chain records block-level targets.
