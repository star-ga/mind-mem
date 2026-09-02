# Copyright 2026 STARGA, Inc.
"""RA.4 — the retention class: how much scrutiny a block's *death* needs.

A governed store has to answer one question before it removes anything:
**what would be lost, and who is allowed to decide that?** The retention
class is that answer, computed as a pure function of fields the block
already carries — no clock, no configuration, no I/O, no learned score.

Three classes, ordered most- to least-protected:

``PROTECTED``
    Existence is structurally load-bearing. A death here is never a
    maintenance by-product; it needs a named, approved decision that says
    what it is giving up. Two sources, both derived from an existing
    authority rather than a list kept here:

    * an **active release decision** — :func:`~mind_mem.admissibility.release_ids`
      returns a non-empty set for it, which means other blocks are admitted
      *because* this one exists. Removing it silently withholds every id it
      released, and the withholding would look like a corpus that simply
      does not contain them.
    * an **operator-authored guardrail** — a ``GR-`` block whose provenance
      :func:`~mind_mem.guardrails.guardrail_provenance_refusal` permits. A
      guardrail bypasses the ranker and fires on a trigger, so a guardrail
      that stops existing stops firing and *nothing else notices*: there is
      no ranked position to go missing from.

``GOVERNED``
    Ordinary admitted corpus content. It may die, but only through the
    proposal path — propose, approve, apply — like every other change to
    admitted content.

``EPHEMERAL``
    Never passed the gate. Its status is one an ingest tier mints for
    withheld content (:data:`~mind_mem.admissibility.UNADMITTED`) or one
    nobody has named at all, which the admissibility allow-list withholds
    by construction. Removing it removes nothing the store ever promised
    to serve. ``EPHEMERAL`` is a statement about *admission*, not about
    worth: quarantined content can be perfectly valuable and is one
    approved release decision away from ``GOVERNED``.

WHAT THIS IS NOT. It is not a verdict on whether a block *should* be
removed, and nothing here removes anything. "Nothing serves it" is
evidence about attention, never about worth — see
:mod:`mind_mem.accountability_views`, which uses this classifier for
exactly one purpose: to refuse to call a ``PROTECTED`` block "waste"
however little it has been served.

PURITY. Every function here reads only the mapping it is handed. The
authorities it defers to are pure too (:func:`release_ids` is a fold over
the blocks it is given; :func:`guardrail_provenance_refusal` reads only
the block), so a classification is reproducible from the block alone, on
any host, on any day. That is what RA.4 means by "no clock in the
classification": a retention class that moved with the wall clock could
not be sealed into a death record.

DELIBERATELY ABSENT: an "importance", "age" or "access count" input.
Each of those is a *decay* signal, and RA's ruling is that decay acts on
attention and never on existence.
"""

from __future__ import annotations

from typing import Any, Final, Mapping

from .admissibility import RELEASE_FIELD, is_admissible_status, release_ids
from .guardrails import GUARDRAIL_ID_PREFIX, guardrail_provenance_refusal

#: Existence is load-bearing: a death needs a named, approved decision.
PROTECTED: Final = "PROTECTED"

#: Admitted content: a death goes through the proposal path.
GOVERNED: Final = "GOVERNED"

#: Never admitted: removing it removes nothing the store promised to serve.
EPHEMERAL: Final = "EPHEMERAL"

#: The closed set, ordered most-protected first. Ordered because a caller
#: that ranks or sorts by class must not have to re-derive the ordering and
#: get it backwards.
RETENTION_CLASSES: Final[tuple[str, str, str]] = (PROTECTED, GOVERNED, EPHEMERAL)

__all__ = [
    "EPHEMERAL",
    "GOVERNED",
    "PROTECTED",
    "RETENTION_CLASSES",
    "protected_reason",
    "retention_class",
]


def _release_probe(block: Mapping[str, Any], status_key: str) -> dict[str, Any]:
    """Re-key *block* into the shape :func:`release_ids` reads.

    :func:`~mind_mem.admissibility.release_ids` is the authority on what an
    active release decision is, and it reads ``Status`` and ``Releases`` by
    those exact names — the corpus spelling. Hit dicts spell the status
    ``status``, so handing one straight to the authority would ask it about
    a field that is not there and get "unstated" back. Re-keying is what
    lets the authority be *called* rather than re-implemented here.
    """
    return {
        "_id": block.get("_id", ""),
        "Status": block.get(status_key),
        RELEASE_FIELD: block.get(RELEASE_FIELD),
    }


def _is_guardrail(block: Mapping[str, Any]) -> bool:
    """True iff *block* is a guardrail :mod:`~mind_mem.guardrails` would honour.

    The ``GR-`` prefix is the module's own recognition rule (a block that
    merely *declares* ``Type: Guardrail`` under some other id is not loaded
    as one), and the provenance refusal is applied for the reason that
    module states: a trigger-bearing block bypasses the ranker, so letting
    external content mint one is an injection primitive. Protecting a block
    the guardrail loader refuses would hand that same primitive a second
    life as an undeletable block.
    """
    block_id = str(block.get("_id") or "")
    if not block_id.startswith(GUARDRAIL_ID_PREFIX):
        return False
    return not guardrail_provenance_refusal(block)


def protected_reason(block: Mapping[str, Any], *, status_key: str = "Status") -> str:
    """Why *block* is :data:`PROTECTED` — ``""`` when it is not.

    A named reason rather than a bare boolean because the reason is the
    useful half: a death record for a protected block has to say what the
    removal is giving up, and "it was protected" says nothing.

    Args:
        block: A parsed block dict (or a recall hit).
        status_key: Field holding the lifecycle status — ``"Status"`` on a
            parsed corpus block, ``"status"`` on an indexed hit.

    Returns:
        A short reason, or ``""``.
    """
    if release_ids([_release_probe(block, status_key)]):
        return "active release decision: other blocks are admitted because this one exists"
    if _is_guardrail(block):
        return "operator-authored guardrail: a prohibition that stops existing stops firing"
    return ""


def retention_class(block: Mapping[str, Any], *, status_key: str = "Status") -> str:
    """Classify *block* as :data:`PROTECTED`, :data:`GOVERNED` or :data:`EPHEMERAL`.

    Pure: reads only *block*. No clock, no config, no I/O.

    Order is load-bearing. ``PROTECTED`` is tested first, so a release
    decision or a guardrail is protected whatever its status says —
    a protection that a status edit can turn off is not a protection.

    Args:
        block: A parsed block dict (or a recall hit).
        status_key: Field holding the lifecycle status — ``"Status"`` on a
            parsed corpus block, ``"status"`` on an indexed hit.

    Returns:
        One of :data:`RETENTION_CLASSES`.
    """
    if protected_reason(block, status_key=status_key):
        return PROTECTED
    if is_admissible_status(block.get(status_key)):
        return GOVERNED
    return EPHEMERAL
