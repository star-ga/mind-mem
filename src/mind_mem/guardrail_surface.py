# Copyright 2026 STARGA, Inc.
"""Ranker bypass for GUARDRAIL blocks — bounded, deterministic surfacing.

:mod:`mind_mem.guardrails` decides *which* guardrails fire for a context.
This module decides *how* they enter a recall response:

1. **Unconditionally first.**  A firing guardrail is prepended ahead of
   every ranked hit, whatever its similarity score — including a score of
   zero, including a guardrail the ranker never retrieved at all.  That is
   the whole point: a prohibition must not have to win a relevance contest
   against the query that is about to violate it.
2. **Bounded displacement.**  At most ``policy.max_surfaced`` guardrails
   are injected (hard cap :data:`~mind_mem.guardrails.MAX_SURFACED_HARD_CAP`),
   and the response keeps its original length, so the number of ranked hits
   pushed out is *at most* ``max_surfaced``.  A guardrail that the ranker had
   already returned is promoted in place and displaces nothing.  A page
   *shorter* than ``max_surfaced`` can be displaced entirely — length is the
   bound that holds unconditionally, and it is the response's own length, not
   the guardrail count.
3. **Marked as a constraint.**  Surfaced hits carry ``guardrail: True``
   plus ``guardrail_severity`` / ``guardrail_triggers`` /
   ``guardrail_constraint`` / ``surfaced_by`` so the consumer can render
   them as constraints instead of as evidence.
4. **Off unless asked.**  With no context supplied — or no guardrail blocks
   in the workspace, or ``recall.guardrails.enabled: false`` — the hit list
   is returned unchanged, the same object, byte for byte.

Hit shape is produced by the shared
:func:`mind_mem._recall_context._block_to_result`, so a guardrail hit is
structurally identical to any other hit plus the ``guardrail_*`` markers.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ._recall_context import _block_to_result
from .guardrails import (
    Guardrail,
    GuardrailContext,
    GuardrailPolicy,
    load_guardrails,
    match_guardrails,
)
from .observability import get_logger

__all__ = [
    "GUARDRAIL_SURFACE_MARKER",
    "apply_guardrail_surfacing",
    "guardrail_hits",
    "guardrail_to_hit",
]

_log = get_logger("guardrail_surface")

#: Value of the ``surfaced_by`` marker on a force-surfaced hit.
GUARDRAIL_SURFACE_MARKER = "guardrail_trigger"


def guardrail_to_hit(
    guardrail: Guardrail,
    matched: Sequence[str],
    *,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render one guardrail as a recall hit.

    Args:
        guardrail: The firing guardrail.
        matched: Trigger dimensions that fired, in report order.
        existing: The ranked hit for the same block, when the ranker had
            already retrieved it.  Its score and fields are preserved so a
            promoted guardrail is not silently rescored.

    Returns:
        A hit dict: the standard recall hit shape plus ``guardrail_*``
        markers.  Never mutates *existing*.
    """
    if existing is not None:
        hit: dict[str, Any] = dict(existing)
    else:
        hit = _block_to_result(dict(guardrail.block), 0.0)
        hit["file"] = guardrail.source_file or hit.get("file", "")
    hit["type"] = "guardrail"
    hit["guardrail"] = True
    hit["guardrail_severity"] = guardrail.severity
    hit["guardrail_triggers"] = list(matched)
    hit["guardrail_constraint"] = guardrail.statement
    hit["surfaced_by"] = GUARDRAIL_SURFACE_MARKER
    return hit


def guardrail_hits(
    workspace: str,
    context: GuardrailContext,
    policy: GuardrailPolicy | None = None,
    *,
    existing_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load, match and render the guardrails firing for *context*.

    Returns at most ``policy.max_surfaced`` hits, ordered by
    ``(severity, block_id)``.  Read-only: no store writes, no scoring
    state, no clock.
    """
    policy = policy or GuardrailPolicy()
    if not policy.enabled or policy.max_surfaced <= 0 or context.is_empty():
        return []
    matches = match_guardrails(load_guardrails(workspace, policy), context)
    if not matches:
        return []
    lookup = existing_by_id or {}
    out: list[dict[str, Any]] = []
    for guardrail, matched in matches[: policy.max_surfaced]:
        out.append(guardrail_to_hit(guardrail, matched, existing=lookup.get(guardrail.block_id)))
    return out


def apply_guardrail_surfacing(
    hits: list[dict],
    *,
    workspace: str | None,
    context: GuardrailContext | None,
    policy: GuardrailPolicy | None = None,
) -> list[dict]:
    """Prepend firing guardrails to *hits* under the displacement bound.

    Returns *hits* unchanged (the same list object) when there is no
    context, no workspace, or no guardrail fires — the zero-regression
    path.  Otherwise returns a new list: guardrails first in
    ``(severity, block_id)`` order, then the surviving ranked hits in
    their original relative order, truncated back to the original
    response length.

    The response never grows, so ranked hits displaced ≤ number of newly
    injected guardrails ≤ ``policy.max_surfaced``.  When *hits* is empty
    the guardrails are still returned: a constraint must fire even when
    recall found nothing.
    """
    if context is None or workspace is None or context.is_empty():
        return hits

    existing_by_id = {str(h.get("_id", "")): h for h in hits}
    surfaced = guardrail_hits(workspace, context, policy, existing_by_id=existing_by_id)
    if not surfaced:
        return hits

    head_ids = {str(h.get("_id", "")) for h in surfaced}
    tail = [h for h in hits if str(h.get("_id", "")) not in head_ids]
    # The response never grows. The ONE documented exception is an empty page:
    # a constraint must fire even when recall found nothing, and there is no
    # length to preserve. ``max(len(hits), len(surfaced))`` also covered that
    # case, but it covered far more — every page SHORTER than the number of
    # firing guardrails. With the default bound of 3, a caller asking for one
    # hit got three, overrunning both its own page size and any downstream
    # max-results budget, one step after the pipeline's own ``[:limit]``.
    budget = len(hits) if hits else len(surfaced)
    result = (surfaced + tail)[:budget]
    # ``budget`` can now be smaller than the head, so the surviving-tail count
    # needs the clamp: a bare ``budget - len(surfaced)`` goes negative there and
    # would report more displaced hits than the page ever held.
    kept_tail = max(0, budget - len(surfaced))
    _log.info(
        "guardrails_surfaced",
        count=len(surfaced),
        promoted=len(hits) - len(tail),
        displaced=len(tail) - kept_tail,
    )
    return result
