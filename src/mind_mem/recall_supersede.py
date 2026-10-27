# Copyright 2026 STARGA, Inc.
"""Supersession by link: a block another block names in ``Supersedes``.

A block whose own status is ``superseded`` is demoted by
:func:`mind_mem._recall_core._rank_replaced_below_live`. A block that stays
``Status: active`` while an admitted block names it in ``Supersedes`` was not
recognised at all, so it out-ranked -- and at a small ``limit`` was returned
instead of -- the block that replaced it.

This module holds the three pieces every recall leg shares:

* :func:`supersede_targets` -- the ids ONE parsed block names, from the
  top-level field (``Supersedes``/``supersedes``, string or list) and from any
  nested mapping that carries the same key (``ConstraintSignatures[].supersedes``,
  ``fields.Supersedes``). Ids are WHOLE tokens; empty markers (``none``,
  ``null``, ``n/a``, ``-``, blank) are not links, and a block never names
  itself. Nothing here pattern-matches an id, so ``D-1-r2`` cannot retire
  ``D-1``.
* :func:`resolve_heads` -- follows each returned id up its chain
  (A <- B <- C) to the live end, with cycle detection. Edges come from a
  ``superseders_of`` callable, so the indexed leg answers from its
  ``supersede_edges`` table and the scan leg from the blocks it already
  parsed; neither parses the corpus a second time.
* :func:`order_by_supersession` -- puts the live head of a retired hit's chain
  directly above it (pulling the head in when the query did not match it),
  with chain members in chain order. With no retired hit the input list is
  returned unchanged, object for object.

Which edges a caller sees is decided by whoever builds ``superseders_of``:
only admitted superseders, and only those the caller may read. A superseder
the caller cannot read neither retires nor reveals anything.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

#: Values that mean "no link". Compared case-insensitively after stripping.
EMPTY_MARKERS = frozenset({"", "none", "null", "nil", "n/a", "na", "-", "--", "—", "–"})

#: Separators between ids in one field value.
_SPLIT = re.compile(r"[\s,;|]+")

#: Punctuation that may wrap an id in prose (``[D-1]``, ``"D-1"``, ``D-1.``).
_WRAP = "[](){}<>\"'`*"

#: Nesting depth the field walk will descend. Parsed blocks are shallow; the
#: bound only stops a pathological document from recursing without end.
_MAX_DEPTH = 6

#: Longest chain followed. A chain this long is malformed data, not history.
_MAX_CHAIN = 64

#: Marker on a hit moved or pulled above a retired block, holding its own score.
LIFTED_MARKER = "_supersede_lifted_from"

#: Marker on a retired hit: its chain ``[s1, ..., head]`` as resolved when it
#: was placed. A later stage that re-sorts the list re-applies the order from
#: this marker alone (:func:`reapply_supersession`), with no edge lookup.
CHAIN_MARKER = "_supersede_chain"


def _field_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def _walk(node: Any, depth: int, out: list[str]) -> None:
    if depth > _MAX_DEPTH:
        return
    if isinstance(node, Mapping):
        for key, value in node.items():
            if not isinstance(key, str) or key.startswith("_"):
                continue
            if key.lower() == "supersedes":
                out.extend(_field_values(value))
            elif isinstance(value, (Mapping, list, tuple)):
                _walk(value, depth + 1, out)
    elif isinstance(node, (list, tuple)):
        for item in node:
            if isinstance(item, (Mapping, list, tuple)):
                _walk(item, depth + 1, out)


def _tokens(raw: str) -> list[str]:
    if raw.strip().lower() in EMPTY_MARKERS:
        return []
    found: list[str] = []
    for piece in _SPLIT.split(raw):
        token = piece.strip(_WRAP).rstrip(".:")
        if token and token.lower() not in EMPTY_MARKERS:
            found.append(token)
    return found


def supersede_targets(block: Mapping[str, Any]) -> tuple[str, ...]:
    """Ids *block* names as superseded, sorted and unique. Whole tokens only."""
    raw: list[str] = []
    _walk(block, 0, raw)
    if not raw:
        return ()
    own = str(block.get("_id") or block.get("id") or "")
    targets = {token for value in raw for token in _tokens(value)}
    targets.discard(own)
    return tuple(sorted(targets))


def edges_from_blocks(blocks: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """``target -> [superseder, ...]`` for already-admitted, already-readable *blocks*."""
    edges: dict[str, set[str]] = {}
    for block in blocks:
        own = block.get("_id")
        if not isinstance(own, str) or not own:
            continue
        for target in supersede_targets(block):
            edges.setdefault(target, set()).add(own)
    return {target: sorted(sups) for target, sups in edges.items()}


SupersedersOf = Callable[[Iterable[str]], Mapping[str, Sequence[str]]]


def resolve_heads(ids: Iterable[str], superseders_of: SupersedersOf) -> dict[str, list[str]]:
    """``id -> [s1, s2, ..., head]`` for every id a live head retires.

    ``s1`` supersedes the id, ``s2`` supersedes ``s1``, and ``head`` is
    superseded by nothing the caller can see. Where a block has several
    superseders, the one whose chain ends at the greatest head id wins
    (ids are date-stamped, so that is the newest); ties go to the greater
    superseder id. An id whose every chain loops back on itself has no live
    head and is not retired. Ids no edge names are absent from the result.
    """
    parents: dict[str, list[str]] = {}
    frontier = {i for i in ids if isinstance(i, str) and i}
    for _ in range(_MAX_CHAIN):
        pending = sorted(frontier - parents.keys())
        if not pending:
            break
        found = superseders_of(pending)
        nxt: set[str] = set()
        for node in pending:
            sups = sorted({s for s in found.get(node, ()) if isinstance(s, str) and s and s != node})
            parents[node] = sups
            nxt.update(sups)
        frontier = nxt

    memo: dict[str, list[str] | None] = {}

    def path_up(node: str, visiting: frozenset[str]) -> list[str] | None:
        """``[s1, ..., head]`` above *node*, ``[]`` when *node* is itself live, None on a loop."""
        if node in memo:
            return memo[node]
        sups = parents.get(node, [])
        if not sups:
            return []
        best: list[str] | None = None
        for sup in sups:
            if sup in visiting:
                continue
            above = path_up(sup, visiting | {node})
            if above is None:
                continue
            candidate = [sup, *above]
            if best is None or (candidate[-1], candidate[0]) > (best[-1], best[0]):
                best = candidate
        # Memoise only answers that did not depend on the current path.
        if not visiting:
            memo[node] = best
        return best

    heads: dict[str, list[str]] = {}
    for node in sorted({i for i in ids if isinstance(i, str) and i}):
        path = path_up(node, frozenset())
        if path:
            heads[node] = path
    return heads


def _score(hit: Mapping[str, Any]) -> float:
    try:
        return float(hit.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def order_by_supersession(
    hits: list[dict],
    heads: Mapping[str, Sequence[str]],
    fetch_head: Callable[[str], dict | None],
) -> list[dict]:
    """Place each retired hit's live head directly above it, chain in order.

    For a retired hit ``h`` with chain ``[s1, ..., head]``: ``head`` is placed
    first (taken from *hits* when present, else from ``fetch_head``), then
    each ``s_i`` that is in *hits* in chain order, then ``h``. Members already
    placed higher stay where they are. A member moved up, or pulled in, takes
    ``h``'s score when its own is lower, so the list stays score-descending;
    its own score is kept under :data:`LIFTED_MARKER`. Hits are never mutated.
    """
    if not heads or not hits:
        return hits
    by_id: dict[str, dict] = {}
    for hit in hits:
        hid = hit.get("_id")
        if isinstance(hid, str) and hid not in by_id:
            by_id[hid] = hit
    if not any(isinstance(h.get("_id"), str) and h.get("_id") in heads for h in hits):
        return hits

    out: list[dict] = []
    placed_objs: set[int] = set()
    placed_ids: set[str] = set()

    def place(member: dict, anchor: float, chain: Sequence[str]) -> None:
        own = _score(member)
        placed_objs.add(id(member))
        update: dict[str, Any] = {}
        if own < anchor:
            update["score"] = anchor
            update[LIFTED_MARKER] = member.get(LIFTED_MARKER, own)
        if chain and list(member.get(CHAIN_MARKER) or ()) != list(chain):
            update[CHAIN_MARKER] = list(chain)
        out.append({**member, **update} if update else member)
        mid = member.get("_id")
        if isinstance(mid, str):
            placed_ids.add(mid)

    for hit in hits:
        if id(hit) in placed_objs:
            continue
        hid = hit.get("_id")
        chain = heads.get(hid) if isinstance(hid, str) else None
        if not chain:
            place(hit, float("-inf"), ())
            continue
        anchor = _score(hit)
        head_id = chain[-1]
        for position in range(len(chain) - 1, -1, -1):
            member_id = chain[position]
            if member_id in placed_ids:
                continue
            member = by_id.get(member_id)
            if member is not None and id(member) in placed_objs:
                continue
            if member is None:
                if member_id != head_id:
                    continue
                member = fetch_head(member_id)
                if member is None:
                    continue
            place(member, anchor, chain[position + 1 :])
        place(hit, float("-inf"), chain if head_id in placed_ids else ())
    return out


def reapply_supersession(hits: list[dict]) -> list[dict]:
    """Restore supersession order on a list a later stage re-sorted.

    Uses only the :data:`CHAIN_MARKER` each retired hit already carries, so
    it needs no edge lookup and cannot pull anything new in: it can only
    move a head or chain member that is still in *hits* back above the block
    it retires. A list with no marked hit is returned unchanged.
    """
    heads: dict[str, list[str]] = {}
    for hit in hits:
        hid = hit.get("_id")
        chain = hit.get(CHAIN_MARKER)
        if isinstance(hid, str) and isinstance(chain, list) and chain and hid not in heads:
            heads[hid] = [c for c in chain if isinstance(c, str)]
    if not heads:
        return hits
    return order_by_supersession(hits, heads, lambda _id: None)


__all__ = [
    "EMPTY_MARKERS",
    "CHAIN_MARKER",
    "LIFTED_MARKER",
    "edges_from_blocks",
    "order_by_supersession",
    "reapply_supersession",
    "resolve_heads",
    "supersede_targets",
]
