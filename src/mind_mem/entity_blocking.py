"""Deterministic blocking for entity resolution: keep the model off easy cases.

ROADMAP ("Blocking + LLM-arbitration hybrid"): "cheap deterministic blocking
(inverted index on name tokens / embedding neighbors) narrows candidates to
50-100-item blocks; the LLM only arbitrates *within* a block. Keeps resolution
sublinear and keeps the expensive model call off the easy cases (typos, casing) that
deterministic logic already handles."

This is the deterministic half, and shipping it alone is the point of the split: an
arbitration layer needs a model, but blocking is an inverted index, and having the
cheap layer means the expensive one is never asked whether "Ada Lovelace" and "ada
lovelace" are the same person.

THE FAILURE MODE THIS MUST NOT HAVE, named by the companion roadmap item: "unmatched
name -> single-element cluster (never silently dropped)". A resolver that drops what
it cannot match loses entities with no signal at all, which is strictly worse than
refusing to resolve them. Every input name therefore appears in exactly one block,
and a test asserts that over the whole set rather than one example.

WHY THE CAP EXISTS AND WHY IT CANNOT DROP. An unbounded block hands the arbitration
layer the whole corpus, which is the cost blocking exists to avoid. But a cap
implemented by truncation would silently reintroduce the dropped-entity failure, so
overflow SPLITS into further blocks instead: coverage is preserved and the bound
holds.

Pure: no model, no clock, no I/O. Blocking decides what a human or model will be
asked about, so it must be replayable -- a reviewer comparing two runs needs to see a
real change rather than a reshuffle.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

#: Upper bound on one block, from the item's own "50-100-item blocks".
MAX_BLOCK_SIZE = 100

#: Tokens too common to discriminate. Blocking on one of these would put every
#: person in one block and hand the arbitration layer the corpus.
_STOP_TOKENS = frozenset({"the", "of", "and", "dr", "mr", "mrs", "ms", "prof", "inc", "ltd"})

_WORD = re.compile(r"[a-z0-9]+")


def name_tokens(name: object) -> frozenset[str]:
    """Discriminating lowercase tokens of *name*.

    Casing, punctuation and spacing are normalised away, because those are exactly
    the "easy cases" the item says deterministic logic should already handle -- a
    model call to decide whether ``O'Brien, Ada`` matches ``ada o brien`` is money
    spent on a solved problem.
    """
    text = str(name or "").lower()
    return frozenset(t for t in _WORD.findall(text) if t not in _STOP_TOKENS and len(t) > 1)


def build_blocks(names: Iterable[object]) -> dict[str, list[str]]:
    """Group *names* into candidate blocks by shared token.

    Returns ``{block_key: [names]}``, every list sorted and every input name
    present in exactly one block. Union-find over shared tokens, so ``A~B`` and
    ``B~C`` put all three together even when A and C share nothing -- a
    transitivity a pairwise scan would miss and a reviewer would have to rediscover.

    A name with no discriminating tokens still gets its own block: unmatched means
    a cluster of one, never a drop.
    """
    cleaned = [str(n).strip() for n in (names or ()) if str(n or "").strip()]

    parent: dict[str, str] = {n: n for n in cleaned}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Lexicographic root keeps the result independent of input order.
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    by_token: dict[str, list[str]] = {}
    for name in cleaned:
        for tok in name_tokens(name):
            by_token.setdefault(tok, []).append(name)
    for members in by_token.values():
        first = members[0]
        for other in members[1:]:
            union(first, other)

    grouped: dict[str, list[str]] = {}
    for name in cleaned:
        grouped.setdefault(find(name), []).append(name)

    # Enforce the cap by SPLITTING, never truncating: a dropped name is the one
    # failure this module is not allowed to have.
    out: dict[str, list[str]] = {}
    for key, members in grouped.items():
        members = sorted(set(members))
        if len(members) <= MAX_BLOCK_SIZE:
            out[key] = members
            continue
        for i in range(0, len(members), MAX_BLOCK_SIZE):
            chunk = members[i : i + MAX_BLOCK_SIZE]
            out[f"{key}#{i // MAX_BLOCK_SIZE}"] = chunk
    return out


def block_for(blocks: Mapping[str, Sequence[str]], name: object) -> list[str]:
    """The block containing *name*, or ``[name]`` when it is in none.

    Returning a single-element list rather than ``[]`` is the enforced failure
    mode: a caller must never receive "nothing" for a name it asked about, because
    "nothing" reads as "no entity" instead of "no match".
    """
    target = str(name or "").strip()
    if not target:
        return []
    for members in blocks.values():
        if target in members:
            return sorted(members)
    return [target]


__all__ = ["MAX_BLOCK_SIZE", "block_for", "build_blocks", "name_tokens"]
