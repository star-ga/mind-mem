#!/usr/bin/env python3
"""Canonical digests the recall attestation commits to.

A leaf module: standard library only, **no first-party imports at all**.
That is the point of splitting it out rather than leaving the encodings
inside :mod:`mind_mem.recall_attestation`.

* The served-set encoding has more than one consumer coming — the served-set
  ledger derives its run id from exactly these bytes. Letting that ledger
  reach the encoding through the attestation module would put a ledger one
  import hop from the scoring path, which is the edge
  ``tests/test_recall_attestation_v2.py`` exists to forbid. A leaf both can
  depend on has no such direction.
* One object, one encoding, one owner. Two spellings of "the served set" is
  precisely the drift a canonical form exists to prevent, and a shared home
  makes a second spelling obvious rather than plausible.

Every function here is pure: same input, same bytes, no clock, no I/O, no
randomness. Each digest carries its **own** domain tag, so a value computed
for one role can never be substituted for another of the same shape.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence


def seq_digest(items: tuple[str, ...]) -> str:
    """Unambiguous SHA-256 over an *ordered* sequence of strings.

    Length-prefixed, then each item folded in as its fixed-width (32-byte)
    SHA-256 digest, so neither element boundaries nor ordering can be forged.
    Mirrors ``fold_attestation.seq_digest`` for the same reason: a leg set
    ``("bm25", "vector")`` must hash distinctly from ``("bm25vector",)``.
    """
    h = hashlib.sha256()
    h.update(str(len(items)).encode("ascii"))
    h.update(b"\x00")
    for it in items:
        h.update(hashlib.sha256(it.encode("utf-8")).digest())
    return h.hexdigest()


#: Domain tag for the served-set digest. Deliberately its OWN tag rather
#: than a slot inside the attestation preimage: the served set is a
#: standalone commitment (the served-set ledger will need the identical
#: bytes), and a shared tag would let a served-set digest be substituted
#: for some other sequence digest of the same shape.
SERVED_SET_TAG = b"MM_SERVED_v1\x00"

#: Domain tag for the query digest, separate from the served-set tag so one
#: string hashed in the two roles can never produce the same value.
QUERY_TAG = b"MM_QUERY_v1\x00"

_U32_MAX = 0xFFFF_FFFF


def _u32be(n: int) -> bytes:
    """Fixed-width big-endian length prefix, range-checked."""
    if not 0 <= n <= _U32_MAX:
        raise ValueError(f"length {n} does not fit a u32 length prefix")
    return struct.pack(">I", n)


def served_set_digest(served_ids: Sequence[str]) -> str:
    """The ONE canonical digest of a served answer: ids in **rank order**.

    ``SHA256(MM_SERVED_v1 \0 ‖ u32be(n) ‖ (u32be(len_i) ‖ utf8(id_i))*)``

    Length-prefixed, so concatenation is unambiguous: ``("AB", "C")`` and
    ``("A", "BC")`` are different answers and hash differently. Order is
    preserved and duplicates are kept — the ranking *is* the thing being
    committed to, and collapsing it to a set would put the collision the
    determinism seam closed straight back.

    Deliberately **not** :func:`seq_digest`, which owns the *leg tuples*.
    One object, one encoding, one owner: two spellings of "the served set"
    is precisely the drift a canonical form exists to prevent. It also
    cannot go through :func:`~mind_mem.preimage.preimage`, whose NUL
    separator rejects any field containing NUL — a length-prefixed frame
    is byte-transparent, so no id can fail to hash.
    """
    digest = hashlib.sha256()
    digest.update(SERVED_SET_TAG)
    ids = tuple(served_ids)
    digest.update(_u32be(len(ids)))
    for served_id in ids:
        raw = str(served_id).encode("utf-8")
        digest.update(_u32be(len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def query_hash(query: str) -> str:
    """Digest of the question a run answered — the last unbound run input.

    ``SHA256(MM_QUERY_v1 \0 ‖ u32be(len) ‖ utf8(query))``. A *digest*, not
    the text: the envelope already carries the query verbatim, so the
    record commits to the question without restating it. Length-prefixed
    and NUL-transparent for the same reason as the served set — a query is
    arbitrary user input and must never be able to break its own hash.
    """
    raw = str(query).encode("utf-8")
    return hashlib.sha256(QUERY_TAG + _u32be(len(raw)) + raw).hexdigest()


def marker_digest(marker: dict[str, str] | None) -> str:
    """Deterministic SHA-256 over a ``.degraded`` marker dict (``""`` when None).

    The marker ``{leg, reason, [variants_degraded, variants_total]}`` is
    serialised with ``sort_keys`` so key order cannot change the digest, then
    hashed. Binding this digest into the preimage means the readable
    ``degraded`` field carried on the attestation cannot be swapped without
    invalidating the hash.
    """
    if not marker:
        return ""
    canonical = json.dumps(marker, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "QUERY_TAG",
    "SERVED_SET_TAG",
    "marker_digest",
    "query_hash",
    "seq_digest",
    "served_set_digest",
]
