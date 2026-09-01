# Copyright 2026 STARGA, Inc.
"""The canonical pure-Python floor for the four MIND hot-path kernels.

MIND kernels are **optional** in mind-mem: no wheel ships a
``libmindmem.so`` and the package is fully functional without one. This
module is what the runtime falls back to, and since 5.1.0 it is the
*only* fallback — :func:`mind_mem.mind_ffi.load_kernels` binds it
directly, so there is one loader and one Python implementation of each
kernel instead of a compiled path and a drifting shadow copy.

The four kernels:

- BM25F scoring          → :func:`bm25f_score`
- SHA3-512 chain verify  → :func:`sha3_512_chain_verify`
- Vector similarity      → :func:`cosine`, :func:`dot`
- RRF fusion             → :func:`rrf_fusion`

Delegation, not duplication
---------------------------
Three of these had a *second* implementation of arithmetic that already
lived — and was already tested — elsewhere in the package. They now call
the incumbent rather than restating it:

- :func:`bm25f_score`   → :mod:`mind_mem._recall_scoring`
  (``compute_weighted_tf`` + ``bm25f_score_terms``), the single source of
  truth for every BM25 computation in recall. The old body summed field
  contributions as ``count(term) * weight`` where the scorer accumulates
  ``+= weight`` per token; for the real ``FIELD_WEIGHTS`` (1.2, 0.8, 0.5,
  0.3 — none of them exact in binary floating point) the two disagree in
  the last bits, so "the fallback matches recall" was true only by
  eyeball. It is now true by construction.
- :func:`cosine`        → :func:`mind_mem.vector_inertness.cosine`.
- :func:`sha3_512_chain_verify` → the entry-hash schemes defined in
  :mod:`mind_mem.hash_chain_v2`, so a change to the canonical preimage
  cannot leave this verifier behind.

:func:`rrf_fusion` keeps its own body: the incumbent
``hybrid_recall.rrf_fuse`` fuses *result dicts* with per-list weights and
emits fusion provenance, which is a different contract from fusing bare
id lists. ``tests/test_mind_kernels_wiring.py`` differences the two so
they cannot drift apart.

Clock discipline
----------------
Every function here is a pure function of its arguments. Nothing in this
module reads a clock or a random source — recall must be a pure function
of (corpus, config, scoring_instant), and a fallback kernel on the scored
path is not the place to break that.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .mind_ffi import Kernels

# Kept as the module's advertised BM25 defaults. Imported lazily inside
# :func:`bm25f_score` (see the note there) so importing this module stays
# as cheap as it was when it had no package dependencies at all.
_DEFAULT_K1 = 1.2
_DEFAULT_B = 0.75


# ---------------------------------------------------------------------------
# BM25F scoring — delegates to the recall scorer
# ---------------------------------------------------------------------------


def bm25f_score(
    query_terms: Sequence[str],
    doc_fields: Mapping[str, Sequence[str]],
    field_weights: Mapping[str, float],
    doc_length: float,
    avg_doc_length: float,
    k1: float = _DEFAULT_K1,
    b: float = _DEFAULT_B,
) -> float:
    """BM25F score for one document, identical to the recall scorer's.

    Args:
        query_terms: Tokenised query terms. A term repeated in the query
            contributes twice, matching ``bm25f_score_terms``.
        doc_fields: ``{field: [tokens]}`` for a single document.
        field_weights: Per-field multipliers. An empty mapping means "use
            the package defaults" (``FIELD_WEIGHTS``), which is what the
            recall scorer does with a falsy ``field_weights``.
        doc_length: Weighted document length. Pass the ``wdl`` that
            ``compute_weighted_tf`` returns for exact agreement with
            recall; a raw token count is accepted but is a different
            normalisation.
        avg_doc_length: Average weighted document length over the corpus.
            Non-positive values are clamped to 1.0 so a single-document
            corpus cannot divide by zero.
        k1: Term-frequency saturation (defaults to ``BM25_K1``).
        b: Length normalisation (defaults to ``BM25_B``).

    Returns:
        The BM25F score with every IDF taken as 1.0. IDF is a corpus-level
        statistic that this per-document entry point has no way to
        compute; a caller holding an ``idf_cache`` should call
        ``bm25f_score_terms`` directly rather than post-multiplying.
    """
    # Imported here, not at module scope: ``mind_ffi`` imports this module
    # eagerly to bind the fallback, and ``category_distiller`` imports
    # ``mind_ffi`` at import time. Pulling the recall scorer in at that
    # point would drag ``_recall_constants`` into every process that only
    # wanted to ask whether a .so exists.
    from ._recall_scoring import bm25f_score_terms, compute_weighted_tf

    if avg_doc_length <= 0:
        avg_doc_length = 1.0
    weighted_tf, _wdl = compute_weighted_tf(
        {field: list(tokens) for field, tokens in doc_fields.items()},
        dict(field_weights) if field_weights else None,
    )
    idf_cache = {term: 1.0 for term in query_terms}
    return bm25f_score_terms(
        list(query_terms),
        weighted_tf,
        float(doc_length),
        idf_cache,
        float(avg_doc_length),
        k1=k1,
        b=b,
    )


# ---------------------------------------------------------------------------
# SHA3-512 chain verify
# ---------------------------------------------------------------------------


def sha3_512_chain_verify(
    entries: Sequence[Mapping[str, object]],
    *,
    previous_hash: str | None = None,
) -> bool:
    """Verify a chain **as a sequence**, downgrade-monotonically.

    Walks ``{entry_id, timestamp, block_id, action, content_hash,
    previous_hash, entry_hash}`` mappings in order and returns True only
    when every entry re-hashes to its stored ``entry_hash`` *and* each
    ``previous_hash`` equals the prior entry's ``entry_hash``.

    The rule this exists for: **once a v3-scheme entry is seen, no later
    entry may verify under the legacy v1 scheme.** The v1 preimage joins
    fields with ``|``, so a field value containing ``|`` can shift a
    boundary without changing the digest; leaving v1 acceptable *after*
    the v2.10.0 upgrade would let an attacker append forged history to a
    hardened chain and keep it verifying. Per-entry verification cannot
    enforce this — it has no memory of what the chain has already
    proven — which is exactly why a sequence verifier is a distinct
    surface and not a loop around ``verify_entry``.

    Args:
        entries: The chain segment, oldest first.
        previous_hash: Digest the first entry must link to. ``None``
            (default) accepts whatever the first entry claims, which is
            right for verifying a detached segment and wrong for
            appending one to a live ledger — pass the ledger head there.

    Returns:
        True if the segment is internally consistent and never downgrades.
        An empty segment is vacuously valid.
    """
    if not entries:
        return True

    # Lazy import so this module stays importable while the package is
    # only partly loaded, and so the one definition of each entry-hash
    # scheme lives with the chain that writes them.
    from .hash_chain_v2 import _compute_entry_hash_v1, _compute_entry_hash_v3

    prev = previous_hash if previous_hash is not None else entries[0].get("previous_hash")
    seen_v3 = False  # downgrade-attack mitigation
    for entry in entries:
        if entry.get("previous_hash") != prev:
            return False
        fields = (
            str(entry.get("entry_id", "")),
            str(entry.get("timestamp", "")),
            str(entry.get("block_id", "")),
            str(entry.get("action", "")),
            str(entry.get("content_hash", "")),
            str(entry.get("previous_hash", "")),
        )
        stored = entry.get("entry_hash")
        if stored == _compute_entry_hash_v3(*fields):
            seen_v3 = True
        elif seen_v3:
            # Downgrade blocked: the chain has already produced a v3
            # entry, so the legacy scheme is not even consulted.
            return False
        elif stored != _compute_entry_hash_v1(*fields):
            return False
        prev = stored
    return True


def sha3_512_hex(data: bytes) -> str:
    """SHA3-512 of *data* as lowercase hex — the digest the chain uses."""
    return hashlib.sha3_512(data).hexdigest()


# ---------------------------------------------------------------------------
# Vector similarity
# ---------------------------------------------------------------------------


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 for an empty, mismatched or zero vector.

    Delegates to :func:`mind_mem.vector_inertness.cosine` — the copy that
    the inertness detector already depends on and that already has tests.
    """
    from .vector_inertness import cosine as _incumbent

    return _incumbent(list(a), list(b))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product; 0.0 for an empty or mismatched pair.

    No incumbent to defer to: every other dot product in the package is
    inlined inside a cosine or a matrix routine, so this stays the one
    standalone definition.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def rrf_fusion(
    ranklists: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over bare id lists.

    ``ranklists`` is a list of ordered id lists (one per retrieval axis);
    each id scores ``sum_i 1 / (k + rank_i)`` over the lists it appears
    in. Returns ``(id, score)`` pairs sorted by descending score, ties
    broken by id so the output is a deterministic function of the input.

    Not merged into ``hybrid_recall.rrf_fuse``: that one fuses result
    *dicts*, applies per-list weights, resolves duplicate metadata by
    recency and emits fusion provenance. Same formula, different
    contract. ``tests/test_mind_kernels_wiring.py`` pins them to the same
    numbers on the inputs both can express.
    """
    scores: dict[str, float] = {}
    for rl in ranklists:
        for rank, bid in enumerate(rl, start=1):
            scores[bid] = scores.get(bid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# Loader — one door, in mind_ffi
# ---------------------------------------------------------------------------


def load_kernels(path: str | None = None) -> "Kernels":
    """Resolve the kernel binding. Delegates to the one loader.

    This used to be a *second* loader: it read ``MIND_MEM_KERNELS_SO``
    and handed the value straight to ``ctypes.CDLL`` with no allowlist,
    so any path in the environment could load any shared object into the
    process — while ``mind_ffi`` had, twenty lines away, an allowlisted
    probe that refused exactly that. Two loaders with two security
    postures is one loader too many, and the weaker one is gone:
    ``MIND_MEM_KERNELS_SO`` is still honoured, but through
    :func:`mind_mem.mind_ffi.load_kernels`, which resolves it against the
    same allowlist as ``MIND_MEM_LIB`` and reports a rejection instead of
    swallowing it.

    Args:
        path: Explicit library path, allowlist-checked like the env vars.

    Returns:
        A :class:`mind_mem.mind_ffi.Kernels` binding. Its four kernel
        callables are the functions in this module in every case; the
        compiled library, when one is present, is exposed as ``.native``
        for the batched ABI (``rrf_fuse``, ``bm25f_batch``, ...).
    """
    from .mind_ffi import load_kernels as _one_loader

    return _one_loader(path)


__all__ = [
    "bm25f_score",
    "sha3_512_chain_verify",
    "sha3_512_hex",
    "cosine",
    "dot",
    "rrf_fusion",
    "load_kernels",
]
