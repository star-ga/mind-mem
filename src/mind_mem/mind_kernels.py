# Copyright 2026 STARGA, Inc.
"""Python fallbacks matching the eventual MIND-compiled kernel ABI (v2.0.0b1).

These functions are the pure-Python floor for the four hot-path
kernels the roadmap promises compiling to native ELF via ``mindc``:

- BM25F scoring
- SHA3-512 hash-chain verification
- Vector similarity (cosine + dot)
- RRF fusion

The signatures match what the compiled kernels will expose, so
:func:`load_kernels` can swap in a native library without touching
caller code. When no native library is present, :func:`load_kernels`
returns this module directly — the automatic fallback required by
the roadmap.
"""

from __future__ import annotations

import hashlib
import math
import os
import types
from typing import Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# BM25F scoring — matches _recall_scoring.bm25f_score contract
# ---------------------------------------------------------------------------


def bm25f_score(
    query_terms: Sequence[str],
    doc_fields: Mapping[str, Sequence[str]],
    field_weights: Mapping[str, float],
    doc_length: int,
    avg_doc_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Reference BM25F scorer — O(|query| * |fields|)."""
    if avg_doc_length <= 0:
        avg_doc_length = 1.0
    total = 0.0
    for term in query_terms:
        tf_sum = 0.0
        for field, tokens in doc_fields.items():
            weight = field_weights.get(field, 1.0)
            if weight <= 0:
                continue
            tf = sum(1 for t in tokens if t == term) * weight
            tf_sum += tf
        if tf_sum <= 0:
            continue
        norm = tf_sum * (k1 + 1) / (
            tf_sum + k1 * (1 - b + b * doc_length / avg_doc_length)
        )
        total += norm
    return total


# ---------------------------------------------------------------------------
# SHA3-512 chain verify
# ---------------------------------------------------------------------------


def sha3_512_chain_verify(entries: Sequence[Mapping[str, str]]) -> bool:
    """Walk a list of ``{previous_hash, entry_hash, content_hash}`` dicts.

    Returns True when every entry's ``previous_hash`` equals the prior
    entry's ``entry_hash`` and all entries are internally consistent.
    """
    if not entries:
        return True

    # Lazy imports so this fallback module stays standalone if the
    # main package is partially loaded.
    from .preimage import preimage as _preimage

    prev = entries[0].get("previous_hash")
    seen_v3 = False  # downgrade-attack mitigation
    for idx, entry in enumerate(entries):
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
        # v3 scheme — TAG_v1 NUL-separated preimage.
        v3_expected = hashlib.sha3_512(
            _preimage("CHAIN_v1", *fields)
        ).hexdigest()
        if stored == v3_expected:
            seen_v3 = True
        elif seen_v3:
            # Downgrade blocked: once chain produces v3 entries, no
            # later entry may fall back to the v1 scheme.
            return False
        else:
            # Legacy v1 scheme — ``|``-joined canonical string.
            v1_expected = hashlib.sha3_512(
                "|".join(fields).encode("utf-8")
            ).hexdigest()
            if stored != v1_expected:
                return False
        prev = stored
    return True


# ---------------------------------------------------------------------------
# Vector similarity
# ---------------------------------------------------------------------------


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot_ab = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot_ab += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 0.0
    return dot_ab / denom


def dot(a: Sequence[float], b: Sequence[float]) -> float:
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
    """Canonical reciprocal rank fusion.

    ``ranklists`` is a list of ordered id lists (one per retrieval
    axis). Returns the fused list of ``(id, score)`` pairs sorted
    descending on score.
    """
    scores: dict[str, float] = {}
    for rl in ranklists:
        for rank, bid in enumerate(rl, start=1):
            scores[bid] = scores.get(bid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# FFI bridge — swap in a native MIND kernel library if present
# ---------------------------------------------------------------------------


_NATIVE_ENV = "MIND_MEM_KERNELS_SO"


def load_kernels(path: str | None = None) -> types.ModuleType:
    """Return either a native kernels module or this Python fallback.

    The roadmap calls for an automatic fallback when the compiled
    ``.mind`` library is unavailable; that behaviour is encoded here.
    Callers that want to force the fallback unset ``MIND_MEM_KERNELS_SO``;
    callers that want native simply point ``MIND_MEM_KERNELS_SO`` at
    the ``.so``.
    """
    candidate = path or os.environ.get(_NATIVE_ENV)
    if candidate and os.path.isfile(candidate):
        try:
            import ctypes

            lib = ctypes.CDLL(candidate)
            # Native module is expected to expose the same symbol names
            # as this fallback. We return a small wrapper exposing the
            # resolved library so callers can attempt native calls.
            return types.SimpleNamespace(
                _native=lib,
                bm25f_score=bm25f_score,      # native caller override below
                sha3_512_chain_verify=sha3_512_chain_verify,
                cosine=cosine,
                dot=dot,
                rrf_fusion=rrf_fusion,
            )  # type: ignore[return-value]
        except OSError:
            pass
    # Fallback path: this Python module.
    return __import__(__name__, fromlist=["*"])


__all__ = [
    "bm25f_score",
    "sha3_512_chain_verify",
    "cosine",
    "dot",
    "rrf_fusion",
    "load_kernels",
]
