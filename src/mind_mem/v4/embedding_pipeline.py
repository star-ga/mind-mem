"""v4 embedding auto-derivation pipeline (Group A — closes the
"caller-supplied" gap from round 2 audit).

Round 2 multi-LLM audit (4/4 model consensus 2026-05-10) flagged that
``surprise_weighted`` and ``consolidation_worker`` both take
embeddings as caller-supplied inputs — meaning the recall log is
unreachable to the kernel layer without external glue. This module
closes that gap.

Strategy:

    Default embedder is **TF-IDF over hashed character n-grams** —
    pure stdlib, no external dependencies, deterministic, fast on
    short text. Good enough as a recall-time signal for surprise
    scoring; not a replacement for sentence-transformers.

    Pluggable: callers register a better embedder via
    :func:`set_embedder` (e.g. one that calls Ollama or sentence-
    transformers). The auto-derivation contract stays the same.

API:

    derive_embedding(text, dim=128) -> list[float]
    derive_embeddings(workspace, block_ids, dim=128) -> {block_id: vec}
    set_embedder(fn) -> register an alternate embedder

Design notes:

    Hashed n-grams use Python's stable ``hash()`` with a per-process
    salt (set once at import time) so the same input produces the
    same vector across calls in one process. Cross-process stability
    requires a stable hash function — production deployments can
    install a ``hashlib``-backed embedder via :func:`set_embedder`.

The default embedder is intentionally simple. The audit's primary
ask was that the *plumbing* for auto-derivation be in place; the
quality of the default embedder is a tunable, not a contract.

Feature-flag gated under ``v4.embedding_pipeline``.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from .feature_flags import require_enabled

__all__ = [
    "FLAG",
    "Embedder",
    "set_embedder",
    "derive_embedding",
    "derive_embeddings",
    "default_embedder",
]


FLAG: str = "embedding_pipeline"

#: An embedder takes (text, dim) and returns a fixed-size vector.
Embedder = Callable[[str, int], list[float]]


def default_embedder(text: str, dim: int = 128) -> list[float]:
    """Hashed TF-IDF over character 3-grams. Deterministic, dependency-free.

    Steps:
        1. Lowercase + strip.
        2. Build the multiset of overlapping 3-grams.
        3. For each n-gram, hash to a bucket index in ``[0, dim)``
           via ``hashlib.blake2b`` (cross-process-stable, unlike
           ``hash()``).
        4. Bucket value = log(1 + count) so frequent grams don't
           swamp rare ones.
        5. L2-normalise so cosine distance becomes the dominant
           similarity signal.

    Returns a vector of zeros for empty / whitespace-only input.
    """
    if not text or not text.strip() or dim <= 0:
        return [0.0] * max(0, dim)
    text = text.lower().strip()
    grams: Counter[str] = Counter()
    if len(text) < 3:
        grams[text] = 1
    else:
        for i in range(len(text) - 2):
            grams[text[i : i + 3]] += 1

    bucket: list[float] = [0.0] * dim
    for gram, count in grams.items():
        h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h, "little") % dim
        bucket[idx] += math.log1p(count)

    norm = math.sqrt(sum(x * x for x in bucket))
    if norm == 0.0:
        return bucket
    return [x / norm for x in bucket]


_active_embedder: Embedder = default_embedder


def set_embedder(fn: Embedder) -> None:
    """Swap the active embedder.

    Production deployments install a real embedder (sentence-
    transformers, Ollama, OpenAI) at startup. The kernel layer calls
    :func:`derive_embedding` / :func:`derive_embeddings` and never
    cares which backend is active.
    """
    require_enabled(FLAG)
    global _active_embedder
    _active_embedder = fn


def derive_embedding(text: str, *, dim: int = 128) -> list[float]:
    """Embed ``text`` with the active embedder."""
    require_enabled(FLAG)
    return _active_embedder(text, dim)


def derive_embeddings(
    workspace: str | Path,
    block_ids: Iterable[str],
    *,
    dim: int = 128,
) -> dict[str, list[float]]:
    """Auto-derive embeddings for the given block IDs from their content.

    Reads block content from the v3 ``blocks(id, content)`` table.
    Missing blocks are skipped (no key in the output). Empty content
    rows produce zero vectors so callers can detect the degenerate
    case.

    Returns ``{block_id: embedding}``. Empty dict when the database
    or the blocks table is missing — fail-soft same as the rest of
    the v4 read surface.
    """
    require_enabled(FLAG)
    out: dict[str, list[float]] = {}
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return out
    ids = list(block_ids)
    if not ids:
        return out
    with sqlite3.connect(db, timeout=30) as conn:
        if not _table_exists(conn, "blocks"):
            return out
        # Pull content for the requested ids.
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, content FROM blocks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    for bid, content in rows:
        out[bid] = _active_embedder(content or "", dim)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


# Type-keepalive for the public Sequence import (used by callers that
# pass arbitrary iterables in).
_keepalive: Sequence[float] | None = None
