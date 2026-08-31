"""v4 embedding auto-derivation pipeline (Group A — closes the
"caller-supplied" gap from round 2 audit).

Round 2 multi-LLM audit (4/4 model consensus 2026-05-10) flagged that
``surprise_weighted`` takes embeddings as a caller-supplied input — meaning the recall log is
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

import contextlib
import hashlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from ..observability import get_logger
from .feature_flags import require_enabled

_log = get_logger("v4.embedding_pipeline")

#: The v3 recall index, relative to the workspace. Its ``blocks`` table
#: has no ``content`` column — block text is rebuilt from ``json_blob``.
_RECALL_DB_REL = ".mind-mem-index/recall.db"

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

    Block text is looked up in two places, in order:

    1. ``<workspace>/index.db``, table ``blocks(id, content)`` — the
       caller-populated table this module was written against. Nothing
       in the v3 write path fills it, so on a stock deployment it is
       absent or empty.
    2. ``<workspace>/.mind-mem-index/recall.db`` — the real v3 index.
       Its ``blocks`` table has no ``content`` column, so the text is
       rebuilt from the field values in ``json_blob``.

    Missing blocks are skipped (no key in the output). Empty content
    rows produce zero vectors so callers can detect the degenerate case.

    Returns ``{block_id: embedding}``. When neither source holds any of
    the requested ids the result is an empty dict — fail-soft like the
    rest of the v4 read surface, but logged rather than silent, because
    at the call site "no content anywhere" and "nothing to embed" look
    identical and only one of them is a broken deployment.
    """
    require_enabled(FLAG)
    ids = list(block_ids)
    if not ids:
        return {}
    root = Path(workspace)
    contents = _content_from_index_db(root / "index.db", ids)
    missing = [bid for bid in ids if bid not in contents]
    if missing:
        contents.update(_content_from_recall_db(root / _RECALL_DB_REL, missing))
    if not contents:
        _log.warning(
            "embedding_pipeline_no_content",
            workspace=str(root),
            requested=len(ids),
            msg="No block content found in index.db(blocks) or the v3 recall index; no embeddings derived.",
        )
        return {}
    return {bid: _active_embedder(text or "", dim) for bid, text in contents.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_from_index_db(db: Path, ids: list[str]) -> dict[str, str]:
    """Content rows from a caller-populated ``blocks(id, content)`` table."""
    if not db.is_file():
        return {}
    with contextlib.closing(sqlite3.connect(db, timeout=30)) as conn:
        if not _table_exists(conn, "blocks"):
            return {}
        placeholders = ",".join("?" * len(ids))
        try:
            rows = conn.execute(
                f"SELECT id, content FROM blocks WHERE id IN ({placeholders})",  # nosec B608 — placeholders is solely "?,?,..,?"; ids are bound parameters
                ids,
            ).fetchall()
        except sqlite3.OperationalError:
            # A ``blocks`` table without a ``content`` column — this is
            # the v3 index schema, handled by _content_from_recall_db.
            return {}
    return {bid: (content or "") for bid, content in rows}


def _content_from_recall_db(db: Path, ids: list[str]) -> dict[str, str]:
    """Block text rebuilt from the v3 recall index (``blocks.json_blob``)."""
    if not db.is_file():
        return {}
    with contextlib.closing(sqlite3.connect(db, timeout=30)) as conn:
        if not _table_exists(conn, "blocks"):
            return {}
        placeholders = ",".join("?" * len(ids))
        try:
            rows = conn.execute(
                f"SELECT id, json_blob FROM blocks WHERE id IN ({placeholders})",  # nosec B608 — placeholders is solely "?,?,..,?"; ids are bound parameters
                ids,
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
    return {bid: _blob_text(blob) for bid, blob in rows}


def _blob_text(json_blob: str | None) -> str:
    """Flatten a stored block into embeddable text.

    Field order follows the stored JSON (insertion order), so the same
    row always renders the same string — the embedder must be
    deterministic on a given corpus. Bookkeeping keys (``_id``,
    ``_line``, ...) are dropped; they carry no content signal.
    """
    if not json_blob:
        return ""
    try:
        blob = json.loads(json_blob)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(blob, dict):
        return ""
    parts: list[str] = []
    for key, value in blob.items():
        if key.startswith("_"):
            continue
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value if isinstance(v, (str, int, float)))
        elif isinstance(value, (int, float)):
            parts.append(str(value))
    return "\n".join(p for p in parts if p)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


# Type-keepalive for the public Sequence import (used by callers that
# pass arbitrary iterables in).
_keepalive: Sequence[float] | None = None
