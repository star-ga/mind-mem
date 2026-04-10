#!/usr/bin/env python3
"""mind-mem 4-layer deduplication filter for search results.

Applies a configurable pipeline of deduplication strategies to remove
redundant, near-duplicate, and over-represented results from recall output.

Layers:
    1. Best chunk per source — keep only the highest-scoring chunk from each
       source block/file.
    2. Cosine similarity dedup — remove results whose text is too similar
       (cosine > threshold) to a higher-scoring result already kept.
    3. Type diversity cap — limit results per entity type (e.g., max N from
       the same block type prefix like DIA-, FACT-, D-).
    4. Per-source chunk cap — limit total chunks from any single source file.

Configuration (mind-mem.json):
    {
      "recall": {
        "dedup": {
          "enabled": true,
          "best_per_source": true,
          "cosine_threshold": 0.85,
          "cosine_enabled": true,
          "type_cap": 3,
          "type_cap_enabled": true,
          "source_cap": 5,
          "source_cap_enabled": true
        }
      }
    }
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .observability import get_logger, metrics

_log = get_logger("dedup")

__all__ = [
    "DedupConfig",
    "deduplicate_results",
    "layer_best_per_source",
    "layer_cosine_dedup",
    "layer_vector_cosine_dedup",
    "layer_type_diversity_cap",
    "layer_source_chunk_cap",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class DedupConfig:
    """Immutable configuration for the 4-layer dedup pipeline.

    All layers default to enabled with sensible thresholds. Individual layers
    can be toggled off via the config dict.
    """

    __slots__ = (
        "enabled",
        "best_per_source",
        "cosine_enabled",
        "cosine_threshold",
        "vector_cosine_enabled",
        "type_cap_enabled",
        "type_cap",
        "source_cap_enabled",
        "source_cap",
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.best_per_source: bool = bool(cfg.get("best_per_source", True))
        self.cosine_enabled: bool = bool(cfg.get("cosine_enabled", True))
        self.cosine_threshold: float = float(cfg.get("cosine_threshold", 0.85))
        self.vector_cosine_enabled: bool = bool(cfg.get("vector_cosine_enabled", False))
        self.type_cap_enabled: bool = bool(cfg.get("type_cap_enabled", True))
        self.type_cap: int = int(cfg.get("type_cap", 3))
        self.source_cap_enabled: bool = bool(cfg.get("source_cap_enabled", True))
        self.source_cap: int = int(cfg.get("source_cap", 5))

        # Clamp thresholds to valid ranges
        self.cosine_threshold = max(0.0, min(1.0, self.cosine_threshold))
        self.type_cap = max(1, self.type_cap)
        self.source_cap = max(1, self.source_cap)

    @staticmethod
    def from_recall_config(recall_cfg: dict[str, Any]) -> "DedupConfig":
        """Extract dedup sub-config from a recall configuration dict."""
        dedup_cfg = recall_cfg.get("dedup")
        if dedup_cfg is None or not isinstance(dedup_cfg, dict):
            return DedupConfig()
        return DedupConfig(dedup_cfg)


# ---------------------------------------------------------------------------
# Text tokenization (lightweight, no external deps)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "it", "its",
        "and", "or", "but", "not", "no", "if", "then", "so", "up", "out",
        "that", "this", "these", "those", "he", "she", "they", "we", "you",
        "i", "me", "my", "your", "his", "her", "our", "their",
    }
)


def _text_tokens(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens, removing stop words."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Layer 1: Best chunk per source
# ---------------------------------------------------------------------------


def _extract_source_key(result: dict[str, Any]) -> str:
    """Derive a source grouping key from a result dict.

    For chunked results (IDs like ``D-20260222-001.0``), the key is the base
    block ID (``D-20260222-001``).  For non-chunked results, the key is the
    ``_id`` itself.
    """
    rid = result.get("_id", "")
    # Chunked IDs have a ``.N`` numeric suffix
    if "." in rid:
        base, suffix = rid.rsplit(".", 1)
        if suffix.isdigit():
            return base
    return rid


def layer_best_per_source(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the highest-scoring result per source block.

    When multiple chunks originate from the same parent block (identified by
    stripping the ``.N`` chunk suffix), only the chunk with the highest score
    is retained.

    Args:
        results: Scored search results sorted by descending score.

    Returns:
        Filtered results preserving original sort order.
    """
    best: dict[str, dict[str, Any]] = {}
    for r in results:
        key = _extract_source_key(r)
        existing = best.get(key)
        if existing is None or r.get("score", 0) > existing.get("score", 0):
            best[key] = r

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in results:
        key = _extract_source_key(r)
        if key not in seen and best.get(key) is r:
            seen.add(key)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Layer 2: Cosine similarity dedup
# ---------------------------------------------------------------------------


def _term_vector(tokens: list[str]) -> dict[str, int]:
    """Build a term-frequency vector from a token list."""
    vec: dict[str, int] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec


def _cosine_similarity(vec_a: dict[str, int], vec_b: dict[str, int]) -> float:
    """Compute cosine similarity between two term-frequency vectors.

    Returns a value in [0, 1]. Returns 0.0 if either vector is empty.
    """
    if not vec_a or not vec_b:
        return 0.0

    # Iterate over the smaller vector for efficiency
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a

    dot = 0
    for term, freq_a in vec_a.items():
        freq_b = vec_b.get(term, 0)
        if freq_b:
            dot += freq_a * freq_b

    if dot == 0:
        return 0.0

    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def _get_result_text(result: dict[str, Any]) -> str:
    """Extract searchable text from a result for similarity comparison."""
    parts = []
    for key in ("excerpt", "content", "tags"):
        val = result.get(key, "")
        if val:
            parts.append(str(val))
    return " ".join(parts)


def layer_cosine_dedup(
    results: list[dict[str, Any]],
    threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """Remove near-duplicate results based on cosine text similarity.

    Iterates through results in score-descending order. For each result, if
    its text has cosine similarity >= ``threshold`` with any already-kept
    result, it is dropped.

    Args:
        results: Scored search results sorted by descending score.
        threshold: Similarity threshold above which a result is considered
            a duplicate. Must be in [0, 1].

    Returns:
        Filtered results with near-duplicates removed.
    """
    if threshold <= 0.0:
        return results
    if threshold > 1.0:
        return results

    kept: list[dict[str, Any]] = []
    kept_vectors: list[dict[str, int]] = []

    for r in results:
        text = _get_result_text(r)
        tokens = _text_tokens(text)
        vec = _term_vector(tokens)

        is_dup = False
        for kv in kept_vectors:
            sim = _cosine_similarity(vec, kv)
            if sim >= threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(r)
            kept_vectors.append(vec)

    return kept


# ---------------------------------------------------------------------------
# Layer 2b: Vector cosine similarity dedup (optional, embedding-based)
# ---------------------------------------------------------------------------


def layer_vector_cosine_dedup(
    results: list[dict[str, Any]],
    threshold: float = 0.85,
    workspace: str | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Remove near-duplicate results using real vector embeddings.

    When the ``recall_vector`` backend is available and a workspace path is
    provided, this layer computes cosine similarity using dense embeddings
    for much better semantic dedup than term-frequency vectors alone.

    Falls back to the existing term-frequency ``layer_cosine_dedup`` when
    embeddings are unavailable.

    Args:
        results: Scored search results sorted by descending score.
        threshold: Similarity threshold above which a result is considered
            a duplicate. Must be in [0, 1].
        workspace: Path to the mind-mem workspace (needed to load the vector
            backend). If None, falls back to term-frequency dedup.
        config: Optional recall config dict passed to VectorBackend.

    Returns:
        Filtered results with near-duplicates removed.
    """
    if not results:
        return results

    if threshold <= 0.0 or threshold > 1.0:
        return results

    # Attempt to use real vector embeddings
    if workspace:
        try:
            from .recall_vector import VectorBackend

            backend = VectorBackend(workspace=workspace, config=config or {})
            texts = [_get_result_text(r) for r in results]
            embeddings = backend.embed(texts)

            if embeddings and len(embeddings) == len(results):
                kept: list[dict[str, Any]] = []
                kept_embeddings: list[list[float]] = []

                for i, r in enumerate(results):
                    emb = embeddings[i]
                    is_dup = False
                    for kept_emb in kept_embeddings:
                        sim = backend.cosine_similarity(emb, kept_emb)
                        if sim >= threshold:
                            is_dup = True
                            break
                    if not is_dup:
                        kept.append(r)
                        kept_embeddings.append(emb)

                _log.info(
                    "dedup_layer_2b_vector_cosine",
                    before=len(results),
                    after=len(kept),
                    removed=len(results) - len(kept),
                    threshold=threshold,
                )
                return kept
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            _log.warning(
                "vector_cosine_dedup_fallback",
                error=str(exc),
                reason="falling_back_to_term_frequency",
            )

    # Fallback: use term-frequency cosine dedup
    return layer_cosine_dedup(results, threshold=threshold)


# ---------------------------------------------------------------------------
# Layer 3: Type diversity cap
# ---------------------------------------------------------------------------


def _get_result_type(result: dict[str, Any]) -> str:
    """Extract the entity type from a result.

    Uses the ``type`` field if present, otherwise derives it from the
    ``_id`` prefix (e.g., ``DIA-``, ``FACT-``, ``D-``).
    """
    rtype = result.get("type", "")
    if rtype:
        return str(rtype)

    rid = result.get("_id", "")
    # Common prefixes: DIA-xxx, FACT-xxx, D-xxx, OBS-xxx, DEC-xxx
    match = re.match(r"^([A-Z]+-)", rid)
    if match:
        return match.group(1).rstrip("-")
    return "unknown"


def layer_type_diversity_cap(
    results: list[dict[str, Any]],
    cap: int = 3,
) -> list[dict[str, Any]]:
    """Limit the number of results per entity type.

    Ensures no single type dominates the result set by capping how many
    results of each type are retained.

    Args:
        results: Scored search results sorted by descending score.
        cap: Maximum number of results to keep per type.

    Returns:
        Filtered results with per-type cap enforced.
    """
    if cap < 1:
        cap = 1

    type_counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []

    for r in results:
        rtype = _get_result_type(r)
        if type_counts[rtype] < cap:
            out.append(r)
            type_counts[rtype] += 1

    return out


# ---------------------------------------------------------------------------
# Layer 4: Per-source file chunk cap
# ---------------------------------------------------------------------------


def layer_source_chunk_cap(
    results: list[dict[str, Any]],
    cap: int = 5,
) -> list[dict[str, Any]]:
    """Limit the number of results from any single source file.

    Prevents a single document from dominating the result set when it
    contains many matching blocks.

    Args:
        results: Scored search results sorted by descending score.
        cap: Maximum number of results to keep per source file.

    Returns:
        Filtered results with per-source cap enforced.
    """
    if cap < 1:
        cap = 1

    source_counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []

    for r in results:
        source = r.get("file", "?")
        if source_counts[source] < cap:
            out.append(r)
            source_counts[source] += 1

    return out


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def deduplicate_results(
    results: list[dict[str, Any]],
    config: DedupConfig | None = None,
    workspace: str | None = None,
) -> list[dict[str, Any]]:
    """Apply the full dedup pipeline to search results.

    Each layer is applied in order. Layers that are disabled in the config
    are skipped. The pipeline preserves score ordering.

    Layers:
        1.  Best chunk per source
        2.  Cosine similarity dedup (term-frequency)
        2b. Vector cosine dedup (embedding-based, optional)
        3.  Type diversity cap
        4.  Per-source chunk cap

    Args:
        results: Scored search results sorted by descending score.
        config: Dedup configuration. Uses defaults if None.
        workspace: Optional workspace path. Required for Layer 2b
            (vector cosine dedup) when ``vector_cosine_enabled`` is True.

    Returns:
        Deduplicated results.
    """
    if not results:
        return results

    cfg = config or DedupConfig()

    if not cfg.enabled:
        _log.info("dedup_disabled")
        return results

    initial_count = len(results)
    current = results

    # Layer 1: Best chunk per source
    if cfg.best_per_source:
        current = layer_best_per_source(current)
        _log.info(
            "dedup_layer_1_best_per_source",
            before=initial_count,
            after=len(current),
            removed=initial_count - len(current),
        )

    # Layer 2: Cosine similarity dedup
    pre_cosine = len(current)
    if cfg.cosine_enabled:
        current = layer_cosine_dedup(current, threshold=cfg.cosine_threshold)
        _log.info(
            "dedup_layer_2_cosine",
            before=pre_cosine,
            after=len(current),
            removed=pre_cosine - len(current),
            threshold=cfg.cosine_threshold,
        )

    # Layer 2b: Vector cosine dedup (optional, embedding-based)
    pre_vector = len(current)
    if cfg.vector_cosine_enabled and workspace:
        current = layer_vector_cosine_dedup(
            current,
            threshold=cfg.cosine_threshold,
            workspace=workspace,
        )
        _log.info(
            "dedup_layer_2b_vector_cosine_pipeline",
            before=pre_vector,
            after=len(current),
            removed=pre_vector - len(current),
        )

    # Layer 3: Type diversity cap
    pre_type = len(current)
    if cfg.type_cap_enabled:
        current = layer_type_diversity_cap(current, cap=cfg.type_cap)
        _log.info(
            "dedup_layer_3_type_cap",
            before=pre_type,
            after=len(current),
            removed=pre_type - len(current),
            cap=cfg.type_cap,
        )

    # Layer 4: Per-source chunk cap
    pre_source = len(current)
    if cfg.source_cap_enabled:
        current = layer_source_chunk_cap(current, cap=cfg.source_cap)
        _log.info(
            "dedup_layer_4_source_cap",
            before=pre_source,
            after=len(current),
            removed=pre_source - len(current),
            cap=cfg.source_cap,
        )

    total_removed = initial_count - len(current)
    _log.info(
        "dedup_complete",
        initial=initial_count,
        final=len(current),
        total_removed=total_removed,
    )
    metrics.inc("dedup_runs")
    metrics.inc("dedup_removed", total_removed)

    return current
