#!/usr/bin/env python3
"""mind-mem Recall Engine (BM25 + TF-IDF + Graph + Stemming). Zero external deps.

Default recall backend: BM25 scoring with Porter stemming, stopword filtering,
query expansion, field boosts, recency weighting, and optional graph-based
neighbor boosting via cross-reference traversal.

For semantic recall (embeddings), see RecallBackend interface below.
Optional vector backends (Qdrant/Pinecone) can be plugged in via config.

Usage:
    python3 scripts/recall.py --query "authentication" --workspace "."
    python3 scripts/recall.py --query "auth" --workspace "." --json --limit 5
    python3 scripts/recall.py --query "deadline" --active-only
    python3 scripts/recall.py --query "database" --graph --workspace .

This module is a facade — all implementation lives in _recall_*.py submodules.
External consumers should import from 'recall', never from '_recall_*' directly.
"""

from __future__ import annotations

# --- Constants (_recall_constants) ---
from _recall_constants import (
    _BLOCK_ID_RE,
    _IRREGULAR_LEMMA,
    _STOPWORDS,
    _VALID_RECALL_KEYS,
    BM25_B,
    BM25_K1,
    CORPUS_FILES,
    FIELD_WEIGHTS,
    GRAPH_BOOST_FACTOR,
    MAX_BLOCKS_PER_QUERY,
    MAX_GRAPH_NEIGHBORS_PER_HOP,
    MAX_RERANK_CANDIDATES,
    SEARCH_FIELDS,
    SIG_FIELDS,
)

# --- Context Packing (_recall_context) ---
from _recall_context import _block_to_result, _parse_dia_id, context_pack

# --- Core Engine (_recall_core) ---
from _recall_core import (
    RecallBackend,
    _load_backend,
    main,
    prefetch_context,
    recall,
)

# --- Detection & Utilities (_recall_detection) ---
from _recall_detection import (
    _INTENT_TO_QUERY_TYPE,
    _QUERY_TYPE_PARAMS,
    _parse_speaker_from_tags,
    chunk_text,
    detect_query_type,
    extract_field_tokens,
    extract_text,
    get_bigrams,
    get_block_type,
    get_excerpt,
    is_skeptical_query,
)

# --- Query Expansion (_recall_expansion) ---
from _recall_expansion import (
    _QUERY_EXPANSIONS,
    _rm3_language_model,
    expand_months,
    expand_query,
    rm3_expand,
)

# --- Reranking (_recall_reranking) ---
from _recall_reranking import rerank_hits

# --- Scoring (_recall_scoring) ---
from _recall_scoring import (
    _category_match_boost,
    _classify_categories,
    _date_proximity_score,
    _detect_negation,
    _extract_bigram_phrases,
    _extract_dates,
    _extract_entities,
    _extract_speaker_names,
    _negation_penalty,
    build_xref_graph,
    date_score,
)

# --- Tokenization (_recall_tokenization) ---
from _recall_tokenization import _stem, tokenize

__all__ = [
    # Constants
    "SEARCH_FIELDS", "SIG_FIELDS", "CORPUS_FILES",
    "BM25_K1", "BM25_B", "FIELD_WEIGHTS",
    "_STOPWORDS", "_IRREGULAR_LEMMA",
    "GRAPH_BOOST_FACTOR", "_BLOCK_ID_RE",
    "MAX_BLOCKS_PER_QUERY", "MAX_GRAPH_NEIGHBORS_PER_HOP", "MAX_RERANK_CANDIDATES",
    "_VALID_RECALL_KEYS",
    # Tokenization
    "_stem", "tokenize",
    # Expansion
    "_QUERY_EXPANSIONS", "expand_months", "expand_query",
    "_rm3_language_model", "rm3_expand",
    # Detection & Utilities
    "extract_text", "extract_field_tokens", "get_bigrams",
    "is_skeptical_query", "detect_query_type",
    "_QUERY_TYPE_PARAMS", "_INTENT_TO_QUERY_TYPE",
    "chunk_text", "get_excerpt", "_parse_speaker_from_tags", "get_block_type",
    # Scoring
    "date_score", "build_xref_graph",
    "_detect_negation", "_negation_penalty",
    "_extract_dates", "_date_proximity_score",
    "_classify_categories", "_category_match_boost",
    "_extract_entities", "_extract_bigram_phrases", "_extract_speaker_names",
    # Reranking
    "rerank_hits",
    # Context Packing
    "context_pack", "_parse_dia_id", "_block_to_result",
    # Core
    "RecallBackend", "recall", "_load_backend", "prefetch_context", "main",
]

if __name__ == "__main__":
    main()
