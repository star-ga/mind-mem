#!/usr/bin/env python3
"""mind-mem Vector Recall Backend (Semantic Search with Embeddings).

Provides semantic/vector-based recall using sentence-transformers embeddings.
Supports local FAISS-like JSON index (zero external deps) or remote vector DBs.

Backends:
- "local": sentence-transformers + local JSON index (default)
- "qdrant": Qdrant vector database (requires qdrant-client)
- "pinecone": Pinecone vector database (requires pinecone-client)

Configuration (mind-mem.json):
    {
      "recall": {
        "backend": "vector",
        "provider": "local",
        "model": "all-MiniLM-L6-v2",
        "index_path": ".mind-mem-vectors",
        "qdrant_url": "http://localhost:6333",
        "pinecone_environment": "us-east-1-aws"
      }
    }

    Pinecone API key must be set via PINECONE_API_KEY environment variable.

Usage:
    # Indexing
    python3 -m mind_mem.recall_vector --index --workspace .

    # Search
    python3 -m mind_mem.recall_vector --query "authentication" --workspace . --limit 5

    # Force reindex
    python3 -m mind_mem.recall_vector --index --force --workspace .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import threading
from datetime import date
from typing import Any, cast

# Import helpers from recall.py
from .block_parser import parse_file
from .corpus_registry import CORPUS_DIRS
from .enums import TaskStatus
from .observability import get_logger, metrics, timed
from .recall import (
    CORPUS_FILES,
    RecallBackend,
    date_score,
    extract_text,
    get_block_type,
    get_excerpt,
)
from .scoring_instant import as_utc_datetime, resolve_scoring_instant
from .v4.circuit_breaker import CircuitBreaker, CircuitOpenError

_log = get_logger("recall_vector")

# Default hard bound (seconds) on a single embedding HTTP round-trip. Recall
# must degrade to BM25 on a cold/slow/unreachable embedder rather than block
# for minutes; override per-workspace via recall.embed_timeout_seconds.
_DEFAULT_EMBED_TIMEOUT_SECONDS = 8.0

# Per-provider embedding circuit-breaker tunables. The embedding fallback
# chain (ollama -> llama_cpp -> fastembed -> sentence-transformers) guards
# each provider with the generic v4 CircuitBreaker so a repeatedly-failing
# provider is short-circuited instead of hammered on every recall. These
# values reproduce the previous ad-hoc breaker (trip after 3 failures, hold
# open ~60s) and add a rolling 60s window so isolated, well-spaced failures
# no longer accumulate into a trip.
_EMBED_CB_THRESHOLD = 3
_EMBED_CB_RECOVERY_S = 60.0
_EMBED_CB_WINDOW_S = 60.0


def _embed_timeout_seconds(config: dict | None) -> float:
    """Return the per-request embedding timeout in seconds (fail-fast bound).

    Reads ``embed_timeout_seconds`` from the recall config, clamped to a
    sane [0.5, 120] range. Invalid / missing values fall back to the default.
    """
    try:
        val = float((config or {}).get("embed_timeout_seconds", _DEFAULT_EMBED_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        return _DEFAULT_EMBED_TIMEOUT_SECONDS
    if val <= 0:
        return _DEFAULT_EMBED_TIMEOUT_SECONDS
    return max(0.5, min(val, 120.0))


# ---------------------------------------------------------------------------
# Vector Backend Implementation
# ---------------------------------------------------------------------------


class VectorBackend(RecallBackend):
    """Semantic search backend using sentence-transformers embeddings.

    Supports local JSON index or remote vector databases (Qdrant, Pinecone).
    Falls back gracefully if sentence-transformers is not installed.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize vector backend from config.

        Args:
            config: recall section from mind-mem.json
                - provider: "local" (default), "qdrant", or "pinecone"
                - model: embedding model name (default: "all-MiniLM-L6-v2")
                - index_path: local index directory (default: ".mind-mem-vectors")
                - dimension: embedding dimension (auto-detected if not set)

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        # Validate config types to prevent injection from malformed mind-mem.json
        if not isinstance(config, dict):
            _log.warning("vector_config_invalid", reason="config is not a dict")
            config = {}
        _VALID_KEYS = {
            "backend",
            "provider",
            "model",
            "index_path",
            "dimension",
            "top_k",
            "pinecone_index",
            "pinecone_namespace",
            "qdrant_url",
            "qdrant_collection",
            "pinecone_environment",
            "rrf_k",
            "bm25_weight",
            "vector_weight",
            "vector_model",
            "vector_enabled",
            "onnx_backend",
            "sqlite_vec_db",
            "llama_cpp_url",
            "ollama_embed_model",
            "ollama_url",
            "embed_timeout_seconds",
            "vector_deadline_seconds",
            "postgres",
        }
        unknown = set(config.keys()) - _VALID_KEYS
        if unknown:
            _log.warning("vector_config_unknown_keys", keys=list(unknown))

        self.config = config
        self.provider = str(config.get("provider", "local"))
        self.model_name = str(config.get("model", "all-MiniLM-L6-v2"))
        self.index_path = str(config.get("index_path", ".mind-mem-vectors"))
        dim = config.get("dimension")
        self.dimension = int(dim) if dim is not None else None

        # Lazy load embedding model (not needed for pinecone/llama_cpp)
        self._model: Any = None
        self._model_loaded = False
        self._fastembed_model: Any = None

        # llama.cpp embedding server URL
        self.llama_cpp_url = str(config.get("llama_cpp_url", "http://localhost:8090"))

        # Pinecone integrated inference config
        self.pinecone_index_name = str(config.get("pinecone_index", "mind-mem"))
        self.pinecone_namespace = str(config.get("pinecone_namespace", "default"))

        # Circuit breakers for the embedding fallback chain — one generic
        # v4 CircuitBreaker per provider (created lazily). Driven via
        # ``guarded_call`` so the protection is always on, independent of
        # the optional ``circuit_breaker`` v4 feature flag. Replaces the
        # previous ad-hoc windowed breaker (``_provider_failures`` dict).
        self._provider_breakers: dict[str, CircuitBreaker] = {}
        self._provider_breakers_lock = threading.Lock()
        _log.info("vector_backend_init", provider=self.provider, model=self.model_name)

    # ------------------------------------------------------------------
    # Embedding Cache (P0) — stdlib only: sqlite3, hashlib, struct
    # ------------------------------------------------------------------

    @staticmethod
    def _content_hash(text: str) -> str:
        """SHA-256 hash of block text for cache key."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_embedding(emb: list[float]) -> bytes:
        """Pack float list to binary blob (4 bytes per float)."""
        return struct.pack(f"{len(emb)}f", *emb)

    @staticmethod
    def _deserialize_embedding(blob: bytes, dim: int) -> list[float]:
        """Unpack binary blob to float list."""
        return list(struct.unpack(f"{dim}f", blob))

    def _ensure_cache_tables(self, conn) -> None:
        """Create embedding_cache and vec_meta_info tables if missing."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                block_id     TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                model_name   TEXT NOT NULL,
                dimension    INTEGER NOT NULL,
                embedding    BLOB NOT NULL,
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (block_id, model_name)
            );

            CREATE TABLE IF NOT EXISTS vec_meta_info (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

    def _get_cached_embedding(self, conn, block_id: str, content_hash: str) -> list[float] | None:
        """Look up cached embedding. Returns None on miss."""
        row = conn.execute(
            "SELECT embedding, dimension FROM embedding_cache WHERE block_id = ? AND content_hash = ? AND model_name = ?",
            (block_id, content_hash, self.model_name),
        ).fetchone()
        if row is None:
            return None
        blob, dim = row[0], row[1]
        try:
            return self._deserialize_embedding(blob, dim)
        except struct.error:
            return None

    def _cache_embedding(self, conn, block_id: str, content_hash: str, embedding: list[float]) -> None:
        """Store embedding in cache (upsert)."""
        dim = len(embedding)
        blob = self._serialize_embedding(embedding)
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (block_id, content_hash, model_name, dimension, embedding) VALUES (?, ?, ?, ?, ?)",
            (block_id, content_hash, self.model_name, dim, blob),
        )

    def _invalidate_cache_for_model(self, conn, model_name: str) -> int:
        """Remove all cached embeddings for a different model. Returns count deleted."""
        cur = conn.execute("DELETE FROM embedding_cache WHERE model_name != ?", (model_name,))
        return int(cur.rowcount)

    def _get_vec_meta(self, conn, key: str) -> str | None:
        """Read a vec_meta_info value."""
        row = conn.execute("SELECT value FROM vec_meta_info WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_vec_meta(self, conn, key: str, value: str) -> None:
        """Write a vec_meta_info value."""
        conn.execute(
            "INSERT OR REPLACE INTO vec_meta_info (key, value) VALUES (?, ?)",
            (key, value),
        )

    def _check_dimension_match(self, conn) -> bool:
        """Check if stored model/dimension match current config.

        Returns True if match (or no prior metadata). Returns False on mismatch.
        """
        stored_model = self._get_vec_meta(conn, "model_name")
        stored_dim = self._get_vec_meta(conn, "dimension")
        if stored_model is None:
            return True  # No prior metadata — first build
        if stored_model != self.model_name:
            _log.warning("vec_model_mismatch", stored=stored_model, current=self.model_name)
            return False
        if stored_dim is not None and self.dimension is not None:
            if int(stored_dim) != self.dimension:
                _log.warning("vec_dimension_mismatch", stored=stored_dim, current=self.dimension)
                return False
        return True

    @property
    def model(self):
        """Lazy-load embedding model on first access."""
        if not self._model_loaded:
            try:
                # Temporarily remove scripts/ from sys.path so that
                # sentence_transformers can find the real `filelock` package
                # (mind_filelock.py shadows it). We remove by realpath
                # to catch both relative and absolute path variants.
                _scripts_real = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
                _saved = list(sys.path)
                sys.path[:] = [p for p in sys.path if os.path.realpath(p) != _scripts_real]
                try:
                    from sentence_transformers import SentenceTransformer
                finally:
                    sys.path[:] = _saved
                cache_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
                device = self.config.get("vector_device", "cpu")
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=cache_dir,
                    device=device,
                )
                self._model_loaded = True
                _log.info("embedding_model_loaded", model=self.model_name)
            except ImportError as e:
                _log.error("sentence_transformers_not_installed", error=str(e))
                raise ImportError("sentence-transformers not installed. Install with: pip install 'mind-mem[embeddings]'") from e
        return self._model

    @staticmethod
    def _augment_for_embedding(block: dict, text: str) -> str:
        """Prepend metadata context to text for better embedding disambiguation.

        A chunk saying "it cost $50" loses context without knowing what "it"
        refers to. Prepending category + speaker + date gives the embedding
        model semantic anchors to disambiguate.

        Args:
            block: Block metadata dict (with Category, Speaker, Date, tags).
            text: Raw searchable text.

        Returns:
            Augmented text with metadata prefix.
        """
        parts = []
        cat = block.get("Category") or block.get("type") or ""
        if cat:
            parts.append(f"[{cat}]")
        speaker = block.get("Speaker") or block.get("speaker") or ""
        if speaker:
            parts.append(f"[{speaker}]")
        date = block.get("Date") or block.get("date") or ""
        if date:
            parts.append(f"[{date}]")
        tags = block.get("Tags") or block.get("tags") or ""
        if tags:
            parts.append(f"[{tags[:50]}]")
        prefix = " ".join(parts)
        return f"{prefix} {text}" if prefix else text

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []

        with timed("embed_batch"):
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            metrics.inc("embeddings_generated", len(texts))
            return [emb.tolist() for emb in embeddings]

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def embed_fastembed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using fastembed (ONNX, no torch required).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            ImportError: If fastembed is not installed
        """
        # Temporarily remove scripts/ from sys.path so fastembed's huggingface_hub
        # import finds the real 'filelock' system package, not mind-mem's local stub.
        import sys as _sys

        _saved = _sys.path[:]
        _scripts_paths = [p for p in _sys.path if "mind-mem/scripts" in p or p.endswith("/scripts")]
        for _p in _scripts_paths:
            _sys.path.remove(_p)
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError("fastembed not installed. Install with: pip install fastembed") from e
        finally:
            _sys.path[:] = _saved

        if not hasattr(self, "_fastembed_model") or self._fastembed_model is None:
            _log.info("fastembed_model_loading", model=self.model_name)
            self._fastembed_model = TextEmbedding(self.model_name)
            _log.info("fastembed_model_loaded", model=self.model_name)

        with timed("embed_fastembed"):
            embeddings = list(self._fastembed_model.embed(texts))
            metrics.inc("embeddings_generated", len(texts))
            return [emb.tolist() for emb in embeddings]

    def embed_llama_cpp(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via a llama.cpp server's /embeddings endpoint.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        import urllib.request as _req
        from urllib.parse import urlparse as _urlparse

        _parsed = _urlparse(self.llama_cpp_url)
        if _parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(f"llama_cpp_url must be localhost, got: {_parsed.hostname}")

        url = self.llama_cpp_url.rstrip("/") + "/embeddings"
        BATCH = 32
        # Fail-fast bound: recall must degrade to BM25 on a cold/slow/down
        # embedder, never block for minutes. Configurable via recall config.
        _embed_timeout = _embed_timeout_seconds(self.config)
        all_embeddings = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            payload = json.dumps({"input": batch}).encode("utf-8")
            req = _req.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with _req.urlopen(req, timeout=_embed_timeout) as resp:  # nosec B310 — URL is hardcoded to http://localhost:*/ (loopback only)
                result = json.loads(resp.read().decode("utf-8"))
            # llama.cpp returns a list of {index, embedding} objects
            if isinstance(result, list):
                for item in result:
                    emb = item.get("embedding", [])
                    if isinstance(emb, list) and emb and isinstance(emb[0], list):
                        emb = emb[0]
                    all_embeddings.append(emb)
            elif isinstance(result, dict) and "data" in result:
                for item in result["data"]:
                    all_embeddings.append(item["embedding"])
            _log.debug("llama_cpp_batch_embedded", batch=i // BATCH + 1, count=len(batch))

        metrics.inc("embeddings_generated", len(texts))
        return all_embeddings

    def embed_ollama(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Ollama's /api/embed endpoint (GPU-accelerated).

        Uses the ``ollama_embed_model`` config key, falling back to ``model``.
        Much faster than ONNX on CPU for local deployments with a GPU.
        The endpoint resolves via ``recall.ollama_url`` config, then the
        ``OLLAMA_HOST`` env var, then ``http://localhost:11434`` — so a
        fleet node can use a central ollama server.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        import urllib.request as _req

        from .ollama_host import ollama_base_url

        # Base URL: recall.ollama_url config key > OLLAMA_HOST env >
        # http://localhost:11434 (see ollama_host.ollama_base_url).
        url = ollama_base_url(self.config) + "/api/embed"
        model = self.config.get("ollama_embed_model", self.model_name)
        MAX_CHARS = 1500  # ~375 tokens, safe for 512-ctx embedding models
        BATCH = 16  # small batches to stay under total context limit
        # Fail-fast bound: recall must degrade to BM25 on a cold/slow/down
        # ollama, never block for minutes. Configurable via recall config.
        _embed_timeout = _embed_timeout_seconds(self.config)
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), BATCH):
            batch = [t[:MAX_CHARS] for t in texts[i : i + BATCH]]
            payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
            req = _req.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with timed("embed_ollama"):
                with _req.urlopen(req, timeout=_embed_timeout) as resp:  # nosec B310 — base URL from operator-controlled config/env only (ollama_base_url enforces http/https), never user or recall input
                    result = json.loads(resp.read().decode("utf-8"))
            embeddings = result.get("embeddings", [])
            all_embeddings.extend(embeddings)
            _log.debug("ollama_batch_embedded", batch=i // BATCH + 1, count=len(batch))

        metrics.inc("embeddings_generated", len(texts))
        return all_embeddings

    def _provider_breaker(self, provider: str) -> CircuitBreaker:
        """Get-or-create the circuit breaker for one embedding provider.

        Thread-safe lazy construction — the embedding path runs under the
        hybrid parallel pool and the multi-query expansion pool, so two
        threads may race to create the same provider's breaker.
        """
        breaker = self._provider_breakers.get(provider)
        if breaker is not None:
            return breaker
        with self._provider_breakers_lock:
            breaker = self._provider_breakers.get(provider)
            if breaker is None:
                breaker = CircuitBreaker(
                    failure_threshold=_EMBED_CB_THRESHOLD,
                    recovery_timeout=_EMBED_CB_RECOVERY_S,
                    failure_window_s=_EMBED_CB_WINDOW_S,
                )
                self._provider_breakers[provider] = breaker
        return breaker

    def _embed_for_provider(self, texts: list[str]) -> list[list[float]]:
        """Route embedding to the configured backend with a fallback chain,
        each provider guarded by its own circuit breaker.

        The generic v4 :class:`CircuitBreaker` (3 failures within a 60s
        window trips OPEN; ~60s recovery) short-circuits a provider that
        is repeatedly failing so a dead embedder is not re-hit on every
        recall. A tripped or failing provider falls through to the next
        one in the chain; every fall-through is logged loudly — the
        degradation is never silent. If the whole chain is exhausted, the
        final sentence-transformers fallback runs unguarded and any
        exception propagates so the caller (the vector leg) can mark the
        recall degraded rather than quietly return nothing.
        """
        backend = self.config.get("onnx_backend", True)

        def _try(provider: str, fn) -> list[list[float]] | None:
            breaker = self._provider_breaker(provider)
            try:
                return cast("list[list[float]] | None", breaker.guarded_call(fn, texts))
            except CircuitOpenError as exc:
                # Breaker OPEN: skip this provider without hitting it, but
                # say so — a short-circuited leg is a degradation, mark it.
                _log.warning(
                    "embed_provider_breaker_open",
                    provider=provider,
                    retry_after=round(exc.retry_after, 2),
                    fallback="next_provider",
                )
                return None
            except Exception as e:
                _log.warning(
                    "embed_provider_failed_fallback",
                    provider=provider,
                    error=str(e),
                )
                return None

        # Try Ollama first (GPU-accelerated, fastest for local)
        result = _try("ollama", self.embed_ollama)
        if result is not None:
            return result

        # Try llama_cpp if configured
        if backend == "llama_cpp" or str(backend).lower() == "llama_cpp":
            result = _try("llama_cpp", self.embed_llama_cpp)
            if result is not None:
                return result

        # Try fastembed (ONNX on CPU)
        result = _try("fastembed", self.embed_fastembed)
        if result is not None:
            return result

        # Final fallback: sentence-transformers (unguarded last resort).
        return self.embed(texts)

    def _sqlite_vec_db_path(self, workspace: str) -> str:
        """Return path to the sqlite-vec DB (shares recall.db with BM25 index)."""
        custom = self.config.get("sqlite_vec_db")
        if custom:
            return os.path.join(workspace, custom)
        return os.path.join(workspace, ".mind-mem-index", "recall.db")

    def _connect_sqlite_vec(self, workspace: str, readonly: bool = False):
        """Open recall.db with sqlite-vec extension loaded.

        Args:
            workspace: Workspace root path
            readonly: Open in read-only mode

        Returns:
            sqlite3 connection with vec0 support enabled
        """
        import sqlite3 as _sqlite3

        try:
            import sqlite_vec
        except ImportError as e:
            raise ImportError("sqlite-vec not installed. Install with: pip install sqlite-vec") from e

        db_path = self._sqlite_vec_db_path(workspace)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        if readonly:
            conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        else:
            conn = _sqlite3.connect(db_path)

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Loadable extensions are not enabled on every Python build — most
        # notably the python.org macOS installer ships without
        # `--enable-loadable-sqlite-extensions`. Degrade gracefully when
        # `enable_load_extension` is unavailable; callers already fall
        # through to BM25-only recall when the vector backend can't init.
        if not hasattr(conn, "enable_load_extension"):
            raise RuntimeError("sqlite was compiled without loadable extensions — vector backend unavailable")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _init_vec_table(self, conn, dimension: int) -> None:
        """Create vec0 virtual table if it doesn't exist.

        Args:
            conn: Connection with sqlite-vec loaded
            dimension: Embedding dimension (e.g. 384 for bge-small)
        """
        if not isinstance(dimension, int) or not (1 <= dimension <= 65536):
            raise ValueError(f"Invalid embedding dimension: {dimension}")
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_blocks USING vec0(
                block_id TEXT PRIMARY KEY,
                embedding FLOAT[{dimension}]
            )
        """)
        conn.commit()

    def _index_sqlite_vec(self, workspace: str, blocks: list[dict], texts: list[str]) -> None:
        """Build sqlite-vec index with embedding cache for incremental updates.

        Uses embedding_cache table to skip re-embedding unchanged blocks.
        Only embeds new or modified blocks, reuses cached embeddings for the rest.

        Args:
            workspace: Workspace root path
            blocks: Block metadata list
            texts: Searchable text for each block
        """
        import sqlite_vec

        _log.info("sqlite_vec_indexing_start", count=len(texts))

        conn = self._connect_sqlite_vec(workspace)
        try:
            self._ensure_cache_tables(conn)

            # Check for model/dimension mismatch — force full rebuild if changed
            model_match = self._check_dimension_match(conn)
            if not model_match:
                _log.info("vec_model_changed_invalidating_cache", model=self.model_name)
                self._invalidate_cache_for_model(conn, self.model_name)
                conn.execute("DROP TABLE IF EXISTS vec_blocks")
                conn.commit()

            # Compute content hashes and check cache
            cache_hits = 0
            cache_misses = 0
            to_embed_indices: list[int] = []
            to_embed_texts: list[str] = []
            embeddings: list[list[float] | None] = [None] * len(texts)

            for i, (block, text) in enumerate(zip(blocks, texts)):
                bid = block["_id"]
                chash = self._content_hash(text)
                cached = self._get_cached_embedding(conn, bid, chash)
                if cached is not None:
                    embeddings[i] = cached
                    cache_hits += 1
                else:
                    to_embed_indices.append(i)
                    to_embed_texts.append(text)
                    cache_misses += 1

            _log.info("embedding_cache_stats", hits=cache_hits, misses=cache_misses, total=len(texts))

            # Embed only the cache misses
            if to_embed_texts:
                new_embeddings = self._embed_for_provider(to_embed_texts)
                if not new_embeddings:
                    _log.error("sqlite_vec_no_embeddings")
                    return
                for idx, emb in zip(to_embed_indices, new_embeddings):
                    embeddings[idx] = emb
                    # Store in cache
                    bid = blocks[idx]["_id"]
                    chash = self._content_hash(texts[idx])
                    self._cache_embedding(conn, bid, chash, emb)

            # Verify all embeddings were filled
            if any(e is None for e in embeddings):
                _log.error("sqlite_vec_incomplete_embeddings")
                return

            filled_embeddings: list[list[float]] = [e for e in embeddings if e is not None]
            dim = len(filled_embeddings[0])
            if self.dimension is None:
                self.dimension = dim

            # Rebuild vec_blocks table (sqlite-vec doesn't support UPDATE well)
            conn.execute("DROP TABLE IF EXISTS vec_blocks")
            self._init_vec_table(conn, dim)
            for block, emb in zip(blocks, filled_embeddings):
                conn.execute(
                    "INSERT INTO vec_blocks(block_id, embedding) VALUES (?, ?)",
                    (block["_id"], sqlite_vec.serialize_float32(emb)),
                )

            # Write metadata for dimension mismatch detection
            from datetime import datetime

            self._set_vec_meta(conn, "model_name", self.model_name)
            self._set_vec_meta(conn, "dimension", str(dim))
            self._set_vec_meta(conn, "built_at", datetime.now().isoformat())
            self._set_vec_meta(conn, "block_count", str(len(blocks)))

            conn.commit()
            _log.info(
                "sqlite_vec_indexed",
                blocks=len(blocks),
                dimension=dim,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )
        finally:
            conn.close()

        # Cache block metadata to a companion JSON (for search result enrichment)
        meta_path = os.path.join(workspace, ".mind-mem-index", "vec_meta.json")
        meta = {b["_id"]: b for b in blocks}
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp_path, meta_path)

    def _search_sqlite_vec(self, workspace: str, query: str, limit: int, active_only: bool) -> list[dict]:
        """ANN search via sqlite-vec vec0 table.

        Args:
            workspace: Workspace root path
            query: Query text
            limit: Max results
            active_only: Filter to active blocks only

        Returns:
            Ranked list of result dicts
        """
        import sqlite_vec

        # Check for dimension mismatch before search
        try:
            check_conn = self._connect_sqlite_vec(workspace, readonly=True)
            self._ensure_cache_tables(check_conn)
            if not self._check_dimension_match(check_conn):
                _log.warning(
                    "vec_search_dimension_mismatch",
                    msg="Index was built with different model — results may be inaccurate. Run reindex.",
                )
            check_conn.close()
        except Exception as exc:
            _log.debug("vec_metadata_check_skipped", error=str(exc))  # non-fatal; don't block search

        # Embed query
        query_emb = self._embed_for_provider([query])[0]
        query_bytes = sqlite_vec.serialize_float32(query_emb)

        # Load block metadata
        meta_path = os.path.join(workspace, ".mind-mem-index", "vec_meta.json")
        meta: dict = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        conn = self._connect_sqlite_vec(workspace, readonly=True)
        try:
            # Audit R-8: adaptive overfetch. The previous fixed ×3
            # under-fetched when the active-block ratio in meta was
            # low — callers asking for limit=10 with a corpus that's
            # 80% inactive got only 6 active hits back, silently
            # truncating recall. We now look up the active ratio from
            # meta and overfetch by ceil(limit / max(ratio, 0.1)),
            # clamped to a sane upper bound to avoid an O(N) scan.
            if active_only:
                total = len(meta) or 1
                active = sum(1 for v in meta.values() if v.get("status") == "active") or 0
                ratio = max(active / total, 0.1) if active else 0.1
                fetch_limit = min(int(limit / ratio) + limit, max(limit * 10, 200))
            else:
                fetch_limit = limit
            rows = conn.execute(
                """
                SELECT block_id, distance
                FROM vec_blocks
                WHERE embedding MATCH ?
                  AND k = ?
                ORDER BY distance
                """,
                (query_bytes, fetch_limit),
            ).fetchall()
        finally:
            conn.close()

        results = []
        for block_id, distance in rows:
            block = meta.get(block_id, {})
            status = block.get("status", "")
            if active_only and status != "active":
                continue
            # Convert cosine distance (0=identical, 2=opposite) to similarity score
            score = round(1.0 - distance / 2.0, 4)
            results.append(
                {
                    "_id": block_id,
                    "type": block.get("type", "unknown"),
                    "score": score,
                    "excerpt": block.get("excerpt", ""),
                    "file": block.get("file", "?"),
                    "line": block.get("line", 0),
                    "status": status,
                }
            )

        return results[:limit]

    def _get_index_path(self, workspace: str) -> str:
        """Get full path to local index file."""
        index_dir = os.path.join(workspace, self.index_path)
        return os.path.join(index_dir, "index.json")

    def _load_local_index(self, workspace: str) -> dict[str, Any] | None:
        """Load local JSON index from disk.

        Returns:
            Index dict with 'blocks' and 'embeddings', or None if not found
        """
        index_file = self._get_index_path(workspace)
        if not os.path.isfile(index_file):
            _log.warning("index_not_found", path=index_file)
            return None

        try:
            with open(index_file, encoding="utf-8") as f:
                index: dict[str, Any] = json.load(f)
            _log.info("index_loaded", path=index_file, blocks=len(index.get("blocks", [])))
            return index
        except (OSError, json.JSONDecodeError) as e:
            _log.error("index_load_failed", path=index_file, error=str(e))
            return None

    def _save_local_index(self, workspace: str, index: dict[str, Any]):
        """Save local JSON index to disk.

        Args:
            workspace: Workspace path
            index: Index dict with 'blocks' and 'embeddings'
        """
        index_dir = os.path.join(workspace, self.index_path)
        os.makedirs(index_dir, exist_ok=True)

        index_file = self._get_index_path(workspace)
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
            _log.info("index_saved", path=index_file, blocks=len(index.get("blocks", [])))
        except OSError as e:
            _log.error("index_save_failed", path=index_file, error=str(e))
            raise

    def index(self, workspace: str):
        """Build or rebuild vector index from workspace files.

        Args:
            workspace: Path to workspace root
        """
        _log.info("indexing_start", workspace=workspace, provider=self.provider)

        with timed("index_build"):
            # Load all blocks from corpus files
            all_blocks = []
            for label, rel_path in CORPUS_FILES.items():
                path = os.path.join(workspace, rel_path)
                if not os.path.isfile(path):
                    _log.debug("corpus_file_missing", file=rel_path)
                    continue

                try:
                    blocks = parse_file(path)
                    _log.debug("corpus_file_parsed", file=rel_path, blocks=len(blocks))
                except (OSError, UnicodeDecodeError, ValueError) as e:
                    _log.warning("corpus_parse_error", file=rel_path, error=str(e))
                    continue

                for b in blocks:
                    b["_source_file"] = rel_path
                    b["_source_label"] = label
                    all_blocks.append(b)

            if not all_blocks:
                _log.warning("no_blocks_found", workspace=workspace)
                return

            _log.info("blocks_loaded", count=len(all_blocks))

            # Extract searchable text from each block
            # Metadata-augmented embeddings: prepend [Category] [Speaker] [Date]
            # to each text so the embedding captures context that raw text loses
            # (e.g. "it cost $50" → "[purchase] [Emma] [2023-05] it cost $50")
            texts = []
            block_metadata = []
            for block in all_blocks:
                text = extract_text(block)
                if not text.strip():
                    continue

                meta = {
                    "_id": block.get("_id", "?"),
                    "type": get_block_type(block.get("_id", "")),
                    "excerpt": get_excerpt(block),
                    "file": block.get("_source_file", "?"),
                    "line": block.get("_line", 0),
                    "status": block.get("Status", ""),
                    "date": block.get("Date", ""),
                }
                texts.append(self._augment_for_embedding(block, text))
                block_metadata.append(meta)

            if not texts:
                _log.warning("no_searchable_text", workspace=workspace)
                return

            # sqlite-vec: local ONNX embeddings stored in recall.db
            if self.provider == "sqlite_vec":
                _log.info("sqlite_vec_indexing", count=len(texts))
                self._index_sqlite_vec(workspace, block_metadata, texts)
                metrics.inc("index_builds")
                _log.info("indexing_complete", blocks=len(texts), provider=self.provider)
                return

            # Pinecone integrated inference: skip local embedding generation
            if self.provider == "pinecone":
                _log.info("pinecone_integrated_inference", count=len(texts))
                self._index_pinecone_integrated(workspace, block_metadata, texts)
                metrics.inc("index_builds")
                _log.info("indexing_complete", blocks=len(texts), provider=self.provider)
                return

            _log.info("generating_embeddings", count=len(texts))

            # Generate embeddings — route based on provider
            if self.provider == "llama_cpp":
                embeddings = self.embed_llama_cpp(texts)
            else:
                embeddings = self.embed(texts)

            if not embeddings:
                _log.error("embedding_generation_failed")
                return

            # Auto-detect dimension if not set
            if self.dimension is None:
                self.dimension = len(embeddings[0])
                _log.info("dimension_detected", dimension=self.dimension)

            # Build index based on provider (llama_cpp uses local JSON storage)
            if self.provider in ("local", "llama_cpp"):
                self._index_local(workspace, block_metadata, embeddings)
            elif self.provider == "qdrant":
                self._index_qdrant(workspace, block_metadata, embeddings)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            metrics.inc("index_builds")
            _log.info("indexing_complete", blocks=len(texts), provider=self.provider)

    def _index_local(self, workspace: str, blocks: list[dict], embeddings: list[list[float]]):
        """Build local JSON index.

        Args:
            workspace: Workspace path
            blocks: List of block metadata dicts
            embeddings: List of embedding vectors
        """
        index = {
            "model": self.model_name,
            "dimension": self.dimension,
            "blocks": blocks,
            "embeddings": embeddings,
        }
        self._save_local_index(workspace, index)

    def _index_qdrant(self, workspace: str, blocks: list[dict], embeddings: list[list[float]]):
        """Build Qdrant vector index.

        Args:
            workspace: Workspace path
            blocks: List of block metadata dicts
            embeddings: List of embedding vectors
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError as e:
            _log.error("qdrant_client_not_installed", error=str(e))
            raise ImportError("qdrant-client not installed. Install with: pip install qdrant-client") from e

        url = self.config.get("qdrant_url", "http://localhost:6333")
        collection = self.config.get("qdrant_collection", "mind-mem")

        client = QdrantClient(url=url)

        # Recreate collection
        try:
            client.delete_collection(collection)
        except Exception as exc:
            _log.debug("qdrant_delete_collection_skipped", collection=collection, error=str(exc))

        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )

        # Upload points
        points = [
            PointStruct(
                id=i,
                vector=emb,
                payload=block,
            )
            for i, (block, emb) in enumerate(zip(blocks, embeddings))
        ]

        client.upsert(collection_name=collection, points=points)
        _log.info("qdrant_indexed", collection=collection, points=len(points))

    def _index_pinecone_integrated(self, workspace: str, blocks: list[dict], texts: list[str]):
        """Index into Pinecone using integrated inference (server-side embeddings).

        Upserts records with text content — Pinecone generates embeddings
        server-side via the model configured on the index (e.g. multilingual-e5-large).
        No local sentence-transformers or torch required.

        Args:
            workspace: Workspace path
            blocks: List of block metadata dicts
            texts: List of text strings (content to embed)
        """
        try:
            from pinecone import Pinecone
        except ImportError as e:
            _log.error("pinecone_not_installed", error=str(e))
            raise ImportError("pinecone not installed. Install with: pip install pinecone") from e

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            # Audit R-9: surface the missing-credential failure as a
            # WARNING-level structured log + counter so dashboards
            # catch silent breakage (the old code raised ValueError
            # but downstream callers swallowed it into a debug log,
            # making the "Pinecone is configured but doing nothing"
            # state invisible).
            _log.warning(
                "pinecone_api_key_missing",
                index=self.pinecone_index_name,
                advice="set PINECONE_API_KEY to enable the Pinecone backend",
            )
            metrics.inc("pinecone_api_key_missing")
            raise ValueError("PINECONE_API_KEY environment variable is required for Pinecone backend")

        pc = Pinecone(api_key=api_key)
        index = pc.Index(self.pinecone_index_name)

        # Build records with text content for integrated inference
        # The index's fieldMap maps "content" → embedding model
        BATCH_SIZE = 96
        total = 0
        for i in range(0, len(blocks), BATCH_SIZE):
            batch_blocks = blocks[i : i + BATCH_SIZE]
            batch_texts = texts[i : i + BATCH_SIZE]
            records: list[dict[str, Any]] = []
            for block, text in zip(batch_blocks, batch_texts):
                record = {
                    "_id": block.get("_id", f"block-{i + len(records)}"),
                    "content": text,
                    "block_type": block.get("type", "unknown"),
                    "excerpt": (block.get("excerpt", "") or "")[:500],
                    "file": block.get("file", ""),
                    "line": block.get("line", 0),
                    "status": block.get("status", ""),
                }
                records.append(record)

            index.upsert_records(self.pinecone_namespace, records)
            total += len(records)
            _log.info("pinecone_batch_upserted", batch=i // BATCH_SIZE + 1, count=len(records))

        _log.info("pinecone_indexed", index=self.pinecone_index_name, vectors=total)

    def search(
        self,
        workspace: str,
        query: str,
        limit: int = 10,
        active_only: bool = False,
        *,
        scoring_instant: date | None = None,
    ) -> list[dict]:
        """Search using semantic/vector similarity.

        Args:
            workspace: Workspace path
            query: Search query text
            limit: Maximum number of results
            active_only: Only return active blocks
            scoring_instant: UTC date the recency boost scores against;
                ``None`` resolves to today in UTC. The local-index provider
                multiplies a recency term into the similarity, so it takes the
                same seam as the BM25 legs — see :mod:`mind_mem.scoring_instant`.

        Returns:
            List of result dicts with _id, type, score, excerpt, file, line, status
        """
        if not query.strip():
            return []

        _log.info("vector_search_start", query=query, limit=limit, provider=self.provider)

        with timed("vector_search"):
            if self.provider == "sqlite_vec":
                results = self._search_sqlite_vec(workspace, query, limit, active_only)
            elif self.provider in ("local", "llama_cpp"):
                results = self._search_local(workspace, query, limit, active_only, scoring_instant=scoring_instant)
            elif self.provider == "qdrant":
                results = self._search_qdrant(workspace, query, limit, active_only)
            elif self.provider == "pinecone":
                results = self._search_pinecone(workspace, query, limit, active_only)
            elif self.provider == "postgres":
                results = self._search_postgres(workspace, query, limit, active_only)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            metrics.inc("vector_searches")
            metrics.inc("vector_results", len(results))
            _log.info(
                "vector_search_complete",
                query=query,
                results=len(results),
                top_score=results[0]["score"] if results else 0,
            )
            return results

    def _search_postgres(self, workspace: str, query: str, limit: int, active_only: bool) -> list[dict]:
        """Vector-only cosine search against the Postgres pgvector store.

        The vector store lives server-side (``recall.provider == "postgres"``)
        — there is no local JSON / sqlite-vec index to consult. Embed the
        query with the configured embedder (honoring the fallback chain +
        circuit breaker) and delegate to ``PostgresBlockStore.vector_search``,
        returning hits in the standard recall shape via ``_pg_block_to_hit``.

        Previously ``provider == "postgres"`` fell through to
        ``raise ValueError("Unknown provider: postgres")``, which
        ``search_batch`` swallowed to ``[]`` — so the HybridBackend/MCP RRF
        path fused BM25 *alone* on every Postgres query while reporting a
        hybrid backend (audit finding 1a). Returns ``[]`` (honest BM25-only
        on the caller's side) when the embedder is unavailable or no block is
        embedded yet — it never raises.
        """
        emb = self._embed_for_provider([query])
        if not emb or not emb[0]:
            _log.warning("pg_vector_search_embed_unavailable_bm25_only", query=query)
            return []
        from ._recall_core import _pg_block_to_hit
        from .storage import get_block_store

        # config=None → get_block_store auto-loads the FULL workspace
        # mind-mem.json (with its block_store/storage section). ``self.config``
        # here is the recall-only section, which lacks the store DSN.
        store = get_block_store(workspace, config=None)
        if not hasattr(store, "vector_search"):
            _log.warning("pg_vector_search_unsupported_store", store=type(store).__name__)
            return []
        blocks = store.vector_search(list(emb[0]), limit=limit)
        return [_pg_block_to_hit(block, block.get("_score", 0.0)) for block in blocks]

    def _search_local(
        self,
        workspace: str,
        query: str,
        limit: int,
        active_only: bool,
        *,
        scoring_instant: date | None = None,
    ) -> list[dict]:
        """Search local JSON index.

        Args:
            workspace: Workspace path
            query: Search query
            limit: Max results
            active_only: Filter to active blocks only
            scoring_instant: UTC date the recency boost scores against.

        Returns:
            Ranked list of results
        """
        _scoring_moment = as_utc_datetime(resolve_scoring_instant(scoring_instant))
        # Load index
        index = self._load_local_index(workspace)
        if not index:
            _log.warning("no_index_available", workspace=workspace)
            return []

        blocks = index.get("blocks", [])
        embeddings = index.get("embeddings", [])

        if not blocks or not embeddings:
            return []

        if len(blocks) != len(embeddings):
            _log.error("index_mismatch", blocks=len(blocks), embeddings=len(embeddings))
            return []

        # Generate query embedding
        if self.provider == "llama_cpp":
            query_emb = self.embed_llama_cpp([query])[0]
        else:
            query_emb = self.embed([query])[0]

        # Compute similarity scores
        results = []
        for block, doc_emb in zip(blocks, embeddings):
            # Filter by status if requested
            if active_only and block.get("status") != "active":
                continue

            # Compute cosine similarity
            similarity = self.cosine_similarity(query_emb, doc_emb)

            # Apply boost factors
            score = similarity

            # Recency boost
            date_str = block.get("date", "")
            if date_str:
                recency = date_score({"Date": date_str}, now=_scoring_moment)
                score *= 0.7 + 0.3 * recency

            # Status boost
            status = block.get("status", "")
            if status == "active":
                score *= 1.2
            elif status in {TaskStatus.TODO, TaskStatus.DOING}:
                score *= 1.1

            result_item = {
                "_id": block.get("_id", "?"),
                "type": block.get("type", "unknown"),
                "score": round(score, 4),
                "excerpt": block.get("excerpt", ""),
                "file": block.get("file", "?"),
                "line": block.get("line", 0),
                "status": status,
            }
            if block.get("speaker"):
                result_item["speaker"] = block["speaker"]
            if block.get("DiaID"):
                result_item["DiaID"] = block["DiaID"]
            if block.get("Date") or block.get("date"):
                result_item["Date"] = block.get("Date") or block.get("date")
            results.append(result_item)

        # Sort by score descending
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def _search_qdrant(self, workspace: str, query: str, limit: int, active_only: bool) -> list[dict]:
        """Search Qdrant vector database.

        Args:
            workspace: Workspace path
            query: Search query
            limit: Max results
            active_only: Filter to active blocks only

        Returns:
            Ranked list of results
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import FieldCondition, Filter, MatchValue
        except ImportError as e:
            raise ImportError("qdrant-client not installed") from e

        url = self.config.get("qdrant_url", "http://localhost:6333")
        collection = self.config.get("qdrant_collection", "mind-mem")

        client = QdrantClient(url=url)

        # Generate query embedding
        query_emb = self.embed([query])[0]

        # Build filter
        query_filter = None
        if active_only:
            query_filter = Filter(must=[FieldCondition(key="status", match=MatchValue(value="active"))])

        # Search
        search_results = client.search(
            collection_name=collection,
            query_vector=query_emb,
            limit=limit,
            query_filter=query_filter,
        )

        # Convert to standard format
        results = []
        for hit in search_results:
            payload = hit.payload
            results.append(
                {
                    "_id": payload.get("_id", "?"),
                    "type": payload.get("type", "unknown"),
                    "score": round(hit.score, 4),
                    "excerpt": payload.get("excerpt", ""),
                    "file": payload.get("file", "?"),
                    "line": payload.get("line", 0),
                    "status": payload.get("status", ""),
                }
            )

        return results

    def _search_pinecone(self, workspace: str, query: str, limit: int, active_only: bool) -> list[dict]:
        """Search Pinecone using integrated inference (server-side embeddings).

        Sends text query directly — Pinecone generates the query embedding
        server-side and returns ranked results. No local embedding required.

        Args:
            workspace: Workspace path
            query: Search query text
            limit: Max results
            active_only: Filter to active blocks only

        Returns:
            Ranked list of results
        """
        try:
            from pinecone import Pinecone
        except ImportError as e:
            raise ImportError("pinecone not installed") from e

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            # Audit R-9: surface the missing-credential failure as a
            # WARNING-level structured log + counter so dashboards
            # catch silent breakage (the old code raised ValueError
            # but downstream callers swallowed it into a debug log,
            # making the "Pinecone is configured but doing nothing"
            # state invisible).
            _log.warning(
                "pinecone_api_key_missing",
                index=self.pinecone_index_name,
                advice="set PINECONE_API_KEY to enable the Pinecone backend",
            )
            metrics.inc("pinecone_api_key_missing")
            raise ValueError("PINECONE_API_KEY environment variable is required for Pinecone backend")

        pc = Pinecone(api_key=api_key)
        index = pc.Index(self.pinecone_index_name)

        # Build filter
        filter_dict = None
        if active_only:
            filter_dict = {"status": {"$eq": "active"}}

        # Search using integrated inference (text query, no local embedding)
        search_results = index.search_records(
            namespace=self.pinecone_namespace,
            query={"inputs": {"text": query}, "top_k": limit, "filter": filter_dict or {}},  # type: ignore[arg-type]
        )

        # Convert to standard format
        results = []
        for hit in search_results.get("result", {}).get("hits", []):
            fields = hit.get("fields", {})
            results.append(
                {
                    "_id": hit.get("_id", "?"),
                    "type": fields.get("block_type", "unknown"),
                    "score": round(hit.get("_score", 0.0), 4),
                    "excerpt": fields.get("excerpt", ""),
                    "file": fields.get("file", "?"),
                    "line": fields.get("line", 0),
                    "status": fields.get("status", ""),
                }
            )

        return results


# ---------------------------------------------------------------------------
# Batch Search (for RRF fusion in hybrid_recall.py)
# ---------------------------------------------------------------------------


def search_batch(
    workspace: str,
    query: str,
    limit: int = 200,
    active_only: bool = False,
    config: dict | None = None,
    *,
    scoring_instant: date | None = None,
) -> list[dict]:
    """Search using VectorBackend with the given (or default) config.

    This is a convenience function used by HybridBackend to fetch a large
    candidate pool for RRF fusion.  Unlike VectorBackend.search(), it
    accepts workspace as the first positional arg so the calling convention
    matches recall.recall().

    Args:
        workspace: Workspace root path.
        query: Search query text.
        limit: Maximum results (default 200 for RRF pool).
        active_only: Only return active blocks.
        config: Optional recall config dict. If None, loads from
            ``<workspace>/mind-mem.json``.
        scoring_instant: UTC date the recency boost scores against;
            ``None`` resolves to today in UTC.

    Returns:
        List of result dicts (same format as VectorBackend.search).
        Returns empty list on any error (e.g. sentence-transformers missing).
    """
    if not query or not query.strip():
        return []

    if config is None:
        config_path = os.path.join(workspace, "mind-mem.json")
        config = {}
        if os.path.isfile(config_path):
            try:
                with open(config_path, encoding="utf-8") as f:
                    full = json.load(f)
                    config = full.get("recall", {})
            except (OSError, json.JSONDecodeError):
                pass

    try:
        backend = VectorBackend(config)
        return backend.search(workspace, query, limit=limit, active_only=active_only, scoring_instant=scoring_instant)
    except ImportError:
        _log.info("search_batch_unavailable", reason="sentence-transformers not installed")
        return []
    except Exception as e:
        _log.error("search_batch_failed", error=str(e))
        return []


# ---------------------------------------------------------------------------
# Index Rebuild (called from MCP reindex tool)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# v4.pq — opt-in product-quantization of the local vector index (default OFF)
# ---------------------------------------------------------------------------

#: How many vectors the PQ codebook is trained on. The default equals the
#: default centroid count on purpose: ``pq._stdlib_kmeans`` short-circuits
#: when ``k >= len(points)``, so training at the default is O(n) rather than
#: O(M x iters x n x K x sub_dim) of pure-Python float math.
#:
#: deferred: raising ``train_sample`` above ``centroids`` runs real Lloyd
#: iterations in pure Python and is minutes-to-hours on a large corpus --
#: upgrade path: install a numpy/sklearn trainer with
#: ``pq.set_codebook_trainer(...)`` first, then raise the knob.
_PQ_DEFAULT_TRAIN_SAMPLE = 256


def _pq_compress(workspace: str, codebook_name: str, ids: list[str], embeddings: list[list[float]]) -> dict | None:
    """PQ-encode ``embeddings`` into ``index.db``; ``None`` when the flag is OFF.

    Wired into :func:`rebuild_index` -- the vector-index rebuild the MCP
    ``reindex(include_vectors=True)`` tool actually calls. (The architect's
    plan named ``VectorBackend._index_local``; that method has no live caller
    in this tree, so wiring it there would have been wiring into dead code.)

    The flag is read ONCE per rebuild, before any per-vector work, with the
    QUIET probe: with ``v4.pq`` off this function reads no config twice, opens
    no database, writes no row and logs nothing, so a rebuild is byte-for-byte
    what it was unwired.

    Deterministic: the training sample is the first ``train_sample`` vectors
    in the rebuild's own (sorted-file, parse-order) sequence, and the codebook
    trainer seeds its k-means++ from ``cfg.kmeans_seed``. No clock, no RNG
    draw that is not seeded.

    Returns a small report dict, or ``None`` if the flag is off or the vectors
    cannot be encoded under the configured geometry.
    """
    try:
        from .v4.feature_flags import is_enabled_quiet

        if not is_enabled_quiet("pq"):
            return None
    except Exception:  # pragma: no cover - defensive: v4 surface absent
        return None

    from .v4 import pq as _pq

    # Filter ids and vectors AS PAIRS. Filtering the vectors alone and then
    # zipping against the unfiltered ids is how a block ends up wearing its
    # neighbour's code — silently, and only for corpora that contain one
    # unembeddable block.
    pairs = [(bid, list(vec)) for bid, vec in zip(ids, embeddings) if vec]
    if not pairs:
        _log.info("pq_compress_skipped", reason="no_usable_vectors")
        return None
    dim = len(pairs[0][1])
    ragged = [bid for bid, vec in pairs if len(vec) != dim]
    if ragged:
        # A mixed-dimension training set makes train_codebook raise, and a
        # codebook trained on the wrong shape encodes every real vector to
        # b"" — an empty index with no error anywhere.
        _log.warning("pq_compress_skipped", reason="ragged_embedding_dimensions", dim=dim, offenders=len(ragged))
        return None

    cfg = _pq._load_config()
    if cfg.subvectors <= 0 or dim % cfg.subvectors != 0:
        # Encoding under a geometry that does not divide the dimension would
        # make encode() return b"" for every vector -- a silently empty index.
        _log.warning("pq_compress_skipped", reason="subvectors_do_not_divide_dim", dim=dim, subvectors=cfg.subvectors)
        return None

    sample_n = max(1, min(len(pairs), _pq_train_sample()))
    codebook = _pq.train_codebook([vec for _bid, vec in pairs[:sample_n]], cfg)
    _pq.store_codebook(workspace, codebook_name, codebook)

    encoded = 0
    for bid, vec in pairs:
        code = _pq.encode(vec, codebook)
        if not code:
            continue
        _pq.store_code(workspace, bid, codebook_name, code)
        encoded += 1

    report = {
        "codebook": codebook_name,
        "dim": dim,
        "subvectors": cfg.subvectors,
        "train_sample": sample_n,
        "encoded": encoded,
        "bytes_per_vector": cfg.subvectors,
    }
    _log.info("pq_compress", **report)
    return report


def _pq_train_sample() -> int:
    """The ``v4.pq.train_sample`` knob, defaulted and floored.

    Separate from ``pq._load_config`` because the cap is a property of THIS
    call site (a pure-Python trainer on a rebuild path), not of the codec.
    """
    try:
        from .v4.feature_flags import flag_config

        raw = flag_config("pq", quiet=True)
        v = raw.get("train_sample", _PQ_DEFAULT_TRAIN_SAMPLE) if isinstance(raw, dict) else _PQ_DEFAULT_TRAIN_SAMPLE
        if isinstance(v, bool):
            return _PQ_DEFAULT_TRAIN_SAMPLE
        return max(1, int(v))
    except (TypeError, ValueError, ImportError):
        return _PQ_DEFAULT_TRAIN_SAMPLE


def rebuild_index(workspace: str) -> int:
    """Rebuild vector index for all blocks in the workspace.

    Parses all block files, generates embeddings, and writes them to the
    local vector index.

    Args:
        workspace: Workspace root path.

    Returns:
        Number of blocks indexed.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    config_path = os.path.join(workspace, "mind-mem.json")
    config: dict = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                full = json.load(f)
                config = full.get("recall", {})
        except (OSError, json.JSONDecodeError):
            pass

    backend = VectorBackend(config)

    # Collect all blocks from workspace
    from .block_parser import parse_file as _parse

    parsed: list[dict] = []
    for subdir in CORPUS_DIRS:
        d = os.path.join(workspace, subdir)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".md"):
                try:
                    parsed.extend(_parse(os.path.join(d, fn)))
                except Exception as exc:
                    _log.debug("corpus_file_parse_skipped", error=str(exc))

    # ADMISSION. This loop parsed the corpus off disk with no status filter at
    # all, so a quarantined or pending block went straight into the vector
    # index and was reachable by similarity even though every text path
    # correctly withholds it. Same defect class as the consolidation loader
    # that SELECTed a status and never filtered on it -- here there was not
    # even a status to ignore.
    #
    # `admit_corpus` is the shared gate and is called, never re-implemented:
    # a hand-rolled status check here would drift from the one table that
    # decides servability, and an unstated status is SERVABLE, so a
    # re-implementation that only rejected known-bad values would admit
    # anything unlabelled.
    from .admissibility import admit_corpus

    blocks = admit_corpus(parsed)
    withheld = len(parsed) - len(blocks)
    if withheld:
        _log.info("rebuild_index_withheld", workspace=workspace, withheld=withheld)

    if not blocks:
        _log.info("rebuild_index_empty", workspace=workspace)
        return 0

    # Build texts for embedding
    texts = []
    for b in blocks:
        text = b.get("Statement", b.get("Description", b.get("Subject", "")))
        texts.append(str(text))

    # Audit R-7: route through _embed_for_provider so configured
    # provider preference + circuit breaker + fallback chain are
    # honored. The old `backend.model.encode(...)` bypass quietly
    # used sentence-transformers regardless of the operator's
    # ``embedder`` config and ignored circuit-broken providers.
    raw_embeddings = backend._embed_for_provider(texts)

    # Write to local index
    index_dir = os.path.join(workspace, backend.index_path)
    os.makedirs(index_dir, exist_ok=True)
    index_data = []
    for i, b in enumerate(blocks):
        emb = raw_embeddings[i]
        # _embed_for_provider returns list[list[float]]; some backends
        # may produce numpy arrays so we coerce defensively.
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        index_data.append(
            {
                "_id": b.get("_id", f"block_{i}"),
                "embedding": list(emb),
                "text": texts[i],
            }
        )

    index_file = os.path.join(index_dir, "index.json")
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f)

    # v4.pq (default OFF): compress the same vectors into 1 byte per
    # subvector position in the v4 side store. One flag read, here, after
    # every per-vector loop above has already finished.
    #
    # The ids are filtered through ``admissibility.admissible`` first. This
    # leg reads blocks, so it calls the shared admission gate -- the rule that
    # exists because a loader once SELECTed a status column and never filtered
    # on it. It matters concretely here: ``rebuild_index`` parses the corpus
    # with ``parse_file`` and no status filter at all, so ``blocks`` above (and
    # therefore ``index.json``) contains quarantined blocks. That is a
    # PRE-EXISTING gap in the JSON index and is deliberately not changed here
    # -- but a new store must not inherit it, so nothing withheld gets a PQ
    # code.
    from .admissibility import admissible as _admissible

    _servable = _admissible(blocks)
    _pq_records = [rec for rec in index_data if rec["_id"] in _servable]
    _pq_compress(
        workspace,
        backend.model_name,
        [rec["_id"] for rec in _pq_records],
        [rec["embedding"] for rec in _pq_records],
    )

    _log.info("rebuild_index_complete", blocks=len(blocks), workspace=workspace)
    return len(blocks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="mind-mem Vector Recall Backend (Semantic Search)")
    parser.add_argument(
        "--workspace",
        "-w",
        default=None,
        help="Workspace path (default: $MIND_MEM_WORKSPACE > nearest mind-mem.json upward > cwd)",
    )
    parser.add_argument("--index", action="store_true", help="Build/rebuild index")
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Max results")
    parser.add_argument("--active-only", action="store_true", help="Only search active blocks")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--force", action="store_true", help="Force reindex")
    args = parser.parse_args()

    # Resolve workspace via the shared resolver so this twin CLI stops
    # diverging from _recall_core.main (--workspace > $MIND_MEM_WORKSPACE >
    # nearest mind-mem.json upward > cwd).
    from ._recall_workspace import resolve_workspace

    workspace = resolve_workspace(args.workspace)

    # Load config
    config_path = os.path.join(workspace, "mind-mem.json")
    config = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                full_config = json.load(f)
                config = full_config.get("recall", {})
        except (OSError, json.JSONDecodeError) as e:
            print(f"recall_vector: config load error: {e}", file=sys.stderr)
            sys.exit(1)

    # Initialize backend
    try:
        backend = VectorBackend(config)
    except ImportError as e:
        print(f"recall_vector: {e}", file=sys.stderr)
        sys.exit(1)

    # Index mode
    if args.index:
        try:
            backend.index(workspace)
            print(f"Index built successfully at {backend._get_index_path(workspace)}")
        except Exception as e:
            print(f"recall_vector: indexing failed: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Search mode
    if not args.query:
        print("recall_vector: --query required for search mode", file=sys.stderr)
        sys.exit(1)

    try:
        results = backend.search(workspace, args.query, args.limit, args.active_only)
    except Exception as e:
        print(f"recall_vector: search failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if not results:
            # Distinguish an empty/missing store from a genuine 0-of-N miss
            # (mirrors _recall_core.main).
            from ._recall_workspace import empty_workspace_warning, probe_block_count

            health = probe_block_count(workspace)
            if health.probe_error is not None:
                print(
                    f"recall_vector: note — could not verify workspace block count ({health.probe_error})",
                    file=sys.stderr,
                )
            elif health.is_empty_or_unbuilt:
                print(empty_workspace_warning(health), file=sys.stderr)
            print("No results found.")
        else:
            for r in results:
                print(f"[{r['score']:.3f}] {r['_id']} ({r['type']}) — {r['excerpt'][:80]}")
                print(f"        {r['file']}:{r['line']}")


if __name__ == "__main__":
    main()
