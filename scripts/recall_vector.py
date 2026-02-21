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
        "pinecone_api_key": "...",
        "pinecone_environment": "us-east-1-aws"
      }
    }

Usage:
    # Indexing
    python3 scripts/recall_vector.py --index --workspace .

    # Search
    python3 scripts/recall_vector.py --query "authentication" --workspace . --limit 5

    # Force reindex
    python3 scripts/recall_vector.py --index --force --workspace .
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import helpers from recall.py
from block_parser import parse_file
from observability import get_logger, metrics, timed
from recall import (
    CORPUS_FILES,
    RecallBackend,
    date_score,
    extract_text,
    get_block_type,
    get_excerpt,
)

_log = get_logger("recall_vector")


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
        _VALID_KEYS = {"backend", "provider", "model", "index_path", "dimension", "top_k",
                       "pinecone_api_key", "pinecone_index", "pinecone_namespace",
                       "qdrant_url", "qdrant_collection", "pinecone_environment",
                       "rrf_k", "bm25_weight", "vector_weight", "vector_model",
                       "vector_enabled", "onnx_backend", "sqlite_vec_db",
                       "llama_cpp_url"}
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
        self._model = None
        self._model_loaded = False

        # llama.cpp embedding server URL
        self.llama_cpp_url = str(config.get("llama_cpp_url", "http://localhost:8090"))

        # Pinecone integrated inference config
        self.pinecone_index_name = str(config.get("pinecone_index", "mind-mem"))
        self.pinecone_namespace = str(config.get("pinecone_namespace", "default"))

        _log.info("vector_backend_init", provider=self.provider, model=self.model_name)

    @property
    def model(self):
        """Lazy-load embedding model on first access."""
        if not self._model_loaded:
            try:
                # Temporarily remove scripts/ from sys.path so that
                # sentence_transformers can find the real `filelock` package
                # (scripts/filelock.py shadows it). We remove by realpath
                # to catch both relative and absolute path variants.
                _scripts_real = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
                _saved = list(sys.path)
                sys.path[:] = [p for p in sys.path
                               if os.path.realpath(p) != _scripts_real]
                try:
                    from sentence_transformers import SentenceTransformer
                finally:
                    sys.path[:] = _saved
                cache_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=cache_dir,
                )
                self._model_loaded = True
                _log.info("embedding_model_loaded", model=self.model_name)
            except ImportError as e:
                _log.error("sentence_transformers_not_installed", error=str(e))
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install 'mind-mem[embeddings]'"
                ) from e
        return self._model

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
        _scripts_paths = [p for p in _sys.path if "mind-mem/scripts" in p or p.endswith("/scripts")]
        for _p in _scripts_paths:
            _sys.path.remove(_p)
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "fastembed not installed. Install with: pip install fastembed"
            ) from e
        finally:
            for _p in _scripts_paths:
                _sys.path.insert(0, _p)

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

        url = self.llama_cpp_url.rstrip("/") + "/embeddings"
        BATCH = 32
        all_embeddings = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            payload = json.dumps({"input": batch}).encode("utf-8")
            req = _req.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with _req.urlopen(req, timeout=120) as resp:
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
            raise ImportError(
                "sqlite-vec not installed. Install with: pip install sqlite-vec"
            ) from e

        db_path = self._sqlite_vec_db_path(workspace)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        if readonly:
            conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        else:
            conn = _sqlite3.connect(db_path)

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
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
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_blocks USING vec0(
                block_id TEXT PRIMARY KEY,
                embedding FLOAT[{dimension}]
            )
        """)
        conn.commit()

    def _index_sqlite_vec(self, workspace: str, blocks: list[dict], texts: list[str]) -> None:
        """Build sqlite-vec index: embed texts locally, store in recall.db.

        Args:
            workspace: Workspace root path
            blocks: Block metadata list
            texts: Searchable text for each block
        """
        import sqlite_vec

        _log.info("sqlite_vec_indexing_start", count=len(texts))

        embeddings = self.embed_fastembed(texts)
        if not embeddings:
            _log.error("sqlite_vec_no_embeddings")
            return

        dim = len(embeddings[0])
        if self.dimension is None:
            self.dimension = dim

        conn = self._connect_sqlite_vec(workspace)
        try:
            self._init_vec_table(conn, dim)
            # Clear existing vectors before rebuild
            conn.execute("DELETE FROM vec_blocks")
            for block, emb in zip(blocks, embeddings):
                conn.execute(
                    "INSERT INTO vec_blocks(block_id, embedding) VALUES (?, ?)",
                    (block["_id"], sqlite_vec.serialize_float32(emb)),
                )
            conn.commit()
            _log.info("sqlite_vec_indexed", blocks=len(blocks), dimension=dim)
        finally:
            conn.close()

        # Cache block metadata to a companion JSON (for search result enrichment)
        meta_path = os.path.join(workspace, ".mind-mem-index", "vec_meta.json")
        meta = {b["_id"]: b for b in blocks}
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def _search_sqlite_vec(
        self, workspace: str, query: str, limit: int, active_only: bool
    ) -> list[dict]:
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

        # Embed query
        query_emb = self.embed_fastembed([query])[0]
        query_bytes = sqlite_vec.serialize_float32(query_emb)

        # Load block metadata
        meta_path = os.path.join(workspace, ".mind-mem-index", "vec_meta.json")
        meta: dict = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        conn = self._connect_sqlite_vec(workspace, readonly=True)
        try:
            fetch_limit = limit * 3 if active_only else limit
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
            results.append({
                "_id": block_id,
                "type": block.get("type", "unknown"),
                "score": score,
                "excerpt": block.get("excerpt", ""),
                "file": block.get("file", "?"),
                "line": block.get("line", 0),
                "status": status,
            })

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
            with open(index_file) as f:
                index = json.load(f)
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
            with open(index_file, "w") as f:
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
            texts = []
            block_metadata = []
            for block in all_blocks:
                text = extract_text(block)
                if not text.strip():
                    continue

                texts.append(text)
                block_metadata.append({
                    "_id": block.get("_id", "?"),
                    "type": get_block_type(block.get("_id", "")),
                    "excerpt": get_excerpt(block),
                    "file": block.get("_source_file", "?"),
                    "line": block.get("_line", 0),
                    "status": block.get("Status", ""),
                    "date": block.get("Date", ""),
                })

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
            raise ImportError(
                "qdrant-client not installed. Install with: pip install qdrant-client"
            ) from e

        url = self.config.get("qdrant_url", "http://localhost:6333")
        collection = self.config.get("qdrant_collection", "mind-mem")

        client = QdrantClient(url=url)

        # Recreate collection
        try:
            client.delete_collection(collection)
        except Exception:
            pass

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
            raise ImportError(
                "pinecone not installed. Install with: pip install pinecone"
            ) from e

        api_key = os.environ.get("PINECONE_API_KEY") or self.config.get("pinecone_api_key")
        if not api_key:
            raise ValueError("pinecone_api_key required (config or PINECONE_API_KEY env var)")

        pc = Pinecone(api_key=api_key)
        index = pc.Index(self.pinecone_index_name)

        # Build records with text content for integrated inference
        # The index's fieldMap maps "content" → embedding model
        BATCH_SIZE = 96
        total = 0
        for i in range(0, len(blocks), BATCH_SIZE):
            batch_blocks = blocks[i:i + BATCH_SIZE]
            batch_texts = texts[i:i + BATCH_SIZE]
            records = []
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

    def search(self, workspace: str, query: str, limit: int = 10, active_only: bool = False) -> list[dict]:
        """Search using semantic/vector similarity.

        Args:
            workspace: Workspace path
            query: Search query text
            limit: Maximum number of results
            active_only: Only return active blocks

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
                results = self._search_local(workspace, query, limit, active_only)
            elif self.provider == "qdrant":
                results = self._search_qdrant(workspace, query, limit, active_only)
            elif self.provider == "pinecone":
                results = self._search_pinecone(workspace, query, limit, active_only)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            metrics.inc("vector_searches")
            metrics.inc("vector_results", len(results))
            _log.info("vector_search_complete", query=query, results=len(results),
                     top_score=results[0]["score"] if results else 0)
            return results

    def _search_local(self, workspace: str, query: str, limit: int, active_only: bool) -> list[dict]:
        """Search local JSON index.

        Args:
            workspace: Workspace path
            query: Search query
            limit: Max results
            active_only: Filter to active blocks only

        Returns:
            Ranked list of results
        """
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
                recency = date_score({"Date": date_str})
                score *= (0.7 + 0.3 * recency)

            # Status boost
            status = block.get("status", "")
            if status == "active":
                score *= 1.2
            elif status in ("todo", "doing"):
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
            query_filter = Filter(
                must=[FieldCondition(key="status", match=MatchValue(value="active"))]
            )

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
            results.append({
                "_id": payload.get("_id", "?"),
                "type": payload.get("type", "unknown"),
                "score": round(hit.score, 4),
                "excerpt": payload.get("excerpt", ""),
                "file": payload.get("file", "?"),
                "line": payload.get("line", 0),
                "status": payload.get("status", ""),
            })

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

        api_key = os.environ.get("PINECONE_API_KEY") or self.config.get("pinecone_api_key")
        if not api_key:
            raise ValueError("pinecone_api_key required (config or PINECONE_API_KEY env var)")

        pc = Pinecone(api_key=api_key)
        index = pc.Index(self.pinecone_index_name)

        # Build filter
        filter_dict = None
        if active_only:
            filter_dict = {"status": {"$eq": "active"}}

        # Search using integrated inference (text query, no local embedding)
        search_results = index.search_records(
            namespace=self.pinecone_namespace,
            query={"inputs": {"text": query}, "top_k": limit, "filter": filter_dict or {}},
        )

        # Convert to standard format
        results = []
        for hit in search_results.get("result", {}).get("hits", []):
            fields = hit.get("fields", {})
            results.append({
                "_id": hit.get("_id", "?"),
                "type": fields.get("block_type", "unknown"),
                "score": round(hit.get("_score", 0.0), 4),
                "excerpt": fields.get("excerpt", ""),
                "file": fields.get("file", "?"),
                "line": fields.get("line", 0),
                "status": fields.get("status", ""),
            })

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
                with open(config_path) as f:
                    full = json.load(f)
                    config = full.get("recall", {})
            except (OSError, json.JSONDecodeError):
                pass

    try:
        backend = VectorBackend(config)
        return backend.search(workspace, query, limit=limit, active_only=active_only)
    except ImportError:
        _log.info("search_batch_unavailable", reason="sentence-transformers not installed")
        return []
    except Exception as e:
        _log.error("search_batch_failed", error=str(e))
        return []


# ---------------------------------------------------------------------------
# Index Rebuild (called from MCP reindex tool)
# ---------------------------------------------------------------------------


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
            with open(config_path) as f:
                full = json.load(f)
                config = full.get("recall", {})
        except (OSError, json.JSONDecodeError):
            pass

    backend = VectorBackend(config)

    # Collect all blocks from workspace
    from block_parser import parse_file as _parse
    blocks: list[dict] = []
    for subdir in ("decisions", "tasks", "entities", "intelligence"):
        d = os.path.join(workspace, subdir)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".md"):
                try:
                    blocks.extend(_parse(os.path.join(d, fn)))
                except Exception:
                    pass

    if not blocks:
        _log.info("rebuild_index_empty", workspace=workspace)
        return 0

    # Build texts for embedding
    texts = []
    for b in blocks:
        text = b.get("Statement", b.get("Description", b.get("Subject", "")))
        texts.append(str(text))

    embeddings = backend.model.encode(texts, show_progress_bar=False)

    # Write to local index
    index_dir = os.path.join(workspace, backend.index_path)
    os.makedirs(index_dir, exist_ok=True)
    index_data = []
    for i, b in enumerate(blocks):
        index_data.append({
            "_id": b.get("_id", f"block_{i}"),
            "embedding": embeddings[i].tolist(),
            "text": texts[i],
        })

    index_file = os.path.join(index_dir, "index.json")
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f)

    _log.info("rebuild_index_complete", blocks=len(blocks), workspace=workspace)
    return len(blocks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="mind-mem Vector Recall Backend (Semantic Search)"
    )
    parser.add_argument("--workspace", "-w", default=".", help="Workspace path")
    parser.add_argument("--index", action="store_true", help="Build/rebuild index")
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Max results")
    parser.add_argument("--active-only", action="store_true", help="Only search active blocks")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--force", action="store_true", help="Force reindex")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.workspace, "mind-mem.json")
    config = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
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
            backend.index(args.workspace)
            print(f"Index built successfully at {backend._get_index_path(args.workspace)}")
        except Exception as e:
            print(f"recall_vector: indexing failed: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Search mode
    if not args.query:
        print("recall_vector: --query required for search mode", file=sys.stderr)
        sys.exit(1)

    try:
        results = backend.search(args.workspace, args.query, args.limit, args.active_only)
    except Exception as e:
        print(f"recall_vector: search failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if not results:
            print("No results found.")
        else:
            for r in results:
                print(f"[{r['score']:.3f}] {r['_id']} ({r['type']}) — {r['excerpt'][:80]}")
                print(f"        {r['file']}:{r['line']}")


if __name__ == "__main__":
    main()
