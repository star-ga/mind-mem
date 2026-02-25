"""Tests for recall_vector.py — VectorBackend semantic search."""

from __future__ import annotations

import json
import math
import os

import pytest

# Ensure scripts/ is on path
_HERE = os.path.dirname(os.path.abspath(__file__))

from mind_mem.recall_vector import VectorBackend, search_batch  # noqa: E402

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_backend(overrides: dict | None = None) -> VectorBackend:
    """Create a VectorBackend with default test config."""
    config = {"provider": "local", "model": "all-MiniLM-L6-v2"}
    if overrides:
        config.update(overrides)
    return VectorBackend(config)


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a minimal workspace with mind-mem.json."""
    config = {"version": "1.1.0", "recall": {"provider": "local"}}
    (tmp_path / "mind-mem.json").write_text(json.dumps(config))
    return str(tmp_path)


# ── VectorBackend.__init__ ───────────────────────────────────────────


class TestVectorBackendInit:
    def test_defaults(self):
        vb = VectorBackend({})
        assert vb.provider == "local"
        assert vb.model_name == "all-MiniLM-L6-v2"
        assert vb.index_path == ".mind-mem-vectors"
        assert vb.dimension is None
        assert vb._model is None
        assert vb._model_loaded is False

    def test_custom_config(self):
        vb = VectorBackend(
            {
                "provider": "qdrant",
                "model": "bge-small-en-v1.5",
                "index_path": "custom-idx",
                "dimension": 384,
            }
        )
        assert vb.provider == "qdrant"
        assert vb.model_name == "bge-small-en-v1.5"
        assert vb.index_path == "custom-idx"
        assert vb.dimension == 384

    def test_non_dict_config_handled(self):
        """Non-dict config should be replaced with empty dict."""
        vb = VectorBackend("not a dict")
        assert vb.provider == "local"
        assert vb.config == {}

    def test_unknown_keys_warning(self, caplog):
        """Unknown config keys should log a warning."""
        VectorBackend({"provider": "local", "bogus_key": True})
        # Warning logged via structured logger — just verify no crash

    def test_llama_cpp_url_default(self):
        vb = VectorBackend({})
        assert vb.llama_cpp_url == "http://localhost:8090"

    def test_llama_cpp_url_custom(self):
        vb = VectorBackend({"llama_cpp_url": "http://gpu-server:9090"})
        assert vb.llama_cpp_url == "http://gpu-server:9090"

    def test_pinecone_defaults(self):
        vb = VectorBackend({})
        assert vb.pinecone_index_name == "mind-mem"
        assert vb.pinecone_namespace == "default"

    def test_pinecone_custom(self):
        vb = VectorBackend(
            {
                "pinecone_index": "my-index",
                "pinecone_namespace": "prod",
            }
        )
        assert vb.pinecone_index_name == "my-index"
        assert vb.pinecone_namespace == "prod"


# ── cosine_similarity ────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vb = _make_backend()
        score = vb.cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        vb = _make_backend()
        score = vb.cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(score) < 1e-6

    def test_opposite_vectors(self):
        vb = _make_backend()
        score = vb.cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(score - (-1.0)) < 1e-6

    def test_empty_vectors(self):
        vb = _make_backend()
        assert vb.cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        vb = _make_backend()
        assert vb.cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector(self):
        vb = _make_backend()
        assert vb.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_known_angle(self):
        """45-degree angle vectors should have cos(45) = sqrt(2)/2."""
        vb = _make_backend()
        score = vb.cosine_similarity([1.0, 0.0], [1.0, 1.0])
        expected = 1.0 / math.sqrt(2)
        assert abs(score - expected) < 1e-6

    def test_high_dimensional(self):
        """Test with 384-dim vectors (typical embedding size)."""
        vb = _make_backend()
        v1 = [1.0] * 384
        v2 = [1.0] * 384
        score = vb.cosine_similarity(v1, v2)
        assert abs(score - 1.0) < 1e-6


# ── _get_index_path ─────────────────────────────────────────────────


class TestGetIndexPath:
    def test_default_path(self):
        vb = _make_backend()
        path = vb._get_index_path("/workspace")
        assert path == os.path.join("/workspace", ".mind-mem-vectors", "index.json")

    def test_custom_path(self):
        vb = _make_backend({"index_path": "my-vectors"})
        path = vb._get_index_path("/ws")
        assert path == os.path.join("/ws", "my-vectors", "index.json")


# ── _load_local_index / _save_local_index ────────────────────────────


class TestLocalIndex:
    def test_load_missing_returns_none(self, tmp_path):
        vb = _make_backend()
        result = vb._load_local_index(str(tmp_path))
        assert result is None

    def test_save_and_load_roundtrip(self, tmp_path):
        vb = _make_backend()
        workspace = str(tmp_path)
        test_index = {
            "model": "test-model",
            "dimension": 3,
            "blocks": [{"_id": "B1", "type": "fact"}],
            "embeddings": [[0.1, 0.2, 0.3]],
        }
        vb._save_local_index(workspace, test_index)

        loaded = vb._load_local_index(workspace)
        assert loaded is not None
        assert loaded["model"] == "test-model"
        assert loaded["dimension"] == 3
        assert len(loaded["blocks"]) == 1
        assert loaded["blocks"][0]["_id"] == "B1"
        assert loaded["embeddings"] == [[0.1, 0.2, 0.3]]

    def test_save_creates_directory(self, tmp_path):
        vb = _make_backend({"index_path": "deep/nested/vectors"})
        workspace = str(tmp_path)
        vb._save_local_index(workspace, {"blocks": [], "embeddings": []})
        assert os.path.isfile(os.path.join(workspace, "deep", "nested", "vectors", "index.json"))

    def test_load_corrupt_json_returns_none(self, tmp_path):
        vb = _make_backend()
        workspace = str(tmp_path)
        idx_dir = os.path.join(workspace, ".mind-mem-vectors")
        os.makedirs(idx_dir)
        with open(os.path.join(idx_dir, "index.json"), "w") as f:
            f.write("{corrupt json!!")
        result = vb._load_local_index(workspace)
        assert result is None


# ── _search_local ────────────────────────────────────────────────────


class TestSearchLocal:
    """Test local search WITHOUT requiring sentence-transformers.

    We pre-build the index with known embeddings and test search logic.
    """

    def _build_index(self, workspace: str, blocks: list[dict], embeddings: list[list[float]]):
        """Write a local index directly."""
        vb = _make_backend()
        index = {
            "model": "test",
            "dimension": len(embeddings[0]) if embeddings else 0,
            "blocks": blocks,
            "embeddings": embeddings,
        }
        vb._save_local_index(workspace, index)

    def test_search_empty_index_returns_empty(self, tmp_path):
        """Search with no index file should return empty."""
        vb = _make_backend()
        # Don't build any index
        # search requires embed() which needs sentence-transformers
        # So we test _load_local_index path only
        result = vb._load_local_index(str(tmp_path))
        assert result is None

    def test_search_empty_query(self, tmp_path):
        vb = _make_backend()
        results = vb.search(str(tmp_path), "", limit=5)
        assert results == []

    def test_search_whitespace_query(self, tmp_path):
        vb = _make_backend()
        results = vb.search(str(tmp_path), "   ", limit=5)
        assert results == []


# ── _embed_for_provider routing ──────────────────────────────────────


class TestEmbedForProviderRouting:
    def test_llama_cpp_routing(self):
        vb = _make_backend({"onnx_backend": "llama_cpp"})
        backend = vb.config.get("onnx_backend")
        assert backend == "llama_cpp"
        # Would call embed_llama_cpp — tested via integration

    def test_fastembed_default_routing(self):
        vb = _make_backend({"onnx_backend": True})
        backend = vb.config.get("onnx_backend")
        assert backend is True
        # Default: calls embed_fastembed


# ── search_batch convenience function ────────────────────────────────


class TestSearchBatch:
    def test_empty_query_returns_empty(self):
        assert search_batch("/nonexistent", "") == []

    def test_whitespace_query_returns_empty(self):
        assert search_batch("/nonexistent", "   ") == []

    def test_none_query_returns_empty(self):
        assert search_batch("/nonexistent", None) == []

    def test_missing_workspace_returns_empty(self):
        """Missing workspace should return empty (not crash)."""
        result = search_batch("/nonexistent/path/xyz", "test query")
        assert result == []

    def test_with_explicit_config(self, tmp_path):
        """Passing explicit config should skip file loading."""
        result = search_batch(str(tmp_path), "test", config={"provider": "local"})
        assert result == []


# ── _sqlite_vec_db_path ──────────────────────────────────────────────


class TestSqliteVecDbPath:
    def test_default_path(self):
        vb = _make_backend()
        path = vb._sqlite_vec_db_path("/ws")
        assert path == os.path.join("/ws", ".mind-mem-index", "recall.db")

    def test_custom_path(self):
        vb = _make_backend({"sqlite_vec_db": "custom/my.db"})
        path = vb._sqlite_vec_db_path("/ws")
        assert os.path.normpath(path) == os.path.normpath("/ws/custom/my.db")


# ── Provider validation ──────────────────────────────────────────────


class TestProviderValidation:
    def test_unknown_provider_raises_on_search(self, tmp_path):
        vb = _make_backend({"provider": "nonexistent"})
        with pytest.raises(ValueError, match="Unknown provider"):
            vb.search(str(tmp_path), "test query", limit=5)

    def test_valid_providers(self):
        for p in ("local", "qdrant", "pinecone", "sqlite_vec", "llama_cpp"):
            vb = _make_backend({"provider": p})
            assert vb.provider == p


# ── Embedding Cache (P0) ─────────────────────────────────────────────


class TestEmbeddingCache:
    """Test embedding cache: sqlite3 + hashlib + struct — all stdlib."""

    def _make_db(self, tmp_path):
        """Create an in-memory-like cache DB for testing."""
        import sqlite3

        db_path = str(tmp_path / "test_cache.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        vb = _make_backend()
        vb._ensure_cache_tables(conn)
        return conn, vb

    def test_content_hash_deterministic(self):
        vb = _make_backend()
        h1 = vb._content_hash("hello world")
        h2 = vb._content_hash("hello world")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_content_hash_different_for_different_text(self):
        vb = _make_backend()
        h1 = vb._content_hash("hello")
        h2 = vb._content_hash("world")
        assert h1 != h2

    def test_serialize_deserialize_roundtrip(self):
        vb = _make_backend()
        original = [0.1, 0.2, 0.3, -1.5, 99.99]
        blob = vb._serialize_embedding(original)
        assert isinstance(blob, bytes)
        assert len(blob) == 5 * 4  # 5 floats * 4 bytes

        restored = vb._deserialize_embedding(blob, 5)
        for a, b in zip(original, restored):
            assert abs(a - b) < 1e-5

    def test_serialize_empty(self):
        vb = _make_backend()
        blob = vb._serialize_embedding([])
        assert blob == b""
        restored = vb._deserialize_embedding(blob, 0)
        assert restored == []

    def test_cache_miss_returns_none(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        result = vb._get_cached_embedding(conn, "block-1", "hash-1")
        assert result is None
        conn.close()

    def test_cache_store_and_hit(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        emb = [0.1, 0.2, 0.3]
        vb._cache_embedding(conn, "block-1", "hash-1", emb)
        conn.commit()

        result = vb._get_cached_embedding(conn, "block-1", "hash-1")
        assert result is not None
        assert len(result) == 3
        for a, b in zip(emb, result):
            assert abs(a - b) < 1e-5
        conn.close()

    def test_cache_miss_on_hash_change(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        vb._cache_embedding(conn, "block-1", "hash-old", [0.1, 0.2])
        conn.commit()

        result = vb._get_cached_embedding(conn, "block-1", "hash-new")
        assert result is None
        conn.close()

    def test_cache_upsert_overwrites(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        vb._cache_embedding(conn, "block-1", "hash-1", [1.0, 2.0])
        conn.commit()

        vb._cache_embedding(conn, "block-1", "hash-2", [3.0, 4.0])
        conn.commit()

        # Old hash should miss
        assert vb._get_cached_embedding(conn, "block-1", "hash-1") is None
        # New hash should hit
        result = vb._get_cached_embedding(conn, "block-1", "hash-2")
        assert result is not None
        assert abs(result[0] - 3.0) < 1e-5
        conn.close()

    def test_invalidate_cache_for_model(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        vb._cache_embedding(conn, "b1", "h1", [1.0])
        vb._cache_embedding(conn, "b2", "h2", [2.0])
        conn.commit()

        # Invalidate for a different model — should delete our entries
        deleted = vb._invalidate_cache_for_model(conn, "different-model")
        conn.commit()
        assert deleted == 2

        # Verify cache is empty
        assert vb._get_cached_embedding(conn, "b1", "h1") is None
        conn.close()

    def test_invalidate_preserves_same_model(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        vb._cache_embedding(conn, "b1", "h1", [1.0])
        conn.commit()

        # Invalidate for the same model — should keep our entries
        deleted = vb._invalidate_cache_for_model(conn, vb.model_name)
        assert deleted == 0

        result = vb._get_cached_embedding(conn, "b1", "h1")
        assert result is not None
        conn.close()

    def test_high_dimensional_cache(self, tmp_path):
        """Test with 384-dim embeddings (typical for bge-small)."""
        conn, vb = self._make_db(tmp_path)
        emb = [float(i) / 384 for i in range(384)]
        vb._cache_embedding(conn, "block-big", "hash-big", emb)
        conn.commit()

        result = vb._get_cached_embedding(conn, "block-big", "hash-big")
        assert result is not None
        assert len(result) == 384
        for a, b in zip(emb, result):
            assert abs(a - b) < 1e-5
        conn.close()


# ── Dimension Mismatch Detection (P0) ────────────────────────────────


class TestDimensionMismatch:
    """Test vec_meta_info tracking and mismatch detection."""

    def _make_db(self, tmp_path):
        import sqlite3

        db_path = str(tmp_path / "test_meta.db")
        conn = sqlite3.connect(db_path)
        vb = _make_backend()
        vb._ensure_cache_tables(conn)
        return conn, vb

    def test_no_prior_metadata_is_match(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        assert vb._check_dimension_match(conn) is True
        conn.close()

    def test_same_model_is_match(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        vb._set_vec_meta(conn, "model_name", vb.model_name)
        vb._set_vec_meta(conn, "dimension", "384")
        vb.dimension = 384
        conn.commit()

        assert vb._check_dimension_match(conn) is True
        conn.close()

    def test_different_model_is_mismatch(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        vb._set_vec_meta(conn, "model_name", "completely-different-model")
        conn.commit()

        assert vb._check_dimension_match(conn) is False
        conn.close()

    def test_different_dimension_is_mismatch(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        vb._set_vec_meta(conn, "model_name", vb.model_name)
        vb._set_vec_meta(conn, "dimension", "1024")
        vb.dimension = 384
        conn.commit()

        assert vb._check_dimension_match(conn) is False
        conn.close()

    def test_meta_get_set_roundtrip(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        vb._set_vec_meta(conn, "test_key", "test_value")
        conn.commit()

        assert vb._get_vec_meta(conn, "test_key") == "test_value"
        assert vb._get_vec_meta(conn, "nonexistent") is None
        conn.close()

    def test_meta_upsert(self, tmp_path):
        conn, vb = self._make_db(tmp_path)
        vb._set_vec_meta(conn, "key", "v1")
        vb._set_vec_meta(conn, "key", "v2")
        conn.commit()

        assert vb._get_vec_meta(conn, "key") == "v2"
        conn.close()

    def test_dimension_none_skips_check(self, tmp_path):
        """When current dimension is None, skip dimension comparison."""
        conn, vb = self._make_db(tmp_path)
        vb._set_vec_meta(conn, "model_name", vb.model_name)
        vb._set_vec_meta(conn, "dimension", "384")
        vb.dimension = None  # Not yet known
        conn.commit()

        assert vb._check_dimension_match(conn) is True
        conn.close()


# ── Circuit Breaker / Fallback ───────────────────────────────────────


class TestCircuitBreaker:
    """Test embedding provider circuit breaker state."""

    def test_initial_state_empty(self):
        vb = _make_backend()
        assert vb._provider_failures == {}

    def test_failures_tracked_across_calls(self):
        """Circuit breaker state persists on the backend instance."""
        vb = _make_backend()
        import time

        vb._provider_failures["test"] = (3, time.time())
        assert "test" in vb._provider_failures
        count, _ = vb._provider_failures["test"]
        assert count == 3
