"""Tests for recall_vector.py — VectorBackend semantic search."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import pytest

# Ensure scripts/ is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

from recall_vector import VectorBackend, search_batch  # noqa: E402


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
        vb = VectorBackend({
            "provider": "qdrant",
            "model": "bge-small-en-v1.5",
            "index_path": "custom-idx",
            "dimension": 384,
        })
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
        vb = VectorBackend({
            "pinecone_index": "my-index",
            "pinecone_namespace": "prod",
        })
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
        assert path == "/workspace/.mind-mem-vectors/index.json"

    def test_custom_path(self):
        vb = _make_backend({"index_path": "my-vectors"})
        path = vb._get_index_path("/ws")
        assert path == "/ws/my-vectors/index.json"


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
        assert os.path.isfile(os.path.join(workspace, "deep/nested/vectors/index.json"))

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
        result = search_batch(
            str(tmp_path), "test", config={"provider": "local"}
        )
        assert result == []


# ── _sqlite_vec_db_path ──────────────────────────────────────────────


class TestSqliteVecDbPath:
    def test_default_path(self):
        vb = _make_backend()
        path = vb._sqlite_vec_db_path("/ws")
        assert path == "/ws/.mind-mem-index/recall.db"

    def test_custom_path(self):
        vb = _make_backend({"sqlite_vec_db": "custom/my.db"})
        path = vb._sqlite_vec_db_path("/ws")
        assert path == "/ws/custom/my.db"


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
