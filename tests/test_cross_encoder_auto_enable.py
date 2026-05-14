"""v3.3.0 Tier 2 #6 — cross-encoder rerank auto-enables on ambiguous queries.

Pre-v3.3.0 the cross-encoder reranker was strictly opt-in via
``cross_encoder.enabled: true`` in ``mind-mem.json``. For multi-hop
and temporal queries — where rerank quality materially changes the
final answer — operators had to know to flip it.

v3.3.0: when ``cross_encoder.auto_enable`` is truthy (default True)
and the query classifies as multi-hop or temporal, the reranker
fires even without ``enabled: true``. Users who don't want this
behaviour set ``auto_enable: false`` or upgrade their config.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mind_mem.hybrid_recall import HybridBackend


def _fake_workspace(tmp_path) -> str:
    """Minimal workspace for HybridBackend.search()."""
    for d in ("decisions", "tasks", "entities", "intelligence", "memory", ".mind-mem-index"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        "[D-20260420-001]\ntype: decision\nStatement: Use PostgreSQL.\n---\n",
        encoding="utf-8",
    )
    return str(tmp_path)


def _build_backend(auto_enable: bool | None = None, enabled: bool = False) -> HybridBackend:
    ce: dict = {"enabled": enabled}
    if auto_enable is not None:
        ce["auto_enable"] = auto_enable
    return HybridBackend(
        {
            "backend": "bm25",
            "rrf_k": 60,
            "bm25_weight": 1.0,
            "vector_weight": 1.0,
            "vector_enabled": False,
            "cross_encoder": ce,
        }
    )


@pytest.fixture
def mock_bm25_hit():
    """One-result BM25 hit — enough to trigger the CE branch."""
    return [{"_id": "D-20260420-001", "score": 5.0, "content": "Use PostgreSQL.", "excerpt": "Use PostgreSQL."}]


@pytest.fixture(autouse=True)
def _short_circuit_llm_query_features(monkeypatch):
    """v4.0.9 hang fix: backend.search() on multi-hop / temporal queries
    triggers TWO LLM-backed expansion paths, each of which makes live
    HTTP calls that hang on CI where no listener is bound:

      1. ``mind_mem.query_planner.decompose_query`` — POSTs to
         ``http://127.0.0.1:8766/v1/chat/completions`` (claude-proxy).
      2. ``mind_mem.query_expansion.expand_queries`` — LLMQueryExpander
         POSTs to ``https://api.openai.com/v1`` (or the configured
         provider) when auto_enable fires on multi-hop / temporal.

    The Windows-3.x hang surfaced on the 2nd path: even with the
    decompose patch, expansion auto-enables on the same multi-hop
    query and spawns a ThreadPoolExecutor whose worker threads block
    on LLM calls. Mocking both keeps multi-hop *detection* intact
    (which is what these tests verify on the cross-encoder branch)
    while keeping the test network-free.
    """
    monkeypatch.setattr(
        "mind_mem.query_planner.decompose_query",
        lambda query, **_: [query],
    )
    monkeypatch.setattr(
        "mind_mem.query_expansion.expand_queries",
        lambda query, **_: [query],
    )


class TestCrossEncoderAutoEnable:
    def test_multihop_query_auto_enables(self, tmp_path, mock_bm25_hit) -> None:
        backend = _build_backend(auto_enable=True, enabled=False)
        with patch.object(backend, "_bm25_search", return_value=mock_bm25_hit):
            with patch("mind_mem.cross_encoder_reranker.CrossEncoderReranker") as mock_ce:
                mock_ce.is_available = MagicMock(return_value=True)
                instance = MagicMock()
                instance.rerank = MagicMock(return_value=mock_bm25_hit)
                mock_ce.return_value = instance
                backend.search(
                    # Multi-hop phrasing triggers detect_query_type → 'multi-hop'
                    "What decision led to the choice that then caused the outage?",
                    _fake_workspace(tmp_path),
                    limit=5,
                )
                mock_ce.assert_called()

    def test_temporal_query_auto_enables(self, tmp_path, mock_bm25_hit) -> None:
        backend = _build_backend(auto_enable=True, enabled=False)
        with patch.object(backend, "_bm25_search", return_value=mock_bm25_hit):
            with patch("mind_mem.cross_encoder_reranker.CrossEncoderReranker") as mock_ce:
                mock_ce.is_available = MagicMock(return_value=True)
                instance = MagicMock()
                instance.rerank = MagicMock(return_value=mock_bm25_hit)
                mock_ce.return_value = instance
                backend.search(
                    "When did we decide to use PostgreSQL yesterday?",
                    _fake_workspace(tmp_path),
                    limit=5,
                )
                mock_ce.assert_called()

    def test_single_hop_query_does_not_auto_enable(self, tmp_path, mock_bm25_hit) -> None:
        """Simple single-hop queries skip CE — auto-enable only for ambiguous."""
        backend = _build_backend(auto_enable=True, enabled=False)
        with patch.object(backend, "_bm25_search", return_value=mock_bm25_hit):
            with patch("mind_mem.cross_encoder_reranker.CrossEncoderReranker") as mock_ce:
                mock_ce.is_available = MagicMock(return_value=True)
                instance = MagicMock()
                instance.rerank = MagicMock(return_value=mock_bm25_hit)
                mock_ce.return_value = instance
                backend.search("PostgreSQL", _fake_workspace(tmp_path), limit=5)
                mock_ce.assert_not_called()

    def test_explicit_enabled_still_works(self, tmp_path, mock_bm25_hit) -> None:
        """``cross_encoder.enabled: true`` takes precedence — no query-type gate."""
        backend = _build_backend(auto_enable=False, enabled=True)
        with patch.object(backend, "_bm25_search", return_value=mock_bm25_hit):
            with patch("mind_mem.cross_encoder_reranker.CrossEncoderReranker") as mock_ce:
                mock_ce.is_available = MagicMock(return_value=True)
                instance = MagicMock()
                instance.rerank = MagicMock(return_value=mock_bm25_hit)
                mock_ce.return_value = instance
                backend.search("PostgreSQL", _fake_workspace(tmp_path), limit=5)
                mock_ce.assert_called()

    def test_auto_enable_false_disables(self, tmp_path, mock_bm25_hit) -> None:
        """``auto_enable: false`` — multi-hop still skips unless ``enabled: true``."""
        backend = _build_backend(auto_enable=False, enabled=False)
        with patch.object(backend, "_bm25_search", return_value=mock_bm25_hit):
            with patch("mind_mem.cross_encoder_reranker.CrossEncoderReranker") as mock_ce:
                mock_ce.is_available = MagicMock(return_value=True)
                instance = MagicMock()
                instance.rerank = MagicMock(return_value=mock_bm25_hit)
                mock_ce.return_value = instance
                backend.search(
                    "What decision led to the choice that then caused the outage?",
                    _fake_workspace(tmp_path),
                    limit=5,
                )
                mock_ce.assert_not_called()
