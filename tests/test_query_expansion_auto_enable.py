"""v3.3.0 Tier 2 #4 — query expansion auto-enables on ambiguous queries.

Same pattern as cross-encoder auto-enable (Tier 2 #6): multi-hop and
temporal queries benefit most from alternative phrasings, so the
hybrid scorer fires ``expand_queries`` even when the operator hasn't
explicitly flipped ``query_expansion.enabled: true`` in the config.
Opt-out via ``query_expansion.auto_enable: false``.

Uses the NLP expander (zero network cost) unless an LLM is configured.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mind_mem.hybrid_recall import HybridBackend


def _fake_workspace(tmp_path) -> str:
    for d in ("decisions", "tasks", "entities", "intelligence", "memory", ".mind-mem-index"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        "[D-20260420-001]\ntype: decision\nStatement: Use PostgreSQL.\n---\n",
        encoding="utf-8",
    )
    return str(tmp_path)


def _build_backend(auto_enable: bool | None = None, enabled: bool = False) -> HybridBackend:
    qe: dict = {"enabled": enabled}
    if auto_enable is not None:
        qe["auto_enable"] = auto_enable
    return HybridBackend(
        {
            "backend": "bm25",
            "rrf_k": 60,
            "bm25_weight": 1.0,
            "vector_weight": 1.0,
            "vector_enabled": False,
            "query_expansion": qe,
        }
    )


@pytest.fixture
def mock_hit():
    return [{"_id": "D-20260420-001", "score": 5.0, "excerpt": "Use PostgreSQL."}]


class TestQueryExpansionAutoEnable:
    def test_multihop_query_auto_enables(self, tmp_path, mock_hit) -> None:
        backend = _build_backend(auto_enable=True, enabled=False)
        expand = MagicMock(return_value=["orig", "paraphrase-1", "paraphrase-2"])
        with patch.object(backend, "_bm25_search", return_value=mock_hit):
            with patch("mind_mem.query_expansion.expand_queries", expand):
                backend.search(
                    "What decision led to the choice that then caused the outage?",
                    _fake_workspace(tmp_path),
                    limit=5,
                )
                expand.assert_called_once()

    def test_temporal_query_auto_enables(self, tmp_path, mock_hit) -> None:
        backend = _build_backend(auto_enable=True, enabled=False)
        expand = MagicMock(return_value=["orig", "paraphrase-1"])
        with patch.object(backend, "_bm25_search", return_value=mock_hit):
            with patch("mind_mem.query_expansion.expand_queries", expand):
                backend.search(
                    "When did we decide to use PostgreSQL yesterday?",
                    _fake_workspace(tmp_path),
                    limit=5,
                )
                expand.assert_called_once()

    def test_single_hop_skips_auto_enable(self, tmp_path, mock_hit) -> None:
        backend = _build_backend(auto_enable=True, enabled=False)
        expand = MagicMock()
        with patch.object(backend, "_bm25_search", return_value=mock_hit):
            with patch("mind_mem.query_expansion.expand_queries", expand):
                backend.search("PostgreSQL", _fake_workspace(tmp_path), limit=5)
                expand.assert_not_called()

    def test_explicit_enable_overrides_query_type(self, tmp_path, mock_hit) -> None:
        backend = _build_backend(auto_enable=False, enabled=True)
        expand = MagicMock(return_value=["PostgreSQL", "Postgres"])
        with patch.object(backend, "_bm25_search", return_value=mock_hit):
            with patch("mind_mem.query_expansion.expand_queries", expand):
                backend.search("PostgreSQL", _fake_workspace(tmp_path), limit=5)
                expand.assert_called_once()

    def test_auto_enable_false_skips_expansion(self, tmp_path, mock_hit) -> None:
        backend = _build_backend(auto_enable=False, enabled=False)
        expand = MagicMock()
        with patch.object(backend, "_bm25_search", return_value=mock_hit):
            with patch("mind_mem.query_expansion.expand_queries", expand):
                backend.search(
                    "What decision led to the choice that then caused the outage?",
                    _fake_workspace(tmp_path),
                    limit=5,
                )
                expand.assert_not_called()
