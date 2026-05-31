"""v3.3.0 Tier 4 #9 — reranker ensemble via Borda count.

``EnsembleReranker`` fuses N independent rerankers' orderings with
Borda-count voting. Tests verify the voting math, graceful fallback
on member failures, and factory config resolution.
"""

from __future__ import annotations

import pytest

from mind_mem.rerank_ensemble import EnsembleReranker, create_ensemble


def _candidates(ids: list[str]) -> list[dict]:
    return [{"_id": cid, "content": f"content-{cid}", "score": 1.0} for cid in ids]


class _StubReranker:
    """Deterministic reranker that returns candidates in a fixed order."""

    def __init__(self, order: list[str]) -> None:
        self._order = order

    def rerank(self, query: str, candidates: list[dict], *, top_k: int = 10, blend_weight: float = 0.6) -> list[dict]:
        by_id = {str(c.get("_id")): c for c in candidates}
        return [by_id[cid] for cid in self._order if cid in by_id][:top_k]


class _FailingReranker:
    def rerank(self, query, candidates, *, top_k=10, blend_weight=0.6):
        raise RuntimeError("boom")


class TestEnsembleVoting:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            EnsembleReranker([])

    def test_single_reranker_passthrough(self) -> None:
        r = _StubReranker(["a", "b", "c"])
        ens = EnsembleReranker([r])
        out = ens.rerank("q", _candidates(["c", "b", "a"]))
        ids = [c["_id"] for c in out]
        # Degenerates to the member's order.
        assert ids == ["a", "b", "c"]

    def test_borda_count_fuses_rankings(self) -> None:
        """Two rerankers: A ranks [x, y, z]; B ranks [y, x, z]. Borda picks x (top-scoring)."""
        r1 = _StubReranker(["x", "y", "z"])
        r2 = _StubReranker(["y", "x", "z"])
        ens = EnsembleReranker([r1, r2])
        out = ens.rerank("q", _candidates(["x", "y", "z"]))
        ids = [c["_id"] for c in out]
        # Scores: x = (3-0) + (3-1) = 5; y = (3-1) + (3-0) = 5; z = 1+1 = 2.
        # Tie between x and y — stable order from first reranker keeps x first.
        assert ids[0] in ("x", "y")
        assert ids[-1] == "z"

    def test_borda_count_majority_wins(self) -> None:
        """Majority agreement wins against minority dissent."""
        r1 = _StubReranker(["x", "y", "z"])
        r2 = _StubReranker(["x", "y", "z"])
        r3 = _StubReranker(["z", "y", "x"])  # contrarian
        ens = EnsembleReranker([r1, r2, r3])
        out = ens.rerank("q", _candidates(["x", "y", "z"]))
        ids = [c["_id"] for c in out]
        assert ids[0] == "x"
        assert ids[-1] == "z"

    def test_borda_score_annotated(self) -> None:
        r1 = _StubReranker(["a", "b"])
        r2 = _StubReranker(["a", "b"])
        ens = EnsembleReranker([r1, r2])
        out = ens.rerank("q", _candidates(["a", "b"]))
        assert "_ensemble_borda" in out[0]
        assert out[0]["_ensemble_borda"] > out[1]["_ensemble_borda"]


class TestEnsembleFailureHandling:
    def test_one_member_fails_ensemble_continues(self) -> None:
        r1 = _StubReranker(["a", "b", "c"])
        r_bad = _FailingReranker()
        r2 = _StubReranker(["a", "b", "c"])
        ens = EnsembleReranker([r1, r_bad, r2])
        out = ens.rerank("q", _candidates(["a", "b", "c"]))
        # Ensemble ignores the failure and uses the two good rerankers.
        assert [c["_id"] for c in out] == ["a", "b", "c"]

    def test_all_members_fail_returns_original(self) -> None:
        ens = EnsembleReranker([_FailingReranker(), _FailingReranker()])
        cands = _candidates(["x", "y"])
        out = ens.rerank("q", cands)
        assert out == cands

    def test_empty_candidates_unchanged(self) -> None:
        ens = EnsembleReranker([_StubReranker([])])
        assert ens.rerank("q", []) == []


class TestCreateEnsemble:
    def test_disabled_returns_none(self) -> None:
        assert create_ensemble({"retrieval": {"reranker_ensemble": {"enabled": False}}}) is None
        assert create_ensemble({}) is None

    def test_unknown_reranker_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown reranker name is logged and skipped, not fatal."""
        # Force CE build to succeed so at least one member is available.
        import mind_mem.rerank_ensemble as mod

        monkeypatch.setattr(mod, "_build_cross_encoder", lambda: _StubReranker(["a"]))
        cfg = {
            "retrieval": {
                "reranker_ensemble": {
                    "enabled": True,
                    "rerankers": ["cross_encoder", "not_a_real_reranker"],
                }
            }
        }
        ens = create_ensemble(cfg)
        assert ens is not None
        # The one valid member is wired.
        assert len(ens._rerankers) == 1

    def test_no_members_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mind_mem.rerank_ensemble as mod

        monkeypatch.setattr(mod, "_build_cross_encoder", lambda: None)
        cfg = {
            "retrieval": {
                "reranker_ensemble": {
                    "enabled": True,
                    "rerankers": ["cross_encoder"],
                }
            }
        }
        assert create_ensemble(cfg) is None

    def test_multi_reranker_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mind_mem.rerank_ensemble as mod

        monkeypatch.setattr(mod, "_build_cross_encoder", lambda: _StubReranker(["a"]))
        monkeypatch.setattr(mod, "_build_bge", lambda: _StubReranker(["b"]))

        # LLM builder sees a fake llm_cfg; build returns a stub.
        def fake_llm(cfg):
            return _StubReranker(["c"])

        monkeypatch.setattr(mod, "_build_llm", fake_llm)

        cfg = {
            "retrieval": {
                "reranker_ensemble": {
                    "enabled": True,
                    "rerankers": ["cross_encoder", "bge", "llm"],
                    "llm": {"base_url": "http://127.0.0.1:8766/v1/chat/completions"},
                }
            }
        }
        ens = create_ensemble(cfg)
        assert ens is not None
        assert len(ens._rerankers) == 3
