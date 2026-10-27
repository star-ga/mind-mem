"""The single-model cross-encoder honours ``cross_encoder.model`` / ``cross_encoder.device``.

Both rerank sites (hybrid post-fusion and the scan/sqlite recall core) used to construct
``CrossEncoderReranker()`` with no arguments, so a configured model (for example
``BAAI/bge-reranker-v2-m3``) was silently replaced by the default MiniLM cross-encoder while the
config, the logs and the operator all believed the configured one ran.
"""

from __future__ import annotations

from unittest.mock import patch

import mind_mem.cross_encoder_reranker as cer
from mind_mem.cross_encoder_reranker import ce_model_args
from mind_mem.hybrid_recall import HybridBackend


class _Recorder:
    calls: list[tuple[str, str]] = []

    def __init__(self, model: str = "default-model", device: str = "cpu"):
        _Recorder.calls.append((model, device))

    @staticmethod
    def is_available() -> bool:
        return True

    def rerank(self, query, candidates, top_k=10, blend_weight=0.6):
        return list(candidates)[:top_k]


def _backend(ce: dict) -> HybridBackend:
    return HybridBackend({"backend": "bm25", "vector_enabled": False, "cross_encoder": ce})


def test_ce_model_args_reads_config():
    assert ce_model_args({"model": "BAAI/bge-reranker-v2-m3", "device": "cuda"}) == {
        "model": "BAAI/bge-reranker-v2-m3",
        "device": "cuda",
    }


def test_ce_model_args_defaults_when_unset_or_malformed():
    assert ce_model_args({}) == {}
    assert ce_model_args({"model": "", "device": 3}) == {}
    assert ce_model_args(None) == {}


def test_hybrid_rerank_uses_configured_model():
    _Recorder.calls = []
    hb = _backend({"enabled": True, "model": "BAAI/bge-reranker-v2-m3", "device": "cpu"})
    cands = [{"_id": f"D-20260101-00{i}", "score": float(10 - i), "excerpt": f"x{i}"} for i in range(5)]
    with patch.object(cer, "CrossEncoderReranker", _Recorder):
        out = hb._maybe_cross_encoder_rerank("q", cands, 3, rerank_depth=5, ce_active=True)
    assert _Recorder.calls == [("BAAI/bge-reranker-v2-m3", "cpu")]
    assert len(out) == 3


def test_hybrid_rerank_default_model_when_unconfigured():
    _Recorder.calls = []
    hb = _backend({"enabled": True})
    cands = [{"_id": "D-20260101-001", "score": 1.0, "excerpt": "x"}, {"_id": "D-20260101-002", "score": 0.5, "excerpt": "y"}]
    with patch.object(cer, "CrossEncoderReranker", _Recorder):
        hb._maybe_cross_encoder_rerank("q", cands, 1, rerank_depth=2, ce_active=True)
    assert _Recorder.calls == [("default-model", "cpu")]
