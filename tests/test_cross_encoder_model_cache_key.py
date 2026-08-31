# Copyright 2026 STARGA, Inc.
"""Regression tests: the reranker must load the model it was asked for.

The loaded cross-encoder lived in a single module-global slot filled by
the first caller, so ``model`` and ``device`` were read only on that first
construction. Every later ``CrossEncoderReranker(model=..., device=...)``
in the process silently reused whatever was already loaded and discarded
the request — with nothing recorded to say which model actually scored.
Both arms of an A/B run then measure the same model while the ``ce_score``
field and the surrounding label attribute the numbers to two.
"""

from __future__ import annotations

from typing import Any

import pytest

from mind_mem import cross_encoder_reranker as ce_mod
from mind_mem.cross_encoder_reranker import CrossEncoderReranker

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_OTHER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class _FakeCrossEncoder:
    """Stands in for the real encoder; records what it was constructed with."""

    loads: list[tuple[str, str]] = []

    def __init__(self, model: str, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        type(self).loads.append((model, device))

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        # Score is a function of the loaded model, so a swapped model is visible.
        return [float(len(self.model) + len(text)) for _query, text in pairs]


@pytest.fixture
def fake_encoder(monkeypatch: pytest.MonkeyPatch):
    """Isolate the module cache and stub the encoder so nothing is downloaded."""
    monkeypatch.setattr(ce_mod, "_CE_MODELS", {})
    monkeypatch.setattr(ce_mod, "_CE_AVAILABLE", True)
    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeCrossEncoder)
    _FakeCrossEncoder.loads = []
    return _FakeCrossEncoder


class TestModelIdentityIsHonoured:
    def test_second_model_is_actually_loaded(self, fake_encoder: Any) -> None:
        first = CrossEncoderReranker()
        second = CrossEncoderReranker(model=_OTHER_MODEL)

        assert fake_encoder.loads == [(_DEFAULT_MODEL, "cpu"), (_OTHER_MODEL, "cpu")]
        assert first._model is not second._model
        assert second._model.model == _OTHER_MODEL

    def test_second_device_is_actually_loaded(self, fake_encoder: Any) -> None:
        CrossEncoderReranker()
        on_gpu = CrossEncoderReranker(device="cuda")

        assert fake_encoder.loads == [(_DEFAULT_MODEL, "cpu"), (_DEFAULT_MODEL, "cuda")]
        assert on_gpu._model.device == "cuda"

    def test_requested_pair_is_recorded_on_the_instance(self, fake_encoder: Any) -> None:
        """Scores must be attributable to the model that produced them."""
        reranker = CrossEncoderReranker(model=_OTHER_MODEL, device="cuda")
        assert reranker.model_name == _OTHER_MODEL
        assert reranker.device == "cuda"

    def test_scores_differ_between_models(self, fake_encoder: Any) -> None:
        """The A/B failure mode itself: two arms must not return one answer."""
        candidates = [{"content": "alpha", "score": 0.5}, {"content": "beta", "score": 0.5}]
        a = CrossEncoderReranker().rerank("q", [dict(c) for c in candidates])
        b = CrossEncoderReranker(model=_OTHER_MODEL).rerank("q", [dict(c) for c in candidates])

        assert [hit["ce_score"] for hit in a] != [hit["ce_score"] for hit in b]


class TestCacheStillCaches:
    def test_same_pair_loads_once(self, fake_encoder: Any) -> None:
        first = CrossEncoderReranker()
        second = CrossEncoderReranker()

        assert fake_encoder.loads == [(_DEFAULT_MODEL, "cpu")]
        assert first._model is second._model

    def test_repeat_of_a_secondary_pair_also_reuses(self, fake_encoder: Any) -> None:
        CrossEncoderReranker()
        one = CrossEncoderReranker(model=_OTHER_MODEL)
        two = CrossEncoderReranker(model=_OTHER_MODEL)

        assert fake_encoder.loads.count((_OTHER_MODEL, "cpu")) == 1
        assert one._model is two._model
