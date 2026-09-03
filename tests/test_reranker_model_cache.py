"""Reranker weights load once per process, not once per query.

``benchmarks/LONGMEMEVAL_FINDINGS_2026-05-19.md`` names this as one of the two
blockers on a full 500-question run: "Cross-encoder reranker has **no model
singleton** ... reloads an 80 MB model every search; a full run drowns in
reloads."

The single-model cross-encoder had since grown a ``(model, device)``-keyed
cache. The **ensemble** path had not: ``create_ensemble`` runs per query and
``_build_bge`` called ``CrossEncoder(...)`` unconditionally, so a configured
``bge`` member reloaded a ~2.2 GB checkpoint on every search.

These tests read a load COUNTER rather than arguing from the code, because a
cache is a performance claim and a performance claim needs a number. They also
check that the cache is keyed rather than single-slot: a one-slot cache makes
the count look perfect while silently serving the wrong model.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Any

import pytest

import mind_mem.cross_encoder_reranker as ce_mod
import mind_mem.rerank_ensemble as ens_mod

#: How many queries a "full run" stands in for here. Any number > 1 exposes a
#: per-query reload; 25 keeps the test fast while making the before/after
#: difference unmistakable (25 loads versus 1).
_QUERIES = 25


class _CountingCrossEncoder:
    """Stand-in for ``sentence_transformers.CrossEncoder`` that counts loads."""

    #: Every (model, device) pair ever constructed, in order.
    constructed: list[tuple[str, str]] = []

    def __init__(self, model: str, device: str = "cpu", **_: Any) -> None:
        self.model = model
        self.device = device
        type(self).constructed.append((model, device))

    def predict(self, pairs: list[tuple[str, str]], **_: Any) -> list[float]:
        return [1.0 / (i + 1) for i in range(len(pairs))]


@pytest.fixture
def counting_st(monkeypatch: pytest.MonkeyPatch) -> type[_CountingCrossEncoder]:
    """Install a counting CrossEncoder and reset every cache and counter."""
    fake = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = _CountingCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    _CountingCrossEncoder.constructed = []
    monkeypatch.setattr(ce_mod, "_CE_MODELS", {}, raising=True)
    monkeypatch.setattr(ce_mod, "_CE_LOAD_LOCK", threading.Lock(), raising=True)
    monkeypatch.setattr(ce_mod, "_CE_AVAILABLE", True, raising=True)
    monkeypatch.setattr(ce_mod, "_CE_LOADS", 0, raising=True)
    monkeypatch.setattr(ens_mod, "_BGE_MODELS", {}, raising=True)
    monkeypatch.setattr(ens_mod, "_BGE_LOAD_LOCK", threading.Lock(), raising=True)
    monkeypatch.setattr(ens_mod, "_BGE_LOADS", 0, raising=True)
    return _CountingCrossEncoder


def test_the_counter_can_count(counting_st: type[_CountingCrossEncoder]) -> None:
    """Positive control: the stub really is what gets constructed.

    Without this, every "loads == 1" assertion below would pass just as
    happily against a path that constructs nothing at all.
    """
    ce_mod.CrossEncoderReranker()
    assert ce_mod.ce_load_count() == 1
    assert counting_st.constructed == [("cross-encoder/ms-marco-MiniLM-L-6-v2", "cpu")]


def test_cross_encoder_loads_once_across_many_queries(counting_st: type[_CountingCrossEncoder]) -> None:
    """The single-model path: one load, whatever the query count."""
    for _ in range(_QUERIES):
        ce_mod.CrossEncoderReranker()
    # Count real constructions FIRST. The internal counter lives inside the
    # cache it is measuring, so a regression that bypasses the cache also
    # bypasses the counter and reports a flattering zero -- the honest
    # measure is what the model class saw.
    assert len(counting_st.constructed) == 1, f"reloaded per query: {len(counting_st.constructed)} loads for {_QUERIES} queries"
    assert ce_mod.ce_load_count() == 1


def test_cross_encoder_cache_is_keyed_not_single_slot(counting_st: type[_CountingCrossEncoder]) -> None:
    """A different model or device must really load, not reuse the first."""
    a = ce_mod.CrossEncoderReranker(model="model-a", device="cpu")
    b = ce_mod.CrossEncoderReranker(model="model-b", device="cpu")
    c = ce_mod.CrossEncoderReranker(model="model-a", device="cuda")
    again = ce_mod.CrossEncoderReranker(model="model-a", device="cpu")

    assert ce_mod.ce_load_count() == 3, counting_st.constructed
    assert a._model is again._model
    assert a._model is not b._model
    assert a._model is not c._model
    assert (a.model_name, a.device) == ("model-a", "cpu")
    assert (c.model_name, c.device) == ("model-a", "cuda")


def test_bge_member_loads_once_across_many_queries(counting_st: type[_CountingCrossEncoder]) -> None:
    """The regression this file was written for: BGE reloaded every query."""
    for _ in range(_QUERIES):
        assert ens_mod._build_bge() is not None
    # Real constructions, not the internal counter: a loader that skips the
    # cache also skips the counter, so the counter alone would report 0
    # loads for 25 reloads -- better-looking than the truth.
    assert len(counting_st.constructed) == 1, f"BGE reloaded per query: {len(counting_st.constructed)} loads for {_QUERIES} queries"
    assert counting_st.constructed == [("BAAI/bge-reranker-v2-m3", "cpu")]
    assert ens_mod.bge_load_count() == 1


def test_bge_cache_is_keyed_by_device(counting_st: type[_CountingCrossEncoder], monkeypatch: pytest.MonkeyPatch) -> None:
    """Switching device must load the weights for that device."""
    ens_mod._build_bge()
    monkeypatch.setenv("MIND_MEM_RERANKER_DEVICE", "cuda")
    ens_mod._build_bge()
    ens_mod._build_bge()
    assert counting_st.constructed == [("BAAI/bge-reranker-v2-m3", "cpu"), ("BAAI/bge-reranker-v2-m3", "cuda")]
    assert ens_mod.bge_load_count() == 2, counting_st.constructed


def test_per_query_ensemble_construction_loads_nothing_after_the_first(
    counting_st: type[_CountingCrossEncoder],
) -> None:
    """The path that actually runs per query.

    ``hybrid_recall._maybe_cross_encoder_rerank`` calls ``create_ensemble``
    on every search. That is the call whose cost the FINDINGS measured, so
    it is the call this test drives -- not the builders underneath it.
    """
    config = {"retrieval": {"reranker_ensemble": {"enabled": True, "rerankers": ["cross_encoder", "bge"]}}}
    for _ in range(_QUERIES):
        ensemble = ens_mod.create_ensemble(config)
        assert ensemble is not None

    loaded = len(counting_st.constructed)
    assert loaded == 2, f"{_QUERIES} queries loaded {loaded} models; expected one per distinct member"
    assert ce_mod.ce_load_count() + ens_mod.bge_load_count() == 2


def test_concurrent_first_queries_load_once(counting_st: type[_CountingCrossEncoder]) -> None:
    """Threads racing the first query must not each load the weights."""
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=30)
            ens_mod._build_bge()
            ce_mod.CrossEncoderReranker()
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    assert len(counting_st.constructed) == 2, counting_st.constructed
    assert ens_mod.bge_load_count() == 1, counting_st.constructed
    assert ce_mod.ce_load_count() == 1, counting_st.constructed
