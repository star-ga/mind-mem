"""The last-resort embedding fallback must not perform an unbounded download.

THE DEFECT, measured on Windows CI (job 102579093360, PR 573):

    tests/test_bench_hybrid_dispatch.py::test_a_servable_dense_leg_moves_the_ranking
      -> eval_adapters._build_vector_index
      -> recall_vector.index -> _embed_for_provider -> embed -> model
      -> sentence_transformers -> huggingface_hub.xet_get -> HUNG
      -> +++ Timeout +++  (pytest-timeout killed the run; exit 1)

ollama refused the connection (WinError 10061) and fastembed was not
installed, so the chain fell through to its last resort, which started a
multi-hundred-megabyte HuggingFace download inside a test.

Two separate mistakes made that reachable:

1. The guard meant to prevent it (``if "/" not in self.model_name``) lives in
   an ``except OSError`` handler -- DOWNSTREAM of the very network call it
   exists to avoid. A hang never raises, so the guard never runs.

2. Its premise is false for the default model. ``all-MiniLM-L6-v2`` has no
   "/", but sentence-transformers resolves bare names against its own org, so
   the load does not fail -- it succeeds slowly, by downloading.

The fix is not a better exception filter. A LAST RESORT, reached only because
every configured provider declined, must be satisfiable from what is already
on disk: cache-only, no network, fail fast and name the real cause.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from mind_mem.recall_vector import EmbeddingProvidersExhausted, VectorBackend


class _Spy:
    """Stands in for SentenceTransformer and records how it was constructed."""

    calls: list[dict[str, Any]] = []

    def __init__(self, name: str, **kwargs: Any) -> None:
        type(self).calls.append({"name": name, **kwargs})
        if not kwargs.get("local_files_only"):
            raise AssertionError(
                "the last resort constructed a model WITHOUT local_files_only -- "
                "that is the unbounded network download this test exists to forbid"
            )

    def encode(self, texts, **_: Any):
        import numpy as np

        return np.zeros((len(texts), 4), dtype="float32")


@pytest.fixture
def spy_sentence_transformers(monkeypatch: pytest.MonkeyPatch):
    _Spy.calls = []
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _Spy  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return _Spy


def _backend_with_every_provider_declining(monkeypatch: pytest.MonkeyPatch) -> VectorBackend:
    backend = VectorBackend({"provider": "ollama", "model": "all-MiniLM-L6-v2"})

    def _decline(*_: Any, **__: Any):
        raise OSError("provider unavailable in this test")

    for name in ("embed_ollama", "embed_llama_cpp", "embed_fastembed"):
        monkeypatch.setattr(backend, name, _decline, raising=False)
    return backend


def test_the_last_resort_never_downloads(spy_sentence_transformers, monkeypatch):
    """The load must be cache-only, or refused -- never an open network call."""
    backend = _backend_with_every_provider_declining(monkeypatch)

    try:
        backend._embed_for_provider(["hello"])
    except EmbeddingProvidersExhausted:
        pass  # refusing outright is also correct

    # POSITIVE CONTROL: prove we actually REACHED the last resort. Without this
    # the test passes when the chain never got there and nothing was proven.
    assert spy_sentence_transformers.calls, "the last resort was never reached, so this test proved nothing about it"
    assert spy_sentence_transformers.calls[-1].get("local_files_only") is True


def test_an_uncached_model_names_the_real_cause(monkeypatch):
    """Not-cached must say the providers declined, not blame HuggingFace."""
    mod = types.ModuleType("sentence_transformers")

    def _not_cached(name: str, **kwargs: Any):
        assert kwargs.get("local_files_only") is True
        raise OSError(f"{name} is not cached and local_files_only=True")

    mod.SentenceTransformer = _not_cached  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)

    backend = _backend_with_every_provider_declining(monkeypatch)
    with pytest.raises(EmbeddingProvidersExhausted) as caught:
        backend._embed_for_provider(["hello"])

    msg = str(caught.value)
    assert "declined" in msg, "the message must name the real cause: every provider declined"
    assert "cache" in msg.lower() or "cached" in msg.lower(), "it must say the model was not on disk"


def test_a_cached_model_still_serves_the_fallback(spy_sentence_transformers, monkeypatch):
    """The legitimate case must keep working: cached model, no network, vectors returned."""
    backend = _backend_with_every_provider_declining(monkeypatch)
    out = backend._embed_for_provider(["hello", "world"])
    assert len(out) == 2 and len(out[0]) == 4
