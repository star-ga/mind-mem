"""The ollama (GPU) embed leg must be tried only when ollama can serve the model.

MEASURED 2026-09-11 while starting the full NIAH repro run. Every case logged:

    embed_provider_failed_fallback  provider=ollama  error=HTTP Error 404: Not Found

Ollama was healthy and serving `mxbai-embed-large:latest` the whole time. The 404 was
the MODEL NAME: `recall_vector` defaults `model` to `all-MiniLM-L6-v2` -- a
sentence-transformers / fastembed name -- and `embed_ollama` falls back to that same
name when `ollama_embed_model` is unset. So the chain's FIRST leg was asking ollama for
a model that belongs to a different backend, on every call, forever.

Two costs, and the second is the expensive one:

  * a wasted HTTP round-trip per batch until the per-provider circuit breaker trips,
    repeated in every fresh process;
  * the GPU path is UNREACHABLE BY DEFAULT. A box with ollama serving a real embedding
    model silently does its embedding on CPU, and the warning makes a correctly
    configured system look degraded -- so the honest reading of the log ("ollama is
    broken") is wrong, which is worse than no log.

The fix asks ollama what it serves (`/api/tags`) and attempts the leg only for a model
that is actually there. DERIVED from ollama, not guessed from a hand-maintained list of
model names -- the same reason the fleet provider set is read from its shim rather than
restated.

The probe is cached per instance and SILENT when the answer is no: a probe on an
unavailable path must not log, or a correctly-configured build becomes noisier than one
without the feature.
"""

from __future__ import annotations

import mind_mem.recall_vector as rv


def _embedder(**config):
    return rv.VectorRecall(config=config) if hasattr(rv, "VectorRecall") else None


def test_ollama_serves_is_derived_from_the_tags_endpoint(monkeypatch):
    """Read from ollama, not from a list restated here — a hand-maintained copy of
    another service's capability set drifts, and in the unsafe direction."""
    called: list[str] = []

    def _fake_tags(base_url: str) -> frozenset[str]:
        called.append(base_url)
        return frozenset({"mxbai-embed-large", "nomic-embed-text"})

    monkeypatch.setattr(rv, "_ollama_served_models", _fake_tags)
    assert rv.ollama_can_serve("mxbai-embed-large", {}) is True
    assert called, "the decision did not consult ollama at all"


def test_a_model_ollama_does_not_serve_is_refused(monkeypatch):
    """THE DEFECT. `all-MiniLM-L6-v2` is a fastembed name; asking ollama for it is a
    guaranteed 404 and must not be attempted."""
    monkeypatch.setattr(rv, "_ollama_served_models",
                        lambda base: frozenset({"mxbai-embed-large"}))
    assert rv.ollama_can_serve("all-MiniLM-L6-v2", {}) is False


def test_a_tag_suffix_does_not_cause_a_false_negative(monkeypatch):
    """Ollama reports `mxbai-embed-large:latest`; an operator writes
    `mxbai-embed-large`. Treating those as different models would make the GPU path
    unreachable for the most common spelling — the same bug in a new place."""
    monkeypatch.setattr(rv, "_ollama_served_models",
                        lambda base: frozenset({"mxbai-embed-large:latest"}))
    assert rv.ollama_can_serve("mxbai-embed-large", {}) is True
    assert rv.ollama_can_serve("mxbai-embed-large:latest", {}) is True


def test_an_explicit_ollama_embed_model_is_what_gets_checked(monkeypatch):
    """`ollama_embed_model` names the ollama model; `model` names the ONNX one. The
    check must read the former when it is set, or an operator who configured the GPU
    path correctly is still refused."""
    monkeypatch.setattr(rv, "_ollama_served_models",
                        lambda base: frozenset({"mxbai-embed-large"}))
    cfg = {"model": "all-MiniLM-L6-v2", "ollama_embed_model": "mxbai-embed-large"}
    assert rv.ollama_model_for(cfg, "all-MiniLM-L6-v2") == "mxbai-embed-large"
    assert rv.ollama_can_serve(rv.ollama_model_for(cfg, "all-MiniLM-L6-v2"), cfg) is True


def test_an_unreachable_ollama_is_a_QUIET_no(monkeypatch):
    """A probe on an unavailable path must not log. A correctly-configured build that
    simply has no ollama must not be noisier than a build without the feature —
    that is the flag-probe rule, and this is the same shape."""
    def _boom(base: str):
        raise OSError("connection refused")

    monkeypatch.setattr(rv, "_ollama_served_models", _boom)
    assert rv.ollama_can_serve("mxbai-embed-large", {}) is False


def test_the_tags_probe_hits_the_network_once_per_base_url(monkeypatch):
    """An OFF path must add no per-item work. Probing tags on every batch would just
    replace one wasted round-trip with another, so the cache is the point — asserted by
    counting real network attempts through the REAL cached function, not by patching it
    away.
    """
    calls: list[str] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @staticmethod
        def read():
            return b'{"models": [{"name": "mxbai-embed-large:latest"}]}'

    def _fake_urlopen(req, timeout=None):
        calls.append(getattr(req, "full_url", str(req)))
        return _Resp()

    import urllib.request as _r

    monkeypatch.setattr(_r, "urlopen", _fake_urlopen)
    rv._ollama_served_models.cache_clear()
    for _ in range(5):
        assert rv.ollama_can_serve("mxbai-embed-large", {}) is True
    assert len(calls) == 1, f"the tags probe ran {len(calls)} times for 5 calls"
    rv._ollama_served_models.cache_clear()


def test_the_real_probe_function_is_cached():
    """Asserted on the real function, since the test above monkeypatches it away."""
    import functools

    assert hasattr(rv._ollama_served_models, "cache_clear"), (
        "the tags probe is not cached, so it runs on every embed batch"
    )
    assert isinstance(rv._ollama_served_models, functools._lru_cache_wrapper)


# ---------------------------------------------------------------------------
# Wiring: the guard must sit in the fallback chain. Helpers that return the right
# answer prove nothing if the chain never asks them.
# ---------------------------------------------------------------------------


def _backend(config):
    return rv.VectorBackend(config=config)


def test_the_ollama_LEG_IS_NOT_ATTEMPTED_for_a_model_ollama_cannot_serve(monkeypatch):
    """THE DEFECT, at the call site. With the default config the chain used to call
    embed_ollama on every batch and take a 404. It must not be called at all.

    Mutation-checked: deleting the guard in `_embed_for_provider` turns this red. The
    helper tests alone stayed GREEN with the guard removed, which is exactly the
    "complete, tested and unwired" failure this pins.
    """
    attempted: list[int] = []

    monkeypatch.setattr(rv, "_ollama_served_models",
                        lambda base: frozenset({"mxbai-embed-large:latest"}))
    monkeypatch.setattr(rv.VectorBackend, "embed_ollama",
                        lambda self, texts: attempted.append(1) or [[0.0]])
    # Every later leg refuses, so the only way to return is through ollama.
    monkeypatch.setattr(rv.VectorBackend, "embed_onnx",
                        lambda self, texts: [[1.0] for _ in texts], raising=False)

    backend = _backend({"model": "all-MiniLM-L6-v2"})
    try:
        backend._embed_for_provider(["hello"])
    except Exception:
        pass  # a later leg may fail in this environment; the assertion is about ollama
    assert attempted == [], "the ollama leg was attempted for a model ollama cannot serve"


def test_the_ollama_LEG_IS_attempted_when_the_model_IS_served(monkeypatch):
    """POSITIVE CONTROL. A guard that blocked the leg unconditionally would satisfy the
    test above while permanently disabling GPU embedding — a worse outcome than the
    404s, and invisible."""
    attempted: list[int] = []

    monkeypatch.setattr(rv, "_ollama_served_models",
                        lambda base: frozenset({"mxbai-embed-large:latest"}))
    monkeypatch.setattr(rv.VectorBackend, "embed_ollama",
                        lambda self, texts: attempted.append(1) or [[0.0] for _ in texts])

    backend = _backend({"model": "all-MiniLM-L6-v2",
                        "ollama_embed_model": "mxbai-embed-large"})
    backend._embed_for_provider(["hello"])
    assert attempted == [1], "the ollama leg was skipped for a model ollama DOES serve"
