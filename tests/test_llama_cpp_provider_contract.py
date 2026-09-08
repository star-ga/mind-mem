# Copyright 2026 STARGA, Inc.
"""llama_cpp indexing goes through the provider contract, like everything else.

`index()` called `embed_llama_cpp` directly whenever the provider was
llama_cpp, bypassing `_embed_for_provider` -- and with it the per-provider
circuit breaker, the logged fall-through and the rest of the chain. Every
other embedding site (`search`, `rebuild_index`, the sqlite-vec builder, the
kind backfill) went through the contract; this one did not.

The bypass existed for a reason, which is why removing it alone would have
broken llama_cpp indexing outright: the chain decided "is this llama_cpp?"
from `config["onnx_backend"]` while `index()` decided it from `self.provider`.
Two names for one fact, so a workspace configured by provider never reached
llama_cpp inside the chain. The chain now accepts either signal, which is what
makes the bypass unnecessary rather than merely absent.
"""

from __future__ import annotations

import inspect
import threading

import pytest

from mind_mem import recall_vector


class _Backend:
    """A minimal stand-in exposing only what the contract touches."""

    def __init__(self, *, provider: str, onnx_backend=None, llama_ok: bool = True):
        self.provider = provider
        self.config = {} if onnx_backend is None else {"onnx_backend": onnx_backend}
        self.calls: list[str] = []
        self._llama_ok = llama_ok
        # The real attribute names the contract's breaker helper uses.
        self._provider_breakers: dict = {}
        self._provider_breakers_lock = threading.Lock()

    # -- the chain's own helpers, reused verbatim from the real class --
    _provider_breaker = recall_vector.VectorBackend._provider_breaker
    _embed_for_provider = recall_vector.VectorBackend._embed_for_provider

    def embed_ollama(self, texts):
        self.calls.append("ollama")
        raise RuntimeError("ollama down")

    def embed_llama_cpp(self, texts):
        self.calls.append("llama_cpp")
        if not self._llama_ok:
            raise RuntimeError("llama_cpp failed")
        return [[0.5] * 4 for _ in texts]

    def embed_fastembed(self, texts):
        self.calls.append("fastembed")
        return [[0.25] * 4 for _ in texts]

    def embed(self, texts):
        self.calls.append("sentence_transformers")
        return [[0.1] * 4 for _ in texts]


def test_the_chain_reaches_llama_cpp_from_the_PROVIDER_signal() -> None:
    """The half that was missing: provider=llama_cpp with no onnx_backend key.

    Before this, such a workspace fell past llama_cpp entirely inside the
    chain -- which is precisely why index() had to call it directly.
    """
    b = _Backend(provider="llama_cpp")
    out = b._embed_for_provider(["a", "b"])
    assert "llama_cpp" in b.calls, f"the chain never tried llama_cpp: {b.calls}"
    assert out == [[0.5] * 4, [0.5] * 4]


def test_the_chain_still_reaches_llama_cpp_from_the_CONFIG_signal() -> None:
    """The half that already worked must keep working."""
    b = _Backend(provider="ollama", onnx_backend="llama_cpp")
    b._embed_for_provider(["a"])
    assert "llama_cpp" in b.calls, f"the config signal regressed: {b.calls}"


def test_a_failing_llama_cpp_falls_through_instead_of_propagating() -> None:
    """The guarantee the bypass gave up: a dead provider degrades, not crashes.

    Calling embed_llama_cpp directly meant one failure propagated out of
    index() and the index was simply not built. Through the contract the
    failure is logged and the next provider answers.
    """
    b = _Backend(provider="llama_cpp", llama_ok=False)
    out = b._embed_for_provider(["a"])
    assert b.calls.count("llama_cpp") == 1
    assert "fastembed" in b.calls, f"no fall-through after llama_cpp failed: {b.calls}"
    assert out == [[0.25] * 4]


def test_index_no_longer_bypasses_the_contract() -> None:
    """Structural: no direct embed_llama_cpp call anywhere in index().

    Behavioural coverage of index() end to end needs a real workspace and a
    real provider; this asserts the specific bypass is gone, which is the
    thing that regressed once and would regress the same way again.
    """
    src = inspect.getsource(recall_vector.VectorBackend.index)
    body = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    offending = [ln.strip() for ln in body if "embed_llama_cpp" in ln]
    assert not offending, f"index() calls the provider directly again: {offending}"
    assert any("_embed_for_provider" in ln for ln in body), "index() no longer routes through the contract"


@pytest.mark.parametrize("site", ["search", "rebuild_index", "index"])
def test_every_embedding_site_routes_through_the_contract(site) -> None:
    """One contract, all callers -- so a future site cannot quietly opt out."""
    fn = getattr(recall_vector.VectorBackend, site, None)
    if fn is None:
        pytest.skip(f"{site} is not a method on this class")
    body = [ln for ln in inspect.getsource(fn).splitlines() if not ln.lstrip().startswith("#")]
    direct = [ln.strip() for ln in body if "self.embed_" in ln and "_embed_for_provider" not in ln]
    assert not direct, f"{site} calls a provider directly: {direct}"
