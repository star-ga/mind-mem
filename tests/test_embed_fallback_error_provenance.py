"""Fallback diagnostics must preserve genuine loader and encoder failures."""

import sys
import types

import pytest

from mind_mem.recall_vector import EmbeddingProvidersExhausted, VectorBackend


def declining_backend(monkeypatch, model="all-MiniLM-L6-v2"):
    backend = VectorBackend({"provider": "ollama", "model": model})

    def decline(*args, **kwargs):
        raise OSError("provider unavailable")

    for name in ("embed_ollama", "embed_fastembed"):
        monkeypatch.setattr(backend, name, decline)
    return backend


@pytest.mark.parametrize("model", ["all-MiniLM-L6-v2", "org/model"])
@pytest.mark.parametrize(
    "failure",
    [
        OSError("corrupt cache: invalid safetensors header"),
        OSError("offline cache contains a corrupt tensor"),
        OSError("permission denied while local_files_only=True"),
        TypeError("encode received invalid batch shape"),
        TypeError("encode() got an unexpected keyword argument 'local_files_only'"),
    ],
)
def test_real_failures_keep_their_identity(monkeypatch, model, failure):
    backend = declining_backend(monkeypatch, model)

    def broken_embed(texts):
        raise failure

    monkeypatch.setattr(backend, "embed", broken_embed)
    with pytest.raises(type(failure)) as caught:
        backend._embed_for_provider(["text"])
    assert caught.value is failure
    assert backend._st_cache_only is False


def test_old_constructor_refuses_without_retrying_online(monkeypatch):
    calls = []

    def old_constructor(name, *, cache_folder=None, device=None):
        calls.append(name)
        raise AssertionError("must not retry without the offline keyword")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=old_constructor),
    )
    backend = declining_backend(monkeypatch)
    with pytest.raises(EmbeddingProvidersExhausted, match="local_files_only"):
        backend._embed_for_provider(["text"])
    assert not calls
    assert backend._st_cache_only is False


@pytest.mark.parametrize(
    "failure",
    [
        OSError("offline cache contains a corrupt tensor"),
        PermissionError("permission denied while local_files_only=True"),
    ],
)
def test_constructor_oserror_is_preserved(monkeypatch, failure):
    calls = []

    def broken_constructor(name, **kwargs):
        calls.append(kwargs)
        raise failure

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=broken_constructor),
    )
    backend = declining_backend(monkeypatch)
    with pytest.raises(type(failure)) as caught:
        backend._embed_for_provider(["text"])
    assert caught.value is failure
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    assert backend._st_cache_only is False


@pytest.mark.parametrize("offline", [False, True])
def test_constructor_internal_typeerror_is_not_version_detection(monkeypatch, offline):
    failure = TypeError("invalid tensor configuration")

    def broken_constructor(name, **kwargs):
        raise failure

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=broken_constructor),
    )
    backend = declining_backend(monkeypatch)
    with pytest.raises(TypeError) as caught:
        if offline:
            backend._embed_for_provider(["text"])
        else:
            backend.model
    assert caught.value is failure
