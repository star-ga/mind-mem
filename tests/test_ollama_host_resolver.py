"""Tests for the shared ollama base-URL resolver (v4.3.1).

Covers :func:`mind_mem.ollama_host.ollama_base_url` — the single source of
truth for every ollama endpoint in mind-mem — plus the call-site threading:
embed (``recall_vector``, ``mm_cli``), extraction (``llm_extractor``),
compression (``compressors``) and rerank (``_recall_reranking`` /
``_recall_core``) must all resolve through it so a fleet node can point at a
central ollama server via ``OLLAMA_HOST`` or a per-section ``ollama_url``
config key.

Documented precedence (asserted below):
    1. explicit config key (``recall.ollama_url`` / ``extraction.ollama_url``)
    2. ``OLLAMA_HOST`` env var (``host:port`` or full ``http[s]://`` URL)
    3. ``http://localhost:11434`` — byte-identical legacy default
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest

from mind_mem.ollama_host import DEFAULT_OLLAMA_URL, ollama_base_url

# ---------------------------------------------------------------------------
# Resolver unit tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_ollama_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from an unset OLLAMA_HOST (the legacy baseline)."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


class TestOllamaBaseUrl:
    def test_default_is_localhost_11434(self) -> None:
        # Backward compatibility: env unset + no config must be
        # byte-identical to the pre-4.3.1 hardcoded URL.
        assert ollama_base_url() == "http://localhost:11434"
        assert DEFAULT_OLLAMA_URL == "http://localhost:11434"

    def test_env_host_port_is_normalized_to_http_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "192.0.2.10:11434")
        assert ollama_base_url() == "http://192.0.2.10:11434"

    def test_env_full_url_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://x:11434")
        assert ollama_base_url() == "http://x:11434"

    def test_env_https_url_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "https://ollama.internal:11434")
        assert ollama_base_url() == "https://ollama.internal:11434"

    def test_trailing_slash_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://192.0.2.10:11434/")
        assert ollama_base_url() == "http://192.0.2.10:11434"

    def test_empty_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "")
        assert ollama_base_url() == DEFAULT_OLLAMA_URL

    def test_whitespace_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "   ")
        assert ollama_base_url() == DEFAULT_OLLAMA_URL

    def test_config_key_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Precedence: explicit workspace config > ambient env var.
        monkeypatch.setenv("OLLAMA_HOST", "http://from-env:11434")
        cfg = {"ollama_url": "http://from-config:11434"}
        assert ollama_base_url(cfg) == "http://from-config:11434"

    def test_config_host_port_normalized(self) -> None:
        assert ollama_base_url({"ollama_url": "192.0.2.10:11434"}) == "http://192.0.2.10:11434"

    def test_empty_config_value_falls_through_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://from-env:11434")
        assert ollama_base_url({"ollama_url": ""}) == "http://from-env:11434"

    def test_missing_config_key_falls_through_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://from-env:11434")
        assert ollama_base_url({"other": 1}) == "http://from-env:11434"

    def test_non_string_config_value_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://from-env:11434")
        assert ollama_base_url({"ollama_url": 11434}) == "http://from-env:11434"

    def test_custom_config_key(self) -> None:
        cfg = {"embed_ollama_url": "http://embed-host:11434"}
        assert ollama_base_url(cfg, config_key="embed_ollama_url") == "http://embed-host:11434"

    def test_bad_scheme_in_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "file:///etc/passwd")
        with pytest.raises(ValueError, match="OLLAMA_HOST"):
            ollama_base_url()

    def test_bad_scheme_in_config_raises(self) -> None:
        with pytest.raises(ValueError, match="ollama_url"):
            ollama_base_url({"ollama_url": "ftp://host:21"})

    def test_hostless_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://")
        with pytest.raises(ValueError, match="host"):
            ollama_base_url()


# ---------------------------------------------------------------------------
# Call-site threading tests — monkeypatch env + urlopen, assert the URL
# ---------------------------------------------------------------------------

_CENTRAL = "http://192.0.2.10:11434"


class _FakeResponse:
    """Minimal urlopen context-manager response."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._raw = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _capture_urlopen(captured: dict[str, Any], payload: dict[str, Any]):
    def _fake(req: urllib.request.Request, timeout: float | None = None, **kw: Any) -> _FakeResponse:
        captured["url"] = req.full_url
        return _FakeResponse(payload)

    return _fake


class TestCallSitesUseResolver:
    def test_recall_vector_embed_ollama_honors_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.recall_vector import VectorBackend

        monkeypatch.setenv("OLLAMA_HOST", "192.0.2.10:11434")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(urllib.request, "urlopen", _capture_urlopen(captured, {"embeddings": [[0.1]]}))
        out = VectorBackend({"provider": "ollama"}).embed_ollama(["hello"])
        assert captured["url"] == f"{_CENTRAL}/api/embed"
        assert out == [[0.1]]

    def test_recall_vector_embed_ollama_honors_config_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.recall_vector import VectorBackend

        monkeypatch.setenv("OLLAMA_HOST", "http://from-env:11434")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(urllib.request, "urlopen", _capture_urlopen(captured, {"embeddings": [[0.1]]}))
        VectorBackend({"provider": "ollama", "ollama_url": _CENTRAL}).embed_ollama(["hello"])
        assert captured["url"] == f"{_CENTRAL}/api/embed"

    def test_llm_extractor_generate_honors_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem import llm_extractor

        monkeypatch.setenv("OLLAMA_HOST", "192.0.2.10:11434")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(urllib.request, "urlopen", _capture_urlopen(captured, {"response": "ok"}))
        assert llm_extractor._query_ollama("prompt", "mind-mem:4b") == "ok"
        assert captured["url"] == f"{_CENTRAL}/api/generate"

    def test_llm_extractor_tags_probe_honors_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem import llm_extractor

        monkeypatch.setenv("OLLAMA_HOST", "192.0.2.10:11434")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(urllib.request, "urlopen", _capture_urlopen(captured, {}))
        assert llm_extractor._ollama_available() is True
        assert captured["url"] == f"{_CENTRAL}/api/tags"

    def test_llm_extractor_explicit_base_url_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem import llm_extractor

        monkeypatch.setenv("OLLAMA_HOST", "http://from-env:11434")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(urllib.request, "urlopen", _capture_urlopen(captured, {"response": "ok"}))
        llm_extractor._query_ollama("p", "m", base_url=_CENTRAL)
        assert captured["url"] == f"{_CENTRAL}/api/generate"

    def test_mm_cli_embed_honors_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.mm_cli import _embed_via_ollama

        monkeypatch.setenv("OLLAMA_HOST", "192.0.2.10:11434")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(urllib.request, "urlopen", _capture_urlopen(captured, {"embeddings": [[0.2]]}))
        assert _embed_via_ollama(["hello"]) == [[0.2]]
        assert captured["url"] == f"{_CENTRAL}/api/embed"

    def test_compressor_default_host_honors_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.compressors import OllamaCompressor

        monkeypatch.setenv("OLLAMA_HOST", "192.0.2.10:11434")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(urllib.request, "urlopen", _capture_urlopen(captured, {"response": "tight"}))
        compressor = OllamaCompressor(model="m")
        assert compressor("current", [{"_id": "D-001", "body": "x"}]) == "tight"
        assert captured["url"] == f"{_CENTRAL}/api/generate"

    def test_compressor_explicit_host_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.compressors import OllamaCompressor

        monkeypatch.setenv("OLLAMA_HOST", "http://from-env:11434")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(urllib.request, "urlopen", _capture_urlopen(captured, {"response": "ok"}))
        OllamaCompressor(model="m", host=_CENTRAL)("x", [{"_id": "D-001", "body": "y"}])
        assert captured["url"] == f"{_CENTRAL}/api/generate"

    def test_llm_rerank_default_url_honors_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem._recall_reranking import llm_rerank

        monkeypatch.setenv("OLLAMA_HOST", "192.0.2.10:11434")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(urllib.request, "urlopen", _capture_urlopen(captured, {"response": "[0.9]"}))
        hits = [{"_id": "B-001", "excerpt": "text", "score": 0.5}]
        llm_rerank("query", hits)
        assert captured["url"] == f"{_CENTRAL}/api/generate"

    def test_llm_rerank_explicit_url_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem._recall_reranking import llm_rerank

        monkeypatch.setenv("OLLAMA_HOST", "http://from-env:11434")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(urllib.request, "urlopen", _capture_urlopen(captured, {"response": "[0.9]"}))
        hits = [{"_id": "B-001", "excerpt": "text", "score": 0.5}]
        llm_rerank("query", hits, url=f"{_CENTRAL}/api/generate")
        assert captured["url"] == f"{_CENTRAL}/api/generate"
