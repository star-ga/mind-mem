"""Regression tests for the MindLLM backend (roadmap v4.0.15).

MindLLM is STARGA's pure-MIND deterministic-inference HTTP server
exposing OpenAI-compatible endpoints (``/v1/chat/completions``,
``/v1/completions``, ``/v1/models``, ``/health``) plus a first-party
RFN classifier endpoint. Bit-identical output across runs; per-token
cryptographic evidence chain.

mind-mem treats MindLLM as just another OpenAI-compatible backend at
the wire level — the value-add is the deterministic + evidence-chain
guarantees, not a new protocol. This test surface just confirms the
``mindllm`` backend type is recognised, routes to MindLLM's default
URL (port 8080), and round-trips through the existing OpenAI-compatible
query path.
"""

from __future__ import annotations

import os

import pytest

from mind_mem import llm_extractor

# ---------------------------------------------------------------------------
# Default URL + env-var override
# ---------------------------------------------------------------------------


def test_mindllm_default_url() -> None:
    """When MIND_MEM_MINDLLM_URL is unset, default to localhost:8080/v1."""
    os.environ.pop("MIND_MEM_MINDLLM_URL", None)
    assert llm_extractor._mindllm_url() == "http://localhost:8080/v1"


def test_mindllm_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """MIND_MEM_MINDLLM_URL overrides the default; trailing slash stripped."""
    monkeypatch.setenv("MIND_MEM_MINDLLM_URL", "http://prod-mindllm.local:8443/v1/")
    assert llm_extractor._mindllm_url() == "http://prod-mindllm.local:8443/v1"


# ---------------------------------------------------------------------------
# Backend type recognition
# ---------------------------------------------------------------------------


def test_mindllm_backend_recognised(monkeypatch: pytest.MonkeyPatch) -> None:
    """``backend="mindllm"`` routes through is_available; aliases accepted."""
    # No MindLLM server running in CI — is_available returns False but
    # MUST NOT raise / return None on the recognised backend name.
    monkeypatch.setenv("MIND_MEM_MINDLLM_URL", "http://127.0.0.1:9 /v1")  # unreachable
    # Replace probe with a known-false to avoid network flake.
    monkeypatch.setattr(llm_extractor, "_openai_compatible_available", lambda url: False)
    assert llm_extractor.is_available("mindllm") is False
    assert llm_extractor.is_available("mind-llm") is False  # alias
    assert llm_extractor.is_available("mind_llm") is False  # alias


def test_mindllm_backend_reports_available_when_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``_openai_compatible_available`` returns True for MindLLM URL,
    ``is_available("mindllm")`` must be True."""
    monkeypatch.setattr(
        llm_extractor,
        "_openai_compatible_available",
        lambda url: url.startswith("http://localhost:8080"),
    )
    assert llm_extractor.is_available("mindllm") is True


# ---------------------------------------------------------------------------
# Auto-discovery order includes MindLLM
# ---------------------------------------------------------------------------


def test_auto_discovery_includes_mindllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``backend="auto"`` must try MindLLM before vLLM / openai-compatible
    so deployments running MindLLM are picked up without explicit config."""
    visited: list[str] = []

    def fake_oai(url: str) -> bool:
        visited.append(url)
        return False

    monkeypatch.setattr(llm_extractor, "_ollama_available", lambda: False)
    monkeypatch.setattr(llm_extractor, "_llama_cpp_available", lambda: False)
    monkeypatch.setattr(llm_extractor, "_transformers_available", lambda: False)
    monkeypatch.setattr(llm_extractor, "_openai_compatible_available", fake_oai)

    # Trigger auto path; result False because everything we patched returns False.
    llm_extractor.is_available("auto")

    # MindLLM must have been probed and BEFORE vLLM (so MindLLM users
    # don't have to set explicit config when both are running).
    assert any("8080" in u for u in visited), f"MindLLM port 8080 not probed: {visited}"
    mindllm_idx = next(i for i, u in enumerate(visited) if "8080" in u)
    vllm_idx_candidates = [i for i, u in enumerate(visited) if "8000" in u]
    if vllm_idx_candidates:
        assert mindllm_idx < vllm_idx_candidates[0], f"MindLLM should be probed before vLLM in auto-discovery; got order {visited}"


# ---------------------------------------------------------------------------
# Pipeline-hash treats MindLLM as a known backend (not the unknown stub)
# ---------------------------------------------------------------------------


def test_mindllm_backend_in_pipeline_hash_known_backends() -> None:
    """``pipeline_hash._BACKEND_SOURCE_FILES`` must include ``mindllm`` so
    the pipeline-hash for a MindLLM-configured workspace doesn't fall
    through to the unknown-backend stub hash."""
    from mind_mem import pipeline_hash

    assert "mindllm" in pipeline_hash._BACKEND_SOURCE_FILES
    # And it points at the same llm_extractor.py source as ollama (same
    # extraction logic; just a different HTTP endpoint).
    assert pipeline_hash._BACKEND_SOURCE_FILES["mindllm"] == pipeline_hash._BACKEND_SOURCE_FILES["ollama"]
