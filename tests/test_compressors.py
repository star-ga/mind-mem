"""Tests for compressors.py — real Compressor implementations.

The `Compressor` contract (`mind_mem.recompaction.Compressor`) is
`(current_text, blocks) -> new_text` and must be pure w.r.t. its inputs for
the recompaction fixed point to mean anything. These tests never touch the
network: the ollama HTTP layer is mocked at `urllib.request.urlopen`.
"""

from __future__ import annotations

import json
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mind_mem.compressors import CompressorError, EchoCompressor, OllamaCompressor

# --- EchoCompressor ---------------------------------------------------------


def test_echo_compressor_returns_input_verbatim():
    """The trivially-converging control: input in, same bytes out."""
    blocks = [{"_id": "D-001", "body": "alpha"}]
    assert EchoCompressor()("current text", blocks) == "current text"


def test_echo_compressor_ignores_blocks_content():
    ec = EchoCompressor()
    assert ec("x", []) == "x"
    assert ec("x", [{"_id": "A"}, {"_id": "B"}]) == "x"


def test_echo_compressor_is_pure_across_repeated_calls():
    ec = EchoCompressor()
    blocks = [{"_id": "D-001", "body": "alpha"}]
    assert ec("same", blocks) == ec("same", blocks)


# --- OllamaCompressor: request shaping --------------------------------------


def _fake_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_ollama_compressor_sets_temperature_zero_and_fixed_seed():
    """Purity is load-bearing for the fixed point — no sampling, no drift."""
    captured: dict[str, Any] = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _fake_response({"response": "tight summary"})

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="mind-mem:4b", seed=42)
        result = compressor("current summary", [{"_id": "D-001", "body": "alpha"}])

    assert result == "tight summary"
    assert captured["body"]["options"]["temperature"] == 0.0
    assert captured["body"]["options"]["seed"] == 42
    assert captured["body"]["model"] == "mind-mem:4b"
    assert captured["body"]["stream"] is False


def test_ollama_compressor_prompt_includes_current_summary_and_sibling_blocks():
    captured: dict[str, Any] = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _fake_response({"response": "ok"})

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="mind-mem:4b")
        compressor("the current summary text", [{"_id": "D-001", "body": "sibling fact one"}])

    prompt = captured["body"]["prompt"]
    assert "the current summary text" in prompt
    assert "sibling fact one" in prompt


def test_ollama_compressor_renders_blocks_without_a_body_field():
    """Real store blocks carry named + list fields, not always a `body` key."""
    captured: dict[str, Any] = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _fake_response({"response": "ok"})

    structured = [{"_id": "D-001", "_source_file": "x.md", "summary": "chose Q16.16", "count": 3}]
    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m")
        compressor("current", structured)

    prompt = captured["body"]["prompt"]
    assert "chose Q16.16" in prompt
    assert "3" in prompt


def test_ollama_compressor_posts_to_configured_host():
    captured: dict[str, Any] = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _fake_response({"response": "ok"})

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m", host="http://example.internal:11434")
        compressor("x", [{"_id": "D-001", "body": "y"}])

    assert captured["url"] == "http://example.internal:11434/api/generate"


# --- OllamaCompressor: response cleaning ------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("plain text response", "plain text response"),
        ("  leading and trailing whitespace  ", "leading and trailing whitespace"),
        ("```\nfenced content\n```", "fenced content"),
        ("```markdown\nfenced content\n```", "fenced content"),
        ("Here is the tighter summary:\nactual summary text", "actual summary text"),
        ("Sure, here's the summary:\n\nactual summary text", "actual summary text"),
        ("Summary:\nactual summary text", "actual summary text"),
    ],
)
def test_ollama_compressor_strips_preamble_and_fences_deterministically(raw, expected):
    """A paraphrasing preamble would prevent convergence forever — strip it."""

    def _fake_urlopen(req, timeout=None):
        return _fake_response({"response": raw})

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m")
        result = compressor("current", [{"_id": "D-001", "body": "b"}])

    assert result == expected


def test_ollama_compressor_cleaning_is_idempotent():
    """Cleaning an already-clean string must be a no-op — required for the fixed point."""

    def _fake_urlopen(req, timeout=None):
        return _fake_response({"response": "already minimal and complete"})

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m")
        first = compressor("already minimal and complete", [{"_id": "D-001", "body": "b"}])
        second = compressor(first, [{"_id": "D-001", "body": "b"}])

    assert first == second == "already minimal and complete"


# --- OllamaCompressor: error handling ---------------------------------------


def test_ollama_compressor_raises_on_non_200_status():
    def _fake_urlopen(req, timeout=None):
        return _fake_response({"response": "x"}, status=500)

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m")
        with pytest.raises(CompressorError, match="500"):
            compressor("x", [{"_id": "D-001", "body": "y"}])


def test_ollama_compressor_raises_on_connection_error():
    def _fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m")
        with pytest.raises(CompressorError, match="connection refused"):
            compressor("x", [{"_id": "D-001", "body": "y"}])


def test_ollama_compressor_raises_on_timeout():
    def _fake_urlopen(req, timeout=None):
        raise TimeoutError("timed out")

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m", timeout=1.0)
        with pytest.raises(CompressorError, match="tim"):
            compressor("x", [{"_id": "D-001", "body": "y"}])


def test_ollama_compressor_raises_on_malformed_json():
    def _fake_urlopen(req, timeout=None):
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b"not json"
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m")
        with pytest.raises(CompressorError, match="malformed|json|decode"):
            compressor("x", [{"_id": "D-001", "body": "y"}])


def test_ollama_compressor_raises_on_missing_response_field():
    def _fake_urlopen(req, timeout=None):
        return _fake_response({"unexpected": "shape"})

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m")
        with pytest.raises(CompressorError, match="response"):
            compressor("x", [{"_id": "D-001", "body": "y"}])


def test_ollama_compressor_never_silently_returns_input_on_error():
    """A silent fallback to the input would fake convergence — must always raise."""

    def _fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("boom")

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m")
        with pytest.raises(CompressorError):
            compressor("the exact input text", [{"_id": "D-001", "body": "y"}])


def test_ollama_compressor_raises_on_non_string_response_field():
    def _fake_urlopen(req, timeout=None):
        return _fake_response({"response": 12345})

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m")
        with pytest.raises(CompressorError, match="not a string"):
            compressor("x", [{"_id": "D-001", "body": "y"}])


# --- constructor validation --------------------------------------------------


def test_ollama_compressor_rejects_empty_model():
    with pytest.raises(ValueError, match="non-empty"):
        OllamaCompressor(model="")


def test_ollama_compressor_rejects_non_zero_temperature():
    """Non-zero sampling temperature breaks purity — the fixed point requires it."""
    with pytest.raises(ValueError, match="temperature"):
        OllamaCompressor(model="m", temperature=0.7)


# --- immutability ------------------------------------------------------------


def test_ollama_compressor_does_not_mutate_blocks():
    blocks = [{"_id": "D-001", "body": "alpha"}]
    before = [dict(b) for b in blocks]

    def _fake_urlopen(req, timeout=None):
        return _fake_response({"response": "ok"})

    with patch("mind_mem.compressors.urllib.request.urlopen", side_effect=_fake_urlopen):
        compressor = OllamaCompressor(model="m")
        compressor("x", blocks)

    assert blocks == before
