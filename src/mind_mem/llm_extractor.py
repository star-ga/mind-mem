#!/usr/bin/env python3
"""mind-mem LLM Entity & Fact Extractor (Optional, config-gated).

Provides LLM-powered extraction of entities and factual claims from text,
using local LLMs via ollama (HTTP API) or llama-cpp-python.  This module
is entirely optional — when no LLM backend is available or extraction is
disabled in config, every function returns empty/passthrough results.

Zero external dependencies by default.  ollama and llama-cpp-python are
detected at runtime; neither is required.

Config (mind-mem.json):
    "extraction": {
        "enabled": false,
        "model": "qwen3.5:9b",
        "backend": "auto"
    }

Backends (tried in order when backend="auto"):
    1. ollama   — HTTP API at localhost:11434
    2. llama-cpp-python — Python bindings for llama.cpp
    3. (none)   — graceful empty results

Usage:
    from .llm_extractor import is_available, extract_entities, extract_facts, enrich_block
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "model": "qwen3.5:9b",
    "backend": "auto",
}


def load_config(workspace: str = ".") -> dict[str, Any]:
    """Load extraction config from mind-mem.json.

    Returns the extraction section merged with defaults.
    """
    config = dict(_DEFAULT_CONFIG)
    config_path = os.path.join(workspace, "mind-mem.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            extraction = cfg.get("extraction", {})
            if isinstance(extraction, dict):
                for key in _DEFAULT_CONFIG:
                    if key in extraction:
                        config[key] = extraction[key]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
    return config


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _ollama_available(model: str = "") -> bool:
    """Check if ollama is running and reachable at localhost:11434."""
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return True
    except (OSError, ValueError):
        pass
    return False


def _llama_cpp_available() -> bool:
    """Check if llama-cpp-python is importable."""
    try:
        import llama_cpp  # noqa: F401

        return True
    except ImportError:
        return False


def is_available(backend: str = "auto") -> bool:
    """Check if the requested LLM backend is reachable.

    Backends:
        ``ollama``              local ollama daemon at :11434
        ``llama-cpp``           in-process llama-cpp-python
        ``vllm``                local vLLM OpenAI-compatible server
        ``openai-compatible``   any OpenAI-compatible HTTP endpoint
        ``transformers``        in-process HF transformers (slowest)
        ``auto``                tries each in order until one answers

    Endpoint and base-url defaults read from env vars
    (``MIND_MEM_LLM_BASE_URL``, ``MIND_MEM_VLLM_URL``).
    """
    if backend == "ollama":
        return _ollama_available()
    if backend in ("llama-cpp", "llama_cpp"):
        return _llama_cpp_available()
    if backend == "vllm":
        return _openai_compatible_available(_vllm_url())
    if backend in ("openai-compatible", "openai_compatible"):
        return _openai_compatible_available(_oai_url())
    if backend == "transformers":
        return _transformers_available()
    if backend == "auto":
        return (
            _ollama_available()
            or _openai_compatible_available(_vllm_url())
            or _openai_compatible_available(_oai_url())
            or _llama_cpp_available()
            or _transformers_available()
        )
    return False


def _vllm_url() -> str:
    return os.environ.get(
        "MIND_MEM_VLLM_URL", "http://localhost:8000/v1"
    ).rstrip("/")


def _oai_url() -> str:
    return os.environ.get(
        "MIND_MEM_LLM_BASE_URL", _vllm_url()
    ).rstrip("/")


def _openai_compatible_available(base_url: str) -> bool:
    """Probe ``GET {base_url}/models`` — works for vLLM, LM Studio,
    text-generation-inference's OpenAI shim, and llama.cpp's
    ``llama-server`` exposed with ``--api`` flag."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(f"{base_url}/models", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return False


def _transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# LLM query helpers
# ---------------------------------------------------------------------------


def _query_ollama(prompt: str, model: str) -> str:
    """Send a prompt to ollama and return the response text."""
    import urllib.request

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }
    ).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    return str(body.get("response", ""))


def _query_llama_cpp(prompt: str, model: str) -> str:
    """Send a prompt via llama-cpp-python and return the response text."""
    import llama_cpp

    # Use a cached model instance per model name
    if not hasattr(_query_llama_cpp, "_models"):
        _query_llama_cpp._models = {}  # type: ignore[attr-defined]
    if model not in _query_llama_cpp._models:  # type: ignore[attr-defined]
        _query_llama_cpp._models[model] = llama_cpp.Llama(model_path=model, n_ctx=2048)  # type: ignore[attr-defined]
    llm = _query_llama_cpp._models[model]  # type: ignore[attr-defined]
    output = llm(prompt, max_tokens=512, temperature=0.1)
    choices = output.get("choices", [])
    if choices:
        return str(choices[0].get("text", ""))
    return ""


def _query_openai_compatible(prompt: str, model: str, base_url: str) -> str:
    """POST ``{base_url}/chat/completions`` with a single user turn.

    Works against vLLM, LM Studio, llama.cpp's ``llama-server --api``,
    text-generation-inference's OpenAI shim, OpenAI itself if
    ``MIND_MEM_LLM_API_KEY`` is set, etc.
    """
    import urllib.request

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 512,
            "stream": False,
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("MIND_MEM_LLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode())
    choices = body.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        return str(msg.get("content", ""))
    return ""


def _query_transformers(prompt: str, model: str) -> str:
    """Load the model in-process and run a single generate call.

    Caches the loaded model + tokenizer per *model* path so subsequent
    calls don't pay the load cost. Use only when no daemon is running;
    much slower than vLLM/Ollama for sustained workloads.
    """
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    cache = getattr(_query_transformers, "_cache", None)
    if cache is None:
        cache = {}
        _query_transformers._cache = cache  # type: ignore[attr-defined]
    if model not in cache:
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        m.eval()
        cache[model] = (tok, m)
    tok, m = cache[model]
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    if hasattr(enc, "to"):
        enc = enc.to(m.device)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=512, do_sample=False)
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def _query_llm(prompt: str, model: str, backend: str = "auto") -> str:
    """Dispatch to the named backend. Returns empty string on failure.

    Order for ``auto`` mode: ollama → vllm → openai-compat → llama-cpp →
    transformers. Each is tried until one returns a non-empty string.
    """
    backends = (
        [backend]
        if backend != "auto"
        else ["ollama", "vllm", "openai-compatible", "llama-cpp", "transformers"]
    )
    for b in backends:
        try:
            if b == "ollama":
                out = _query_ollama(prompt, model)
            elif b in ("llama-cpp", "llama_cpp"):
                out = _query_llama_cpp(prompt, model)
            elif b == "vllm":
                out = _query_openai_compatible(prompt, model, _vllm_url())
            elif b in ("openai-compatible", "openai_compatible"):
                out = _query_openai_compatible(prompt, model, _oai_url())
            elif b == "transformers":
                out = _query_transformers(prompt, model)
            else:
                continue
            if out:
                return out
        except (OSError, ValueError, RuntimeError, ImportError):
            continue
    return ""


# ---------------------------------------------------------------------------
# JSON extraction from LLM output
# ---------------------------------------------------------------------------


def _parse_json_from_response(text: str) -> list[dict]:
    """Extract a JSON array from LLM output, tolerating markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Find the JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

_ENTITY_PROMPT = """\
Extract entities from the following text. Return a JSON array of objects, \
each with keys: "name" (string), "type" (one of: person, place, date, \
organization, decision, tool, project), "context" (short phrase).

Only return the JSON array, no explanation.

Text: {text}

JSON:"""


def extract_entities(text: str, model: str = "qwen3.5:9b", backend: str = "auto") -> list[dict]:
    """Extract entities (people, places, dates, decisions) from text.

    Args:
        text: Input text to extract entities from.
        model: LLM model name (default: qwen3.5:9b).
        backend: Backend to use ("auto", "ollama", "llama-cpp").

    Returns:
        List of entity dicts with keys: name, type, context.
        Returns empty list if no LLM backend is available.
    """
    if not text or not text.strip():
        return []
    if not is_available(backend):
        return []
    prompt = _ENTITY_PROMPT.format(text=text[:2000])
    _start = time.monotonic()
    response = _query_llm(prompt, model, backend)
    _latency_ms = (time.monotonic() - _start) * 1000.0
    if not response:
        _record_extraction_feedback(model, "entities", len(text), 0, _latency_ms)
        return []
    entities = _parse_json_from_response(response)
    # Validate required keys
    validated = []
    for ent in entities:
        if "name" in ent and "type" in ent:
            validated.append(
                {
                    "name": str(ent["name"]),
                    "type": str(ent["type"]),
                    "context": str(ent.get("context", "")),
                }
            )
    _record_extraction_feedback(model, "entities", len(text), len(validated), _latency_ms)
    return validated


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

_FACT_PROMPT = """\
Extract factual claims from the following text. Return a JSON array of \
objects, each with keys: "claim" (string, one sentence), "confidence" \
(float 0-1), "category" (one of: identity, event, preference, relation, \
negation, plan, state).

Only return the JSON array, no explanation.

Text: {text}

JSON:"""


def extract_facts(text: str, model: str = "qwen3.5:9b", backend: str = "auto") -> list[dict]:
    """Extract factual claims from text.

    Args:
        text: Input text to extract facts from.
        model: LLM model name (default: qwen3.5:9b).
        backend: Backend to use ("auto", "ollama", "llama-cpp").

    Returns:
        List of fact dicts with keys: claim, confidence, category.
        Returns empty list if no LLM backend is available.
    """
    if not text or not text.strip():
        return []
    if not is_available(backend):
        return []
    prompt = _FACT_PROMPT.format(text=text[:2000])
    _start = time.monotonic()
    response = _query_llm(prompt, model, backend)
    _latency_ms = (time.monotonic() - _start) * 1000.0
    if not response:
        _record_extraction_feedback(model, "facts", len(text), 0, _latency_ms)
        return []
    facts = _parse_json_from_response(response)
    # Validate required keys
    validated = []
    for fact in facts:
        if "claim" in fact:
            conf = fact.get("confidence", 0.5)
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            validated.append(
                {
                    "claim": str(fact["claim"]),
                    "confidence": conf,
                    "category": str(fact.get("category", "state")),
                }
            )
    _record_extraction_feedback(model, "facts", len(text), len(validated), _latency_ms)
    return validated


def _record_extraction_feedback(
    model: str, operation: str, input_length: int, output_count: int, latency_ms: float
) -> None:
    """Best-effort ExtractionFeedback.record wrapper. Never raises."""
    try:
        from .extraction_feedback import ExtractionFeedback

        ExtractionFeedback().record(
            model=model,
            operation=operation,
            input_length=input_length,
            output_count=output_count,
            latency_ms=latency_ms,
        )
    except Exception:  # pragma: no cover — best-effort telemetry
        pass


# ---------------------------------------------------------------------------
# Block enrichment
# ---------------------------------------------------------------------------


def enrich_block(
    block: dict,
    model: str = "qwen3.5:9b",
    backend: str = "auto",
    enabled: bool = False,
) -> dict:
    """Add LLM-extracted metadata to a memory block.

    When disabled or no LLM is available, returns the block unchanged.

    Args:
        block: A recall result dict (must have "excerpt" or "content" key).
        model: LLM model name.
        backend: Backend to use.
        enabled: Whether LLM extraction is enabled.

    Returns:
        The block dict, potentially with added "llm_entities" and "llm_facts" keys.
    """
    if not enabled:
        return block
    text = block.get("excerpt", block.get("content", ""))
    if not text:
        return block
    if not is_available(backend):
        return block
    entities = extract_entities(text, model=model, backend=backend)
    facts = extract_facts(text, model=model, backend=backend)
    if entities:
        block["llm_entities"] = entities
    if facts:
        block["llm_facts"] = facts
    return block


def enrich_results(
    results: list[dict],
    workspace: str = ".",
) -> list[dict]:
    """Enrich a list of recall results using LLM extraction per config.

    Reads extraction config from mind-mem.json.  When disabled (default),
    returns results unchanged with zero overhead.

    Args:
        results: List of recall result dicts.
        workspace: Workspace root path for config loading.

    Returns:
        The same list, potentially with LLM metadata added to each block.
    """
    config = load_config(workspace)
    if not config.get("enabled", False):
        return results
    model = config.get("model", "qwen3.5:9b")
    backend = config.get("backend", "auto")
    for block in results:
        enrich_block(block, model=model, backend=backend, enabled=True)
    return results
