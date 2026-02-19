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
        "model": "phi3:mini",
        "backend": "auto"
    }

Backends (tried in order when backend="auto"):
    1. ollama   — HTTP API at localhost:11434
    2. llama-cpp-python — Python bindings for llama.cpp
    3. (none)   — graceful empty results

Usage:
    from llm_extractor import is_available, extract_entities, extract_facts, enrich_block
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "model": "phi3:mini",
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
    except Exception:
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
    """Check if any LLM backend is accessible.

    Args:
        backend: "auto" tries ollama then llama-cpp; "ollama" or "llama-cpp"
                 checks only that specific backend.

    Returns:
        True if at least one backend is available.
    """
    if backend == "ollama":
        return _ollama_available()
    if backend in ("llama-cpp", "llama_cpp"):
        return _llama_cpp_available()
    if backend == "auto":
        return _ollama_available() or _llama_cpp_available()
    # Unknown backend name — not supported
    return False


# ---------------------------------------------------------------------------
# LLM query helpers
# ---------------------------------------------------------------------------

def _query_ollama(prompt: str, model: str) -> str:
    """Send a prompt to ollama and return the response text."""
    import urllib.request
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    return body.get("response", "")


def _query_llama_cpp(prompt: str, model: str) -> str:
    """Send a prompt via llama-cpp-python and return the response text."""
    import llama_cpp
    # Use a cached model instance per model name
    if not hasattr(_query_llama_cpp, "_models"):
        _query_llama_cpp._models = {}
    if model not in _query_llama_cpp._models:
        _query_llama_cpp._models[model] = llama_cpp.Llama(model_path=model, n_ctx=2048)
    llm = _query_llama_cpp._models[model]
    output = llm(prompt, max_tokens=512, temperature=0.1)
    choices = output.get("choices", [])
    if choices:
        return choices[0].get("text", "")
    return ""


def _query_llm(prompt: str, model: str, backend: str = "auto") -> str:
    """Query the best available LLM backend. Returns empty string on failure."""
    if backend in ("ollama", "auto"):
        try:
            return _query_ollama(prompt, model)
        except Exception:
            if backend == "ollama":
                return ""
    if backend in ("llama-cpp", "llama_cpp", "auto"):
        try:
            return _query_llama_cpp(prompt, model)
        except Exception:
            return ""
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


def extract_entities(text: str, model: str = "phi3:mini", backend: str = "auto") -> list[dict]:
    """Extract entities (people, places, dates, decisions) from text.

    Args:
        text: Input text to extract entities from.
        model: LLM model name (default: phi3:mini).
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
    response = _query_llm(prompt, model, backend)
    if not response:
        return []
    entities = _parse_json_from_response(response)
    # Validate required keys
    validated = []
    for ent in entities:
        if "name" in ent and "type" in ent:
            validated.append({
                "name": str(ent["name"]),
                "type": str(ent["type"]),
                "context": str(ent.get("context", "")),
            })
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


def extract_facts(text: str, model: str = "phi3:mini", backend: str = "auto") -> list[dict]:
    """Extract factual claims from text.

    Args:
        text: Input text to extract facts from.
        model: LLM model name (default: phi3:mini).
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
    response = _query_llm(prompt, model, backend)
    if not response:
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
            validated.append({
                "claim": str(fact["claim"]),
                "confidence": conf,
                "category": str(fact.get("category", "state")),
            })
    return validated


# ---------------------------------------------------------------------------
# Block enrichment
# ---------------------------------------------------------------------------

def enrich_block(
    block: dict,
    model: str = "phi3:mini",
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
    model = config.get("model", "phi3:mini")
    backend = config.get("backend", "auto")
    for block in results:
        enrich_block(block, model=model, backend=backend, enabled=True)
    return results
