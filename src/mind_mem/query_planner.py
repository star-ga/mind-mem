"""Query decomposition for multi-hop questions (v3.3.0 Tier 1 #1).

LoCoMo multi-hop queries like "What did X say about Y after Z?" need
their reasoning steps split into retrievable sub-queries. The raw
BM25 / hybrid recall stacks can't span multi-hop evidence in a single
pass — each sub-clause is better answered by a dedicated retrieval.

This module provides two decomposers:

* :class:`NLPQueryDecomposer` — regex/rule-based, zero network cost,
  enabled by default. Handles conjunction ("A and B"), temporal
  ordering ("A after B"), comparative ("A vs B"), and causal
  ("A because B") patterns.
* :class:`LLMQueryDecomposer` — LLM-backed via an OpenAI-compatible
  endpoint (includes claude-proxy CLI auth for free local routing).
  Opt-in via ``retrieval.query_decomposition.provider``.

``decompose_query()`` is the public entry point — it auto-selects the
decomposer from config and always includes the original query as the
first element of the returned list so RRF fusion treats the original
as another signal rather than discarding it.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from .observability import get_logger

_log = get_logger("query_planner")


# ---------------------------------------------------------------------------
# Public Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class QueryDecomposer(Protocol):
    """Protocol for query decomposers. Returns [original, ...sub_queries]."""

    def decompose(self, query: str, max_subqueries: int = 4) -> list[str]: ...


# ---------------------------------------------------------------------------
# NLP (rule-based) decomposer
# ---------------------------------------------------------------------------


# Patterns that reliably split a multi-hop query. Order matters — the
# first match wins, so more-specific patterns come first.
_SPLIT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("temporal_after", re.compile(r"\s+(?:after|following|once)\s+", re.IGNORECASE)),
    ("temporal_before", re.compile(r"\s+(?:before|prior to|until)\s+", re.IGNORECASE)),
    ("causal", re.compile(r"\s+(?:because|since|due to|as a result of|caused by)\s+", re.IGNORECASE)),
    ("contrastive", re.compile(r"\s+(?:but|however|although|though|whereas)\s+", re.IGNORECASE)),
    ("comparison", re.compile(r"\s+(?:vs\.?|versus|compared to|compared with)\s+", re.IGNORECASE)),
    ("conjunction", re.compile(r"\s+(?:and then|then|and also|and)\s+", re.IGNORECASE)),
]


# Minimum length (in words) each sub-query must have to be usable —
# avoids producing "X" from "A and X" where X is a one-letter tail.
_MIN_SUBQUERY_WORDS = 2


class NLPQueryDecomposer:
    """Rule-based decomposer. Zero network cost, always available."""

    def decompose(self, query: str, max_subqueries: int = 4) -> list[str]:
        if not query or not query.strip():
            return [query] if query else [""]
        original = query.strip()
        parts = self._split_on_patterns(original, max_subqueries)
        # Always lead with the original so the retrieval pipeline keeps
        # the verbatim question as one RRF signal.
        results: list[str] = [original]
        seen: set[str] = {original.lower()}
        for p in parts:
            p_clean = p.strip(" ?.!,;")
            if len(p_clean.split()) < _MIN_SUBQUERY_WORDS:
                continue
            key = p_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(p_clean)
            if len(results) >= max_subqueries:
                break
        return results

    def _split_on_patterns(self, query: str, max_out: int) -> list[str]:
        """Return the split fragments in source order, or [query] on no match."""
        for name, pat in _SPLIT_PATTERNS:
            if pat.search(query):
                fragments = pat.split(query)
                if len(fragments) > 1:
                    _log.info(
                        "query_decomposed",
                        pattern=name,
                        fragments=len(fragments),
                    )
                    return fragments
        # Last-resort heuristic: multiple question marks → split on ?
        if query.count("?") > 1:
            fragments = [f.strip() for f in query.split("?") if f.strip()]
            _log.info("query_decomposed", pattern="multi_qmark", fragments=len(fragments))
            return fragments
        return []


# ---------------------------------------------------------------------------
# LLM-backed decomposer (opt-in)
# ---------------------------------------------------------------------------


class LLMQueryDecomposer:
    """LLM-backed decomposer via OpenAI-compatible endpoint.

    Pointed at ``http://127.0.0.1:8766/v1/chat/completions`` with
    model ``claude-proxy/claude-opus-4-7`` for free OAuth routing,
    or at any other OpenAI-compatible endpoint with an API key.

    Fails open to :class:`NLPQueryDecomposer` on any error.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.base_url: str = cfg.get("base_url", "http://127.0.0.1:8766/v1/chat/completions")
        self.model: str = cfg.get("model", "claude-proxy/claude-opus-4-7")
        self.api_key_env: str = cfg.get("api_key_env", "")
        self.timeout: float = float(cfg.get("timeout", 20.0))
        self._fallback = NLPQueryDecomposer()

    def decompose(self, query: str, max_subqueries: int = 4) -> list[str]:
        if not query or not query.strip():
            return [query] if query else [""]
        try:
            subs = self._call_llm(query.strip(), max_subqueries - 1)
        except Exception as exc:
            _log.warning("llm_decomposition_failed", error=str(exc), fallback="nlp")
            return self._fallback.decompose(query, max_subqueries)
        results: list[str] = [query.strip()]
        seen: set[str] = {query.strip().lower()}
        for s in subs:
            s_clean = s.strip(" ?.!,;")
            if len(s_clean.split()) < _MIN_SUBQUERY_WORDS:
                continue
            key = s_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(s_clean)
            if len(results) >= max_subqueries:
                break
        if len(results) == 1:
            return self._fallback.decompose(query, max_subqueries)
        return results

    def _call_llm(self, query: str, n: int) -> list[str]:
        import json
        import os
        import urllib.request

        prompt = (
            "Decompose this multi-hop search query into up to "
            f"{n} atomic sub-queries that a retrieval system could answer independently. "
            "Preserve entities and dates verbatim. Return one sub-query per line, "
            "no numbering, no extra commentary.\n\n"
            f"Query: {query}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 400,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        return [ln.strip() for ln in text.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Factory + public entry point
# ---------------------------------------------------------------------------


def create_decomposer(config: dict[str, Any] | None = None) -> QueryDecomposer:
    """Return the appropriate decomposer per config.

    ``retrieval.query_decomposition.provider`` selects:
      * ``"nlp"`` (default) — :class:`NLPQueryDecomposer`
      * ``"llm"`` — :class:`LLMQueryDecomposer`

    Unknown providers fall back to NLP.
    """
    cfg = (config or {}).get("retrieval", {}) if config else {}
    dec_cfg = cfg.get("query_decomposition", {}) if isinstance(cfg, dict) else {}
    provider = dec_cfg.get("provider", "nlp")
    if provider == "llm":
        return LLMQueryDecomposer(dec_cfg)
    return NLPQueryDecomposer()


def decompose_query(
    query: str,
    config: dict[str, Any] | None = None,
    max_subqueries: int = 4,
) -> list[str]:
    """Return [original, ...sub_queries].

    When ``config`` omits ``retrieval.query_decomposition`` entirely,
    NLP decomposition is used. Length of the returned list is at most
    ``max_subqueries``. On any failure, returns ``[query]`` alone.
    """
    if not query or not query.strip():
        return [query] if query else [""]
    try:
        return create_decomposer(config).decompose(query, max_subqueries)
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("decompose_query_failed", error=str(exc))
        return [query.strip()]


__all__ = [
    "QueryDecomposer",
    "NLPQueryDecomposer",
    "LLMQueryDecomposer",
    "create_decomposer",
    "decompose_query",
]
