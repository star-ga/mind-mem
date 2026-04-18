"""Multi-query expansion for improved recall.

Generates 2-3 alternative phrasings of a query before searching, improving
recall by matching documents that use different terminology or phrasing.

Two expansion modes:
  - NLP-based (default): synonym substitution, query decomposition, and
    morphological variants using zero external dependencies.
  - LLM-backed (optional): uses a configurable LLM provider to generate
    rephrasings. Disabled by default; enable via config.

Configuration (mind-mem.json):
    {
      "recall": {
        "query_expansion": {
          "enabled": true,
          "max_expansions": 3,
          "llm": {
            "enabled": false,
            "provider": "anthropic",
            "model": "claude-haiku",
            "api_key_env": "ANTHROPIC_API_KEY"
          }
        }
      }
    }
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from .observability import get_logger

_log = get_logger("query_expansion")

__all__ = [
    "QueryExpander",
    "NLPQueryExpander",
    "LLMQueryExpander",
    "expand_queries",
    "create_expander",
]


# ---------------------------------------------------------------------------
# Synonym map — maps common terms to alternatives for full-query rewriting.
# Unlike _recall_expansion._QUERY_EXPANSIONS (which operates on stemmed
# tokens within BM25), this map operates on surface-form words for
# generating human-readable alternative queries.
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, list[str]] = {
    # Actions
    "add": ["create", "insert"],
    "remove": ["delete", "drop"],
    "update": ["modify", "change"],
    "fix": ["repair", "resolve", "patch"],
    "find": ["search", "locate", "look up"],
    "get": ["retrieve", "fetch", "obtain"],
    "set": ["configure", "assign"],
    "show": ["display", "list", "view"],
    "build": ["compile", "construct"],
    "run": ["execute", "start", "launch"],
    "stop": ["halt", "terminate", "kill"],
    "send": ["transmit", "dispatch"],
    "receive": ["accept", "get"],
    "check": ["verify", "validate", "inspect"],
    "deploy": ["release", "ship", "publish"],
    # Concepts
    "error": ["exception", "failure", "bug"],
    "bug": ["defect", "issue", "error"],
    "issue": ["problem", "bug", "defect"],
    "performance": ["speed", "latency", "throughput"],
    "security": ["protection", "safety", "auth"],
    "authentication": ["auth", "login", "sign-in"],
    "authorization": ["permissions", "access control"],
    "database": ["db", "data store", "storage"],
    "configuration": ["config", "settings", "setup"],
    "documentation": ["docs", "guide", "manual"],
    "test": ["spec", "check", "verify"],
    "user": ["account", "member", "client"],
    "server": ["backend", "service", "host"],
    "client": ["frontend", "browser", "UI"],
    "cache": ["memoize", "store", "buffer"],
    "log": ["record", "trace", "journal"],
    "message": ["notification", "alert", "event"],
    "request": ["call", "query", "petition"],
    "response": ["reply", "answer", "result"],
    "migration": ["upgrade", "transition", "conversion"],
    # Adjectives / qualifiers
    "slow": ["sluggish", "laggy", "unresponsive"],
    "fast": ["quick", "rapid", "efficient"],
    "broken": ["failing", "malfunctioning", "down"],
    "new": ["recent", "latest", "fresh"],
    "old": ["legacy", "outdated", "previous"],
}

# Precompute a lowercase lookup for case-insensitive matching
_SYNONYMS_LOWER: dict[str, list[str]] = {k.lower(): v for k, v in _SYNONYMS.items()}

# Question word rewrite patterns: map question forms to alternative phrasings
_QUESTION_REWRITES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^how (?:do|does|can|to) (.+?)[\?\.]?\s*$", re.IGNORECASE), r"steps to \1"),
    (re.compile(r"^what is (.+?)[\?\.]?\s*$", re.IGNORECASE), r"\1 definition"),
    (re.compile(r"^what are (.+?)[\?\.]?\s*$", re.IGNORECASE), r"\1 overview"),
    (re.compile(r"^why (?:does|do|is|are) (.+?)[\?\.]?\s*$", re.IGNORECASE), r"reason for \1"),
    (re.compile(r"^when (?:did|does|was|is) (.+?)[\?\.]?\s*$", re.IGNORECASE), r"\1 timeline"),
    (re.compile(r"^where (?:is|are|can) (.+?)[\?\.]?\s*$", re.IGNORECASE), r"\1 location"),
    (re.compile(r"^who (?:is|was|are) (.+?)[\?\.]?\s*$", re.IGNORECASE), r"\1 identity"),
]


# ---------------------------------------------------------------------------
# Expander protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class QueryExpander(Protocol):
    """Protocol for query expansion implementations."""

    def expand(self, query: str, max_expansions: int = 3) -> list[str]:
        """Return a list of alternative query phrasings.

        The original query is always included as the first element.
        """
        ...


# ---------------------------------------------------------------------------
# NLP-based expander (zero external deps)
# ---------------------------------------------------------------------------


class NLPQueryExpander:
    """Generate alternative query phrasings using rule-based NLP techniques.

    Strategies applied (in order, up to max_expansions total):
      1. Synonym substitution: replace key terms with synonyms.
      2. Question rewriting: convert question forms to declarative.
      3. Term reordering: rearrange multi-word queries for different emphasis.
    """

    def expand(self, query: str, max_expansions: int = 3) -> list[str]:
        """Generate up to max_expansions alternative phrasings.

        Args:
            query: Original search query.
            max_expansions: Maximum number of results including the original.

        Returns:
            List of query strings starting with the original, followed by
            up to (max_expansions - 1) alternatives. Duplicates are removed.
        """
        if not query or not query.strip():
            return [query] if query else [""]

        query = query.strip()
        results: list[str] = [query]
        seen: set[str] = {_normalize_for_dedup(query)}

        # Strategy 1: Synonym substitution
        synonym_alt = self._synonym_substitute(query)
        if synonym_alt:
            norm = _normalize_for_dedup(synonym_alt)
            if norm not in seen:
                results.append(synonym_alt)
                seen.add(norm)

        # Strategy 2: Question rewriting
        if len(results) < max_expansions:
            rewrite = self._question_rewrite(query)
            if rewrite:
                norm = _normalize_for_dedup(rewrite)
                if norm not in seen:
                    results.append(rewrite)
                    seen.add(norm)

        # Strategy 3: Keyword extraction (declarative form)
        if len(results) < max_expansions:
            keywords = self._extract_keywords(query)
            if keywords:
                norm = _normalize_for_dedup(keywords)
                if norm not in seen:
                    results.append(keywords)
                    seen.add(norm)

        return results[:max_expansions]

    def _synonym_substitute(self, query: str) -> str | None:
        """Replace the first substitutable word with a synonym."""
        words = query.split()
        for i, word in enumerate(words):
            clean = re.sub(r"[^\w]", "", word.lower())
            syns = _SYNONYMS_LOWER.get(clean)
            if syns:
                replacement = syns[0]
                # Preserve original casing style
                if word[0].isupper():
                    replacement = replacement.capitalize()
                new_words = list(words)
                # Preserve trailing punctuation
                trailing = ""
                if word and not word[-1].isalnum():
                    trailing = word[-1]
                new_words[i] = replacement + trailing
                return " ".join(new_words)
        return None

    def _question_rewrite(self, query: str) -> str | None:
        """Rewrite question-form queries into declarative form."""
        for pattern, replacement in _QUESTION_REWRITES:
            match = pattern.match(query)
            if match:
                result = pattern.sub(replacement, query)
                return result.strip()
        return None

    def _extract_keywords(self, query: str) -> str | None:
        """Extract content-bearing keywords, dropping function words."""
        stopwords = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "about",
            "against",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "each",
            "every",
            "all",
            "any",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "than",
            "too",
            "very",
            "just",
            "how",
            "what",
            "when",
            "where",
            "why",
            "who",
            "which",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "they",
            "them",
            "their",
        }
        words = re.findall(r"\b\w+\b", query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        if len(keywords) >= 2 and keywords != words:
            return " ".join(keywords)
        return None


# ---------------------------------------------------------------------------
# LLM-backed expander (optional, config-gated)
# ---------------------------------------------------------------------------


class LLMQueryExpander:
    """Generate alternative query phrasings using an LLM.

    Requires an API key and network access. Disabled by default.
    Falls back to NLP expansion on any failure.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.provider: str = cfg.get("provider", "anthropic")
        self.model: str = cfg.get("model", "claude-haiku")
        self.api_key_env: str = cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        self.base_url: str = cfg.get("base_url", "https://api.openai.com/v1")
        self._fallback = NLPQueryExpander()

    def expand(self, query: str, max_expansions: int = 3) -> list[str]:
        """Generate alternative phrasings via LLM, with NLP fallback.

        Args:
            query: Original search query.
            max_expansions: Maximum number of results including the original.

        Returns:
            List of query strings starting with the original.
        """
        if not query or not query.strip():
            return [query] if query else [""]

        query = query.strip()

        try:
            alternatives = self._call_llm(query, max_expansions - 1)
        except Exception as exc:
            _log.warning(
                "llm_expansion_failed",
                error=str(exc),
                fallback="nlp",
            )
            return self._fallback.expand(query, max_expansions)

        results: list[str] = [query]
        seen: set[str] = {_normalize_for_dedup(query)}
        for alt in alternatives:
            alt = alt.strip()
            if not alt:
                continue
            norm = _normalize_for_dedup(alt)
            if norm not in seen:
                results.append(alt)
                seen.add(norm)
            if len(results) >= max_expansions:
                break

        return results[:max_expansions]

    def _call_llm(self, query: str, n: int) -> list[str]:
        """Call the configured LLM to generate alternative phrasings.

        Returns a list of alternative query strings (not including the original).
        Raises on any failure (caller handles fallback).
        """
        import os as _os

        api_key = _os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"LLM expansion requires {self.api_key_env} environment variable")

        prompt = (
            f"Generate exactly {n} alternative phrasings of this search query. "
            f"Each alternative should use different words but preserve the same "
            f"search intent. Return only the alternatives, one per line, with no "
            f"numbering or extra text.\n\n"
            f"Query: {query}"
        )

        if self.provider == "anthropic":
            return self._call_anthropic(api_key, prompt)
        else:
            # All other providers use OpenAI-compatible chat completions API
            return self._call_openai_compatible(api_key, prompt)

    def _call_anthropic(self, api_key: str, prompt: str) -> list[str]:
        """Call Anthropic API for query expansion."""
        import json as _json
        import urllib.request

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        body = _json.dumps(
            {
                "model": self.model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))

        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        return [line.strip() for line in text.strip().splitlines() if line.strip()]

    def _call_openai_compatible(self, api_key: str, prompt: str) -> list[str]:
        """Call an OpenAI-compatible chat completions API for query expansion.

        Works with OpenAI, xAI, Mistral, DeepSeek, NVIDIA NIM, and any other
        provider exposing the ``/chat/completions`` endpoint.

        Args:
            api_key: Bearer token for the API.
            prompt: The expansion prompt.

        Returns:
            List of alternative query strings parsed from the response.

        Raises:
            RuntimeError: On network or parsing failures.
        """
        import json as _json
        import urllib.request

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = _json.dumps(
            {
                "model": self.model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))

        # Extract text from the first choice
        choices = data.get("choices", [])
        if not choices:
            return []
        message = choices[0].get("message", {})
        text = message.get("content", "")

        return [line.strip() for line in text.strip().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_for_dedup(text: str) -> str:
    """Normalize query text for deduplication comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Factory and convenience API
# ---------------------------------------------------------------------------


def create_expander(config: dict[str, Any] | None = None) -> QueryExpander:
    """Create a QueryExpander from configuration.

    Args:
        config: The ``query_expansion`` section of the recall config.
            When None or when ``llm.enabled`` is False, returns an
            NLP-based expander.

    Returns:
        A QueryExpander instance.
    """
    if config is None:
        return NLPQueryExpander()

    llm_cfg = config.get("llm", {})
    if isinstance(llm_cfg, dict) and llm_cfg.get("enabled", False):
        _log.info("using_llm_expander", provider=llm_cfg.get("provider", "anthropic"))
        return LLMQueryExpander(config=llm_cfg)

    return NLPQueryExpander()


def expand_queries(
    query: str,
    config: dict[str, Any] | None = None,
    max_expansions: int = 3,
) -> list[str]:
    """Expand a query into multiple alternative phrasings.

    Convenience function that creates an expander from config and runs it.

    Args:
        query: Original search query.
        config: The ``query_expansion`` section of the recall config.
        max_expansions: Maximum number of query variants to generate
            (including the original).

    Returns:
        List of query strings, starting with the original.
    """
    cfg = config or {}
    max_exp = int(cfg.get("max_expansions", max_expansions))
    expander = create_expander(cfg)
    return expander.expand(query, max_expansions=max_exp)
