"""Multi-query expansion for improved recall.

Generates 2-3 alternative phrasings of a query before searching, improving
recall by matching documents that use different terminology or phrasing.

Two expansion modes:
  - NLP-based (default): synonym substitution, query decomposition, and
    morphological variants using zero external dependencies.
  - LLM-backed (optional): calls whatever HTTP endpoint the operator
    configures. Disabled by default; enable via config. There is no
    built-in provider, endpoint or model — every one of those is
    configuration, so the code privileges no vendor and contacts no
    host the operator did not name.

Configuration (mind-mem.json):
    {
      "recall": {
        "query_expansion": {
          "enabled": true,
          "max_expansions": 3,
          "llm": {
            "enabled": false,
            "base_url": "https://your-endpoint.example/v1",
            "model": "your-model-id",
            "api_key_env": "YOUR_API_KEY_ENV_VAR",
            "provider": "whatever-you-call-it",
            "endpoint_path": "/chat/completions",
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
            "headers": {},
            "response_path": ["choices", 0, "message", "content"],
            "response_item_filter": {}
          }
        }
      }
    }

``base_url``, ``model`` and ``api_key_env`` are REQUIRED. With any of them
missing the expander logs ``llm_expander_unconfigured`` / raises an
actionable error and falls back to the NLP expander — it never guesses an
endpoint. ``provider`` is a free-form label used only in log lines.

The remaining keys describe the REQUEST SHAPE as data; their defaults (shown
above) describe the widely implemented chat-completions shape, and the code
has no per-endpoint branch. ``response_path`` walks the decoded JSON reply:
strings are dict keys, integers are list indices, and ``"*"`` iterates a list
and concatenates the text found at the remainder of the path.
``response_item_filter`` restricts which items of that list contribute (every
key/value pair must match). So an endpoint that answers with a list of typed
content parts, and authenticates with a bare key header plus a static version
header, is reachable purely by configuration:

    "endpoint_path": "<the path that endpoint documents>",
    "auth_header": "x-api-key",
    "auth_prefix": "",
    "headers": {"<header that endpoint requires>": "<value>"},
    "response_path": ["content", "*", "text"],
    "response_item_filter": {"type": "text"}
"""

from __future__ import annotations

import re
from collections.abc import Sequence
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
    """Generate alternative query phrasings using a configured HTTP endpoint.

    Requires ``base_url``, ``model`` and ``api_key_env`` in config plus
    network access. Disabled by default. Falls back to NLP expansion on any
    failure — including the "not configured" failure, so an operator who
    flips ``llm.enabled`` without supplying an endpoint gets a logged
    warning and working NLP expansion, never a request to a host the code
    picked on its own.

    The request shape is data (see the module docstring): endpoint path,
    auth header name, auth prefix, extra static headers and the response
    text location all come from config, so an endpoint whose shape differs
    from chat-completions needs configuration, not a code branch.
    """

    #: Defaults describing the widely implemented chat-completions shape.
    #: These are SHAPE defaults, not endpoint defaults — there is
    #: deliberately no default ``base_url``, ``model`` or ``api_key_env``.
    DEFAULT_ENDPOINT_PATH = "/chat/completions"
    DEFAULT_AUTH_HEADER = "Authorization"
    DEFAULT_AUTH_PREFIX = "Bearer "
    DEFAULT_RESPONSE_PATH: tuple[Any, ...] = ("choices", 0, "message", "content")

    #: Config keys with no default — absence is a configuration error.
    REQUIRED_KEYS = ("base_url", "model", "api_key_env")

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        # Free-form operator label; used only for log lines.
        self.provider: str = str(cfg.get("provider", "") or "")
        # No vendor default: an unset value is a configuration error that
        # surfaces at call time (see ``_require_configured``).
        self.model: str = str(cfg.get("model", "") or "")
        self.api_key_env: str = str(cfg.get("api_key_env", "") or "")
        self.base_url: str = str(cfg.get("base_url", "") or "")
        # Request-shape data.
        self.endpoint_path: str = str(cfg.get("endpoint_path", self.DEFAULT_ENDPOINT_PATH) or "")
        self.auth_header: str = str(cfg.get("auth_header", self.DEFAULT_AUTH_HEADER) or "")
        self.auth_prefix: str = str(cfg.get("auth_prefix", self.DEFAULT_AUTH_PREFIX))
        extra = cfg.get("headers", {})
        self.headers: dict[str, str] = {str(k): str(v) for k, v in extra.items()} if isinstance(extra, dict) else {}
        path = cfg.get("response_path", self.DEFAULT_RESPONSE_PATH)
        self.response_path: tuple[Any, ...] = tuple(path) if isinstance(path, (list, tuple)) else self.DEFAULT_RESPONSE_PATH
        item_filter = cfg.get("response_item_filter", {})
        self.response_item_filter: dict[str, Any] = dict(item_filter) if isinstance(item_filter, dict) else {}
        self._fallback = NLPQueryExpander()

    def missing_config(self) -> list[str]:
        """Return the required config keys that were not supplied."""
        return [key for key in self.REQUIRED_KEYS if not getattr(self, key)]

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

    def _require_configured(self) -> None:
        """Raise an actionable error when the endpoint was never configured.

        Deliberately NOT a fallback to some built-in host: the operator
        asked for LLM expansion without saying where, and the honest
        answer is to say which keys are missing.
        """
        missing = self.missing_config()
        if missing:
            keys = ", ".join(f"recall.query_expansion.llm.{key}" for key in missing)
            raise RuntimeError(
                f"LLM query expansion is not configured: set {keys} in mind-mem.json "
                f"(there is no built-in endpoint or model default)"
            )

    def _call_llm(self, query: str, n: int) -> list[str]:
        """Call the configured endpoint to generate alternative phrasings.

        Returns a list of alternative query strings (not including the original).
        Raises on any failure (caller handles fallback).
        """
        import json as _json
        import os as _os
        import urllib.request

        self._require_configured()

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

        base = self.base_url.rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise ValueError(f"LLMQueryExpander: invalid URL scheme for base_url: {base!r}")
        url = f"{base}{self.endpoint_path}"

        headers = {"Content-Type": "application/json"}
        headers.update(self.headers)
        headers[self.auth_header] = f"{self.auth_prefix}{api_key}"

        body = _json.dumps(
            {
                "model": self.model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 — scheme validated above to http/https only
            data = _json.loads(resp.read().decode("utf-8"))

        text = _extract_response_text(data, self.response_path, self.response_item_filter)
        return [line.strip() for line in text.strip().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_for_dedup(text: str) -> str:
    """Normalize query text for deduplication comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_response_text(
    data: Any,
    path: Sequence[Any],
    item_filter: dict[str, Any],
) -> str:
    """Pull the generated text out of a decoded JSON response by data path.

    This is what replaces the per-endpoint response parsers. ``path``
    elements are dict keys (str), list indices (int), or the wildcard
    ``"*"``, which iterates the list at that position, keeps the items
    matching every key/value pair in ``item_filter``, resolves the rest of
    the path against each, and concatenates the strings.

    Returns "" for any path that does not resolve — a malformed or empty
    response yields no alternatives rather than an exception, matching the
    previous per-endpoint parsers.
    """
    node: Any = data
    for idx, step in enumerate(path):
        if step == "*":
            if not isinstance(node, list):
                return ""
            rest = tuple(path)[idx + 1 :]
            parts: list[str] = []
            for item in node:
                if item_filter:
                    if not isinstance(item, dict):
                        continue
                    if any(item.get(key) != value for key, value in item_filter.items()):
                        continue
                piece = _extract_response_text(item, rest, {})
                if piece:
                    parts.append(piece)
            return "".join(parts)
        if isinstance(step, bool):
            return ""
        if isinstance(step, int):
            if not isinstance(node, list) or not -len(node) <= step < len(node):
                return ""
            node = node[step]
            continue
        if isinstance(node, dict) and step in node:
            node = node[step]
            continue
        return ""
    return node if isinstance(node, str) else ""


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
        expander = LLMQueryExpander(config=llm_cfg)
        missing = expander.missing_config()
        if missing:
            # Say it once, at wiring time, with the keys to add — rather
            # than only on the first query, and never by substituting an
            # endpoint the operator did not choose.
            _log.warning(
                "llm_expander_unconfigured",
                missing=missing,
                fallback="nlp",
                hint="set recall.query_expansion.llm.{" + ",".join(missing) + "} in mind-mem.json",
            )
        else:
            _log.info("using_llm_expander", provider=expander.provider or "unspecified")
        return expander

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
