#!/usr/bin/env python3
"""Single source of truth for the ollama base URL.

Every ollama endpoint in mind-mem — embed (``recall_vector``, ``mm_cli``),
extraction (``llm_extractor`` ``/api/generate`` + ``/api/tags``), compression
(``compressors``) and rerank (``_recall_reranking`` / ``_recall_core``) —
resolves its base URL through :func:`ollama_base_url` so a fleet node can
point at a central ollama server instead of the historical hardcoded
``http://localhost:11434``.

Resolution precedence (first match wins):

    1. **Explicit config key** — ``config_section[config_key]``
       (``ollama_url`` by default, e.g. ``recall.ollama_url`` /
       ``extraction.ollama_url`` in ``mind-mem.json``). A workspace-level
       config value is the most specific operator intent, so it beats the
       ambient environment.
    2. **``OLLAMA_HOST`` env var** — ollama's own standard client/server
       env var. Accepts both ``host:port`` (``192.168.0.193:11434``) and a
       full URL (``http://192.168.0.193:11434``); ``host:port`` is
       normalized to ``http://host:port``.
    3. **Default** — ``http://localhost:11434``, byte-identical to the
       pre-4.3.1 hardcoded behavior, so existing installs are unaffected.

Security note: the resolved value is *operator-controlled deployment
config* (workspace ``mind-mem.json`` or the process environment), never
user/recall input — the same trust class as the pre-existing
``llm_rerank_url`` / ``MIND_MEM_VLLM_URL`` keys. The scheme is validated
to ``http``/``https`` here so ``urllib.request.urlopen`` at the call
sites can never be reached with ``file://`` or another transport.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

DEFAULT_OLLAMA_URL = "http://localhost:11434"

_ALLOWED_SCHEMES = ("http", "https")


def _normalize(raw: str, *, source: str) -> str:
    """Normalize an operator-supplied ollama endpoint to a base URL.

    Accepts ``host:port`` or a full ``http[s]://host[:port]`` URL; strips
    any trailing slash. Raises :class:`ValueError` on a non-http(s) scheme
    or a hostless value — a misconfiguration must fail loudly at the
    boundary, not surface later as a confusing connection error (or worse,
    let ``urlopen`` touch a non-HTTP transport).
    """
    value = raw.strip()
    if "://" not in value:
        value = f"http://{value}"
    value = value.rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"ollama base URL from {source} must use http or https, got {raw!r}")
    if not parsed.hostname:
        raise ValueError(f"ollama base URL from {source} has no host: {raw!r}")
    return value


def ollama_base_url(
    config_section: Mapping[str, Any] | None = None,
    *,
    config_key: str = "ollama_url",
) -> str:
    """Resolve the ollama base URL (no trailing slash, no path).

    Precedence: ``config_section[config_key]`` > ``OLLAMA_HOST`` env >
    :data:`DEFAULT_OLLAMA_URL`. See the module docstring for the full
    contract. Empty/whitespace-only and non-string values at a higher
    tier fall through to the next tier rather than erroring — only a
    *present, malformed* value raises :class:`ValueError`.

    Args:
        config_section: An already-loaded config mapping (e.g. the
            ``recall`` or ``extraction`` section of ``mind-mem.json``).
        config_key: Key holding the endpoint override in that section.

    Returns:
        Base URL such as ``http://192.168.0.193:11434`` — append the API
        path (``/api/embed``, ``/api/generate``, ``/api/tags``) at the
        call site.
    """
    if config_section is not None:
        configured = config_section.get(config_key)
        if isinstance(configured, str) and configured.strip():
            return _normalize(configured, source=f"config key {config_key!r}")
    env_value = os.environ.get("OLLAMA_HOST", "")
    if env_value.strip():
        return _normalize(env_value, source="OLLAMA_HOST")
    return DEFAULT_OLLAMA_URL
