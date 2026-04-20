"""HTTP bearer-token authentication helpers for the MCP surface.

Extracted from ``mcp_server.py`` in the v3.2.0 §1.2 decomposition
(see docs/v3.2.0-mcp-decomposition-plan.md PR-1). Provides:

* :func:`_check_token` — resolve the user token from
  ``MIND_MEM_TOKEN``, with ``None`` meaning "auth disabled".
* :func:`verify_token` — constant-time header verification for
  ``Authorization: Bearer`` and ``X-MindMem-Token``.
* :func:`_build_http_auth_tokens` — assemble the
  FastMCP ``StaticTokenVerifier`` map from the configured user
  and admin tokens.

Behavior is bit-for-bit identical to the pre-move version — the
metric name ``mcp_http_auth_failures`` is preserved, the two
header-name spellings (lower-case and Title-Case) are both still
tried, and both tokens use ``hmac.compare_digest`` to keep the
comparison constant-time.
"""

from __future__ import annotations

import hmac
import os
from typing import Any

from mind_mem.observability import metrics


def _check_token() -> str | None:
    """Get token from environment. Returns None if no auth configured."""
    return os.environ.get("MIND_MEM_TOKEN")


_MAX_TOKEN_LEN = 4096
# Recommended minimum token length for production deployments.
# Shorter tokens are accepted (for backward compatibility and testing)
# but a startup warning is emitted by check_token_strength().
RECOMMENDED_MIN_TOKEN_LEN = 32


def check_token_strength() -> list[str]:
    """Return a list of security warnings about the configured token.

    Call this once at server startup to surface weak-token warnings in logs.
    Returns an empty list when no issues are found.
    """
    warnings: list[str] = []
    token = _check_token()
    if token is not None and len(token) < RECOMMENDED_MIN_TOKEN_LEN:
        warnings.append(
            f"MIND_MEM_TOKEN is only {len(token)} characters; recommend ≥{RECOMMENDED_MIN_TOKEN_LEN} chars (e.g. openssl rand -hex 32)"
        )
    admin = os.environ.get("MIND_MEM_ADMIN_TOKEN")
    if admin is not None and len(admin) < RECOMMENDED_MIN_TOKEN_LEN:
        warnings.append(
            f"MIND_MEM_ADMIN_TOKEN is only {len(admin)} characters; "
            f"recommend ≥{RECOMMENDED_MIN_TOKEN_LEN} chars (e.g. openssl rand -hex 32)"
        )
    return warnings


def verify_token(headers: dict) -> bool:
    """Verify Bearer token from request headers. Constant-time compare.

    Returns True if:
      - No token is configured (open access), or
      - Token matches via Authorization: Bearer <token> or X-MindMem-Token header.

    Returns False if token is configured but missing/invalid.
    Tokens longer than _MAX_TOKEN_LEN are rejected to prevent DoS via
    oversized header values.
    """
    expected = _check_token()
    if expected is None:
        return True  # No auth configured — allow

    # Try Authorization: Bearer <token>
    auth = headers.get("authorization", headers.get("Authorization", ""))
    if auth.startswith("Bearer "):
        provided = auth[7:]
        if len(provided) <= _MAX_TOKEN_LEN and hmac.compare_digest(provided, expected):
            return True

    # Try X-MindMem-Token header
    alt = headers.get("x-mindmem-token", headers.get("X-MindMem-Token", ""))
    if alt and len(alt) <= _MAX_TOKEN_LEN and hmac.compare_digest(alt, expected):
        return True

    metrics.inc("mcp_http_auth_failures")
    return False


def _build_http_auth_tokens() -> dict[str, dict[str, Any]]:
    """Build StaticTokenVerifier token metadata from environment variables."""
    tokens: dict[str, dict[str, Any]] = {}

    user_token = _check_token()
    if user_token:
        tokens[user_token] = {
            "client_id": "mind-mem-user",
            "scopes": ["user"],
            "sub": "mind-mem-user",
        }

    admin_token = os.environ.get("MIND_MEM_ADMIN_TOKEN")
    if admin_token:
        tokens[admin_token] = {
            "client_id": "mind-mem-admin",
            "scopes": ["user", "admin"],
            "sub": "mind-mem-admin",
        }

    return tokens
