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
    """Get the user token, normalizing unset and empty to no credential."""
    return os.environ.get("MIND_MEM_TOKEN") or None


# Env var name for the explicit "I know there is no auth and I accept that
# only because I'm bound to loopback" opt-out. Set by the
# ``--allow-unauthenticated-localhost`` CLI flag on both ``mind-mem-mcp``
# and ``mm serve``; tests using the in-process FastAPI/MCP TestClient set
# it directly because TestClient does not bind a real port. Any other
# value (or absence) means HTTP/REST auth is fail-CLOSED. v3.7.0 H4.
ALLOW_UNAUTH_ENV = "MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST"


#: One dependency-light declaration of the authentication mechanisms used by
#: the REST startup and request gates. A tuple is a conjunction; the outer
#: tuple is a disjunction. Truthiness matches credential construction.
AUTH_ENV_GROUPS: tuple[tuple[str, ...], ...] = (
    ("MIND_MEM_TOKEN",),
    ("MIND_MEM_ADMIN_TOKEN",),
    ("MIND_MEM_API_KEY_DB",),
    ("OIDC_ISSUER", "OIDC_AUDIENCE"),
)


def auth_is_configured() -> bool:
    """Return whether at least one usable REST auth mechanism is configured."""
    return any(all(os.environ.get(var) for var in group) for group in AUTH_ENV_GROUPS)


def _other_auth_configured() -> bool:
    """True when a usable auth mechanism OTHER than ``MIND_MEM_TOKEN`` is set.

    Truthiness, not presence — matching ``rest._auth_is_configured``, so an
    exported-but-empty variable is not a credential in either place.
    """
    return any(all(os.environ.get(var) for var in group) for group in AUTH_ENV_GROUPS[1:])


def _unauthenticated_explicitly_allowed() -> bool:
    """Return True when the operator has opted into unauthenticated localhost.

    The opt-in is deliberately scoped — ``verify_token`` and
    ``_verify_bearer`` consult this when no token is configured; the
    CLI startup paths additionally enforce that binding is loopback-only
    before honouring it.
    """
    val = os.environ.get(ALLOW_UNAUTH_ENV, "").strip().lower()
    return val in ("1", "true", "yes", "on")


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


def verify_token(headers: dict, *, allow_unauthenticated: bool | None = None) -> bool:
    """Verify Bearer token from request headers. Constant-time compare.

    v3.7.0 H4: fail-CLOSED by default when no token is configured.
    Pre-v3.7.0 returned True in that case ("auth disabled — allow"),
    which left HTTP/REST exposed unauthenticated for any operator who
    forgot to set ``MIND_MEM_TOKEN``. The new contract:

    * Token configured + matching header → True
    * Token configured + missing/wrong header → False
    * No token configured + ``MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST=1``
      → True (operator opted in for loopback-only deployments / tests)
    * No token configured + opt-in absent → False (fail-closed)

    Tokens longer than ``_MAX_TOKEN_LEN`` are rejected before any
    compare to prevent DoS via oversized header values.

    ``allow_unauthenticated=None`` preserves the MCP helper's environment
    contract. REST always supplies an explicit app-scoped decision produced by
    its bind validator, so a direct ASGI import cannot gain access from the
    environment variable alone.
    """
    expected = _check_token()
    if expected is None:
        # v3.7.0 H4: refuse unauthenticated requests unless the operator
        # has explicitly opted in via the env var. The CLI binds the
        # server to loopback when the matching ``--allow-unauthenticated-
        # localhost`` flag is passed; tests set the env var directly
        # because the in-process TestClient skips network binding.
        # The opt-in asserts "there is NO auth and I accept that because I am on
        # loopback". If another mechanism IS configured that assertion is false,
        # and honouring it hands out anonymous access on a server whose operator
        # configured authentication.
        #
        # This was reachable, not theoretical: `rest._auth_is_configured` counts
        # MIND_MEM_ADMIN_TOKEN, so `_enforce_fail_closed` returned EARLY on it and
        # never reached its loopback check -- letting `--host 0.0.0.0` bind -- while
        # `_check_token` reads only MIND_MEM_TOKEN, so every request landed here and
        # was allowed. Measured on 28b9d7f1: verify_token({}) and a wrong bearer
        # both returned True.
        #
        # The admin token gains no reach from this: it does not become a user
        # credential, it only revokes the anonymous branch.
        anonymous_allowed = _unauthenticated_explicitly_allowed() if allow_unauthenticated is None else allow_unauthenticated
        if anonymous_allowed and not _other_auth_configured():
            return True
        metrics.inc("mcp_http_auth_failures")
        return False

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
