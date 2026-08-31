"""REST API layer for mind-mem (v3.2.0, v3.2.1 hardening).

FastAPI application that mirrors the MCP tool surface with:
- Pydantic-validated request models
- Bearer-token auth via MIND_MEM_TOKEN / MIND_MEM_ADMIN_TOKEN env vars
- OIDC JWT auth (v3.2.1) — JWT scopes drive the admin gate
- Per-client sliding-window rate limiting
- Per-request workspace scoping via ContextVar (v3.2.1 — replaces
  earlier ``os.environ`` mutation which raced under concurrent load)
- Optional Prometheus exposition at /v1/metrics
- OpenAPI docs at /openapi.json (FastAPI default)

Launch via: mm serve [--port 8080] [--host 127.0.0.1]
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from contextvars import ContextVar
from typing import Annotated, Any, cast

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, status
    from fastapi.responses import JSONResponse, PlainTextResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field, field_validator
except ImportError as _err:  # pragma: no cover
    raise ImportError("mind-mem REST API requires the 'api' extra: pip install 'mind-mem[api]'") from _err

from mind_mem import __version__ as _PACKAGE_VERSION
from mind_mem.mcp.infra.constants import MCP_SCHEMA_VERSION
from mind_mem.mcp.infra.http_auth import (
    ALLOW_UNAUTH_ENV,
    _check_token,
    verify_token,
)
from mind_mem.mcp.infra.rate_limit import SlidingWindowRateLimiter, _get_client_rate_limiter
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.schema_version import CURRENT_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Agent-ID context variable (set by auth dependencies, read by audit chain)
# ---------------------------------------------------------------------------

#: The current request's authenticated agent ID.  Defaults to "anonymous".
#: Set this from MCP tool entry points to propagate identity through the stack.
#:
#: NOTE: ContextVar writes inside sync FastAPI dependencies run in
#: threadpool workers and do not propagate back to the calling request
#: context. Use :data:`fastapi.Request.state` when cross-dependency
#: state is required (see ``_require_auth``'s ``request.state.oidc_scopes``
#: handoff). ``current_agent_id`` is kept for audit-chain callers that
#: read it within the same sync dependency frame.
current_agent_id: ContextVar[str] = ContextVar("current_agent_id", default="anonymous")


def _oidc_admin_scope_names() -> tuple[str, ...]:
    """Scope names that grant admin access via OIDC.

    Configured via ``MIND_MEM_OIDC_ADMIN_SCOPES`` env var (comma- or
    space-separated). Defaults to ``"mind-mem.admin admin"`` so common
    OIDC IdP conventions (Okta custom scopes, Auth0 roles) work without
    extra configuration.
    """
    raw = os.environ.get("MIND_MEM_OIDC_ADMIN_SCOPES", "mind-mem.admin admin")
    parts = [s.strip() for s in raw.replace(",", " ").split() if s.strip()]
    return tuple(parts)


#: Process-wide cache of OIDC providers, keyed by issuer configuration.
#: Bounded, and holds only configuration-derived objects — nothing whose
#: owner can go away — so it cannot pin per-request or per-thread state.
_OIDC_PROVIDER_CACHE: dict[tuple[str, str, str, str], Any] = {}
_OIDC_PROVIDER_LOCK = threading.Lock()
_OIDC_PROVIDER_CACHE_MAX = 8


def _oidc_provider(issuer: str, client_id: str, client_secret: str, audience: str) -> Any:
    """Return the process-cached ``OIDCProvider`` for this configuration.

    ``OIDCProvider`` caches the issuer's JWKS on the *instance*
    (``self._jwks``) and its docstring promises that cache lives "for the
    lifetime of the process" — but both call sites built a fresh provider
    per call, so the cache was per-request and every authenticated
    request made its own blocking 10-second-timeout HTTPS fetch to the
    IdP. When the IdP then throttled, verification failed and the caller
    saw a silent 401.

    Keyed on the four configuration values so rotating any of them yields
    a new provider (never a stale JWKS) rather than requiring a restart.
    Past ``_OIDC_PROVIDER_CACHE_MAX`` distinct configurations the cache is
    dropped wholesale: a deployment has one configuration, and a bound is
    cheaper than an eviction policy nobody will tune.
    """
    from mind_mem.api.auth import OIDCConfig, OIDCProvider  # noqa: PLC0415

    key = (issuer, client_id, client_secret, audience)
    with _OIDC_PROVIDER_LOCK:
        cached = _OIDC_PROVIDER_CACHE.get(key)
        if cached is not None:
            return cached
        if len(_OIDC_PROVIDER_CACHE) >= _OIDC_PROVIDER_CACHE_MAX:
            _OIDC_PROVIDER_CACHE.clear()
        provider = OIDCProvider(OIDCConfig(issuer=issuer, client_id=client_id, client_secret=client_secret, audience=audience))
        _OIDC_PROVIDER_CACHE[key] = provider
        return provider


def _verify_oidc_token(token: str) -> tuple[str, tuple[str, ...]] | None:
    """Validate *token* as an OIDC JWT. Returns (agent_id, scopes) or None.

    Returns None when OIDC is not configured, the token isn't a JWT,
    or validation fails. Never raises — falls back cleanly so static
    token auth keeps working in deployments without OIDC.
    """
    issuer = os.environ.get("OIDC_ISSUER")
    audience = os.environ.get("OIDC_AUDIENCE", "")
    if not issuer or not audience:
        return None
    if token.count(".") != 2:
        # Fast reject: JWTs have exactly two dots. Avoids the cost of
        # dragging in python-jose for every bearer token.
        return None
    try:
        from mind_mem.api.auth import AuthError  # noqa: PLC0415
    except ImportError:
        return None

    client_id = os.environ.get("OIDC_CLIENT_ID", "")
    client_secret = os.environ.get("OIDC_CLIENT_SECRET", "")
    provider = _oidc_provider(issuer, client_id, client_secret, audience)
    try:
        claims = provider.verify(token)
    except AuthError:
        return None
    except Exception:
        return None
    scopes = tuple(provider.extract_scopes(claims))
    agent_id = str(claims.get("sub") or claims.get("email") or "oidc-user")
    return agent_id, scopes


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


def _extract_bearer(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> str | None:
    """Return raw token string from Authorization: Bearer <token>, or None."""
    if credentials is not None:
        return str(credentials.credentials)
    return None


def _client_id_from_token(token: str | None) -> str:
    """Derive a stable per-client identifier from the bearer token.

    N-12: a truncated token is not a "non-sensitive" key. The old
    ``token[-16:]`` put a suffix of live credential material into the
    rate limiter's dict keys, and from there into every log line, metric
    label and 429 diagnostic that echoes a bucket — and for a token
    shorter than 16 characters it returned the whole secret verbatim. A
    digest identifies a client exactly as well and carries none of it.
    """
    if token is None:
        return "anonymous"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _verify_bearer(token: str | None) -> tuple[bool, str, tuple[str, ...]]:
    """Return (valid, agent_id, oidc_scopes) for a bearer/API-key token.

    ``oidc_scopes`` is empty unless the token was validated as an OIDC
    JWT. Callers hand the scopes to :func:`_has_admin_scope` and/or
    stash them on ``request.state`` for cross-dependency reads.

    Priority:
    1. MIND_MEM_ADMIN_TOKEN match → agent_id "admin"
    2. ``mmk_live_*`` or ``mmk_test_*`` → look up in APIKeyStore
    3. OIDC JWT (when OIDC_ISSUER + OIDC_AUDIENCE configured) →
       agent_id from ``sub``/``email`` claim, scopes extracted from
       ``scope``/``scopes``/``roles`` claims.
    4. MIND_MEM_TOKEN match → agent_id "user"
    5. No auth configured → fail CLOSED (v3.7.0 H4); operator must opt
       in via ``MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST=1`` for tests
       or loopback-only deployments. Pre-v3.7.0 anonymous access is
       no longer the default.
    """
    if token is None:
        headers: dict[str, str] = {}
        return verify_token(headers), "anonymous", ()

    # 1. Admin token — fast path (checks env directly, not ContextVar)
    admin = os.environ.get("MIND_MEM_ADMIN_TOKEN")
    if admin is not None:
        import hmac

        if hmac.compare_digest(token, admin):
            return True, "admin", ()

    # 2. API key (mmk_live_* / mmk_test_*)
    if token.startswith("mmk_live_") or token.startswith("mmk_test_"):
        record = _resolve_api_key_store_record(token)
        if record is not None:
            return True, record["agent_id"], ()
        return False, "anonymous", ()

    # 3. OIDC JWT (v3.2.1): validate and return scopes for admin-gate eval.
    oidc_configured = os.environ.get("OIDC_ISSUER") is not None and os.environ.get("OIDC_AUDIENCE") is not None
    oidc = _verify_oidc_token(token)
    if oidc is not None:
        agent_id, scopes = oidc
        return True, agent_id, scopes
    if oidc_configured and token.count(".") == 2:
        # OIDC is configured and the token looks like a JWT (two dots)
        # but validation failed. Reject — don't silently fall through
        # to the static-token path which would accept any token when
        # MIND_MEM_TOKEN is unset.
        return False, "anonymous", ()

    # 4. User bearer token
    headers = {"Authorization": f"Bearer {token}"}
    if verify_token(headers):
        return True, "user", ()
    return False, "anonymous", ()


def _resolve_api_key_store_record(raw_key: str) -> dict | None:
    """Verify an mmk_* API key and return its record, or None.

    Propagates :class:`APIKeyStoreUnavailable` rather than folding a
    broken store into "no such key": a configured-but-unopenable store
    is an operator problem (503), not a bad credential (401).
    """
    store = _get_api_key_store()
    if store is None:
        return None
    return cast(dict[str, Any] | None, store.verify(raw_key))


class APIKeyStoreUnavailable(RuntimeError):
    """``MIND_MEM_API_KEY_DB`` is set but the store could not be opened."""


#: Process-wide cache of API-key stores, keyed by (db path, production).
#: Bounded and configuration-derived; holds no per-request state.
_API_KEY_STORE_CACHE: dict[tuple[str, bool], Any] = {}
_API_KEY_STORE_LOCK = threading.Lock()
_API_KEY_STORE_CACHE_MAX = 8


def _get_api_key_store() -> Any:
    """Return the process-cached ``APIKeyStore``, or ``None`` if unconfigured.

    ``None`` now means exactly one thing: ``MIND_MEM_API_KEY_DB`` is
    unset. It used to mean that *or* "the store is configured but failed
    to open", because a bare ``except Exception`` returned ``None`` and
    recorded nothing. ``APIKeyStore.__init__`` does real I/O —
    ``os.makedirs`` plus a connect and ``CREATE TABLE`` — so an
    unwritable parent directory silently 401'd every ``mmk_*`` key and
    made ``GET /v1/admin/api_keys`` answer 501 "API key store not
    configured (set MIND_MEM_API_KEY_DB)" while the variable held
    exactly that path, pointing the operator away from the real cause.
    Construction failure now raises :class:`APIKeyStoreUnavailable` and
    is logged with the failing path and the exception type.

    Also cached: the store was rebuilt (and a sqlite file opened) on
    every key-authenticated request.

    Raises:
        APIKeyStoreUnavailable: the path is configured but unusable.
    """
    db_path = os.environ.get("MIND_MEM_API_KEY_DB")
    if not db_path:
        return None
    production = os.environ.get("MIND_MEM_ENV", "production") == "production"
    key = (db_path, production)
    with _API_KEY_STORE_LOCK:
        cached = _API_KEY_STORE_CACHE.get(key)
        if cached is not None:
            return cached
    try:
        from mind_mem.api.api_keys import APIKeyStore  # noqa: PLC0415

        store = APIKeyStore(db_path, production=production)
    except Exception as exc:
        from mind_mem.observability import get_logger as _get_logger  # noqa: PLC0415

        _get_logger("rest").error(
            "api_key_store_unavailable",
            path=db_path,
            error=type(exc).__name__,
        )
        raise APIKeyStoreUnavailable(f"cannot open API key store at {db_path!r} ({type(exc).__name__})") from exc
    with _API_KEY_STORE_LOCK:
        if len(_API_KEY_STORE_CACHE) >= _API_KEY_STORE_CACHE_MAX:
            _API_KEY_STORE_CACHE.clear()
        _API_KEY_STORE_CACHE[key] = store
    return store


def _require_api_key_store() -> Any:
    """Admin-route helper: the store, or an HTTP error naming the real cause.

    Splits the two states ``_get_api_key_store`` used to conflate:
    503 when the configured store cannot be opened (with the path and
    exception type — these routes are already behind the admin gate),
    501 only when nothing is configured at all.
    """
    try:
        store = _get_api_key_store()
    except APIKeyStoreUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="API key store not configured (set MIND_MEM_API_KEY_DB)",
        )
    return store


def _has_admin_scope(token: str | None, oidc_scopes: tuple[str, ...] = ()) -> bool:
    """Return True when the token grants admin access.

    Admin is granted by either:
    * Static ``MIND_MEM_ADMIN_TOKEN`` env var match (constant-time compare).
    * An OIDC JWT that carries one of the configured admin scopes
      (see :func:`_oidc_admin_scope_names`). Scopes must be passed in
      by the caller (read off ``request.state.oidc_scopes`` which
      :func:`_require_auth` populates after validating the JWT).
    """
    if token is None:
        return False

    # Static admin token match
    admin = os.environ.get("MIND_MEM_ADMIN_TOKEN")
    if admin is not None:
        import hmac

        if hmac.compare_digest(token, admin):
            return True

    # OIDC scope match — empty tuple is a cheap no-op for non-JWT tokens.
    if oidc_scopes:
        admin_scopes = _oidc_admin_scope_names()
        if any(s in admin_scopes for s in oidc_scopes):
            return True
    return False


def _api_key_has_admin_scope(token: str | None) -> bool:
    """Return True when an mmk_* key carries the admin scope."""
    if token is None or not (token.startswith("mmk_live_") or token.startswith("mmk_test_")):
        return False
    record = _resolve_api_key_store_record(token)
    if record is None:
        return False
    scopes: list[str] = record.get("scopes", [])
    # One definition, no fallback: mind_mem.scopes has no dependencies, so a
    # REST-only install resolves it without the MCP extra. A duplicated literal
    # here would silently diverge the day either copy changes.
    from ..scopes import ADMIN_SCOPES

    return bool(set(scopes) & ADMIN_SCOPES)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def _require_auth(
    request: Request,
    token: Annotated[str | None, Depends(_extract_bearer)],
) -> str | None:
    """Dependency: require valid bearer/API-key token when auth is configured.

    Sets the ``current_agent_id`` contextvar so audit records carry the
    authenticated identity. Also stashes any OIDC scopes from the
    validated JWT on ``request.state.oidc_scopes`` for
    :func:`_require_admin` to consult — ContextVar writes inside sync
    deps don't propagate back to the request context, so request.state
    is the authoritative handoff.
    """
    try:
        valid, agent_id, oidc_scopes = _verify_bearer(token)
    except APIKeyStoreUnavailable as exc:
        # The credential may well be valid; we cannot tell. 401 would
        # blame the caller for an operator-side failure (and send them
        # off rotating a working key).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key store unavailable",
        ) from exc
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    current_agent_id.set(agent_id)
    request.state.oidc_scopes = oidc_scopes
    return token


def _admin_gate_is_configured() -> bool:
    """Return True when :func:`_require_admin` must enforce admin scope.

    Deliberately composed from :func:`_auth_is_configured` — the
    predicate the startup bind gate uses — rather than restating the
    list of mechanisms. The two *were* written out separately and
    drifted: the bind gate counted ``MIND_MEM_API_KEY_DB`` and this one
    did not, so an API-key-only deployment started as "authenticated,
    bind allowed" while every admin endpoint ran with its scope check
    skipped. Any valid low-privilege ``mmk_*`` key could then mint
    itself an admin key through ``POST /v1/admin/api_keys``. Composing
    the predicates makes that class of drift impossible: a mechanism
    added to the bind gate can never go missing here.

    The extra ``is not None`` terms are a fail-closed superset covering
    present-but-empty misconfiguration (e.g. ``OIDC_ISSUER=""``), which
    the bind gate reads as "no auth" but which must still resolve the
    admin gate *closed*.
    """
    if _auth_is_configured():
        return True
    return (
        _check_token() is not None
        or os.environ.get("MIND_MEM_ADMIN_TOKEN") is not None
        or os.environ.get("MIND_MEM_API_KEY_DB") is not None
        or (os.environ.get("OIDC_ISSUER") is not None and os.environ.get("OIDC_AUDIENCE") is not None)
    )


def _require_admin(
    request: Request,
    token: Annotated[str | None, Depends(_require_auth)],
) -> str | None:
    """Dependency: require admin-scope token (bearer, mmk_*, or OIDC JWT).

    The gate is enforced whenever *any* auth mechanism is configured —
    ``MIND_MEM_TOKEN`` (user-tier), ``MIND_MEM_ADMIN_TOKEN`` (static
    admin), ``MIND_MEM_API_KEY_DB`` (per-agent ``mmk_*`` keys), or OIDC
    (``OIDC_ISSUER`` + ``OIDC_AUDIENCE``) — see
    :func:`_admin_gate_is_configured`. Using ``_check_token()`` alone
    (MIND_MEM_TOKEN) was wrong: a deployment that skips MIND_MEM_TOKEN
    but sets MIND_MEM_ADMIN_TOKEN would silently bypass the admin
    check. Same bug applied pre-v3.2.1 for OIDC-only deployments, and
    again for API-key-only deployments until the two configuration
    predicates were composed instead of duplicated.
    """
    if _admin_gate_is_configured():
        oidc_scopes: tuple[str, ...] = getattr(request.state, "oidc_scopes", ())
        is_admin = _has_admin_scope(token, oidc_scopes=oidc_scopes) or _api_key_has_admin_scope(token)
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin scope required",
            )
    return token


def _rate_limit_bucket(request: Request, token: str | None) -> str:
    """Pick the sliding-window bucket key for this request.

    A token identifies a client better than an address does (one client
    behind a proxy still gets its own bucket), so it wins when present.
    With no token there is nothing client-specific in
    :func:`_client_id_from_token` — it returns the constant
    ``"anonymous"`` — so the *address* is the only per-client signal left
    and must be used instead of collapsing every caller into one bucket.

    The old expression, ``_client_id_from_token(token) or
    request.client.host if request.client else "unknown"``, parsed as
    ``(_client_id_from_token(token) or request.client.host) if
    request.client else "unknown"``: `or` binds tighter than the
    conditional, and its left arm is never falsy, so the host branch was
    unreachable. Under
    ``MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST=1`` every caller shared
    the ``"anonymous"`` bucket, and one client's burst returned 429 to
    all the others — the starvation per-client buckets exist to prevent.
    """
    if token is not None:
        return _client_id_from_token(token)
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(bucket: str) -> None:
    """Raise 429 when *bucket* is over its sliding window."""
    limiter: SlidingWindowRateLimiter = _get_client_rate_limiter(bucket)
    allowed, retry_after = limiter.allow()
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(retry_after + 1))},
        )


def _rate_limit(request: Request, token: Annotated[str | None, Depends(_require_auth)]) -> None:
    """Dependency: enforce per-client sliding-window rate limit (auth required)."""
    _enforce_rate_limit(_rate_limit_bucket(request, token))


def _public_rate_limit(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> None:
    """Rate limit for routes that must stay reachable without a token.

    Same sliding window and same bucket rule as :func:`_rate_limit`, but
    it does not depend on :func:`_require_auth`, so an unauthenticated
    caller gets a 429 rather than a 401. Used by the liveness endpoint
    and by the OIDC callback, which is how a caller *becomes*
    authenticated and therefore cannot require a validated token.
    """
    token = str(credentials.credentials) if credentials is not None else None
    _enforce_rate_limit(_rate_limit_bucket(request, token))


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

_BACKEND_CHOICES = ("auto", "bm25", "hybrid")


class CreateAPIKeyRequest(BaseModel):
    """Request body for POST /v1/admin/api_keys."""

    agent_id: str = Field(..., min_length=1, max_length=128, description="Agent identifier")
    scopes: list[str] = Field(default_factory=list, description="Access scopes")
    expires_in_days: int = Field(90, ge=1, le=3650, description="Key validity in days")


class RecallRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8192, description="Search query")
    limit: int = Field(10, ge=1, le=200, description="Maximum number of results")
    active_only: bool = Field(False, description="Only return active (non-superseded) blocks")
    backend: str = Field("auto", description="Retrieval backend: auto | bm25 | hybrid")
    scoring_instant: str = Field(
        "",
        max_length=10,
        description=(
            "UTC date (YYYY-MM-DD) the recency layer scores against. Recall is deterministic "
            "given (corpus, config, scoring_instant); pass back the instant a previous run's "
            "attestation recorded to replay it. Empty means today in UTC."
        ),
    )

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        if v not in _BACKEND_CHOICES:
            raise ValueError(f"backend must be one of {_BACKEND_CHOICES}")
        return v


class ProposeUpdateRequest(BaseModel):
    block_type: str = Field(..., description="Block type: decision | task")
    statement: str = Field(..., min_length=1, max_length=500, description="The proposal statement")
    rationale: str = Field("", max_length=2000, description="Rationale for the proposal")
    tags: str = Field("", max_length=500, description="Comma-separated tags")
    confidence: str = Field("medium", description="Confidence level: low | medium | high")

    @field_validator("block_type")
    @classmethod
    def _validate_block_type(cls, v: str) -> str:
        if v not in ("decision", "task"):
            raise ValueError("block_type must be 'decision' or 'task'")
        return v

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: str) -> str:
        if v not in ("low", "medium", "high"):
            raise ValueError("confidence must be 'low', 'medium', or 'high'")
        return v


class ApproveApplyRequest(BaseModel):
    proposal_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^P-\d{8}-\d{3}$",
        description="Proposal ID in format P-YYYYMMDD-NNN",
    )
    dry_run: bool = Field(True, description="Dry-run (True) or commit (False)")


class RollbackProposalRequest(BaseModel):
    receipt_ts: str = Field(
        ...,
        min_length=15,
        max_length=15,
        pattern=r"^\d{8}-\d{6}$",
        description="Receipt timestamp in format YYYYMMDD-HHMMSS",
    )
    reason: str = Field(
        ...,
        min_length=8,
        max_length=2000,
        description="Why this rollback is being performed. Required for audit trail (issue #510 / N-02).",
    )


# ---------------------------------------------------------------------------
# Workspace helper
# ---------------------------------------------------------------------------


def _active_workspace(workspace: str | None) -> str:
    if workspace:
        return os.path.abspath(workspace)
    return os.environ.get("MIND_MEM_WORKSPACE", os.getcwd())


def _set_workspace_env(workspace: str) -> None:
    """Export workspace so MCP tool functions resolve it.

    .. deprecated:: 3.2.1
        Kept for compatibility with existing callers. New code should
        wrap request handlers in :func:`mind_mem.mcp.infra.workspace.use_workspace`
        to avoid mutating process-global state. A FastAPI middleware in
        :func:`create_app` now sets a per-request ``ContextVar`` override
        which takes precedence over this env var.
    """
    os.environ["MIND_MEM_WORKSPACE"] = workspace


_SENSITIVE_LOG_KEYS = frozenset({"log", "traceback", "stack_trace", "exception"})


def _strip_sensitive_fields(data: Any) -> Any:
    """Recursively remove internal log/traceback fields before returning to clients.

    Prevents stack-trace-exposure (CodeQL py/stack-trace-exposure) by ensuring
    that fields carrying server-side diagnostic text never reach the wire.
    """
    if isinstance(data, dict):
        return {k: _strip_sensitive_fields(v) for k, v in data.items() if k not in _SENSITIVE_LOG_KEYS}
    if isinstance(data, list):
        return [_strip_sensitive_fields(item) for item in data]
    return data


def _parse_tool_json(raw: str) -> Any:
    """Parse JSON string returned by MCP tool functions.

    Strips internal diagnostic fields (log, traceback, etc.) so that
    server-side stack traces are never forwarded to REST callers.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"raw": raw}
    return _strip_sensitive_fields(parsed)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


# These four live above ``create_app`` rather than beside the launcher
# they otherwise belong to: the module ends with a module-level
# ``app = create_app()``, so anything the constructor consults must
# already be bound by the time that line executes.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _auth_is_configured() -> bool:
    """Return True when at least one authentication mechanism is *usable*.

    Every test here is a truthiness test, matching the code that turns
    these variables into credentials: ``http_auth._build_http_auth_tokens``
    registers a token only ``if user_token:`` / ``if admin_token:``, and
    ``verify_token`` treats an empty ``MIND_MEM_TOKEN`` as no token at
    all.

    This gate used to test *presence* (``is not None``) for the two
    static tokens, so ``MIND_MEM_ADMIN_TOKEN=""`` — exported but empty —
    certified a routable bind as authenticated while no credential
    existed: :func:`_enforce_fail_closed` returned without raising and
    every subsequent request 401'd, because the constant-time compare
    could only match an empty bearer that ``HTTPBearer`` never produces.
    A gate must not certify a property it did not check, whichever
    direction the mismatch happens to fail in today.

    Note that :func:`_admin_gate_is_configured` deliberately keeps its
    extra ``is not None`` terms: that one is a fail-closed *superset*,
    so present-but-empty must still resolve the admin gate closed.
    """
    if _check_token():
        return True
    if os.environ.get("MIND_MEM_ADMIN_TOKEN"):
        return True
    if os.environ.get("MIND_MEM_API_KEY_DB"):
        return True
    if os.environ.get("OIDC_ISSUER") and os.environ.get("OIDC_AUDIENCE"):
        return True
    return False


BIND_HOST_ENV = "MIND_MEM_BIND_HOST"


def _docs_enabled(host: str | None = None) -> bool:
    """Whether ``/docs``, ``/redoc`` and ``/openapi.json`` are served.

    N-13: the schema enumerates every route, including the admin ones,
    to any unauthenticated caller. On a loopback bind that is a
    development convenience worth keeping. On a routable bind, once the
    server is authenticated, it is free reconnaissance — so the docs go
    away and the API keeps working.

    ``MIND_MEM_API_DOCS=on|off`` overrides in both directions. With no
    known bind (``create_app`` called directly by a test or an ASGI
    factory) the docs stay on: there is nothing to judge, and silently
    dropping the schema would be a worse surprise than serving it.
    """
    override = os.environ.get("MIND_MEM_API_DOCS", "").strip().lower()
    if override in ("1", "true", "on", "yes"):
        return True
    if override in ("0", "false", "off", "no"):
        return False
    if host is None:
        host = os.environ.get(BIND_HOST_ENV, "").strip()
    if not host:
        return True
    if host.strip("[]").lower() in _LOOPBACK_HOSTS:
        return True
    return not _auth_is_configured()


def create_app(workspace: str | None = None) -> FastAPI:
    """Create and return the configured FastAPI application.

    Parameters
    ----------
    workspace:
        Absolute path to the mind-mem workspace.  When *None* the
        ``MIND_MEM_WORKSPACE`` environment variable (or cwd) is used.
    """
    resolved_ws = _active_workspace(workspace)

    # Persist into env so tool functions see it even if called without context
    _set_workspace_env(resolved_ws)

    application = FastAPI(
        title="mind-mem REST API",
        description="REST API that mirrors the mind-mem MCP tool surface.",
        # The package version, not a hand-maintained constant: three
        # different versions used to be advertised at once (this string,
        # the health endpoint's "3.2.0", and the real package version),
        # so a client capability-gating on either number negotiated
        # against a value frozen years before the code it was talking to.
        version=_PACKAGE_VERSION,
        # N-13: gated — see _docs_enabled().
        docs_url="/docs" if _docs_enabled() else None,
        redoc_url="/redoc" if _docs_enabled() else None,
        openapi_url="/openapi.json" if _docs_enabled() else None,
    )

    # ------------------------------------------------------------------
    # Per-request workspace scoping (v3.2.1)
    # ------------------------------------------------------------------
    #
    # Wrap every request in ``use_workspace`` so tool calls read the
    # workspace from a request-local ``ContextVar`` instead of mutating
    # ``os.environ`` on every handler. ContextVar is task-local under
    # asyncio and propagates through Starlette's thread pool for sync
    # handlers, so concurrent requests can no longer race on workspace
    # state (v3.2.0 audit finding → v3.2.1).
    @application.middleware("http")
    async def _scope_workspace(request: Request, call_next):  # type: ignore[no-untyped-def]
        with use_workspace(resolved_ws):
            return await call_next(request)

    # Audit-header propagation (roadmap v4.0.0 Group D). Each request
    # carries a server-assigned ``X-MindMem-Request-Id`` (UUID-4) for
    # log correlation; the client may set ``X-MindMem-Actor`` and
    # ``X-MindMem-Purpose`` to identify the calling agent + intent.
    # All three are echoed on the response so a downstream proxy /
    # SIEM can stitch traces without parsing the body. The values are
    # sanitised against CRLF / control-char injection (same pattern as
    # ``v4.federation._safe`` after alert #192) before they touch the
    # log or response headers.
    @application.middleware("http")
    async def _audit_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        import re as _re
        import uuid as _uuid

        _CTRL = _re.compile(r"[\x00-\x1f\x7f]")

        def _safe_hdr(raw: str | None, *, default: str = "", max_len: int = 256) -> str:
            """Return *raw* with CR/LF/NUL/control bytes stripped, bounded length.

            CodeQL-friendly: leads with explicit ``.replace`` so the stock
            ``py/log-injection`` and ``py/header-injection`` queries
            recognise this as a sanitiser node.
            """
            if not raw:
                return default
            cleaned = raw.replace("\r", "").replace("\n", "")
            cleaned = _CTRL.sub("", cleaned)
            return cleaned[:max_len]

        # Capture raw presence-vs-absence on the request: an absent
        # client header should NOT echo a synthetic "anonymous" string
        # on the response (operators read header absence as
        # "unattributed", not as a literal value).
        raw_actor = request.headers.get("x-mindmem-actor")
        raw_purpose = request.headers.get("x-mindmem-purpose")

        request_id = _safe_hdr(
            request.headers.get("x-mindmem-request-id"),
            default=str(_uuid.uuid4()),
            max_len=64,
        )
        actor = _safe_hdr(raw_actor, default="anonymous")
        purpose = _safe_hdr(raw_purpose, default="")
        # Stash on request.state for any downstream handler that wants
        # to record actor/purpose in the audit chain. ``actor`` defaults
        # to "anonymous" here so the chain has a stable string to record;
        # the response-echo path below uses the raw presence check.
        request.state.mindmem_request_id = request_id
        request.state.mindmem_actor = actor
        request.state.mindmem_purpose = purpose
        response = await call_next(request)
        # Echo on the response — operators stitch upstream logs by these.
        response.headers["X-MindMem-Request-Id"] = request_id
        if raw_actor and actor:
            response.headers["X-MindMem-Actor"] = actor
        if raw_purpose and purpose:
            response.headers["X-MindMem-Purpose"] = purpose
        return response

    # ------------------------------------------------------------------
    # Global exception handler — prevent stack-trace-exposure
    # (CodeQL py/stack-trace-exposure): unhandled exceptions must never
    # leak server-side tracebacks to REST callers.  FastAPI's default
    # 500 handler already omits stack traces in non-debug mode, but we
    # add an explicit handler here so CodeQL's taint analysis sees a
    # sanitisation point that breaks the data-flow from tool calls to
    # the HTTP response body.
    # ------------------------------------------------------------------

    from fastapi.responses import JSONResponse as _JSONResponse

    @application.exception_handler(Exception)
    async def _sanitise_unhandled(request: Request, exc: Exception) -> _JSONResponse:  # type: ignore[misc]
        """Return a generic 500 body — never echo exception messages or tracebacks."""
        from mind_mem.observability import get_logger as _get_logger

        _get_logger("rest").warning("unhandled_exception", path=str(request.url.path), error=type(exc).__name__)
        return _JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "detail": "An unexpected error occurred. See server logs."},
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @application.get(
        "/v1/health",
        tags=["observability"],
        summary="Workspace health and schema version",
        # Deliberately reachable without a token — this is the liveness
        # probe, and an orchestrator that cannot call it restarts a
        # healthy server. It is rate limited by source address so it
        # cannot be used as a free amplifier, and the parts of the body
        # that describe the host (the workspace path and whether it
        # exists) are withheld from unauthenticated callers.
        dependencies=[Depends(_public_rate_limit)],
    )
    def health(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    ) -> JSONResponse:
        ws = _active_workspace(workspace)
        body: dict[str, Any] = {
            "_schema_version": MCP_SCHEMA_VERSION,
            "status": "ok",
            "schema_version": CURRENT_SCHEMA_VERSION,
            # The real package version. This was a hardcoded "3.2.0" that
            # had not moved since that release, so it could not even
            # distinguish a pre-v3.7.0 fail-OPEN server from the current
            # fail-closed one — while /openapi.json advertised a third,
            # different number for the same app.
            "api_version": _PACKAGE_VERSION,
        }
        token = str(credentials.credentials) if credentials is not None else None
        try:
            authenticated, _agent_id, _scopes = _verify_bearer(token)
        except APIKeyStoreUnavailable:
            authenticated = False
        if authenticated:
            body["workspace"] = ws
            body["workspace_exists"] = os.path.isdir(ws)
        return JSONResponse(body)

    @application.get(
        "/v1/metrics",
        tags=["observability"],
        summary="Prometheus metrics exposition (requires prometheus_client)",
        # The default Prometheus registry includes `mcp_http_auth_failures`
        # — i.e. whether credential guessing is underway — plus workspace
        # and traffic shape. Unlike liveness, nothing needs this
        # anonymously, so it takes the authenticated limiter.
        dependencies=[Depends(_rate_limit)],
    )
    def metrics_endpoint() -> PlainTextResponse:
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # type: ignore[import-untyped]

            output = generate_latest()
            return PlainTextResponse(
                content=output.decode("utf-8") if isinstance(output, bytes) else output,
                media_type=CONTENT_TYPE_LATEST,
            )
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prometheus metrics not available. Install: pip install 'mind-mem[otel]'",
            )

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    @application.post(
        "/v1/recall",
        tags=["recall"],
        summary="Search memory with BM25/hybrid backend",
        dependencies=[Depends(_rate_limit)],
    )
    def recall(body: RecallRequest, _token: Annotated[str | None, Depends(_require_auth)]) -> Any:
        from mind_mem.mcp.tools.recall import _recall_impl

        raw = _recall_impl(
            query=body.query,
            limit=body.limit,
            active_only=body.active_only,
            backend=body.backend,
            scoring_instant=body.scoring_instant or None,
        )
        return _parse_tool_json(raw)

    @application.get(
        "/v1/block/{block_id}",
        tags=["recall"],
        summary="Retrieve a single block by ID",
        dependencies=[Depends(_rate_limit)],
    )
    def get_block(block_id: str, _token: Annotated[str | None, Depends(_require_auth)]) -> Any:
        from mind_mem.mcp.tools.memory_ops import get_block as _get_block

        raw = _get_block(block_id)
        parsed = _parse_tool_json(raw)
        if isinstance(parsed, dict) and parsed.get("found") is False:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=parsed)
        return parsed

    # ------------------------------------------------------------------
    # Governance
    # ------------------------------------------------------------------

    @application.post(
        "/v1/propose_update",
        tags=["governance"],
        summary="Stage a new decision or task proposal",
        dependencies=[Depends(_rate_limit)],
    )
    def propose_update(body: ProposeUpdateRequest, _token: Annotated[str | None, Depends(_require_admin)]) -> Any:
        from mind_mem.mcp.tools.governance import propose_update as _propose_update

        raw = _propose_update(
            block_type=body.block_type,
            statement=body.statement,
            rationale=body.rationale,
            tags=body.tags,
            confidence=body.confidence,
        )
        return _parse_tool_json(raw)

    @application.post(
        "/v1/approve_apply",
        tags=["governance"],
        summary="Apply a staged proposal (admin scope required)",
        dependencies=[Depends(_rate_limit)],
    )
    def approve_apply(body: ApproveApplyRequest, _token: Annotated[str | None, Depends(_require_admin)]) -> Any:
        from mind_mem.mcp.tools.governance import approve_apply as _approve_apply

        raw = _approve_apply(proposal_id=body.proposal_id, dry_run=body.dry_run)
        return _parse_tool_json(raw)

    @application.post(
        "/v1/rollback_proposal",
        tags=["governance"],
        summary="Rollback an applied proposal (admin scope required)",
        dependencies=[Depends(_rate_limit)],
    )
    def rollback_proposal(body: RollbackProposalRequest, _token: Annotated[str | None, Depends(_require_admin)]) -> Any:
        from mind_mem.mcp.tools.governance import rollback_proposal as _rollback

        raw = _rollback(receipt_ts=body.receipt_ts, reason=body.reason)
        return _parse_tool_json(raw)

    @application.get(
        "/v1/scan",
        tags=["governance"],
        summary="Run workspace integrity scan",
        dependencies=[Depends(_rate_limit)],
    )
    def scan(_token: Annotated[str | None, Depends(_require_auth)]) -> Any:
        from mind_mem.mcp.tools.governance import scan as _scan

        raw = _scan()
        return _parse_tool_json(raw)

    @application.get(
        "/v1/contradictions",
        tags=["governance"],
        summary="List detected contradictions with resolution analysis",
        dependencies=[Depends(_rate_limit)],
    )
    def list_contradictions(_token: Annotated[str | None, Depends(_require_auth)]) -> Any:
        from mind_mem.mcp.tools.governance import list_contradictions as _list_contradictions

        raw = _list_contradictions()
        return _parse_tool_json(raw)

    # ------------------------------------------------------------------
    # OIDC / SSO callback
    # ------------------------------------------------------------------

    @application.post(
        "/v1/auth/oidc/callback",
        tags=["auth"],
        summary="Exchange an OIDC access_token for a validated session",
        # This route intentionally has no auth dependency (it is how a
        # caller *becomes* authenticated), which made it the one endpoint
        # an anonymous party could use to drive an outbound IdP request
        # per HTTP request they sent. It is rate limited per client
        # instead, by source address — `_public_rate_limit` does not
        # require a token.
        dependencies=[Depends(_public_rate_limit)],
    )
    def oidc_callback(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    ) -> JSONResponse:
        """Validate an OIDC JWT supplied via ``Authorization: Bearer <token>``.

        When valid, returns the decoded claims so callers can confirm their
        identity was accepted.  Configure the issuer via ``OIDC_ISSUER``,
        ``OIDC_CLIENT_ID``, ``OIDC_AUDIENCE`` environment variables.
        """
        oidc_issuer = os.environ.get("OIDC_ISSUER")
        oidc_client_id = os.environ.get("OIDC_CLIENT_ID", "")
        oidc_client_secret = os.environ.get("OIDC_CLIENT_SECRET", "")
        oidc_audience = os.environ.get("OIDC_AUDIENCE", "")

        if not oidc_issuer:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="OIDC not configured (set OIDC_ISSUER, OIDC_CLIENT_ID, OIDC_AUDIENCE)",
            )
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            from mind_mem.api.auth import AuthError  # noqa: PLC0415

            provider = _oidc_provider(oidc_issuer, oidc_client_id, oidc_client_secret, oidc_audience)
            claims = provider.verify(credentials.credentials)
            scopes = provider.extract_scopes(claims)
            agent_id = claims.get("sub", "oidc-user")
            current_agent_id.set(agent_id)
        except AuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

        return JSONResponse({"authenticated": True, "agent_id": agent_id, "scopes": scopes})

    # ------------------------------------------------------------------
    # Admin: per-agent API key management
    # ------------------------------------------------------------------

    @application.post(
        "/v1/admin/api_keys",
        tags=["admin"],
        summary="Create a new per-agent API key (admin scope required)",
    )
    def create_api_key(
        body: CreateAPIKeyRequest,
        _token: Annotated[str | None, Depends(_require_admin)],
    ) -> JSONResponse:
        store = _require_api_key_store()
        raw_key = store.create(
            agent_id=body.agent_id,
            scopes=body.scopes,
            expires_in_days=body.expires_in_days,
        )
        return JSONResponse(
            {"key": raw_key, "agent_id": body.agent_id, "scopes": body.scopes},
            status_code=status.HTTP_201_CREATED,
        )

    @application.get(
        "/v1/admin/api_keys",
        tags=["admin"],
        summary="List API keys (admin scope required)",
    )
    def list_api_keys(
        _token: Annotated[str | None, Depends(_require_admin)],
        agent_id: str = "",
    ) -> JSONResponse:
        store = _require_api_key_store()
        keys = store.list_keys(agent_id=agent_id)
        return JSONResponse({"keys": keys})

    @application.delete(
        "/v1/admin/api_keys/{key_id}",
        tags=["admin"],
        summary="Revoke an API key (admin scope required)",
    )
    def revoke_api_key(
        key_id: str,
        _token: Annotated[str | None, Depends(_require_admin)],
    ) -> JSONResponse:
        store = _require_api_key_store()
        revoked = store.revoke(key_id)
        if not revoked:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key not found or already revoked: {key_id}",
            )
        return JSONResponse({"revoked": True, "key_id": key_id})

    @application.post(
        "/v1/admin/api_keys/{key_id}/rotate",
        tags=["admin"],
        summary="Rotate an API key — revoke old, issue new (admin scope required)",
    )
    def rotate_api_key(
        key_id: str,
        _token: Annotated[str | None, Depends(_require_admin)],
    ) -> JSONResponse:
        store = _require_api_key_store()
        try:
            new_key = store.rotate(key_id)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key not found: {key_id}",
            )
        return JSONResponse({"key": new_key, "rotated_from": key_id})

    return application


# ---------------------------------------------------------------------------
# Default application instance (used by `mm serve` and in tests)
# ---------------------------------------------------------------------------

app = create_app()

# ---------------------------------------------------------------------------
# Convenience launcher
# ---------------------------------------------------------------------------


def _enforce_fail_closed(host: str, allow_unauthenticated_localhost: bool) -> None:
    """v3.7.0 H4: refuse to bind a network port without authentication.

    Raises :class:`SystemExit` with a structured message when:

    * No auth is configured AND ``--allow-unauthenticated-localhost``
      is not set → unauthenticated bind is forbidden.
    * ``--allow-unauthenticated-localhost`` is set BUT ``host`` is not
      a loopback interface → routable unauthenticated bind is forbidden.
    """
    if _auth_is_configured():
        return
    if not allow_unauthenticated_localhost:
        raise SystemExit(
            "mind-mem REST: refusing to start without authentication.\n"
            "  Set MIND_MEM_TOKEN, MIND_MEM_ADMIN_TOKEN, MIND_MEM_API_KEY_DB,\n"
            "  or OIDC_ISSUER+OIDC_AUDIENCE — or pass\n"
            "  --allow-unauthenticated-localhost to bind 127.0.0.1 only."
        )
    if host not in _LOOPBACK_HOSTS:
        raise SystemExit(
            "mind-mem REST: --allow-unauthenticated-localhost requires a loopback bind.\n"
            f"  Refusing to listen on host={host!r} without auth.\n"
            "  Use --host 127.0.0.1 (or localhost / ::1)."
        )


def run(
    host: str = "127.0.0.1",
    port: int = 8080,
    workspace: str | None = None,
    *,
    allow_unauthenticated_localhost: bool = False,
) -> None:
    """Launch the REST API with uvicorn.

    Parameters
    ----------
    host:
        Interface to bind (default ``127.0.0.1``).
    port:
        TCP port (default ``8080``).
    workspace:
        mind-mem workspace path; falls back to ``MIND_MEM_WORKSPACE`` or cwd.
    allow_unauthenticated_localhost:
        v3.7.0 H4: explicit operator opt-in to skip authentication.
        Permitted only when ``host`` is a loopback interface; routable
        binds without auth are refused at startup.
    """
    try:
        import uvicorn  # type: ignore[import-untyped]
    except ImportError as err:  # pragma: no cover
        raise ImportError("uvicorn is required to run the REST API server. Install: pip install 'mind-mem[api]'") from err

    _enforce_fail_closed(host, allow_unauthenticated_localhost)
    if allow_unauthenticated_localhost and not _auth_is_configured():
        os.environ[ALLOW_UNAUTH_ENV] = "1"
    # N-13: create_app() has no view of the bind, so publish it here.
    os.environ[BIND_HOST_ENV] = host

    server_app = create_app(workspace)
    uvicorn.run(server_app, host=host, port=port)
