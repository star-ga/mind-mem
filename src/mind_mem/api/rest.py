"""REST API layer for mind-mem (v3.2.0).

FastAPI application that mirrors the MCP tool surface with:
- Pydantic-validated request models
- Bearer-token auth via existing MIND_MEM_TOKEN / MIND_MEM_ADMIN_TOKEN env vars
- Per-client sliding-window rate limiting
- Optional Prometheus exposition at /v1/metrics
- OpenAPI docs at /openapi.json (FastAPI default)

Launch via: mm serve [--port 8080] [--host 127.0.0.1]
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from typing import Annotated, Any, cast

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, status
    from fastapi.responses import JSONResponse, PlainTextResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field, field_validator
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "mind-mem REST API requires the 'api' extra: "
        "pip install 'mind-mem[api]'"
    ) from _err

from mind_mem.mcp.infra.constants import MCP_SCHEMA_VERSION
from mind_mem.mcp.infra.http_auth import _check_token, verify_token
from mind_mem.mcp.infra.rate_limit import SlidingWindowRateLimiter, _get_client_rate_limiter
from mind_mem.schema_version import CURRENT_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Agent-ID context variable (set by auth dependencies, read by audit chain)
# ---------------------------------------------------------------------------

#: The current request's authenticated agent ID.  Defaults to "anonymous".
#: Set this from MCP tool entry points to propagate identity through the stack.
current_agent_id: ContextVar[str] = ContextVar("current_agent_id", default="anonymous")

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
    """Derive a stable per-client identifier from the bearer token."""
    if token is None:
        return "anonymous"
    # Use last 16 chars as a non-sensitive bucket key
    return token[-16:] if len(token) >= 16 else token


def _verify_bearer(token: str | None) -> tuple[bool, str]:
    """Return (valid, agent_id) for a bearer/API-key token.

    Priority:
    1. MIND_MEM_ADMIN_TOKEN match → agent_id "admin"
    2. ``mmk_live_*`` or ``mmk_test_*`` → look up in APIKeyStore
    3. MIND_MEM_TOKEN match → agent_id "user"
    4. No auth configured → anonymous access allowed, agent_id "anonymous"
    """
    if token is None:
        headers: dict[str, str] = {}
        return verify_token(headers), "anonymous"

    # 1. Admin token — fast path
    if _has_admin_scope(token):
        return True, "admin"

    # 2. API key (mmk_live_* / mmk_test_*)
    if token.startswith("mmk_live_") or token.startswith("mmk_test_"):
        record = _resolve_api_key_store_record(token)
        if record is not None:
            return True, record["agent_id"]
        return False, "anonymous"

    # 3. User bearer token
    headers = {"Authorization": f"Bearer {token}"}
    if verify_token(headers):
        return True, "user"
    return False, "anonymous"


def _resolve_api_key_store_record(raw_key: str) -> dict | None:
    """Verify an mmk_* API key and return its record, or None."""
    store = _get_api_key_store()
    if store is None:
        return None
    return cast(dict[str, Any] | None, store.verify(raw_key))


def _get_api_key_store() -> Any:
    """Lazily load the APIKeyStore if configured."""
    db_path = os.environ.get("MIND_MEM_API_KEY_DB")
    if not db_path:
        return None
    try:
        from mind_mem.api.api_keys import APIKeyStore  # noqa: PLC0415

        production = os.environ.get("MIND_MEM_ENV", "production") == "production"
        return APIKeyStore(db_path, production=production)
    except Exception:
        return None


def _has_admin_scope(token: str | None) -> bool:
    """Return True when the provided token is the MIND_MEM_ADMIN_TOKEN."""
    if token is None:
        return False
    admin = os.environ.get("MIND_MEM_ADMIN_TOKEN")
    if admin is None:
        return False
    import hmac

    return hmac.compare_digest(token, admin)


def _api_key_has_admin_scope(token: str | None) -> bool:
    """Return True when an mmk_* key carries the admin scope."""
    if token is None or not (token.startswith("mmk_live_") or token.startswith("mmk_test_")):
        return False
    record = _resolve_api_key_store_record(token)
    if record is None:
        return False
    scopes: list[str] = record.get("scopes", [])
    return "admin" in scopes


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def _require_auth(
    token: Annotated[str | None, Depends(_extract_bearer)],
) -> str | None:
    """Dependency: require valid bearer/API-key token when auth is configured.

    Sets the ``current_agent_id`` contextvar so audit records carry the
    authenticated identity.
    """
    valid, agent_id = _verify_bearer(token)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    current_agent_id.set(agent_id)
    return token


def _require_admin(
    token: Annotated[str | None, Depends(_require_auth)],
) -> str | None:
    """Dependency: require admin-scope token (bearer or mmk_* with admin scope).

    The gate is enforced when *either* MIND_MEM_TOKEN *or*
    MIND_MEM_ADMIN_TOKEN is configured.  Using ``_check_token()`` alone
    (MIND_MEM_TOKEN) was wrong: a deployment that skips MIND_MEM_TOKEN
    but sets MIND_MEM_ADMIN_TOKEN would silently bypass the admin check.
    """
    token_configured = _check_token() is not None or os.environ.get("MIND_MEM_ADMIN_TOKEN") is not None
    if token_configured:
        is_admin = _has_admin_scope(token) or _api_key_has_admin_scope(token)
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin scope required",
            )
    return token


def _rate_limit(request: Request, token: Annotated[str | None, Depends(_require_auth)]) -> None:
    """Dependency: enforce per-client sliding-window rate limit."""
    client_id = _client_id_from_token(token) or request.client.host if request.client else "unknown"
    limiter: SlidingWindowRateLimiter = _get_client_rate_limiter(client_id)
    allowed, retry_after = limiter.allow()
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(retry_after + 1))},
        )


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


# ---------------------------------------------------------------------------
# Workspace helper
# ---------------------------------------------------------------------------


def _active_workspace(workspace: str | None) -> str:
    if workspace:
        return os.path.abspath(workspace)
    return os.environ.get("MIND_MEM_WORKSPACE", os.getcwd())


def _set_workspace_env(workspace: str) -> None:
    """Temporarily export workspace so MCP tool functions resolve it."""
    os.environ["MIND_MEM_WORKSPACE"] = workspace


def _parse_tool_json(raw: str) -> Any:
    """Parse JSON string returned by MCP tool functions."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"raw": raw}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


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
        version="3.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @application.get(
        "/v1/health",
        tags=["observability"],
        summary="Workspace health and schema version",
    )
    def health() -> JSONResponse:
        ws = _active_workspace(workspace)
        return JSONResponse(
            {
                "_schema_version": MCP_SCHEMA_VERSION,
                "status": "ok",
                "workspace": ws,
                "workspace_exists": os.path.isdir(ws),
                "schema_version": CURRENT_SCHEMA_VERSION,
                "api_version": "3.2.0",
            }
        )

    @application.get(
        "/v1/metrics",
        tags=["observability"],
        summary="Prometheus metrics exposition (requires prometheus_client)",
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
        _set_workspace_env(resolved_ws)
        from mind_mem.mcp.tools.recall import _recall_impl

        raw = _recall_impl(
            query=body.query,
            limit=body.limit,
            active_only=body.active_only,
            backend=body.backend,
        )
        return _parse_tool_json(raw)

    @application.get(
        "/v1/block/{block_id}",
        tags=["recall"],
        summary="Retrieve a single block by ID",
        dependencies=[Depends(_rate_limit)],
    )
    def get_block(block_id: str, _token: Annotated[str | None, Depends(_require_auth)]) -> Any:
        _set_workspace_env(resolved_ws)
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
    def propose_update(
        body: ProposeUpdateRequest, _token: Annotated[str | None, Depends(_require_admin)]
    ) -> Any:
        _set_workspace_env(resolved_ws)
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
    def approve_apply(
        body: ApproveApplyRequest, _token: Annotated[str | None, Depends(_require_admin)]
    ) -> Any:
        _set_workspace_env(resolved_ws)
        from mind_mem.mcp.tools.governance import approve_apply as _approve_apply

        raw = _approve_apply(proposal_id=body.proposal_id, dry_run=body.dry_run)
        return _parse_tool_json(raw)

    @application.post(
        "/v1/rollback_proposal",
        tags=["governance"],
        summary="Rollback an applied proposal (admin scope required)",
        dependencies=[Depends(_rate_limit)],
    )
    def rollback_proposal(
        body: RollbackProposalRequest, _token: Annotated[str | None, Depends(_require_admin)]
    ) -> Any:
        _set_workspace_env(resolved_ws)
        from mind_mem.mcp.tools.governance import rollback_proposal as _rollback

        raw = _rollback(receipt_ts=body.receipt_ts)
        return _parse_tool_json(raw)

    @application.get(
        "/v1/scan",
        tags=["governance"],
        summary="Run workspace integrity scan",
        dependencies=[Depends(_rate_limit)],
    )
    def scan(_token: Annotated[str | None, Depends(_require_auth)]) -> Any:
        _set_workspace_env(resolved_ws)
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
        _set_workspace_env(resolved_ws)
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
            from mind_mem.api.auth import AuthError, OIDCConfig, OIDCProvider  # noqa: PLC0415

            config = OIDCConfig(
                issuer=oidc_issuer,
                client_id=oidc_client_id,
                client_secret=oidc_client_secret,
                audience=oidc_audience,
            )
            provider = OIDCProvider(config)
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
        store = _get_api_key_store()
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="API key store not configured (set MIND_MEM_API_KEY_DB)",
            )
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
        store = _get_api_key_store()
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="API key store not configured (set MIND_MEM_API_KEY_DB)",
            )
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
        store = _get_api_key_store()
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="API key store not configured (set MIND_MEM_API_KEY_DB)",
            )
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
        store = _get_api_key_store()
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="API key store not configured (set MIND_MEM_API_KEY_DB)",
            )
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


def run(host: str = "127.0.0.1", port: int = 8080, workspace: str | None = None) -> None:
    """Launch the REST API with uvicorn.

    Parameters
    ----------
    host:
        Interface to bind (default ``127.0.0.1``).
    port:
        TCP port (default ``8080``).
    workspace:
        mind-mem workspace path; falls back to ``MIND_MEM_WORKSPACE`` or cwd.
    """
    try:
        import uvicorn  # type: ignore[import-untyped]
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "uvicorn is required to run the REST API server. "
            "Install: pip install 'mind-mem[api]'"
        ) from err

    server_app = create_app(workspace)
    uvicorn.run(server_app, host=host, port=port)
