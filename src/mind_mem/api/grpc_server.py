"""gRPC wire protocol for mind-mem (v4.0 prep).

REST (FastAPI) is fine for interactive clients, but service-to-service
callers in a cluster want lower-latency typed RPCs. This module is a
minimal gRPC stub that exposes the same recall / governance surface
as the REST layer.

Design notes:
* Protocol shape is defined inline as typed dataclasses and *they are
  the schema of record*. This repo ships no ``.proto`` file and no
  generated code; an operator who wants real protobufs mirrors these
  dataclasses by hand and drops the protoc output in as
  ``mind_mem.api.grpc_generated`` (see :func:`serve`).
* No grpcio dependency at import time. The handler functions are
  plain Python and take dicts; a thin gRPC servicer adapts them in
  :func:`serve` when grpcio is available.
* **No authentication and no tenant routing.** Unlike the REST layer,
  this transport runs no API-key check and never enters
  ``mcp.infra.workspace.use_workspace`` — every call resolves the
  process-wide workspace. ``tenant_id`` is consequently *rejected*
  rather than accepted and silently served from the wrong corpus (see
  :func:`handle_recall` and :func:`handle_governance`), and
  :func:`serve` binds loopback only.

There is no ``mm`` subcommand for this surface. An operator starts it
in their own process::

    python -c "from mind_mem.api.grpc_server import serve; serve(50051)"

or, under their own supervision (systemd, a k8s Deployment), imports
the handler functions and binds them in their own server loop.
Requires ``grpcio``, which mind-mem does not depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mind_mem.observability import get_logger

_log = get_logger("grpc_server")

#: Why a populated ``tenant_id`` is refused instead of ignored. This transport
#: has no per-tenant routing: it never enters ``use_workspace``, so every call
#: resolves the one process-wide workspace. Accepting the field and dropping it
#: would answer a scoped request from an unscoped corpus — and answer it with
#: ``ok``. Fail closed until real routing exists.
TENANT_UNSUPPORTED = (
    "tenant_id is not supported by the gRPC transport: it performs no "
    "per-tenant routing and would serve the request from the process-wide "
    "workspace. Omit tenant_id, or use a transport that scopes the workspace."
)


def _reject_unsupported_tenant(tenant_id: str | None) -> None:
    """Raise if the caller asked for a tenant this transport cannot honour."""
    if tenant_id:
        raise ValueError(TENANT_UNSUPPORTED)


# ---------------------------------------------------------------------------
# Typed request / response dataclasses — shape the .proto mirrors.
# ---------------------------------------------------------------------------


@dataclass
class RecallRequest:
    query: str
    limit: int = 10
    active_only: bool = False
    backend: str = "auto"
    format: str = "blocks"  # or "bundle"
    tenant_id: str | None = None
    #: UTC date (``YYYY-MM-DD``) the recency layer scores against. Recall is
    #: deterministic given (corpus, config, scoring_instant); replaying an
    #: attested run means sending back the instant it recorded. Empty = today
    #: in UTC.
    scoring_instant: str = ""


@dataclass
class RecallResponse:
    payload: str  # JSON-encoded recall result (blocks or bundle)
    took_ms: float = 0.0


@dataclass
class GovernanceRequest:
    operation: str  # "propose" | "approve" | "rollback" | "scan"
    args: dict[str, Any] = field(default_factory=dict)
    tenant_id: str | None = None


@dataclass
class GovernanceResponse:
    ok: bool
    payload: str  # JSON-encoded result
    error: str | None = None


@dataclass
class HealthResponse:
    status: str
    workspace: str
    schema_version: str


# ---------------------------------------------------------------------------
# Handlers — pure Python, grpcio-free.
# ---------------------------------------------------------------------------


def handle_recall(request: RecallRequest) -> RecallResponse:
    """Recall handler — delegates to the shared ``_recall_impl``.

    Pulled out of the servicer so the same function can be exercised
    from tests + reused by REST / MCP / gRPC transports.

    Raises:
        ValueError: If ``request.tenant_id`` is set. :class:`RecallResponse`
            has no error channel, so a request this transport cannot scope
            fails loudly rather than returning another tenant's corpus.
    """
    import time

    _reject_unsupported_tenant(request.tenant_id)
    t0 = time.perf_counter()
    from mind_mem.mcp.tools.recall import _recall_impl

    payload = _recall_impl(
        query=request.query,
        limit=request.limit,
        active_only=request.active_only,
        backend=request.backend,
        format=request.format,
        scoring_instant=request.scoring_instant or None,
    )
    took_ms = (time.perf_counter() - t0) * 1000.0
    return RecallResponse(payload=payload, took_ms=round(took_ms, 3))


def handle_governance(request: GovernanceRequest) -> GovernanceResponse:
    """Route a governance op to the existing MCP tool impl.

    A populated ``tenant_id`` is refused (see :data:`TENANT_UNSUPPORTED`) —
    these operations mutate the governed corpus, so running them against the
    process-wide workspace because the requested tenant could not be honoured
    is the one outcome worth failing closed on.
    """
    import json

    if request.tenant_id:
        return GovernanceResponse(ok=False, payload="", error=TENANT_UNSUPPORTED)
    ops = {
        "propose": ("mind_mem.mcp.tools.governance", "propose_update"),
        "approve": ("mind_mem.mcp.tools.governance", "approve_apply"),
        "rollback": ("mind_mem.mcp.tools.governance", "rollback_proposal"),
        "scan": ("mind_mem.mcp.tools.governance", "scan"),
    }
    if request.operation not in ops:
        return GovernanceResponse(ok=False, payload="", error=f"unknown operation: {request.operation}")
    mod_name, fn_name = ops[request.operation]
    try:
        import importlib

        mod = importlib.import_module(mod_name)
        fn: Callable[..., str] = getattr(mod, fn_name)
        payload = fn(**request.args)
    except Exception as exc:
        return GovernanceResponse(ok=False, payload="", error=str(exc))
    # Unwrap JSON-string responses so the caller gets typed access.
    try:
        parsed = json.loads(payload)
        ok = not (isinstance(parsed, dict) and "error" in parsed)
    except json.JSONDecodeError:
        ok = True
    return GovernanceResponse(ok=ok, payload=payload)


def handle_health() -> HealthResponse:
    from mind_mem.mcp.infra.constants import MCP_SCHEMA_VERSION
    from mind_mem.mcp.infra.workspace import _workspace

    return HealthResponse(
        status="ok",
        workspace=_workspace(),
        schema_version=MCP_SCHEMA_VERSION,
    )


# ---------------------------------------------------------------------------
# Servicer adapter — only loaded when grpcio is installed.
# ---------------------------------------------------------------------------


def _build_servicer() -> Any:
    """Return a grpcio-compatible servicer. Raises on missing deps."""
    try:
        import grpc  # type: ignore  # noqa: F401
    except ImportError as exc:
        # 'grpcio' is not a mind-mem dependency and there is no extra that
        # pulls it in — name the package the operator actually installs.
        raise RuntimeError("mind-mem gRPC server requires 'grpcio'. Install with: pip install grpcio") from exc

    # The real .proto-generated servicer classes live in
    # ``mind_mem.api.grpc_generated`` — shipped as a sibling package
    # with the operator's choice of protoc output. This function just
    # wraps our handler funcs into a dispatcher that package can call.
    class _Servicer:
        def Recall(self, request_dict: dict, context: Any) -> dict:
            return handle_recall(RecallRequest(**request_dict)).__dict__

        def Governance(self, request_dict: dict, context: Any) -> dict:
            return handle_governance(GovernanceRequest(**request_dict)).__dict__

        def Health(self, request_dict: dict, context: Any) -> dict:
            return handle_health().__dict__

    return _Servicer()


def _register_generated_services(server: Any, servicer: Any) -> bool:
    """Mount the operator-supplied generated servicer on ``server``.

    ``mind_mem.api.grpc_generated`` is not shipped by this package — it is
    the protoc output an operator drops in alongside it — so its absence is
    a supported state, not an error. It is *not* a silent one: without it
    the server binds a port and answers ``UNIMPLEMENTED`` to every RPC, and
    nothing on the server side would otherwise say why.

    Returns:
        True when services were registered, False when the generated
        package is absent.
    """
    try:
        from mind_mem.api import grpc_generated  # type: ignore
    except ImportError as exc:
        _log.warning(
            "grpc_generated_missing",
            detail=str(exc),
            effect="no services registered — every RPC will answer UNIMPLEMENTED",
            remedy="generate protoc output from the dataclasses in this module and install it as mind_mem.api.grpc_generated",
        )
        return False
    grpc_generated.register(server, servicer)
    return True


def serve(port: int = 50051) -> None:
    """Start a blocking gRPC server on ``port``.

    Requires ``grpcio`` installed. Operators that prefer their own
    supervision (systemd, k8s Deployment) import the handler
    functions directly and bind them in their own server loop.

    There is no ``mm`` subcommand that calls this; see the module
    docstring for how to launch it.
    """
    servicer = _build_servicer()
    # Actual .serve-and-block happens in the operator's
    # grpc_generated adapter; we surface the servicer as the mount
    # point so they don't re-implement the handlers.
    from concurrent import futures

    import grpc  # type: ignore

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    # Real gRPC needs generated servicer classes. Since those are
    # operator-provided, a missing package is logged (not swallowed) and
    # the server still starts, so an operator running their own registration
    # against this process is unaffected.
    _register_generated_services(server, servicer)
    # Bind loopback by default, NOT [::] (all interfaces). This gRPC surface
    # has no TLS and no auth interceptor and drives governance mutations
    # (approve/rollback) via **kwargs, so exposing it network-wide lets any
    # reachable party defeat the HITL gate. Operators who genuinely need a
    # non-loopback bind must opt in explicitly via MIND_MEM_GRPC_HOST (and
    # should front it with TLS + token auth).
    import os as _os

    host = _os.environ.get("MIND_MEM_GRPC_HOST", "127.0.0.1")
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    server.wait_for_termination()


__all__ = [
    "TENANT_UNSUPPORTED",
    "RecallRequest",
    "RecallResponse",
    "GovernanceRequest",
    "GovernanceResponse",
    "HealthResponse",
    "handle_recall",
    "handle_governance",
    "handle_health",
    "serve",
]
