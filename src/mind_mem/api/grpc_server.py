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
  :func:`serve` binds loopback only — and now *enforces* it:
  a non-loopback ``MIND_MEM_GRPC_HOST`` is refused unless the operator
  sets ``MIND_MEM_GRPC_ALLOW_INSECURE_BIND=true``, mirroring the REST
  layer's fail-closed bind check.

There is no ``mm`` subcommand for this surface. An operator starts it
in their own process::

    python -c "from mind_mem.api.grpc_server import serve; serve(50051)"

or, under their own supervision (systemd, a k8s Deployment), imports
the handler functions and binds them in their own server loop.
Requires ``grpcio``, which mind-mem does not depend on.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple

from mind_mem import audit_context as _audit_ctx
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
# Audit-header propagation (roadmap v4.0.0 Group D)
# ---------------------------------------------------------------------------


def audit_context_from_metadata(metadata: Optional[Sequence[Tuple[str, Any]]]) -> "_audit_ctx.AuditContext":
    """Build a request-scoped audit context from gRPC invocation metadata.

    gRPC metadata keys are lowercase ASCII, which is already the shape
    :func:`mind_mem.audit_context.context_from_headers` looks up, so
    the same three names carry across both transports and one correlation
    id survives a REST-to-gRPC hop.

    Binary metadata (a ``-bin`` key, whose value is ``bytes``) is ignored
    rather than coerced: these three headers are text, and a ``bytes``
    value here means the caller sent something this transport does not
    define.

    Note what is deliberately absent: nothing here fills
    ``agent_authenticated``. This transport performs no authentication at
    all (see the module docstring), so every identity it sees is a claim,
    and recording a claim as an authenticated identity is how attribution
    becomes forgery.
    """
    pairs: dict[str, str] = {}
    for key, value in metadata or ():
        if isinstance(value, str):
            pairs[str(key).lower()] = value
    return _audit_ctx.context_from_headers(pairs.get, transport="grpc")


@contextmanager
def bind_call_context(context: Any) -> Iterator[Optional["_audit_ctx.AuditContext"]]:
    """Bind the audit context for one RPC, from its ``ServicerContext``.

    A servicer context that cannot produce metadata (a plain object in a
    test harness, a generated stub that passes ``None``) yields ``None``
    and binds nothing — the handler then behaves exactly as it did before
    this existed.
    """
    getter = getattr(context, "invocation_metadata", None)
    if getter is None:
        yield None
        return
    try:
        metadata = getter()
    except Exception:
        yield None
        return
    ctx = audit_context_from_metadata(metadata)
    with _audit_ctx.bind_audit_context(ctx):
        yield ctx


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
            with bind_call_context(context):
                return handle_recall(RecallRequest(**request_dict)).__dict__

        def Governance(self, request_dict: dict, context: Any) -> dict:
            with bind_call_context(context):
                return handle_governance(GovernanceRequest(**request_dict)).__dict__

        def Health(self, request_dict: dict, context: Any) -> dict:
            with bind_call_context(context):
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


_GRPC_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _enforce_grpc_bind(host: str) -> None:
    """Refuse a routable bind on this unauthenticated transport.

    The REST layer refuses to open a network port without authentication
    (``rest._enforce_fail_closed``). This surface is strictly more
    exposed -- no auth to configure, no TLS, and ``approve``/``rollback``
    reachable through ``**kwargs`` -- and yet the only thing standing
    against ``MIND_MEM_GRPC_HOST=0.0.0.0`` was a comment saying an
    operator "should" front it with TLS. Same mistake as REST, failing
    the opposite way.

    Loopback binds pass. Anything else requires
    ``MIND_MEM_GRPC_ALLOW_INSECURE_BIND=true``, so exposing the HITL gate
    to the network is something an operator states rather than something
    they reach by setting a hostname.
    """
    import os as _os

    if host.strip("[]").lower() in _GRPC_LOOPBACK_HOSTS:
        return
    if _os.environ.get("MIND_MEM_GRPC_ALLOW_INSECURE_BIND", "").strip().lower() in ("1", "true", "yes"):
        _log.warning("grpc_insecure_bind_acknowledged", host=host)
        return
    raise RuntimeError(
        f"mind-mem gRPC: refusing to bind {host!r}.\n"
        "  This transport has no authentication and no TLS, and it drives\n"
        "  governance mutations (approve/rollback), so a routable bind lets\n"
        "  any reachable party defeat the HITL gate.\n"
        "  Use MIND_MEM_GRPC_HOST=127.0.0.1, or set\n"
        "  MIND_MEM_GRPC_ALLOW_INSECURE_BIND=true to accept that risk."
    )


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
    import os as _os
    from concurrent import futures

    # Before grpc is even imported: a check that runs after
    # add_insecure_port would leave the socket listening.
    _enforce_grpc_bind(_os.environ.get("MIND_MEM_GRPC_HOST", "127.0.0.1"))

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
    host = _os.environ.get("MIND_MEM_GRPC_HOST", "127.0.0.1")
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    server.wait_for_termination()


__all__ = [
    "TENANT_UNSUPPORTED",
    "_enforce_grpc_bind",
    "audit_context_from_metadata",
    "bind_call_context",
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
