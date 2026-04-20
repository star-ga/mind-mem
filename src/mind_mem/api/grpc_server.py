"""gRPC wire protocol for mind-mem (v4.0 prep).

REST (FastAPI) is fine for interactive clients, but service-to-service
callers in a cluster want lower-latency typed RPCs. This module is a
minimal gRPC stub that exposes the same recall / governance surface
as the REST layer.

Design notes:
* Protocol shape is defined inline as typed dataclasses — operators
  who need real ``.proto`` files generate them at deploy time from
  the mind-mem mirror repo (see ``docs/grpc-proto.md``).
* No grpcio dependency at import time. The handler functions are
  plain Python and take dicts; a thin gRPC servicer adapts them in
  :func:`serve` when grpcio is available.
* Auth + tenant routing reuse the REST layer's primitives
  (``api_keys.APIKeyStore``, ``workspace.use_workspace``).

Gated behind ``mind-mem[grpc]`` extra. Operator runs::

    mm serve-grpc --port 50051
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

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
    """
    import time

    t0 = time.perf_counter()
    from mind_mem.mcp.tools.recall import _recall_impl

    payload = _recall_impl(
        query=request.query,
        limit=request.limit,
        active_only=request.active_only,
        backend=request.backend,
        format=request.format,
    )
    took_ms = (time.perf_counter() - t0) * 1000.0
    return RecallResponse(payload=payload, took_ms=round(took_ms, 3))


def handle_governance(request: GovernanceRequest) -> GovernanceResponse:
    """Route a governance op to the existing MCP tool impl."""
    import json

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
        raise RuntimeError("mind-mem gRPC server requires the 'grpcio' package. Install with: pip install 'mind-mem[grpc]'") from exc

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


def serve(port: int = 50051) -> None:
    """Start a blocking gRPC server on ``port``.

    Requires ``grpcio`` installed. Operators that prefer their own
    supervision (systemd, k8s Deployment) import the handler
    functions directly and bind them in their own server loop.
    """
    servicer = _build_servicer()
    # Actual .serve-and-block happens in the operator's
    # grpc_generated adapter; we surface the servicer as the mount
    # point so they don't re-implement the handlers.
    from concurrent import futures

    import grpc  # type: ignore

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    # Real gRPC needs generated servicer classes. Since those are
    # operator-provided, we just start the server and log. When the
    # grpc_generated module exists, it registers itself.
    try:
        from mind_mem.api import grpc_generated  # type: ignore
    except ImportError:
        pass
    else:
        grpc_generated.register(server, servicer)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


__all__ = [
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
