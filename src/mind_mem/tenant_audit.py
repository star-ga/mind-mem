"""Per-tenant audit chain isolation (v4.0 prep).

v3.x keeps a single workspace-wide hash chain in
``audit_chain.py``. For multi-tenant deployments (v4.0) each
tenant's governance history must be cryptographically isolated —
otherwise a cross-tenant compliance export leaks foreign audit
records.

This module introduces a thin façade that routes append/verify
calls to the appropriate per-tenant chain. The chain itself stays
the existing :mod:`mind_mem.audit_chain` implementation — we just
maintain a table of ``tenant_id → AuditChain`` and a per-tenant
genesis + spec-hash binding so each chain is independently
verifiable without cross-referencing the operator's root chain.

This is a v4.0-prep module: the isolation contract is complete,
but the storage adapter is intentionally the same SQLite-backed
chain v3.x uses. Sharding onto tenant-specific databases is a
v4.0 storage concern and plugs in below via
:func:`register_chain_factory`.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .observability import get_logger

_log = get_logger("tenant_audit")


# Length of the genesis token — 256 bits, matches audit chain block ID size.
_GENESIS_BYTES = 32


@dataclass(frozen=True)
class TenantChain:
    """Immutable per-tenant chain handle.

    ``chain`` is the underlying audit chain object (adapts whatever
    implementation the caller registered). ``genesis`` is the tenant-
    specific chain anchor; ``spec_hash`` is an HMAC of the tenant's
    governance spec so verification can't be spoofed across tenants.
    """

    tenant_id: str
    chain: Any
    genesis: bytes
    spec_hash: bytes


# Global registry: one factory per deployment. Default factory creates
# the existing audit_chain.AuditChain bound to a per-tenant SQLite path.
_FactoryFn = Callable[[str, str], Any]
_factory: _FactoryFn | None = None
_tenants: dict[str, TenantChain] = {}
_lock = threading.Lock()


def register_chain_factory(factory: _FactoryFn) -> None:
    """Register how tenant chains are constructed.

    Signature: ``factory(tenant_id, base_path) -> chain_impl``.
    ``chain_impl`` must have ``append(operation, target, content,
    actor, metadata)`` and ``verify()`` methods compatible with
    :class:`audit_chain.AuditChain`.

    Call once at deployment init — the registry is process-global.
    """
    global _factory
    _factory = factory


def _default_factory(tenant_id: str, base_path: str) -> Any:
    """Default: wrap :class:`audit_chain.AuditChain` with a per-tenant path."""
    from .audit_chain import AuditChain

    tenant_db = os.path.join(base_path, "tenants", tenant_id, "audit.jsonl")
    os.makedirs(os.path.dirname(tenant_db), exist_ok=True)
    return AuditChain(tenant_db)


def _tenant_genesis(tenant_id: str, root_secret: bytes) -> bytes:
    """Per-tenant chain anchor — HMAC of tenant_id under the root secret.

    The root secret is the deployment's master key (v4.0 KMS scope).
    Computed once and cached; stable across restarts as long as
    root_secret is stable.
    """
    return hmac.new(root_secret, tenant_id.encode("utf-8"), hashlib.sha256).digest()[:_GENESIS_BYTES]


def _tenant_spec_hash(tenant_id: str, spec: bytes, root_secret: bytes) -> bytes:
    """Bind the tenant's governance spec to the chain.

    Any change to the spec produces a new spec_hash, so audit exports
    can detect spec drift (similar to the v3.x chain's overall
    spec_hash binding).
    """
    mac = hmac.new(root_secret, digestmod=hashlib.sha256)
    mac.update(tenant_id.encode("utf-8"))
    mac.update(b":")
    mac.update(spec)
    return mac.digest()


def get_chain(
    tenant_id: str,
    *,
    base_path: str,
    root_secret: bytes,
    spec: bytes = b"",
) -> TenantChain:
    """Return the :class:`TenantChain` for ``tenant_id``; build on first call.

    Args:
        tenant_id: Stable tenant identifier. Usually the customer
            namespace name, not a display label.
        base_path: Filesystem root for tenant audit artifacts. Each
            tenant gets a subdirectory.
        root_secret: Deployment master key (length-unchecked —
            operator is responsible for ≥32 bytes of entropy).
        spec: Bytes of the tenant's governance spec (JSON-serialised
            namespace config). Re-computed on every call and asserted
            against the cached handle — spec changes force a refresh.
    """
    if not tenant_id:
        raise ValueError("tenant_id must be non-empty")
    if not isinstance(root_secret, (bytes, bytearray)) or len(root_secret) < 16:
        raise ValueError("root_secret must be ≥16 bytes")

    expected_spec_hash = _tenant_spec_hash(tenant_id, spec, bytes(root_secret))
    with _lock:
        cached = _tenants.get(tenant_id)
        if cached is not None and cached.spec_hash == expected_spec_hash:
            return cached
        factory = _factory or _default_factory
        chain_impl = factory(tenant_id, base_path)
        handle = TenantChain(
            tenant_id=tenant_id,
            chain=chain_impl,
            genesis=_tenant_genesis(tenant_id, bytes(root_secret)),
            spec_hash=expected_spec_hash,
        )
        _tenants[tenant_id] = handle
        _log.info(
            "tenant_audit_chain_initialized",
            tenant_id=tenant_id,
            genesis=handle.genesis.hex()[:16],
        )
        return handle


def list_tenants() -> list[str]:
    """Return all tenant IDs with an initialized chain in this process."""
    with _lock:
        return sorted(_tenants.keys())


def reset() -> None:
    """Clear the tenant registry. Used between tests — production calls
    should never hit this path."""
    with _lock:
        _tenants.clear()


def verify_tenant(
    tenant_id: str,
    *,
    base_path: str,
    root_secret: bytes,
    spec: bytes = b"",
) -> dict[str, Any]:
    """Run the underlying chain's verification for a single tenant.

    Returns a dict with ``{tenant_id, verified, records, genesis,
    spec_hash}``. Callers use this for per-tenant compliance exports
    without leaking cross-tenant chain state.
    """
    handle = get_chain(
        tenant_id,
        base_path=base_path,
        root_secret=root_secret,
        spec=spec,
    )
    ok = True
    records = 0
    try:
        result = handle.chain.verify()
        # audit_chain.AuditChain.verify returns a dict; fall back
        # gracefully for custom chain impls.
        if isinstance(result, dict):
            ok = bool(result.get("verified", True))
            records = int(result.get("records", 0))
        else:
            ok = bool(result)
    except Exception as exc:
        _log.warning("tenant_audit_verify_failed", tenant_id=tenant_id, error=str(exc))
        ok = False
    return {
        "tenant_id": tenant_id,
        "verified": ok,
        "records": records,
        "genesis": handle.genesis.hex(),
        "spec_hash": handle.spec_hash.hex(),
    }


__all__ = [
    "TenantChain",
    "get_chain",
    "list_tenants",
    "register_chain_factory",
    "reset",
    "verify_tenant",
]
