"""Raft-style consensus wrapper for governance writes (v4.0 prep).

v3.x runs governance (append to audit chain, apply proposal) on a
single node. v4.0 cluster deployments need strong consistency on
writes — two agents applying the same proposal on two nodes should
either both succeed with the same resulting chain state or neither
succeeds. Recall reads stay eventual-consistency (stale-okay).

This module is the façade: a governance operation is proposed as a
:class:`Proposal`, the :class:`ConsensusLog` replicates it to a
quorum, and only after commit does the caller run the side effect
(chain append, snapshot, rollback). For dev/test the module ships
an in-process :class:`LocalConsensusLog` that commits immediately
— v4.0 will bolt on a real Raft implementation (etcd-raft,
OpenRaft, hashicorp/raft over gRPC) via ``register_consensus_log``.

Pure Python, zero network. Real Raft transports plug in as:

    from mind_mem.governance_raft import register_consensus_log
    register_consensus_log(lambda cfg: EtcdRaftConsensusLog(cfg))

The governance layer only sees the :class:`ConsensusLog` Protocol
— swapping implementations doesn't touch caller code.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from .observability import get_logger

_log = get_logger("governance_raft")


@dataclass(frozen=True)
class Proposal:
    """Governance operation proposed to the consensus log.

    ``operation`` is the canonical op name (``APPEND``, ``APPLY``,
    ``ROLLBACK``, ``SNAPSHOT``). ``payload`` is the op-specific body
    (the audit record being appended, the proposal ID to apply, etc.).
    ``client_id`` identifies the proposer — every follower validates
    proposals were signed by an authorised client before committing.
    """

    operation: str
    payload: dict[str, Any]
    client_id: str
    # Monotonic proposal timestamp — tie-breaks concurrent proposals
    # in the commit order (Raft itself handles this but the façade
    # records it for audit).
    ts_monotonic: float = field(default_factory=time.monotonic)

    def digest(self) -> bytes:
        """Stable SHA-256 of the proposal contents — used as the replica ID."""
        body = json.dumps(
            {
                "operation": self.operation,
                "payload": self.payload,
                "client_id": self.client_id,
                "ts_monotonic": self.ts_monotonic,
            },
            default=str,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(body).digest()


@dataclass(frozen=True)
class CommitResult:
    """Outcome of submitting a proposal to the consensus log.

    ``committed`` is true only after the quorum acks the proposal.
    ``term`` and ``index`` follow Raft semantics — monotonically
    increasing across the cluster. ``digest`` echoes the proposal
    digest so callers can correlate.
    """

    committed: bool
    term: int
    index: int
    digest: bytes
    reason: str = ""


@runtime_checkable
class ConsensusLog(Protocol):
    """Interface between mind-mem governance and the underlying
    consensus implementation.

    The real v4.0 implementation is a Raft log backed by a cluster
    of mind-mem nodes. Callers submit a proposal, wait for the
    CommitResult, and only then run the operation locally — the
    cluster's commit event drives side effects on every node.
    """

    def submit(self, proposal: Proposal, *, timeout_seconds: float = 5.0) -> CommitResult: ...

    def subscribe(self, handler: Callable[[Proposal, CommitResult], None]) -> None:
        """Register a callback invoked on every committed proposal."""
        ...

    def current_term(self) -> int: ...

    def is_leader(self) -> bool: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# LocalConsensusLog — in-process, single-node. For dev/test.
# ---------------------------------------------------------------------------


class LocalConsensusLog:
    """Single-node stand-in for a real Raft cluster.

    Every submission commits immediately (there's no quorum to ack).
    Term stays at 1 and index monotonically increments. Useful for
    tests and single-node deployments — v4.0 production callers
    replace this with a real Raft log via ``register_consensus_log``.
    """

    def __init__(self) -> None:
        self._index = 0
        self._lock = threading.Lock()
        self._handlers: list[Callable[[Proposal, CommitResult], None]] = []
        self._log: list[tuple[Proposal, CommitResult]] = []

    def submit(self, proposal: Proposal, *, timeout_seconds: float = 5.0) -> CommitResult:
        with self._lock:
            self._index += 1
            result = CommitResult(
                committed=True,
                term=1,
                index=self._index,
                digest=proposal.digest(),
                reason="local_immediate",
            )
            self._log.append((proposal, result))
            handlers = list(self._handlers)
        for handler in handlers:
            try:
                handler(proposal, result)
            except Exception as exc:
                _log.warning("consensus_handler_failed", error=str(exc))
        _log.info(
            "consensus_local_commit",
            operation=proposal.operation,
            index=result.index,
        )
        return result

    def subscribe(self, handler: Callable[[Proposal, CommitResult], None]) -> None:
        with self._lock:
            self._handlers.append(handler)

    def current_term(self) -> int:
        return 1

    def is_leader(self) -> bool:
        # Single-node is always the leader.
        return True

    def close(self) -> None:
        with self._lock:
            self._handlers.clear()

    # --- introspection helpers (tests only; not on the Protocol) -----------

    def log_entries(self) -> list[tuple[Proposal, CommitResult]]:
        with self._lock:
            return list(self._log)


# ---------------------------------------------------------------------------
# Pluggable registry
# ---------------------------------------------------------------------------


_FactoryFn = Callable[[dict[str, Any]], ConsensusLog]
_factory: _FactoryFn | None = None


def register_consensus_log(factory: _FactoryFn) -> None:
    """Override the default log factory at deployment init.

    Called once by the real Raft-backed implementation (e.g. OpenRaft
    adapter in ``mind-mem-cluster``) so mind-mem's governance layer
    picks it up without source changes.
    """
    global _factory
    _factory = factory


def create_consensus_log(config: dict[str, Any] | None = None) -> ConsensusLog:
    """Return a consensus log per the current registered factory.

    Defaults to :class:`LocalConsensusLog` — safe for every v3.x
    single-node deployment.
    """
    cfg = config or {}
    if _factory is not None:
        return _factory(cfg)
    return LocalConsensusLog()


def sign_proposal(proposal: Proposal, secret: bytes) -> bytes:
    """HMAC-SHA-256 signature over a proposal's digest.

    v4.0 production Raft clusters require every proposal be signed
    by an authorised client; followers reject unsigned proposals.
    Shipped here so the signing contract is stable across transport
    impls.
    """
    if not isinstance(secret, (bytes, bytearray)) or len(secret) < 16:
        raise ValueError("signing secret must be ≥16 bytes")
    return hmac.new(bytes(secret), proposal.digest(), hashlib.sha256).digest()


def verify_proposal(proposal: Proposal, signature: bytes, secret: bytes) -> bool:
    if not isinstance(secret, (bytes, bytearray)) or len(secret) < 16:
        return False
    expected = sign_proposal(proposal, secret)
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Helpers for use-at-a-callsite
# ---------------------------------------------------------------------------


def replicate(
    log: ConsensusLog,
    proposals: Iterable[Proposal],
    *,
    timeout_seconds: float = 5.0,
) -> list[CommitResult]:
    """Submit a batch of proposals sequentially, returning each result.

    Real Raft implementations may batch internally; this helper is
    just a convenience for callers that want per-proposal results.
    """
    return [log.submit(p, timeout_seconds=timeout_seconds) for p in proposals]


__all__ = [
    "ConsensusLog",
    "CommitResult",
    "LocalConsensusLog",
    "Proposal",
    "create_consensus_log",
    "register_consensus_log",
    "replicate",
    "sign_proposal",
    "verify_proposal",
]
