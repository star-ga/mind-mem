# Copyright 2026 STARGA, Inc.
"""P2P memory mesh (v2.6.0) — explicit peer list, 7 sync scopes.

The roadmap calls for an mDNS-discovered distributed mesh with
last-write-wins for hot tiers and governance-gated merges for cold
tiers. This module ships the scope-tracking + conflict-resolution
core plus an explicit-peer-list configuration; mDNS discovery itself
stays optional / deferred.

In a production deployment the transport layer (HTTP, gRPC, QUIC)
slots in around this core — the core holds no network dependencies,
so it unit-tests cleanly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional


class SyncScope(str, Enum):
    """Seven independently-toggleable scopes (one per roadmap bullet)."""

    MEMORIES = "memories"
    ACTIONS = "actions"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    RELATIONS = "relations"
    GRAPH = "graph"
    GOVERNANCE = "governance"


class ConflictResolution(str, Enum):
    LAST_WRITE_WINS = "last_write_wins"
    GOVERNANCE_GATED = "governance_gated"


@dataclass(frozen=True)
class Peer:
    peer_id: str
    endpoint: str
    scopes: tuple[SyncScope, ...] = tuple()
    last_seen: Optional[str] = None


@dataclass
class SyncEvent:
    """Single audit-log entry for a mesh sync exchange."""

    timestamp: str
    peer_id: str
    scope: SyncScope
    blocks_transferred: int
    conflicts_resolved: int
    resolution: ConflictResolution


class MemoryMesh:
    """Tracks peers + scope policy + an append-only sync audit log."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._peers: dict[str, Peer] = {}
        self._policy: dict[SyncScope, ConflictResolution] = {
            SyncScope.MEMORIES: ConflictResolution.LAST_WRITE_WINS,
            SyncScope.ACTIONS: ConflictResolution.LAST_WRITE_WINS,
            SyncScope.SEMANTIC: ConflictResolution.GOVERNANCE_GATED,
            SyncScope.PROCEDURAL: ConflictResolution.GOVERNANCE_GATED,
            SyncScope.RELATIONS: ConflictResolution.GOVERNANCE_GATED,
            SyncScope.GRAPH: ConflictResolution.GOVERNANCE_GATED,
            SyncScope.GOVERNANCE: ConflictResolution.GOVERNANCE_GATED,
        }
        self._log: list[SyncEvent] = []

    def add_peer(
        self,
        peer_id: str,
        endpoint: str,
        scopes: Iterable[SyncScope] = (),
    ) -> Peer:
        if not peer_id.strip():
            raise ValueError("peer_id must be non-empty")
        if not endpoint.strip():
            raise ValueError("endpoint must be non-empty")
        peer = Peer(
            peer_id=peer_id.strip(),
            endpoint=endpoint.strip(),
            scopes=tuple(scopes),
            last_seen=None,
        )
        with self._lock:
            self._peers[peer.peer_id] = peer
        return peer

    def remove_peer(self, peer_id: str) -> bool:
        with self._lock:
            return self._peers.pop(peer_id, None) is not None

    def peers(self) -> list[Peer]:
        with self._lock:
            return list(self._peers.values())

    def set_policy(self, scope: SyncScope, policy: ConflictResolution) -> None:
        with self._lock:
            self._policy[scope] = policy

    def resolve_conflict(
        self,
        scope: SyncScope,
        local: Mapping[str, Any],
        remote: Mapping[str, Any],
    ) -> dict:
        """Apply the scope's resolution policy.

        Returns the winning record. For ``LAST_WRITE_WINS`` the
        ``updated_at`` key (or ``timestamp``) breaks ties; for
        governance-gated scopes the local copy wins and a proposal is
        emitted via ``metadata.requires_review=True``.
        """
        policy = self._policy.get(scope, ConflictResolution.GOVERNANCE_GATED)
        if policy is ConflictResolution.LAST_WRITE_WINS:
            lt = str(local.get("updated_at") or local.get("timestamp") or "")
            rt = str(remote.get("updated_at") or remote.get("timestamp") or "")
            return dict(remote if rt > lt else local)
        # Governance-gated — keep local, surface review marker.
        merged = dict(local)
        meta = dict(merged.get("metadata") or {})
        meta["requires_review"] = True
        meta["pending_remote"] = dict(remote)
        merged["metadata"] = meta
        return merged

    def log_sync(
        self,
        peer_id: str,
        scope: SyncScope,
        blocks_transferred: int,
        conflicts_resolved: int,
    ) -> SyncEvent:
        resolution = self._policy.get(scope, ConflictResolution.GOVERNANCE_GATED)
        event = SyncEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            peer_id=peer_id,
            scope=scope,
            blocks_transferred=int(blocks_transferred),
            conflicts_resolved=int(conflicts_resolved),
            resolution=resolution,
        )
        with self._lock:
            self._log.append(event)
            if peer_id in self._peers:
                self._peers[peer_id] = Peer(
                    peer_id=peer_id,
                    endpoint=self._peers[peer_id].endpoint,
                    scopes=self._peers[peer_id].scopes,
                    last_seen=event.timestamp,
                )
        return event

    def audit_log(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "timestamp": e.timestamp,
                    "peer_id": e.peer_id,
                    "scope": e.scope.value,
                    "blocks_transferred": e.blocks_transferred,
                    "conflicts_resolved": e.conflicts_resolved,
                    "resolution": e.resolution.value,
                }
                for e in self._log
            ]

    def status(self) -> dict:
        with self._lock:
            return {
                "peer_count": len(self._peers),
                "peers": [
                    {
                        "peer_id": p.peer_id,
                        "endpoint": p.endpoint,
                        "scopes": [s.value for s in p.scopes],
                        "last_seen": p.last_seen,
                    }
                    for p in self._peers.values()
                ],
                "policy": {s.value: p.value for s, p in self._policy.items()},
                "events_logged": len(self._log),
            }


__all__ = ["SyncScope", "ConflictResolution", "Peer", "SyncEvent", "MemoryMesh"]
