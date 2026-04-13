# Copyright 2026 STARGA, Inc.
"""Optional external-ledger anchoring (v2.0.0rc1).

The roadmap calls for periodically publishing the Merkle root to an
external ledger (Ethereum L2 or similar). Actually *posting* to a
chain requires web3 keys + network access, which isn't something we
want baked into a retrieval library. Instead we ship:

- A local :class:`AnchorHistory` that records every root the caller
  asked to anchor, with block number, chain id, and the tx hash the
  external poster produced. When no external poster is wired, the
  record carries ``status="pending"`` and ``tx_hash=None``, still
  giving a complete local audit trail.
- :func:`anchor_root` — append an entry and return the manifest.
- ``anchor_history`` MCP tool (registered in mcp_server).

Callers integrating with a real chain wrap their poster around
:func:`anchor_root` and pass the transaction hash when it clears.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class AnchorEntry:
    merkle_root: str
    block_height: int
    timestamp: str
    chain: str
    tx_hash: Optional[str]
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "merkle_root": self.merkle_root,
            "block_height": self.block_height,
            "timestamp": self.timestamp,
            "chain": self.chain,
            "tx_hash": self.tx_hash,
            "status": self.status,
        }


class AnchorHistory:
    """Append-only JSONL history of anchored Merkle roots."""

    def __init__(self, path: str) -> None:
        if not path or not path.strip():
            raise ValueError("path must be a non-empty string")
        self._path = os.path.abspath(path)
        parent = os.path.dirname(self._path) or "."
        os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()

    @property
    def path(self) -> str:
        return self._path

    def record(
        self,
        merkle_root: str,
        *,
        block_height: int,
        chain: str = "local",
        tx_hash: Optional[str] = None,
        status: str = "pending",
    ) -> AnchorEntry:
        if not merkle_root or len(merkle_root) < 16:
            raise ValueError("merkle_root must be a non-trivial hash")
        if block_height < 0:
            raise ValueError("block_height must be >= 0")
        entry = AnchorEntry(
            merkle_root=merkle_root,
            block_height=int(block_height),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            chain=chain,
            tx_hash=tx_hash,
            status=status,
        )
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(entry.as_dict(), separators=(",", ":")) + "\n"
                )
                fh.flush()
                os.fsync(fh.fileno())
        return entry

    def all(self) -> list[AnchorEntry]:
        if not os.path.isfile(self._path):
            return []
        out: list[AnchorEntry] = []
        with open(self._path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                try:
                    out.append(
                        AnchorEntry(
                            merkle_root=str(data["merkle_root"]),
                            block_height=int(data["block_height"]),
                            timestamp=str(data["timestamp"]),
                            chain=str(data.get("chain", "local")),
                            tx_hash=data.get("tx_hash"),
                            status=str(data.get("status", "pending")),
                        )
                    )
                except (KeyError, ValueError, TypeError):
                    continue
        return out

    def latest(self) -> Optional[AnchorEntry]:
        entries = self.all()
        return entries[-1] if entries else None


def anchor_root(
    history: AnchorHistory,
    merkle_root: str,
    *,
    block_height: int,
    chain: str = "local",
    tx_hash: Optional[str] = None,
) -> AnchorEntry:
    return history.record(
        merkle_root,
        block_height=block_height,
        chain=chain,
        tx_hash=tx_hash,
        status="confirmed" if tx_hash else "pending",
    )


__all__ = ["AnchorEntry", "AnchorHistory", "anchor_root"]
