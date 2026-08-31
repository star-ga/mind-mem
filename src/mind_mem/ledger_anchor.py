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
from dataclasses import dataclass
from typing import Any, Optional

from .observability import get_logger

_log = get_logger("ledger_anchor")


class AnchorHistoryDamagedError(RuntimeError):
    """Raised by :meth:`AnchorHistory.all` (strict) on an unreadable record."""


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
    """Append-only JSONL history of anchored Merkle roots.

    A line the reader cannot turn back into an :class:`AnchorEntry` is
    *damage*, not absence: dropping it silently shortens the history and
    makes :meth:`latest` hand back the preceding anchor as if it were
    current. Every unreadable line is therefore reported — as a warning
    log on each read, through :meth:`problems`, and as a hard error from
    :meth:`all` when the caller asks for ``strict=True``.
    """

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
                fh.write(json.dumps(entry.as_dict(), separators=(",", ":")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return entry

    def _scan(self) -> tuple[list[AnchorEntry], list[str]]:
        """Read the log once, returning ``(entries, problems)``.

        *problems* holds one ``"line N: <reason>"`` string per record
        that could not be reconstructed.
        """
        if not os.path.isfile(self._path):
            return [], []
        out: list[AnchorEntry] = []
        problems: list[str] = []
        with open(self._path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    problems.append(f"line {lineno}: not valid JSON ({exc.msg})")
                    continue
                if not isinstance(data, dict):
                    problems.append(f"line {lineno}: expected a JSON object, got {type(data).__name__}")
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
                except (KeyError, ValueError, TypeError) as exc:
                    problems.append(f"line {lineno}: malformed anchor record ({type(exc).__name__}: {exc})")
        if problems:
            _log.warning(
                "anchor_history_damaged",
                path=self._path,
                damaged=len(problems),
                readable=len(out),
                first=problems[0],
            )
        return out, problems

    def problems(self) -> list[str]:
        """Unreadable records in the log, one description per damaged line."""
        return self._scan()[1]

    def all(self, *, strict: bool = False) -> list[AnchorEntry]:
        """Every readable anchor, oldest first.

        Args:
            strict: raise :class:`AnchorHistoryDamagedError` instead of
                skipping when any record is unreadable. Callers that
                treat this file as an audit trail — rather than as a
                best-effort cache — should pass ``True``.
        """
        entries, problems = self._scan()
        if problems and strict:
            raise AnchorHistoryDamagedError(f"{self._path}: {len(problems)} unreadable record(s): {'; '.join(problems[:3])}")
        return entries

    def latest(self, *, strict: bool = False) -> Optional[AnchorEntry]:
        """Most recent readable anchor.

        With ``strict=False`` a damaged tail silently yields the previous
        anchor, which a caller comparing today's Merkle root would read
        as a self-consistent older answer — pass ``strict=True`` to make
        that damage an error instead.
        """
        entries = self.all(strict=strict)
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


__all__ = ["AnchorEntry", "AnchorHistory", "AnchorHistoryDamagedError", "anchor_root"]
