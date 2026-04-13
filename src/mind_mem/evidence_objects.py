# Copyright 2026 STARGA, Inc.
"""mind-mem Evidence Objects — structured, tamper-evident governance records.

Every governance decision (propose, apply, rollback, contradiction, drift,
resolve, verify) gets an immutable evidence record that is self-hashing and
chain-linked. The chain can be verified at any point to detect tampering.

Hash computation:
    payload_hash   = sha256(canonical_payload_bytes)
    evidence_hash  = sha256(
        "{evidence_id}:{timestamp_iso}:{action}:{actor}:{target_block_id}:"
        "{payload_hash}:{previous_hash}"
    )

Integration points (do not import here — document only):
    audit_chain.py         — EvidenceChain extends the hash-chain concept with
                             structured, typed records per governance action.
    apply_engine.py        — Create APPLY evidence on successful proposal apply.
    contradiction_detector.py — Create CONTRADICT evidence when conflicts found.
    drift_detector.py      — Create DRIFT evidence when belief evolution detected.

Zero external deps — dataclasses, enum, hashlib, json, os, uuid (all stdlib).

Usage:
    from .evidence_objects import EvidenceChain, EvidenceAction

    chain = EvidenceChain()
    ev = chain.create(
        action=EvidenceAction.PROPOSE,
        actor="auto_resolver",
        target_block_id="D-20260401-007",
        target_file="decisions/DECISIONS.md",
        payload=b"proposed new content here",
        metadata={"proposal": "update priority to 5"},
        confidence=0.88,
    )
    assert chain.verify(ev)
    ok, broken = chain.verify_chain()
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Union
from uuid import uuid4

from .observability import get_logger, metrics

_log = get_logger("evidence_objects")

# Genesis seed — matches audit_chain.py convention
_GENESIS_HASH = "0" * 64


# ---------------------------------------------------------------------------
# EvidenceAction
# ---------------------------------------------------------------------------


class EvidenceAction(str, Enum):
    """Enumeration of governance actions that produce evidence records."""

    PROPOSE = "PROPOSE"
    APPLY = "APPLY"
    ROLLBACK = "ROLLBACK"
    CONTRADICT = "CONTRADICT"
    DRIFT = "DRIFT"
    RESOLVE = "RESOLVE"
    VERIFY = "VERIFY"


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _compute_payload_hash(payload: Union[bytes, str, dict, None]) -> str:
    """SHA256 hex digest of a governance payload.

    Accepts bytes, str (UTF-8 encoded), dict (JSON-serialised with sorted
    keys for determinism), or None (empty).
    """
    if payload is None:
        data = b""
    elif isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _compute_evidence_hash(
    evidence_id: str,
    timestamp_iso: str,
    action: str,
    actor: str,
    target_block_id: str,
    payload_hash: str,
    previous_hash: str,
    target_file: str = "",
    metadata: dict | None = None,
    confidence: float = 1.0,
) -> str:
    """SHA256 hex digest of the canonical evidence fields.

    Canonical form:
        "{evidence_id}:{timestamp_iso}:{action}:{actor}:{target_block_id}:{payload_hash}:{previous_hash}:{target_file}:{metadata_json}:{confidence}"
    """
    metadata_json = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))
    canonical = (
        f"{evidence_id}:{timestamp_iso}:{action}:{actor}:"
        f"{target_block_id}:{payload_hash}:{previous_hash}:"
        f"{target_file}:{metadata_json}:{confidence}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# EvidenceObject
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EvidenceObject:
    """Immutable, self-hashing evidence record for a governance action.

    All fields are set at construction time and cannot be modified.
    Tampering with any field will cause `EvidenceChain.verify()` to fail
    because `evidence_hash` will no longer match the recomputed value.
    """

    evidence_id: str
    """UUID4 string uniquely identifying this evidence record."""

    timestamp: datetime
    """UTC datetime when this evidence was created."""

    action: EvidenceAction
    """Governance action that generated this evidence."""

    actor: str
    """Who/what triggered this action (e.g. "user", "auto_resolver")."""

    target_block_id: str
    """ID of the memory block this evidence relates to."""

    target_file: str
    """Relative path of the file containing the target block."""

    payload_hash: str
    """SHA256 of the content being acted on (computed at creation)."""

    previous_hash: str
    """Evidence hash of the preceding record in the chain (genesis = "000...0")."""

    evidence_hash: str
    """SHA256 self-hash of this record's canonical fields (tamper-detection)."""

    metadata: dict
    """Action-specific ancillary data (proposal text, contradiction details, etc.)."""

    confidence: float
    """How confident the actor is in this decision (0.0–1.0)."""

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "evidence_id": self.evidence_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "actor": self.actor,
            "target_block_id": self.target_block_id,
            "target_file": self.target_file,
            "payload_hash": self.payload_hash,
            "previous_hash": self.previous_hash,
            "evidence_hash": self.evidence_hash,
            "metadata": self.metadata,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvidenceObject:
        """Deserialise from a dict (as produced by `to_dict`)."""
        return cls(
            evidence_id=d["evidence_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            action=EvidenceAction(d["action"]),
            actor=d["actor"],
            target_block_id=d["target_block_id"],
            target_file=d["target_file"],
            payload_hash=d["payload_hash"],
            previous_hash=d["previous_hash"],
            evidence_hash=d["evidence_hash"],
            metadata=d.get("metadata", {}),
            confidence=float(d.get("confidence", 1.0)),
        )


# ---------------------------------------------------------------------------
# EvidenceChain
# ---------------------------------------------------------------------------


class EvidenceChain:
    """Ordered, append-only chain of EvidenceObjects.

    Maintains an in-memory list of records.  When `store_path` is provided
    the chain is persisted as JSONL — one JSON object per line.  On
    initialisation with an existing file the chain is loaded and verified.

    Args:
        store_path: Optional path to a JSONL file for persistence.
                    The directory is created if it does not exist.
    """

    def __init__(self, store_path: str | None = None) -> None:
        self._store_path: str | None = store_path
        self._entries: list[EvidenceObject] = []
        self._integrity_compromised: bool = False

        if store_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)
            if os.path.isfile(store_path):
                self._load_from_file(store_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        action: EvidenceAction,
        actor: str,
        target_block_id: str,
        target_file: str,
        payload: Union[bytes, str, dict, None] = None,
        metadata: dict | None = None,
        confidence: float = 1.0,
        spec_hash: str | None = None,
    ) -> EvidenceObject:
        """Create and append a new evidence record.

        Args:
            action: Governance action being recorded.
            actor: Identity of the agent/user performing the action.
            target_block_id: Block ID the action targets.
            target_file: File path containing the target block.
            payload: Raw content being acted on (hashed, not stored raw).
            metadata: Arbitrary action-specific data dict.
            confidence: Decision confidence in [0.0, 1.0].

        Returns:
            The newly created and appended EvidenceObject.

        Raises:
            ValueError: If confidence is outside [0.0, 1.0].
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence!r}")

        # Merge spec_hash into metadata when provided
        effective_metadata: dict = dict(metadata or {})
        if spec_hash is not None:
            effective_metadata["spec_hash"] = spec_hash
        metadata = effective_metadata

        evidence_id = str(uuid4())
        now = datetime.now(timezone.utc)
        timestamp_iso = now.isoformat()

        payload_hash = _compute_payload_hash(payload)
        previous_hash = self._entries[-1].evidence_hash if self._entries else _GENESIS_HASH

        evidence_hash = _compute_evidence_hash(
            evidence_id,
            timestamp_iso,
            action.value,
            actor,
            target_block_id,
            payload_hash,
            previous_hash,
            target_file=target_file,
            metadata=metadata or {},
            confidence=confidence,
        )

        ev = EvidenceObject(
            evidence_id=evidence_id,
            timestamp=now,
            action=action,
            actor=actor,
            target_block_id=target_block_id,
            target_file=target_file,
            payload_hash=payload_hash,
            previous_hash=previous_hash,
            evidence_hash=evidence_hash,
            metadata=metadata or {},
            confidence=confidence,
        )

        self._entries.append(ev)

        if self._store_path is not None:
            self._append_to_file(ev)

        _log.info("evidence_created", action=action.value, actor=actor, target_block_id=target_block_id)
        metrics.inc("evidence_objects_created")
        return ev

    def __len__(self) -> int:
        """Return the number of entries currently in the chain."""
        with self._lock:
            return len(self._entries)

    def verify(self, evidence: EvidenceObject) -> bool:
        """Verify that an evidence object's self-hash matches its fields.

        Recomputes `evidence_hash` from the record's canonical fields and
        compares to the stored `evidence_hash`.  Returns False if any field
        has been tampered with.

        Args:
            evidence: The EvidenceObject to verify.

        Returns:
            True if the evidence is intact, False if tampered.
        """
        expected = _compute_evidence_hash(
            evidence.evidence_id,
            evidence.timestamp.isoformat(),
            evidence.action.value,
            evidence.actor,
            evidence.target_block_id,
            evidence.payload_hash,
            evidence.previous_hash,
            target_file=evidence.target_file,
            metadata=evidence.metadata,
            confidence=evidence.confidence,
        )
        return evidence.evidence_hash == expected

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the entire chain's integrity.

        Checks:
        1. Whether any records were silently dropped during load (tamper indicator).
        2. Each record's self-hash is valid (via `verify()`).
        3. Each record's `previous_hash` matches the preceding record's
           `evidence_hash` (or the genesis hash for the first record).

        Returns:
            (is_valid, broken_evidence_ids) — broken_evidence_ids is empty
            when the chain is fully intact.
        """
        if self._integrity_compromised:
            return False, ["load_integrity_compromised"]

        broken: list[str] = []

        if not self._entries:
            return True, []

        prev_hash = _GENESIS_HASH
        for ev in self._entries:
            if not self.verify(ev):
                broken.append(ev.evidence_id)
                # Still advance prev_hash to the stored value so subsequent
                # linkage errors are also caught accurately.
            if ev.previous_hash != prev_hash:
                if ev.evidence_id not in broken:
                    broken.append(ev.evidence_id)
            prev_hash = ev.evidence_hash

        is_valid = len(broken) == 0
        if is_valid:
            _log.info("evidence_chain_verify_ok", entries=len(self._entries))
        else:
            _log.warning("evidence_chain_verify_failed", broken=len(broken))
        return is_valid, broken

    def get_evidence_for_block(self, block_id: str) -> list[EvidenceObject]:
        """Return all evidence records for a given block ID.

        Args:
            block_id: The target block ID to filter on.

        Returns:
            Evidence records in creation order.
        """
        return [e for e in self._entries if e.target_block_id == block_id]

    def get_evidence_by_action(self, action: EvidenceAction) -> list[EvidenceObject]:
        """Return all evidence records for a given action type.

        Args:
            action: EvidenceAction to filter on.

        Returns:
            Evidence records in creation order.
        """
        return [e for e in self._entries if e.action == action]

    def get_latest(self, n: int = 10) -> list[EvidenceObject]:
        """Return the n most recently appended evidence records.

        Args:
            n: Maximum number of records to return (default 10).

        Returns:
            Up to n records, oldest-to-newest within the returned slice.
        """
        return self._entries[-n:] if n > 0 else []

    def export_jsonl(self, path: str) -> None:
        """Export the chain as JSONL (one JSON object per line).

        Args:
            path: Output file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for ev in self._entries:
                fh.write(json.dumps(ev.to_dict(), separators=(",", ":")) + "\n")
        _log.info("evidence_chain_exported", entries=len(self._entries), path=path)

    def import_jsonl(self, path: str) -> None:
        """Load and verify a JSONL chain file, replacing current state.

        Each record is verified for self-hash integrity and chain linkage.
        Raises ValueError if any record fails verification (tamper detected).

        Args:
            path: JSONL file path to import.

        Raises:
            ValueError: If any record in the file has been tampered with.
        """
        loaded: list[EvidenceObject] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    d = json.loads(stripped)
                    ev = EvidenceObject.from_dict(d)
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise ValueError(f"Line {line_no}: invalid evidence record — {exc}") from exc

                if not self.verify(ev):
                    raise ValueError(
                        f"Line {line_no}: tamper detected in evidence_id={ev.evidence_id}"
                    )
                loaded.append(ev)

        # Verify chain linkage after all records are parsed
        prev_hash = _GENESIS_HASH
        for idx, ev in enumerate(loaded, 1):
            if ev.previous_hash != prev_hash:
                raise ValueError(
                    f"Chain linkage broken at record {idx} (evidence_id={ev.evidence_id}): "
                    f"previous_hash mismatch"
                )
            prev_hash = ev.evidence_hash

        self._entries = loaded
        _log.info("evidence_chain_imported", entries=len(loaded), path=path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_to_file(self, ev: EvidenceObject) -> None:
        """Append a single evidence record to the JSONL store file."""
        with open(self._store_path, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
            fh.write(json.dumps(ev.to_dict(), separators=(",", ":")) + "\n")

    _MAX_LOAD_ENTRIES: int = 1_000_000
    _MAX_LOAD_LINE_BYTES: int = 1_048_576  # 1 MiB per JSONL line

    def _load_from_file(self, path: str) -> None:
        """Load all records from a JSONL file, verifying each entry's hash.

        Stops at the first integrity failure rather than silently skipping
        entries: continuing past a broken entry makes every downstream
        linkage check look broken, hiding the actual failure point and
        letting callers operate on a chain whose prefix is verified but
        whose suffix is untrusted.

        Enforces per-line size cap and total-entry cap to protect against
        pathologically large chains that would OOM the process.
        """
        previous_hash: str | None = None
        with open(path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                if line_num > self._MAX_LOAD_ENTRIES:
                    _log.warning(
                        "evidence_load_cap_reached",
                        cap=self._MAX_LOAD_ENTRIES,
                        path=path,
                    )
                    self._integrity_compromised = True
                    return
                if len(line.encode("utf-8")) > self._MAX_LOAD_LINE_BYTES:
                    _log.warning(
                        "evidence_load_line_too_large",
                        line=line_num,
                        cap=self._MAX_LOAD_LINE_BYTES,
                    )
                    self._integrity_compromised = True
                    return
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    ev = EvidenceObject.from_dict(json.loads(stripped))
                except (json.JSONDecodeError, KeyError, ValueError):
                    self._integrity_compromised = True
                    continue
                if not self.verify(ev):
                    _log.warning(
                        "evidence_hash_mismatch",
                        evidence_id=getattr(ev, "evidence_id", "?"),
                    )
                    self._integrity_compromised = True
                    continue
                if previous_hash is not None and ev.previous_hash != previous_hash:
                    _log.warning(
                        "evidence_chain_break",
                        evidence_id=getattr(ev, "evidence_id", "?"),
                        expected=previous_hash,
                        got=ev.previous_hash,
                    )
                    self._integrity_compromised = True
                    continue
                previous_hash = ev.evidence_hash
                self._entries.append(ev)
