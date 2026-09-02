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
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import NoReturn, Union
from uuid import uuid4

from .admission import GovernanceBypassError
from .mind_filelock import FileLock, LockTimeout
from .observability import get_logger, metrics
from .preimage import preimage
from .q1616 import hex_q16_16

_log = get_logger("evidence_objects")

# Genesis seed — matches audit_chain.py convention
_GENESIS_HASH = "0" * 64

#: Schema tag stamped into ``metadata`` of every record this release writes.
#: The preimage already covers ``metadata``, so the tag is tamper-evident for
#: free. Absence of the key means a record written before the tag existed —
#: readers must treat a missing tag as "<= v3.0", never as an error.
EVIDENCE_SCHEMA_VERSION = "v3.1"

#: Seconds a writer waits for the cross-process append lock before giving up.
#: Generous: a governance chain is low-volume, and a writer that gives up
#: early would be back to guessing the tail — the thing this lock exists to
#: stop.
_APPEND_LOCK_TIMEOUT_SECONDS = 30.0

#: Bytes read per backwards step when resolving the on-disk tail record, so
#: the cost of an append does not grow with the length of the chain.
_TAIL_CHUNK_BYTES = 8192


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EvidenceChainCompromisedError(GovernanceBypassError):
    """A chain whose stored history did not load intact refuses to be written.

    :meth:`EvidenceChain._load_from_file` stops at the first record it
    cannot trust and leaves the in-memory chain **empty** — deliberately,
    so no caller mistakes a verified prefix for the whole history. That
    emptiness is exactly what makes an append catastrophic: the next
    record would take ``_GENESIS_HASH`` as its ``previous_hash`` and be
    written after the untrusted tail, so the file holds two chains rooted
    at genesis and the whole history stops verifying rather than merely
    its tail — the fork ``governance_gate.evict_gate`` documents as the
    thing that must not happen.

    A subclass of :class:`~mind_mem.admission.GovernanceBypassError`
    because that is what a refused governed write is; the gate's existing
    callers already abort on it.
    """


# ---------------------------------------------------------------------------
# EvidenceAction
# ---------------------------------------------------------------------------


class UnknownAction(str):
    """A governance verb this reader has no member for.

    A newer writer may record an action a older reader was never taught.
    Strict parsing (``EvidenceAction(raw)``) turns that into a
    ``ValueError``, which :meth:`EvidenceChain._load_from_file` reads as
    "unreadable record" and answers by freezing the whole chain — so one
    new verb would take the ledger offline for every older process
    sharing the workspace.

    So the action is **verified from its raw string** (it always was:
    the preimage hashes ``action.value``) and only *dispatched* through
    the enum. An unrecognised verb round-trips byte-identically and
    verifies; code that needs semantics matches on
    :class:`EvidenceAction` members and treats an ``UnknownAction`` as
    "something happened that I do not model", never as absence.

    A ``str`` subclass carrying the enum's ``.value``/``.name`` shape,
    following the ``Predicate.register`` precedent in
    ``knowledge_graph.py``.
    """

    __slots__ = ()

    @property
    def value(self) -> str:
        """The raw recorded verb — the same shape as ``EvidenceAction.value``."""
        return str(self)

    @property
    def name(self) -> str:
        """The raw recorded verb — the same shape as ``EvidenceAction.name``."""
        return str(self)

    def __repr__(self) -> str:
        return f"UnknownAction({str(self)!r})"


class EvidenceAction(str, Enum):
    """Enumeration of governance actions that produce evidence records.

    Adding a member is additive for writers and **breaking for a reader
    that parses strictly**; every reader in this package goes through
    :meth:`parse`, which never raises. A release older than the one that
    introduced :class:`UnknownAction` still parses strictly, so a chain
    carrying a member added after it will freeze *that* reader — which is
    why a new member and the writer that emits it are separate landings,
    reader first.
    """

    PROPOSE = "PROPOSE"
    APPLY = "APPLY"
    ROLLBACK = "ROLLBACK"
    CONTRADICT = "CONTRADICT"
    DRIFT = "DRIFT"
    RESOLVE = "RESOLVE"
    VERIFY = "VERIFY"

    @classmethod
    def parse(cls, raw: str) -> Union["EvidenceAction", UnknownAction]:
        """Return the member for *raw*, or an :class:`UnknownAction` sentinel.

        Never raises. This is the only supported way to turn a recorded
        action string back into something dispatchable.
        """
        try:
            return cls(raw)
        except ValueError:
            metrics.inc("evidence_unknown_action_read")
            return UnknownAction(raw)


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


def _compute_evidence_hash_v1(
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
    """Legacy v1 evidence-hash scheme (schema v2 and earlier).

    Uses JSON sorted-key canonical form. Preserved so that chains written
    before v2.10.0 still verify without migration. New code should call
    :func:`_compute_evidence_hash_v3` instead.
    """
    canonical_obj = {
        "evidence_id": evidence_id,
        "timestamp": timestamp_iso,
        "action": action,
        "actor": actor,
        "target_block_id": target_block_id,
        "payload_hash": payload_hash,
        "previous_hash": previous_hash,
        "target_file": target_file,
        "metadata": metadata or {},
        "confidence": confidence,
    }
    canonical = json.dumps(canonical_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_evidence_hash_v3(
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
    """v3 evidence-hash scheme (v2.10.0+).

    Uses :func:`preimage.preimage` for NUL-separated collision-resistant
    encoding and Q16.16 fixed-point for the ``confidence`` float so the
    digest is byte-identical across CPU architectures. The ``metadata``
    dict is still JSON-sorted into a single string slot — callers that
    want per-key hashing should expand the metadata before invoking.
    """
    metadata_json = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))
    pre = preimage(
        "EV_v1",
        evidence_id,
        timestamp_iso,
        action,
        actor,
        target_block_id,
        payload_hash,
        previous_hash,
        target_file,
        metadata_json,
        hex_q16_16(confidence),
    )
    return hashlib.sha256(pre).hexdigest()


# Public alias — new records hash via v3 by default. Legacy loaders that
# need to re-verify pre-v2.10.0 chains should explicitly call
# ``_compute_evidence_hash_v1``.
_compute_evidence_hash = _compute_evidence_hash_v3


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

    action: Union[EvidenceAction, UnknownAction]
    """Governance action that generated this evidence.

    An :class:`EvidenceAction` member when this reader models the verb,
    an :class:`UnknownAction` when a newer writer recorded one it does
    not. Both expose ``.value``, and the raw string is what the hash
    covers, so an unmodelled verb still verifies.
    """

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
            action=EvidenceAction.parse(d["action"]),
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

    A chain whose stored history does not load intact is **frozen**:
    :meth:`create` and :meth:`export_jsonl` raise
    :class:`EvidenceChainCompromisedError` instead of operating on the
    empty in-memory chain a failed load leaves behind.  Appending would
    fork the ledger at genesis; exporting would present that emptiness as
    the history.  Reads still work and :meth:`verify_chain` still reports
    the failure.

    **Cross-process safety.** A store file is shared by every process
    that opens the same workspace, so an in-memory tail is only ever a
    guess about what is on disk. Each persisted append therefore takes
    the cross-process :class:`~mind_mem.mind_filelock.FileLock` on the
    store, absorbs whatever other writers appended
    (:meth:`_refresh_from_store`), and cross-checks its tail against the
    record actually last on disk before linking to it. A writer that
    still believes it is at genesis while the store holds records is
    refused rather than allowed to start a second chain behind the first
    — the fork that leaves a file full of zero-``previous_hash`` rows.

    Args:
        store_path: Optional path to a JSONL file for persistence.
                    The directory is created if it does not exist.
    """

    def __init__(self, store_path: str | None = None) -> None:
        self._store_path: str | None = store_path
        self._entries: list[EvidenceObject] = []
        self._integrity_compromised: bool = False
        self._load_failure: str | None = None
        # Serialize concurrent create() calls so the in-memory chain and the
        # on-disk JSONL cannot interleave entries or diverge from each other.
        self._lock = threading.RLock()
        # Bytes of the store this chain has already read. Everything past it
        # was appended by somebody else and must be absorbed before we can
        # know the tail. 0 for "nothing read yet", which is also the truth
        # for a store that does not exist.
        self._store_offset: int = 0

        if store_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)
            if os.path.isfile(store_path):
                try:
                    # Under the same lock the writers use: a load racing an
                    # append would otherwise read a half-written last line
                    # and freeze a perfectly healthy chain.
                    with self._store_lock():
                        self._read_store(store_path)
                except (OSError, LockTimeout):
                    # Reading is allowed to degrade; writing is not. A chain
                    # archived read-only after a re-anchor is exactly the
                    # thing an auditor must still be able to verify, and it
                    # is a directory no lockfile can be created in. Loading
                    # unlocked is what every release before this one did, so
                    # the fallback is never worse than the status quo — while
                    # create() keeps failing closed on the same condition,
                    # because an unserialised append is how the ledger forks.
                    _log.info("evidence_load_without_lock", path=store_path)
                    self._read_store(store_path)

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
            EvidenceChainCompromisedError: If the stored chain did not load
                intact, if another writer's records do not link, or if this
                writer's view of the store is stale (including the genesis
                case: this chain believes it is empty while the store holds
                records). Appending is refused rather than forking the ledger.
            ValueError: If confidence is outside [0.0, 1.0].
            LockTimeout: If the cross-process append lock on the store could
                not be taken within ``_APPEND_LOCK_TIMEOUT_SECONDS``.
        """
        # Refuse BEFORE validating arguments: a chain that could not be read
        # in full has no trustworthy tail to link to, so there is no such
        # thing as a well-formed append to it. `_entries` is empty after a
        # failed load, so an unguarded append would silently restart the
        # chain at `_GENESIS_HASH` behind the untrusted records already on
        # disk. verify_chain() alone is not a guard — nothing on the write
        # path consults it.
        self._raise_if_compromised("append to")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence!r}")

        # Merge spec_hash into metadata when provided
        effective_metadata: dict = dict(metadata or {})
        if spec_hash is not None:
            effective_metadata["spec_hash"] = spec_hash
        # Stamp the schema this record was written under. setdefault, not
        # assignment: a deliberate re-anchoring operation writes its own tag
        # and must not have it silently overwritten.
        effective_metadata.setdefault("evidence_schema", EVIDENCE_SCHEMA_VERSION)
        metadata = effective_metadata

        payload_hash = _compute_payload_hash(payload)

        with self._lock:
            if self._store_path is None:
                # Memory-only chain: no file, so nothing to serialise against
                # and no disk tail to consult.
                ev = self._forge(
                    previous_hash=(self._entries[-1].evidence_hash if self._entries else _GENESIS_HASH),
                    action=action,
                    actor=actor,
                    target_block_id=target_block_id,
                    target_file=target_file,
                    payload_hash=payload_hash,
                    metadata=metadata,
                    confidence=confidence,
                )
                self._entries.append(ev)
            else:
                # One writer at a time, across processes, for the whole
                # read-tail → compute → append sequence. Splitting the read
                # from the append is what let two processes each believe they
                # owned the tail.
                with self._store_lock():
                    ev = self._forge(
                        previous_hash=self._linkable_previous_hash(),
                        action=action,
                        actor=actor,
                        target_block_id=target_block_id,
                        target_file=target_file,
                        payload_hash=payload_hash,
                        metadata=metadata,
                        confidence=confidence,
                    )
                    # Persist FIRST so an I/O failure cannot leave in-memory
                    # state ahead of the on-disk chain. Only append to
                    # _entries once the durable write has landed.
                    self._append_to_file(ev)
                    self._entries.append(ev)

        _log.info("evidence_created", action=action.value, actor=actor, target_block_id=target_block_id)
        metrics.inc("evidence_objects_created")
        return ev

    def __len__(self) -> int:
        """Return the number of entries currently in the chain."""
        with self._lock:
            return len(self._entries)

    @property
    def integrity_compromised(self) -> bool:
        """True when the stored chain did not load intact.

        A frozen chain: reads are served from whatever loaded, but
        :meth:`create` and :meth:`export_jsonl` refuse.
        """
        return self._integrity_compromised

    @property
    def load_failure(self) -> str | None:
        """Why the stored chain could not be loaded in full, or None."""
        return self._load_failure

    def verify(self, evidence: EvidenceObject) -> bool:
        """Verify that an evidence object's self-hash matches its fields.

        Convenience wrapper around :meth:`_verify_scheme`. Returns
        ``True`` when either the v3 or v1 scheme matches, so callers
        that don't care about which scheme passed still get a single-
        bool answer. Chain-level downgrade detection lives in
        :meth:`verify_chain`.
        """
        return self._verify_scheme(evidence) is not None

    def _verify_scheme(self, evidence: EvidenceObject) -> str | None:
        """Return which scheme verified the record, or None.

        ``"v3"`` if the v2.10.0+ scheme matched, ``"v1"`` for the
        legacy scheme, ``None`` if neither. :meth:`verify_chain` uses
        this to enforce no-downgrade-after-v3.
        """
        args = (
            evidence.evidence_id,
            evidence.timestamp.isoformat(),
            evidence.action.value,
            evidence.actor,
            evidence.target_block_id,
            evidence.payload_hash,
            evidence.previous_hash,
        )
        if evidence.evidence_hash == _compute_evidence_hash_v3(
            *args,
            target_file=evidence.target_file,
            metadata=evidence.metadata,
            confidence=evidence.confidence,
        ):
            return "v3"
        if evidence.evidence_hash == _compute_evidence_hash_v1(
            *args,
            target_file=evidence.target_file,
            metadata=evidence.metadata,
            confidence=evidence.confidence,
        ):
            return "v1"
        return None

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
        seen_v3 = False  # downgrade-attack mitigation: once v3 scheme
        # has signed a record in this chain, no later
        # record is permitted to verify only under the
        # separator-injection-vulnerable v1 scheme.
        for ev in self._entries:
            scheme = self._verify_scheme(ev)
            if scheme is None:
                broken.append(ev.evidence_id)
            elif scheme == "v3":
                seen_v3 = True
            elif scheme == "v1" and seen_v3:
                # downgrade detected
                broken.append(ev.evidence_id)
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

        Raises:
            EvidenceChainCompromisedError: If the stored chain did not load
                intact — the in-memory chain is then empty, and writing it
                out would publish that emptiness as the history (and would
                truncate the store outright were *path* the store itself).
        """
        self._raise_if_compromised("export")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for ev in self._entries:
                fh.write(json.dumps(ev.to_dict(), separators=(",", ":")) + "\n")
        _log.info("evidence_chain_exported", entries=len(self._entries), path=path)

    def import_jsonl(self, path: str) -> None:
        """Load and verify a JSONL chain file, replacing current state.

        Each record is verified for self-hash integrity and chain linkage.
        Raises ValueError if any record fails verification (tamper detected).

        A successful import does **not** unfreeze a chain that was frozen
        by a failed load: ``store_path`` still points at the file that
        could not be read, and that file — not the imported one — is what
        :meth:`create` would append to. Repair the store instead.

        Nor does it make the imported history appendable to a *different*
        store: the next :meth:`create` resolves its link against the
        store's own last record, so importing a foreign chain (or an
        empty one) over a populated store is refused at the next append
        rather than silently forking it.

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
                    raise ValueError(f"Line {line_no}: tamper detected in evidence_id={ev.evidence_id}")
                loaded.append(ev)

        # Verify chain linkage after all records are parsed
        prev_hash = _GENESIS_HASH
        for idx, ev in enumerate(loaded, 1):
            if ev.previous_hash != prev_hash:
                raise ValueError(f"Chain linkage broken at record {idx} (evidence_id={ev.evidence_id}): previous_hash mismatch")
            prev_hash = ev.evidence_hash

        self._entries = loaded
        _log.info("evidence_chain_imported", entries=len(loaded), path=path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_compromised(self, reason: str) -> None:
        """Freeze the chain and record why the stored history was rejected.

        Called from :meth:`_load_from_file` only, before it abandons the
        load. Every reason — tamper, unparseable line, broken linkage, or
        a size/entry cap — means the same thing for the write path: the
        real tail of the chain is unknown, so nothing may be appended.
        """
        self._integrity_compromised = True
        self._load_failure = reason
        metrics.inc("evidence_load_integrity_compromised")

    def _read_store(self, store_path: str) -> None:
        """Load the stored chain and record how far into the file we read."""
        self._load_from_file(store_path)
        try:
            self._store_offset = os.path.getsize(store_path)
        except OSError:  # pragma: no cover - stat failing right after a read
            self._store_offset = 0

    def _store_lock(self) -> FileLock:
        """The cross-process lock guarding this chain's store file.

        Every reader and writer of the store takes it, so a load never
        observes a half-written line and two appends can never resolve the
        same tail. ``FileLock``'s per-path thread lock is **not**
        reentrant, so this must never be nested with itself.
        """
        return FileLock(str(self._store_path), timeout=_APPEND_LOCK_TIMEOUT_SECONDS)

    def _freeze_and_raise(self, reason: str) -> NoReturn:
        """Freeze the chain for *reason* and refuse the write.

        Every caller has established that the real tail of the stored
        chain is not what this process thought it was. Writing anyway is
        exactly the fork this class exists to prevent, and repairing the
        history by rewriting hashes is never this code's decision.
        """
        self._mark_compromised(reason)
        raise EvidenceChainCompromisedError(
            f"EvidenceChain refuses to append to {self._store_path!r}: {reason}. "
            "No hash is ever rewritten to repair this — archive the stored "
            "chain and re-anchor it as a deliberate operation."
        )

    def _read_disk_tail_hash(self) -> str | None:
        """``evidence_hash`` of the last record on disk, or None if there is none.

        Reads backwards from the end so an append does not cost a pass
        over the whole chain. ``None`` means the store holds no record
        (absent, empty, or nothing but blank lines) — never "could not
        tell": a tail that exists but cannot be parsed freezes the chain,
        because a writer that treated it as absence would restart at
        genesis behind it.

        Caller holds :meth:`_store_lock`.
        """
        path = self._store_path
        if path is None:
            return None
        buf = b""
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                pos = fh.tell()
                while pos > 0:
                    step = min(_TAIL_CHUNK_BYTES, pos)
                    pos -= step
                    fh.seek(pos)
                    buf = fh.read(step) + buf
                    # A newline before the final byte means the last
                    # non-blank line is bounded on both sides and therefore
                    # complete; otherwise keep walking backwards.
                    if b"\n" in buf[:-1]:
                        break
        except FileNotFoundError:
            return None
        except OSError as exc:
            self._freeze_and_raise(f"the store could not be read ({exc})")
        lines = [line for line in buf.split(b"\n") if line.strip()]
        if not lines:
            return None
        try:
            record = json.loads(lines[-1].decode("utf-8"))
            return str(record["evidence_hash"])
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
            self._freeze_and_raise(f"the store's last line is not a readable evidence record ({exc})")

    def _refresh_from_store(self) -> None:
        """Absorb records other writers appended since this chain last read.

        The append lock serialises writers, but each still holds its own
        copy of the chain in memory. Without this, a second writer's idea
        of the tail stays whatever it read at construction time — the
        stale tail that forks the ledger.

        Only the bytes past :attr:`_store_offset` are read, so following a
        busy store costs the new records, not the whole file. Every
        absorbed record is verified and linked; anything else freezes the
        chain rather than being skipped.

        Caller holds :meth:`_store_lock`.
        """
        path = self._store_path
        if path is None:
            return
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            if self._entries:
                self._freeze_and_raise("the store this chain was loaded from no longer exists")
            self._store_offset = 0
            return
        except OSError as exc:
            self._freeze_and_raise(f"the store could not be stat'd ({exc})")
        if size == self._store_offset:
            return
        if size < self._store_offset:
            self._freeze_and_raise(f"the store shrank from {self._store_offset} to {size} bytes — append-only history was rewritten")

        prev_hash = self._entries[-1].evidence_hash if self._entries else _GENESIS_HASH
        absorbed: list[EvidenceObject] = []
        with open(path, "rb") as fh:
            fh.seek(self._store_offset)
            for raw in fh:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    ev = EvidenceObject.from_dict(json.loads(stripped.decode("utf-8")))
                except (json.JSONDecodeError, UnicodeDecodeError, KeyError, ValueError) as exc:
                    self._freeze_and_raise(f"a record appended by another writer is unreadable ({exc})")
                if not self.verify(ev):
                    self._freeze_and_raise(f"a record appended by another writer fails its own hash (evidence_id={ev.evidence_id})")
                if ev.previous_hash != prev_hash:
                    self._freeze_and_raise(f"a record appended by another writer does not link (evidence_id={ev.evidence_id})")
                prev_hash = ev.evidence_hash
                absorbed.append(ev)

        self._entries.extend(absorbed)
        self._store_offset = size
        if absorbed:
            metrics.inc("evidence_records_absorbed")
            _log.info("evidence_chain_absorbed", records=len(absorbed), path=path)

    def _linkable_previous_hash(self) -> str:
        """The hash the next record must carry as ``previous_hash``.

        Resolved against the store, never taken on trust from memory:
        the in-memory tail is refreshed from disk first and then checked
        against the record actually last on disk.

        Two disagreements are refused, and they are different failures:

        * **genesis into a non-empty chain** — this writer believes the
          chain is empty while the store holds records. That is the live
          fork signature: every such writer restarts at
          ``_GENESIS_HASH`` and the file ends up holding several chains
          rooted at genesis. Reachable without any concurrency at all
          (``import_jsonl`` of an empty file leaves exactly this state).
        * **a tail that is not the store's tail** — this writer has
          history the store does not, or a different history.

        Caller holds :meth:`_store_lock`.
        """
        self._refresh_from_store()
        disk_tail = self._read_disk_tail_hash()
        memory_tail = self._entries[-1].evidence_hash if self._entries else None

        if memory_tail is None:
            if disk_tail is not None:
                self._freeze_and_raise(
                    "genesis into non-empty chain — this writer holds no history "
                    f"while the store's last record is {disk_tail[:16]}…; appending "
                    "would root a second chain at the genesis hash behind the first"
                )
            return _GENESIS_HASH

        if memory_tail != disk_tail:
            self._freeze_and_raise(
                f"this writer's tail {memory_tail[:16]}… is not the store's last "
                f"record ({'nothing on disk' if disk_tail is None else disk_tail[:16] + '…'})"
            )
        return memory_tail

    def _forge(
        self,
        *,
        previous_hash: str,
        action: EvidenceAction,
        actor: str,
        target_block_id: str,
        target_file: str,
        payload_hash: str,
        metadata: dict,
        confidence: float,
    ) -> EvidenceObject:
        """Build one self-hashing record linked to *previous_hash*.

        Pure construction — no I/O, no chain state — so the caller decides
        under which locks the record is minted and appended.
        """
        evidence_id = str(uuid4())
        now = datetime.now(timezone.utc)
        timestamp_iso = now.isoformat()
        evidence_hash = _compute_evidence_hash(
            evidence_id,
            timestamp_iso,
            action.value,
            actor,
            target_block_id,
            payload_hash,
            previous_hash,
            target_file=target_file,
            metadata=metadata,
            confidence=confidence,
        )
        return EvidenceObject(
            evidence_id=evidence_id,
            timestamp=now,
            action=action,
            actor=actor,
            target_block_id=target_block_id,
            target_file=target_file,
            payload_hash=payload_hash,
            previous_hash=previous_hash,
            evidence_hash=evidence_hash,
            metadata=metadata,
            confidence=confidence,
        )

    def _raise_if_compromised(self, operation: str) -> None:
        """Refuse *operation* when the stored chain did not load intact."""
        if self._integrity_compromised:
            raise EvidenceChainCompromisedError(
                f"EvidenceChain refuses to {operation} {self._store_path!r}: "
                f"its stored history did not load intact ({self._load_failure}). "
                "The in-memory chain is empty, so proceeding would fork the "
                "ledger at the genesis hash and invalidate the whole audit "
                "history. Repair or archive the stored chain first."
            )

    def _append_to_file(self, ev: EvidenceObject) -> None:
        """Append a single evidence record to the JSONL store file.

        Writes are flushed and fsync'd so that a crash between create() and
        the next event cannot lose the record. Callers must hold self._lock
        *and* :meth:`_store_lock` — the first stops parallel create() calls
        in this process interleaving, the second stops another process
        appending between our tail read and our write.

        The record we just wrote is one this chain has read, so
        :attr:`_store_offset` moves past it; otherwise the next append
        would re-absorb our own record and see it as somebody else's.
        """
        line = json.dumps(ev.to_dict(), separators=(",", ":")) + "\n"
        with open(self._store_path, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._store_offset += len(line.encode("utf-8"))

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
        loaded: list[EvidenceObject] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                if line_num > self._MAX_LOAD_ENTRIES:
                    _log.warning(
                        "evidence_load_cap_reached",
                        cap=self._MAX_LOAD_ENTRIES,
                        path=path,
                    )
                    self._mark_compromised(f"entry cap of {self._MAX_LOAD_ENTRIES} exceeded at line {line_num}")
                    return
                if len(line.encode("utf-8")) > self._MAX_LOAD_LINE_BYTES:
                    _log.warning(
                        "evidence_load_line_too_large",
                        line=line_num,
                        cap=self._MAX_LOAD_LINE_BYTES,
                    )
                    self._mark_compromised(f"line {line_num} exceeds the {self._MAX_LOAD_LINE_BYTES}-byte line cap")
                    return
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    ev = EvidenceObject.from_dict(json.loads(stripped))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    _log.warning("evidence_load_parse_error", line=line_num, error=str(exc))
                    self._mark_compromised(f"line {line_num} is not a readable evidence record")
                    return
                if not self.verify(ev):
                    _log.warning(
                        "evidence_hash_mismatch",
                        line=line_num,
                        evidence_id=getattr(ev, "evidence_id", "?"),
                    )
                    self._mark_compromised(f"hash mismatch at line {line_num} — record was altered")
                    return
                if previous_hash is not None and ev.previous_hash != previous_hash:
                    _log.warning(
                        "evidence_chain_break",
                        line=line_num,
                        evidence_id=getattr(ev, "evidence_id", "?"),
                        expected=previous_hash,
                        got=ev.previous_hash,
                    )
                    self._mark_compromised(f"chain linkage breaks at line {line_num}")
                    return
                previous_hash = ev.evidence_hash
                loaded.append(ev)
        self._entries = loaded
