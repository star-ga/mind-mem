# Copyright 2026 STARGA, Inc.
"""Redactable-leaf tombstones — deletion that survives tamper-evidence.

``delete_memory_item`` removes a block from the corpus, but the audit
surfaces (``hash_chain_v2``, ``evidence_objects``, ``audit_chain``, the
Merkle tree built from :func:`sqlite_index.merkle_leaves`) are
append-only. Deleting a block therefore used to do two contradictory
things at once: it dropped the block's Merkle leaf (so every previously
issued inclusion proof stopped verifying) while *keeping* a full
plaintext copy in ``memory/deleted_blocks.jsonl`` (so the content was
never actually forgotten).

A tombstone resolves the collision the way redactable ledgers do:

* the block's **content is destroyed** — corpus text, index rows,
  cached embeddings, and any pre-existing plaintext recovery receipt;
* the block's **leaf is preserved** — the SHA-256 content hash that
  went into the Merkle tree stays in this ledger and is re-supplied to
  :func:`sqlite_index.merkle_leaves`, so the root and every historical
  inclusion proof still verify;
* the **deletion event is itself chained** — actor, reason and the
  destroyed content's digests are admitted through
  :class:`~mind_mem.governance_gate.GovernanceGate` (evidence chain +
  SHA3-512 hash chain) and recorded in the SHA-256
  :class:`~mind_mem.audit_chain.AuditChain`; the ledger record binds
  those receipts and links to the previous tombstone.

Ledger: ``<workspace>/.mind-mem-audit/tombstones.jsonl`` — one JSON
object per line, in the same directory as the v1 audit chain.

**Opt-in.** Everything here is inert unless the workspace enables
``v4.redactable_tombstones``; with the flag off, deletion keeps its
historical (recoverable) behaviour byte-for-byte. Every read helper
short-circuits when the ledger file is absent, so a workspace that has
never redacted pays a single ``os.stat``.

Zero external deps — hashlib, json, os, dataclasses (all stdlib).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import threading
from typing import Any, Final

from .mind_filelock import FileLock
from .observability import get_logger, metrics
from .preimage import preimage

_log = get_logger("tombstone")

#: v4 feature-flag key. OFF unless ``mind-mem.json`` says otherwise.
TOMBSTONE_FLAG: Final[str] = "redactable_tombstones"

#: Ledger location, relative to the workspace root.
LEDGER_REL_PATH: Final[str] = os.path.join(".mind-mem-audit", "tombstones.jsonl")

#: Genesis sentinel for the ledger's own back-linkage (SHA-256 width).
GENESIS_HASH: Final[str] = "0" * 64

#: Where the preserved Merkle leaf hash came from, most authoritative first.
LEAF_SOURCES: Final[tuple[str, ...]] = ("index_meta", "parsed_block", "raw_text")

_BLOCK_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z]+-[a-zA-Z0-9_.-]+$")
_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

_cache_lock = threading.RLock()
#: ``{ledger_path: (stat_key, records)}`` — invalidated by mtime_ns + size.
_cache: dict[str, tuple[tuple[int, int], tuple["Tombstone", ...]]] = {}


class TombstoneError(ValueError):
    """Raised when a tombstone record is malformed or a redaction is refused."""


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Tombstone:
    """Immutable proof that a block existed, was redacted, and by whom.

    Carries no content — only digests of it. ``leaf_hash`` is the exact
    value the redacted block contributed to the Merkle tree so the root
    is unchanged by the redaction.
    """

    block_id: str
    redacted_at: str
    actor: str
    reason: str
    leaf_hash: str
    leaf_source: str
    content_sha3_512: str
    content_bytes: int
    source_file: str
    evidence_id: str
    evidence_hash: str
    chain_entry_id: str
    chain_entry_hash: str
    audit_seq: int
    audit_entry_hash: str
    previous_hash: str
    record_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise with a fixed key order (deterministic JSONL bytes)."""
        return {
            "schema": "mind-mem-tombstone-v1",
            "block_id": self.block_id,
            "redacted_at": self.redacted_at,
            "actor": self.actor,
            "reason": self.reason,
            "leaf_hash": self.leaf_hash,
            "leaf_source": self.leaf_source,
            "content_sha3_512": self.content_sha3_512,
            "content_bytes": self.content_bytes,
            "source_file": self.source_file,
            "evidence_id": self.evidence_id,
            "evidence_hash": self.evidence_hash,
            "chain_entry_id": self.chain_entry_id,
            "chain_entry_hash": self.chain_entry_hash,
            "audit_seq": self.audit_seq,
            "audit_entry_hash": self.audit_entry_hash,
            "previous_hash": self.previous_hash,
            "record_hash": self.record_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Tombstone":
        """Rebuild from a ledger line. Raises :class:`TombstoneError` on gaps."""
        try:
            return cls(
                block_id=str(d["block_id"]),
                redacted_at=str(d["redacted_at"]),
                actor=str(d["actor"]),
                reason=str(d["reason"]),
                leaf_hash=str(d["leaf_hash"]),
                leaf_source=str(d["leaf_source"]),
                content_sha3_512=str(d["content_sha3_512"]),
                content_bytes=int(d["content_bytes"]),
                source_file=str(d["source_file"]),
                evidence_id=str(d["evidence_id"]),
                evidence_hash=str(d["evidence_hash"]),
                chain_entry_id=str(d["chain_entry_id"]),
                chain_entry_hash=str(d["chain_entry_hash"]),
                audit_seq=int(d["audit_seq"]),
                audit_entry_hash=str(d["audit_entry_hash"]),
                previous_hash=str(d["previous_hash"]),
                record_hash=str(d["record_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TombstoneError(f"malformed tombstone record: {exc}") from exc

    def public_view(self) -> dict[str, Any]:
        """Operator-facing summary — what a tombstoned block looks like.

        Deliberately distinguishable from a never-existed block: the
        block ID is present, ``status`` is ``tombstoned``, and the
        preserved leaf lets the caller re-verify the old Merkle proof.
        """
        return {
            "block_id": self.block_id,
            "status": "tombstoned",
            "redacted_at": self.redacted_at,
            "actor": self.actor,
            "reason": self.reason,
            "leaf_hash": self.leaf_hash,
            "content_bytes": self.content_bytes,
            "source_file": self.source_file,
            "evidence_hash": self.evidence_hash,
            "chain_entry_hash": self.chain_entry_hash,
            "audit_entry_hash": self.audit_entry_hash,
            "record_hash": self.record_hash,
        }


def compute_record_hash(
    *,
    block_id: str,
    redacted_at: str,
    actor: str,
    reason: str,
    leaf_hash: str,
    leaf_source: str,
    content_sha3_512: str,
    content_bytes: int,
    source_file: str,
    evidence_id: str,
    evidence_hash: str,
    chain_entry_id: str,
    chain_entry_hash: str,
    audit_seq: int,
    audit_entry_hash: str,
    previous_hash: str,
) -> str:
    """SHA-256 over a ``TOMB_v1`` NUL-separated preimage.

    Same construction as the v3 evidence/audit hashes: the tag prevents
    cross-class collisions and the NUL separator makes field boundaries
    unambiguous even when a free-text ``reason`` contains punctuation.
    """
    pre = preimage(
        "TOMB_v1",
        block_id,
        redacted_at,
        actor,
        reason,
        leaf_hash,
        leaf_source,
        content_sha3_512,
        content_bytes,
        source_file,
        evidence_id,
        evidence_hash,
        chain_entry_id,
        chain_entry_hash,
        audit_seq,
        audit_entry_hash,
        previous_hash,
    )
    return hashlib.sha256(pre).hexdigest()


# ---------------------------------------------------------------------------
# Flag + ledger location
# ---------------------------------------------------------------------------


def tombstones_enabled(workspace: str | None = None) -> bool:
    """Return True iff redactable tombstones are switched on.

    Workspace-local ``mind-mem.json`` wins (a compliance switch belongs
    to the corpus it governs); otherwise the global v4 flag registry
    decides. Fail-closed: any read/parse problem means OFF.
    """
    if workspace:
        local = os.path.join(os.path.abspath(workspace), "mind-mem.json")
        if os.path.isfile(local):
            try:
                with open(local, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                return False
            v4 = data.get("v4")
            if isinstance(v4, dict):
                sub = v4.get(TOMBSTONE_FLAG)
                if isinstance(sub, dict):
                    return sub.get("enabled") is True
            return False
    from .v4.feature_flags import is_enabled

    return is_enabled(TOMBSTONE_FLAG)


def ledger_path(workspace: str) -> str:
    """Absolute path of the workspace's tombstone ledger."""
    return os.path.join(os.path.abspath(workspace), LEDGER_REL_PATH)


def ledger_exists(workspace: str) -> bool:
    """True when a non-empty ledger exists (the cheap short-circuit)."""
    try:
        return os.path.getsize(ledger_path(workspace)) > 0
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def load_tombstones(workspace: str) -> tuple[Tombstone, ...]:
    """Return every ledger record in write order (empty when no ledger).

    Cached per path and invalidated on ``(mtime_ns, size)`` so repeated
    recall queries don't re-read the file. Unparsable lines are skipped
    here and reported by :func:`verify_ledger`.
    """
    path = ledger_path(workspace)
    try:
        st = os.stat(path)
    except OSError:
        with _cache_lock:
            _cache.pop(path, None)
        return ()
    stat_key = (st.st_mtime_ns, st.st_size)
    with _cache_lock:
        hit = _cache.get(path)
        if hit is not None and hit[0] == stat_key:
            return hit[1]

    records: list[Tombstone] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(Tombstone.from_dict(json.loads(stripped)))
                except (ValueError, TombstoneError):
                    continue
    except OSError:
        return ()

    frozen = tuple(records)
    with _cache_lock:
        _cache[path] = (stat_key, frozen)
    return frozen


def invalidate_cache(workspace: str | None = None) -> None:
    """Drop the ledger cache (all workspaces when *workspace* is None)."""
    with _cache_lock:
        if workspace is None:
            _cache.clear()
        else:
            _cache.pop(ledger_path(workspace), None)


def tombstoned_ids(workspace: str) -> frozenset[str]:
    """Block IDs that have been redacted in *workspace*."""
    return frozenset(t.block_id for t in load_tombstones(workspace))


def get_tombstone(workspace: str, block_id: str) -> Tombstone | None:
    """Most recent tombstone for *block_id*, or None."""
    found: Tombstone | None = None
    for record in load_tombstones(workspace):
        if record.block_id == block_id:
            found = record
    return found


def tombstone_leaves(workspace: str) -> list[tuple[str, str]]:
    """Preserved ``(block_id, leaf_hash)`` pairs, sorted by block ID.

    A block redacted twice (re-created, re-deleted) contributes its
    latest leaf — the one the tree held when the redaction happened.
    """
    latest: dict[str, str] = {}
    for record in load_tombstones(workspace):
        latest[record.block_id] = record.leaf_hash
    return sorted(latest.items())


def merge_leaves(
    live: list[tuple[str, str]],
    tombstoned: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Union live and preserved leaves, live winning on a shared block ID.

    Deterministic: output is sorted by block ID. A live leaf wins
    because a block that exists again has real content in the tree; the
    tombstone leaf only fills the hole a redaction would otherwise
    leave (which is what keeps historical proofs verifying).
    """
    if not tombstoned:
        return live
    merged = {bid: h for bid, h in tombstoned}
    merged.update({bid: h for bid, h in live})
    return sorted(merged.items())


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


def _validate(field: str, value: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TombstoneError(f"{field} must be a non-empty string")
    if "\x00" in value:
        raise TombstoneError(f"{field} must not contain NUL")
    if pattern is not None and not pattern.match(value):
        raise TombstoneError(f"{field} has an invalid format: {value!r}")
    return value


def append_tombstone(
    workspace: str,
    *,
    block_id: str,
    redacted_at: str,
    actor: str,
    reason: str,
    leaf_hash: str,
    leaf_source: str,
    content_sha3_512: str,
    content_bytes: int,
    source_file: str,
    evidence_id: str,
    evidence_hash: str,
    chain_entry_id: str,
    chain_entry_hash: str,
    audit_seq: int,
    audit_entry_hash: str,
) -> Tombstone:
    """Append a tombstone, linking it to the previous ledger record.

    Boundary validation is strict — a tombstone is the only surviving
    proof that the block existed, so a malformed one is refused rather
    than written. Raises :class:`TombstoneError`.
    """
    _validate("block_id", block_id, _BLOCK_ID_RE)
    _validate("actor", actor)
    _validate("reason", reason)
    _validate("leaf_hash", leaf_hash, _HEX64_RE)
    _validate("redacted_at", redacted_at)
    if leaf_source not in LEAF_SOURCES:
        raise TombstoneError(f"leaf_source must be one of {LEAF_SOURCES}, got {leaf_source!r}")
    if not isinstance(content_bytes, int) or content_bytes < 0:
        raise TombstoneError("content_bytes must be a non-negative int")

    path = ledger_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with FileLock(path + ".lock"):
        existing = load_tombstones(workspace)
        previous_hash = existing[-1].record_hash if existing else GENESIS_HASH
        record_hash = compute_record_hash(
            block_id=block_id,
            redacted_at=redacted_at,
            actor=actor,
            reason=reason,
            leaf_hash=leaf_hash,
            leaf_source=leaf_source,
            content_sha3_512=content_sha3_512,
            content_bytes=content_bytes,
            source_file=source_file,
            evidence_id=evidence_id,
            evidence_hash=evidence_hash,
            chain_entry_id=chain_entry_id,
            chain_entry_hash=chain_entry_hash,
            audit_seq=audit_seq,
            audit_entry_hash=audit_entry_hash,
            previous_hash=previous_hash,
        )
        record = Tombstone(
            block_id=block_id,
            redacted_at=redacted_at,
            actor=actor,
            reason=reason,
            leaf_hash=leaf_hash,
            leaf_source=leaf_source,
            content_sha3_512=content_sha3_512,
            content_bytes=content_bytes,
            source_file=source_file,
            evidence_id=evidence_id,
            evidence_hash=evidence_hash,
            chain_entry_id=chain_entry_id,
            chain_entry_hash=chain_entry_hash,
            audit_seq=audit_seq,
            audit_entry_hash=audit_entry_hash,
            previous_hash=previous_hash,
            record_hash=record_hash,
        )
        line = json.dumps(record.to_dict(), separators=(",", ":")) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        invalidate_cache(workspace)

    _log.info("tombstone_appended", block_id=block_id, actor=actor)
    metrics.inc("tombstones_appended")
    return record


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_ledger(workspace: str) -> tuple[bool, list[str]]:
    """Verify every record's self-hash and back-linkage.

    Returns ``(ok, errors)``. An absent ledger is valid-and-empty. Line
    level parse failures are errors here (unlike :func:`load_tombstones`,
    which skips them) — a corrupted tombstone line is exactly the tamper
    an auditor needs to see.
    """
    path = ledger_path(workspace)
    if not os.path.isfile(path):
        return True, []

    errors: list[str] = []
    prev_hash = GENESIS_HASH
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = Tombstone.from_dict(json.loads(stripped))
                except (ValueError, TombstoneError) as exc:
                    errors.append(f"line {line_num}: unreadable record: {exc}")
                    continue

                expected = compute_record_hash(
                    block_id=record.block_id,
                    redacted_at=record.redacted_at,
                    actor=record.actor,
                    reason=record.reason,
                    leaf_hash=record.leaf_hash,
                    leaf_source=record.leaf_source,
                    content_sha3_512=record.content_sha3_512,
                    content_bytes=record.content_bytes,
                    source_file=record.source_file,
                    evidence_id=record.evidence_id,
                    evidence_hash=record.evidence_hash,
                    chain_entry_id=record.chain_entry_id,
                    chain_entry_hash=record.chain_entry_hash,
                    audit_seq=record.audit_seq,
                    audit_entry_hash=record.audit_entry_hash,
                    previous_hash=record.previous_hash,
                )
                if record.record_hash != expected:
                    errors.append(f"line {line_num} ({record.block_id}): record_hash tampered")
                if record.previous_hash != prev_hash:
                    errors.append(f"line {line_num} ({record.block_id}): previous_hash mismatch — a tombstone was removed or reordered")
                prev_hash = record.record_hash
    except OSError as exc:
        errors.append(f"cannot read tombstone ledger: {exc}")

    ok = not errors
    if not ok:
        _log.warning("tombstone_ledger_verify_failed", errors=len(errors))
    return ok, errors


__all__ = [
    "GENESIS_HASH",
    "LEAF_SOURCES",
    "LEDGER_REL_PATH",
    "TOMBSTONE_FLAG",
    "Tombstone",
    "TombstoneError",
    "append_tombstone",
    "compute_record_hash",
    "get_tombstone",
    "invalidate_cache",
    "ledger_exists",
    "ledger_path",
    "load_tombstones",
    "merge_leaves",
    "tombstone_leaves",
    "tombstoned_ids",
    "tombstones_enabled",
    "verify_ledger",
]
