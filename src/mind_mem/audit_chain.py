#!/usr/bin/env python3
"""mind-mem field-level audit sidecar — tamper-evident append-only ledger.

**Not the ledger of record.** What a mutation was allowed to do is
recorded by the gate in ``hash_chain_v2``; why it was allowed is
recorded in ``evidence_chain``. This module is a *sidecar* of those
two: it carries field-granular detail for the four doors that write to
it -- three distinct verbs between them -- and nothing else reaches it. It is one of the four ledgers
``mind-mem-verify`` walks, and the only one whose rows name individual
fields.

Everything that writes here, and nothing else does:

* ``field_audit.record_change``          -> ``update_field``
* ``compliance.audit.record_redaction``  -> ``update_field``
* ``auto_resolver`` (resolution applied) -> ``apply_proposal``
* ``importers.quarantine``               -> ``create_block``

:data:`VALID_OPERATIONS` is exactly that written set, and
``tests/test_ledger_hierarchy.py`` re-derives it from the source by
AST, so the two cannot drift without failing the build. Do not read a
missing verb as a missing record: a delete is recorded by the gate and
the ``deleted_blocks`` ledger, a rollback by ``apply_engine.rollback``
in the evidence chain — neither lands here, and until 5.0.2 this
docstring claimed both did. The verbs this module advertised but never
wrote are kept in :data:`RETIRED_OPERATIONS`, so a pre-5.0.2 ledger
carrying them still reads and still verifies; :meth:`AuditChain.append`
refuses to mint a new one.

Each entry names the gate admission it happened under —
``admission_entry_id``, the ``hash_chain`` entry id published by
:func:`~mind_mem.admission.current_admission` — whenever a scope is
open. That is the link from a sidecar row back to the chain of record,
and it is read from the ambient scope rather than passed by the
caller, so a new writer gets it without knowing it exists. A row
appended outside any admission scope omits the field rather than
inventing a link.

deferred: ``admission_entry_id`` sits outside the entry-hash preimage,
exactly like ``fields_changed``, so it is recorded but not
authenticated — an edit to the link alone does not break the chain.
Upgrade path: a v4 preimage (new TAG, ``AUDIT_v2``) covering both
fields, added reader-first so an older verifier is never handed a row
it cannot check.

Ledger location: workspace/.mind-mem-audit/chain.jsonl
Each line is a JSON object with:
    seq, timestamp, operation, target, agent, reason,
    payload_hash, prev_hash, entry_hash
    [, fields_changed] [, admission_entry_id]

Usage:
    from .audit_chain import AuditChain
    chain = AuditChain(workspace)
    chain.append("update_field", "decisions/DECISIONS.md",
                 agent="claude", reason="User corrected priority",
                 payload={"field": "Priority", "old": "3", "new": "5"})
    ok, errors = chain.verify()

Zero external deps — hashlib, json, os (all stdlib).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

from .admission import current_admission
from .mind_filelock import FileLock
from .observability import get_logger, metrics

_log = get_logger("audit_chain")

# Genesis block seed — fixed, never changes. Sixty-four ASCII zeros: a
# well-known constant anchor, NOT the digest of any string. export()
# writes this literal into the exported payload so an external
# verifier reads the anchor rather than recomputing it.
_GENESIS_HASH = "0" * 64

#: The operations a door actually appends here. Derived from the source
#: by ``tests/test_ledger_hierarchy.py``: adding a member without a
#: writer, or a writer whose verb is not a member, fails the build.
VALID_OPERATIONS = frozenset(
    {
        "create_block",
        "update_field",
        "apply_proposal",
    }
)

#: Verbs a pre-5.0.2 ``VALID_OPERATIONS`` accepted that no door ever
#: wrote. Kept so an existing ledger carrying one still parses and still
#: verifies -- :meth:`AuditChain.verify` never consults either set, and
#: an operator's own ``mm``-era rows must not become unreadable because
#: we corrected the contract. Refused for NEW appends: minting one now
#: would re-advertise a record this ledger does not keep. Each is
#: recorded elsewhere (the gate's hash chain, the evidence chain, or
#: ``deleted_blocks``); none of them is lost.
RETIRED_OPERATIONS = frozenset(
    {
        "delete_block",
        "supersede",
        "append_block",
        "set_status",
        "replace_range",
        "append_list_item",
        "insert_after_block",
        "restore_backup",
        "rollback",
    }
)

#: Everything a reader may legitimately encounter on disk.
READABLE_OPERATIONS = VALID_OPERATIONS | RETIRED_OPERATIONS


def _compute_hash(data: str) -> str:
    """SHA256 hex digest of UTF-8 encoded data."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _payload_hash(payload: dict | str | None) -> str:
    """Compute hash of mutation payload for content verification."""
    if payload is None:
        return _compute_hash("")
    if isinstance(payload, str):
        return _compute_hash(payload)
    return _compute_hash(json.dumps(payload, sort_keys=True, default=str))


class AuditEntry:
    """Single entry in the hash chain."""

    __slots__ = (
        "seq",
        "timestamp",
        "operation",
        "target",
        "agent",
        "reason",
        "payload_hash",
        "prev_hash",
        "entry_hash",
        "fields_changed",
        "admission_entry_id",
    )

    def __init__(
        self,
        seq: int,
        timestamp: str,
        operation: str,
        target: str,
        agent: str,
        reason: str,
        payload_hash: str,
        prev_hash: str,
        entry_hash: str,
        fields_changed: list[str] | None = None,
        admission_entry_id: str | None = None,
    ):
        self.seq = seq
        self.timestamp = timestamp
        self.operation = operation
        self.target = target
        self.agent = agent
        self.reason = reason
        self.payload_hash = payload_hash
        self.prev_hash = prev_hash
        self.entry_hash = entry_hash
        self.fields_changed = fields_changed or []
        #: ``hash_chain`` entry id of the gate admission this row was
        #: appended under, or ``None`` when no scope was open. Never
        #: fabricated: absent means "not written inside an admission",
        #: which is a fact about the row, not a gap in it.
        self.admission_entry_id = admission_entry_id

    def to_dict(self) -> dict:
        d: dict = {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "target": self.target,
            "agent": self.agent,
            "reason": self.reason,
            "payload_hash": self.payload_hash,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }
        if self.fields_changed:
            d["fields_changed"] = self.fields_changed
        if self.admission_entry_id:
            d["admission_entry_id"] = self.admission_entry_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AuditEntry:
        return cls(
            seq=d["seq"],
            timestamp=d["timestamp"],
            operation=d["operation"],
            target=d["target"],
            agent=d.get("agent", ""),
            reason=d.get("reason", ""),
            payload_hash=d["payload_hash"],
            prev_hash=d["prev_hash"],
            entry_hash=d["entry_hash"],
            fields_changed=d.get("fields_changed"),
            admission_entry_id=d.get("admission_entry_id"),
        )

    @staticmethod
    def compute_entry_hash_v1(
        seq: int,
        timestamp: str,
        operation: str,
        target: str,
        agent: str,
        reason: str,
        payload_hash: str,
        prev_hash: str,
    ) -> str:
        """Legacy v1 audit-entry hash (``|``-joined).

        Preserved so pre-v2.10.0 audit ledgers continue to verify.

        Audit FP-9: this preimage is structurally ambiguous — a
        ``|`` byte inside any field (reason free-text in particular)
        collides with the field separator and lets a hostile writer
        forge two semantically different entries that hash the same.
        v3 (compute_entry_hash_v3) closed this with TAG_v1 NUL-separated
        length-prefixed preimages. v1 is kept ONLY for backward-compat
        verification of pre-v2.10.0 ledgers; we emit a diagnostic
        warning when any input field contains ``|`` so operators can
        flag the historical entry for re-signing.
        """
        for name, value in (
            ("timestamp", timestamp),
            ("operation", operation),
            ("target", target),
            ("agent", agent),
            ("reason", reason),
            ("payload_hash", payload_hash),
            ("prev_hash", prev_hash),
        ):
            if isinstance(value, str) and "|" in value:
                logging.getLogger("mind_mem.audit_chain").warning(
                    "LEGACY_SEPARATOR_AMBIGUITY",
                    extra={
                        "field": name,
                        "seq": seq,
                        "advice": (
                            "v1 preimage uses '|' as a separator; this field "
                            "contains a literal '|' which makes the hash "
                            "ambiguous. Re-sign with v3 (TAG_v1) preimage."
                        ),
                    },
                )
        canonical = f"{seq}|{timestamp}|{operation}|{target}|{agent}|{reason}|{payload_hash}|{prev_hash}"
        return _compute_hash(canonical)

    @staticmethod
    def compute_entry_hash_v3(
        seq: int,
        timestamp: str,
        operation: str,
        target: str,
        agent: str,
        reason: str,
        payload_hash: str,
        prev_hash: str,
    ) -> str:
        """v3 audit-entry hash (v2.10.0+): TAG_v1 NUL-separated preimage."""
        from .preimage import preimage

        pre = preimage(
            "AUDIT_v1",
            seq,
            timestamp,
            operation,
            target,
            agent,
            reason,
            payload_hash,
            prev_hash,
        )
        return hashlib.sha256(pre).hexdigest()

    # New entries hash via v3 by default; verify tries both.
    @staticmethod
    def compute_entry_hash(
        seq: int,
        timestamp: str,
        operation: str,
        target: str,
        agent: str,
        reason: str,
        payload_hash: str,
        prev_hash: str,
    ) -> str:
        """Dispatch to the active (v3) hash scheme."""
        return AuditEntry.compute_entry_hash_v3(seq, timestamp, operation, target, agent, reason, payload_hash, prev_hash)


class AuditChainCorruptedError(OSError):
    """The ledger tail exists but cannot be read, so the next link is unknown.

    Subclasses :class:`OSError` because that is what the condition means to
    a caller: the chain on disk could not be read back, whether the file
    refused the read or its last line is not an entry. Existing governed
    writers that already convert a ledger-write failure into their own
    refusal (``importers.quarantine`` catches ``OSError``/``ValueError``
    and raises ``ImportQuarantineError``) therefore keep working, instead
    of a fresh exception class slipping past their handler.

    Raised by :meth:`AuditChain.append`. An unreadable or unparsable last
    line is *not* the same thing as an empty chain: collapsing the two
    makes the next append restart at ``seq 1`` with the genesis
    ``prev_hash`` on top of the entries already on disk. That forks the
    linkage silently — the writer gets an ordinary :class:`AuditEntry`
    back and only a later :meth:`AuditChain.verify` reveals that the
    append-only ledger has been growing across a break. Appending fails
    closed instead; run ``verify()`` to locate the damage.
    """


class AuditChain:
    """Append-only hash-chained audit ledger.

    Thread-safe via FileLock. Each append acquires the lock,
    reads the last entry's hash, computes the new entry hash,
    and appends atomically.
    """

    def __init__(self, workspace: str) -> None:
        self.workspace = os.path.realpath(workspace)
        self._audit_dir = os.path.join(self.workspace, ".mind-mem-audit")
        self._chain_path = os.path.join(self._audit_dir, "chain.jsonl")
        os.makedirs(self._audit_dir, exist_ok=True)

    def _last_entry(self) -> AuditEntry | None:
        """Read the last entry from the chain file.

        ``None`` means the chain genuinely holds no entry — no file, or a
        file with no non-blank line. A tail that exists but cannot be read
        or parsed raises :class:`AuditChainCorruptedError` rather than
        returning ``None``; see that class for why the two cases must not
        collapse into one.
        """
        if not os.path.isfile(self._chain_path):
            return None
        last_line = ""
        try:
            with open(self._chain_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        last_line = stripped
        except (OSError, UnicodeDecodeError) as exc:
            raise AuditChainCorruptedError(f"cannot read audit chain {self._chain_path}: {exc}") from exc
        if not last_line:
            return None
        try:
            return AuditEntry.from_dict(json.loads(last_line))
        except (json.JSONDecodeError, KeyError) as exc:
            raise AuditChainCorruptedError(
                f"last line of audit chain {self._chain_path} is not a readable entry ({exc}); "
                "refusing to append on top of a broken link — run verify() to locate the damage"
            ) from exc

    def _next_seq(self, last: AuditEntry | None) -> int:
        """Get the next sequence number."""
        if last is None:
            return 1
        return last.seq + 1

    def append(
        self,
        operation: str,
        target: str,
        *,
        agent: str = "",
        reason: str = "",
        payload: dict | str | None = None,
        fields_changed: list[str] | None = None,
    ) -> AuditEntry:
        """Append a new mutation entry to the chain.

        Args:
            operation: One of :data:`VALID_OPERATIONS`. A
                :data:`RETIRED_OPERATIONS` verb is refused with the
                ledger that does record it named in the message.
            target: Relative path of the affected file/block.
            agent: Identity of who made the change.
            reason: Governance justification for the change.
            payload: Mutation content (hashed, not stored raw).
            fields_changed: List of field names that were modified.

        Returns:
            The created AuditEntry.

        Raises:
            ValueError: If operation is not valid.
            AuditChainCorruptedError: If the existing chain's last entry
                cannot be read or parsed. The append is refused rather
                than restarted at seq 1 on top of the damaged tail.
        """
        if operation not in VALID_OPERATIONS:
            if operation in RETIRED_OPERATIONS:
                raise ValueError(
                    f"Operation '{operation}' is retired: no door has ever written it to the "
                    f"field-audit sidecar, and minting one now would advertise a record this "
                    f"ledger does not keep. It is recorded in the chain of record instead "
                    f"(hash_chain_v2 + evidence_chain; deletes also in deleted_blocks). "
                    f"Appendable here: {sorted(VALID_OPERATIONS)}"
                )
            raise ValueError(f"Invalid operation '{operation}'. Must be one of: {sorted(VALID_OPERATIONS)}")

        # Normalize target to relative path
        if os.path.isabs(target):
            try:
                target = os.path.relpath(target, self.workspace)
            except ValueError:
                pass  # Different drives on Windows

        with FileLock(self._chain_path):
            last = self._last_entry()
            prev_hash = last.entry_hash if last else _GENESIS_HASH
            seq = self._next_seq(last)
            ts = datetime.now(timezone.utc).isoformat()
            p_hash = _payload_hash(payload)

            entry_hash = AuditEntry.compute_entry_hash(seq, ts, operation, target, agent, reason, p_hash, prev_hash)

            # The link to the chain of record is taken from the ambient
            # scope, not from an argument: a caller cannot forget to pass
            # it, and a future door gets the link without knowing the
            # field exists. A ContextVar read -- no syscall, no parse, no
            # clock -- so a row written outside any scope costs exactly
            # what it cost before this field existed.
            receipt = current_admission()
            admission_entry_id = getattr(receipt, "entry_id", None) if receipt is not None else None

            entry = AuditEntry(
                seq=seq,
                timestamp=ts,
                operation=operation,
                target=target,
                agent=agent,
                reason=reason,
                payload_hash=p_hash,
                prev_hash=prev_hash,
                entry_hash=entry_hash,
                fields_changed=fields_changed,
                admission_entry_id=admission_entry_id,
            )

            with open(self._chain_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")

        _log.info(
            "audit_append",
            seq=seq,
            operation=operation,
            target=target,
            agent=agent,
        )
        metrics.inc("audit_entries_appended")
        return entry

    def verify(self) -> tuple[bool, list[str]]:
        """Verify the integrity of the entire chain.

        Returns:
            (is_valid, list_of_errors)
        """
        errors: list[str] = []
        if not os.path.isfile(self._chain_path):
            return True, []  # Empty chain is valid

        prev_hash = _GENESIS_HASH
        expected_seq = 1
        seen_v3 = False  # downgrade-attack mitigation

        try:
            with open(self._chain_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue

                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: invalid JSON: {e}")
                        continue

                    try:
                        entry = AuditEntry.from_dict(data)
                    except KeyError as e:
                        errors.append(f"Line {line_num}: missing field: {e}")
                        continue

                    # Check sequence
                    if entry.seq != expected_seq:
                        errors.append(f"Line {line_num}: seq {entry.seq}, expected {expected_seq}")

                    # Check prev_hash linkage
                    if entry.prev_hash != prev_hash:
                        errors.append(
                            f"Line {line_num} (seq {entry.seq}): prev_hash mismatch. "
                            f"Expected {prev_hash[:16]}..., got {entry.prev_hash[:16]}..."
                        )

                    # Recompute entry_hash — try v3 (current) first. Only
                    # fall back to the v1 legacy scheme if no v3 entry
                    # has been seen yet; once the chain has a v3 entry,
                    # all subsequent entries MUST verify v3 (downgrade
                    # attack mitigation — v1's `|`-separator scheme is
                    # injection-vulnerable).
                    args = (
                        entry.seq,
                        entry.timestamp,
                        entry.operation,
                        entry.target,
                        entry.agent,
                        entry.reason,
                        entry.payload_hash,
                        entry.prev_hash,
                    )
                    v3_hash = AuditEntry.compute_entry_hash_v3(*args)
                    expected_hash = v3_hash
                    if entry.entry_hash == v3_hash:
                        seen_v3 = True
                    elif seen_v3:
                        errors.append(f"Line {line_num} (seq {entry.seq}): downgrade to v1 scheme after v3 entry — rejected")
                    else:
                        legacy = AuditEntry.compute_entry_hash_v1(*args)
                        if entry.entry_hash == legacy:
                            expected_hash = legacy
                    if entry.entry_hash != expected_hash:
                        errors.append(
                            f"Line {line_num} (seq {entry.seq}): entry_hash tampered. "
                            f"Expected {expected_hash[:16]}..., got {entry.entry_hash[:16]}..."
                        )

                    prev_hash = entry.entry_hash
                    expected_seq = entry.seq + 1

        except OSError as e:
            errors.append(f"Cannot read chain file: {e}")

        is_valid = len(errors) == 0
        if is_valid:
            _log.info("audit_verify_ok", entries=expected_seq - 1)
        else:
            _log.warning("audit_verify_failed", errors=len(errors))
        metrics.inc("audit_verifications")
        return is_valid, errors

    def entries(self, *, last_n: int = 0) -> list[AuditEntry]:
        """Read entries from the chain.

        Args:
            last_n: If > 0, return only the last N entries.

        Returns:
            List of AuditEntry objects.
        """
        if not os.path.isfile(self._chain_path):
            return []

        all_entries: list[AuditEntry] = []
        try:
            with open(self._chain_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        all_entries.append(AuditEntry.from_dict(json.loads(stripped)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            return []

        if last_n > 0:
            return all_entries[-last_n:]
        return all_entries

    def query(
        self,
        *,
        target: str | None = None,
        operation: str | None = None,
        agent: str | None = None,
        field: str | None = None,
        since: str | None = None,
    ) -> list[AuditEntry]:
        """Query the audit chain with filters.

        Args:
            target: Filter by target path (substring match).
            operation: Filter by operation type.
            agent: Filter by agent identity.
            field: Filter by field name in fields_changed.
            since: ISO timestamp — only entries after this time.

        Returns:
            Matching AuditEntry objects.
        """
        results = []
        for entry in self.entries():
            if target and target not in entry.target:
                continue
            if operation and entry.operation != operation:
                continue
            if agent and entry.agent != agent:
                continue
            if field and field not in entry.fields_changed:
                continue
            if since and entry.timestamp < since:
                continue
            results.append(entry)
        return results

    def export(self, output_path: str) -> int:
        """Export the full chain to a file for external audit.

        Args:
            output_path: Path to write the exported chain.

        Returns:
            Number of entries exported.
        """
        entries = self.entries()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "format": "mind-mem-audit-chain",
                        "version": "1.0",
                        "exported_at": datetime.now(timezone.utc).isoformat(),
                        "genesis_hash": _GENESIS_HASH,
                        "entry_count": len(entries),
                    }
                )
                + "\n"
            )
            for entry in entries:
                f.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
        _log.info("audit_export", entries=len(entries), output=output_path)
        return len(entries)

    def entry_count(self) -> int:
        """Count entries without loading all of them."""
        if not os.path.isfile(self._chain_path):
            return 0
        count = 0
        try:
            with open(self._chain_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except OSError:
            pass
        return count
