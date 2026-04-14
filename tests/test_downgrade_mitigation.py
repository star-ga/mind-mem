# Copyright 2026 STARGA, Inc.
"""v2.10.0 downgrade-attack mitigation tests.

Scenario: an attacker who sees that mind-mem still accepts v1 hashes
for backward-compat could forge a new entry using the v1 scheme (which
has the ``|``-separator injection vulnerability) and slip it past the
v3 verifier. The mitigation is: once any entry in a chain verifies
under v3, no later entry may verify only under v1.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from mind_mem.audit_chain import AuditChain, AuditEntry
from mind_mem.mind_kernels import sha3_512_chain_verify


class TestAuditChainDowngradeBlocked:
    def test_pure_v1_chain_still_verifies(self, tmp_path: Path) -> None:
        """Chains written entirely pre-v2.10.0 still verify OK."""
        audit_dir = tmp_path / ".mind-mem-audit"
        audit_dir.mkdir()
        chain_file = audit_dir / "chain.jsonl"
        # Synthesize a pure-v1 chain by writing entries whose hash matches
        # the legacy ``|``-joined scheme.
        entries = []
        prev = "0" * 64
        for seq in (1, 2, 3):
            fields = (
                seq, f"2026-04-{seq:02d}T00:00:00Z", "update",
                f"D-{seq}", "alice", "reason", "payload_hash", prev,
            )
            entry_hash = AuditEntry.compute_entry_hash_v1(*fields)
            entries.append({
                "seq": seq,
                "timestamp": fields[1],
                "operation": "update",
                "target": fields[3],
                "agent": "alice",
                "reason": "reason",
                "payload_hash": "payload_hash",
                "prev_hash": prev,
                "entry_hash": entry_hash,
            })
            prev = entry_hash
        chain_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        chain = AuditChain(workspace=str(tmp_path))
        ok, errors = chain.verify()
        assert ok, errors

    def test_downgrade_after_v3_is_rejected(self, tmp_path: Path) -> None:
        """Once a v3 entry appears, a later v1-only entry is rejected."""
        audit_dir = tmp_path / ".mind-mem-audit"
        audit_dir.mkdir()
        chain_file = audit_dir / "chain.jsonl"
        entries = []
        prev = "0" * 64

        # Entry 1: v3
        fields1 = (
            1, "2026-04-13T00:00:00Z", "update",
            "D-1", "alice", "reason", "payload_hash", prev,
        )
        h1 = AuditEntry.compute_entry_hash_v3(*fields1)
        entries.append({
            "seq": 1, "timestamp": fields1[1], "operation": "update",
            "target": "D-1", "agent": "alice", "reason": "reason",
            "payload_hash": "payload_hash", "prev_hash": prev, "entry_hash": h1,
        })
        prev = h1

        # Entry 2: v1 only (the downgrade attempt)
        fields2 = (
            2, "2026-04-14T00:00:00Z", "update",
            "D-2", "mallory", "forged", "payload_hash", prev,
        )
        h2 = AuditEntry.compute_entry_hash_v1(*fields2)
        entries.append({
            "seq": 2, "timestamp": fields2[1], "operation": "update",
            "target": "D-2", "agent": "mallory", "reason": "forged",
            "payload_hash": "payload_hash", "prev_hash": prev, "entry_hash": h2,
        })

        chain_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        chain = AuditChain(workspace=str(tmp_path))
        ok, errors = chain.verify()
        assert not ok
        assert any("downgrade" in e.lower() for e in errors), errors


class TestMindKernelsDowngradeBlocked:
    def test_v3_then_v1_rejected(self) -> None:
        """The mind_kernels.sha3_512_chain_verify fallback verifier
        also refuses a v1 entry after a v3 entry."""
        from mind_mem.preimage import preimage

        # Entry 1: v3 via preimage
        prev = "0" * 128
        fields1 = ("id1", "t1", "b1", "create", "ch1", prev)
        h1 = hashlib.sha3_512(preimage("CHAIN_v1", *fields1)).hexdigest()
        entry1 = {
            "entry_id": "id1", "timestamp": "t1", "block_id": "b1",
            "action": "create", "content_hash": "ch1",
            "previous_hash": prev, "entry_hash": h1,
        }

        # Entry 2: v1 via |-join (the downgrade)
        fields2 = ("id2", "t2", "b2", "update", "ch2", h1)
        h2 = hashlib.sha3_512("|".join(fields2).encode()).hexdigest()
        entry2 = {
            "entry_id": "id2", "timestamp": "t2", "block_id": "b2",
            "action": "update", "content_hash": "ch2",
            "previous_hash": h1, "entry_hash": h2,
        }

        assert sha3_512_chain_verify([entry1]) is True
        assert sha3_512_chain_verify([entry1, entry2]) is False
