# Copyright 2026 STARGA, Inc.
"""Adversarial memory corpus tests (v3.0.0 — GH #507).

Feeds crafted malicious inputs to mind-mem's governance components and
verifies each one either rejects the input cleanly or processes it
without corrupting state. Targets:

    - preimage.py       — NUL injection, None, NaN, type smuggling
    - q1616.py          — overflow, NaN, inf
    - evidence_objects  — oversized metadata, forged v1 hashes in v3 chains
    - hash_chain_v2     — separator injection via block_id, non-UTF8
    - memory_tiers      — negative access counts, future timestamps
    - audit_chain       — malformed JSONL lines, huge payloads
    - recall            — SQL-injection-flavoured queries

A passing test means the component either rejected the adversarial
input with a typed error or absorbed it without damaging its chain /
DB state. A failing test means we found a real attack surface.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


class TestPreimageAttackSurface:
    def test_nul_byte_in_string_field(self) -> None:
        from mind_mem.preimage import preimage

        with pytest.raises(ValueError):
            preimage("T_v1", "hello\x00world")

    def test_nul_byte_in_bytes_field(self) -> None:
        from mind_mem.preimage import preimage

        with pytest.raises(ValueError):
            preimage("T_v1", b"hello\x00world")

    def test_nul_byte_in_tag(self) -> None:
        from mind_mem.preimage import preimage

        with pytest.raises(ValueError):
            preimage("bad\x00tag", "x")

    def test_none_rejection(self) -> None:
        from mind_mem.preimage import preimage

        with pytest.raises(ValueError):
            preimage("T_v1", None)

    def test_nan_rejection(self) -> None:
        from mind_mem.preimage import preimage

        with pytest.raises(ValueError):
            preimage("T_v1", float("nan"))

    def test_unsupported_type_rejection(self) -> None:
        from mind_mem.preimage import preimage

        class Evil:
            def __bytes__(self) -> bytes:
                return b"hello\x00world"

        with pytest.raises(TypeError):
            preimage("T_v1", Evil())

    def test_same_body_different_tag_diverges(self) -> None:
        from mind_mem.preimage import preimage

        # An attacker can't replay a CHAIN preimage as an EV preimage.
        assert preimage("CHAIN_v1", "a", "b") != preimage("EV_v1", "a", "b")


class TestQ1616AttackSurface:
    def test_infinity_saturates(self) -> None:
        from mind_mem.q1616 import to_q16_16

        assert to_q16_16(float("inf")) == 0x7FFFFFFF
        assert to_q16_16(float("-inf")) == -0x80000000

    def test_massive_values_saturate(self) -> None:
        from mind_mem.q1616 import to_q16_16

        # 10^20 would overflow int32 without saturation
        assert to_q16_16(1e20) == 0x7FFFFFFF

    def test_hex_stays_8_chars_always(self) -> None:
        from mind_mem.q1616 import hex_q16_16

        for v in (0.0, 1.0, -1.0, 1e10, -1e10, float("inf"), float("nan")):
            assert len(hex_q16_16(v)) == 8


class TestEvidenceChainDowngradeBlocked:
    def test_forged_v1_after_v3_rejected(self, tmp_path: Path) -> None:
        from mind_mem.evidence_objects import (
            EvidenceAction,
            EvidenceChain,
            EvidenceObject,
            _compute_evidence_hash_v1,
        )

        chain = EvidenceChain(store_path=str(tmp_path / "evidence.jsonl"))
        # Legitimate v3 entry
        real = chain.create(
            action=EvidenceAction.PROPOSE,
            actor="alice",
            target_block_id="D-1",
            target_file="d.md",
            payload=b"ok",
        )
        # Now attempt to append a v1-hashed forgery after real
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        forged_args = {
            "evidence_id": "forged",
            "timestamp_iso": now.isoformat(),
            "action": "APPLY",
            "actor": "mallory",
            "target_block_id": "D-2",
            "payload_hash": "dead" * 16,
            "previous_hash": real.evidence_hash,
            "target_file": "d.md",
            "metadata": {},
            "confidence": 1.0,
        }
        forged_v1_hash = _compute_evidence_hash_v1(**forged_args)
        forged_ev = EvidenceObject(
            evidence_id="forged",
            timestamp=now,
            action=EvidenceAction.APPLY,
            actor="mallory",
            target_block_id="D-2",
            target_file="d.md",
            payload_hash="dead" * 16,
            previous_hash=real.evidence_hash,
            metadata={},
            confidence=1.0,
            evidence_hash=forged_v1_hash,
        )
        # Inject the forgery directly so we can run verify_chain
        chain._entries.append(forged_ev)
        ok, broken = chain.verify_chain()
        assert not ok
        assert "forged" in broken


class TestHashChainSeparatorInjection:
    def test_pipe_in_block_id_does_not_forge(self, tmp_path: Path) -> None:
        """Legacy v1 hash is `|`-separated. Our v3 upgrade uses NUL-sep
        preimages so block_id="a|b|ev" can't collide with block_id="a"
        + action="b|ev" under v3.
        """
        from mind_mem.hash_chain_v2 import (
            _compute_entry_hash_v1,
            _compute_entry_hash_v3,
        )

        a = _compute_entry_hash_v3("id", "ts", "a|b", "ev", "ch", "prev")
        b = _compute_entry_hash_v3("id", "ts", "a", "b|ev", "ch", "prev")
        # v3 MUST differ — v1 might collide (the vulnerability we fixed)
        assert a != b

        # Sanity: v1 classic injection — block_id "a|b" with action "ev"
        # vs block_id "a" with action "b|ev" DO collide in v1.
        legacy_a = _compute_entry_hash_v1("id", "ts", "a|b", "ev", "ch", "prev")
        legacy_b = _compute_entry_hash_v1("id", "ts", "a", "b|ev", "ch", "prev")
        assert legacy_a == legacy_b, (
            "If this assertion fails, the v1 vulnerability may have been "
            "silently fixed; this test exists to document the attack we "
            "mitigated in v2.10.0."
        )


class TestAuditChainMalformedInput:
    def test_malformed_jsonl_line_does_not_crash_verify(self, tmp_path: Path) -> None:
        from mind_mem.audit_chain import AuditChain

        ad = tmp_path / ".mind-mem-audit"
        ad.mkdir()
        chain_file = ad / "chain.jsonl"
        chain_file.write_text(
            '{"seq":1,"timestamp":"t","operation":"update_field","target":"x",'
            '"agent":"a","reason":"r","payload_hash":"p","prev_hash":"",'
            '"entry_hash":"deadbeef"}\n'
            "this is not json\n"
            '{"seq":2}\n'
        )
        chain = AuditChain(workspace=str(tmp_path))
        ok, errors = chain.verify()
        # verify() doesn't crash; returns errors.
        assert not ok
        assert len(errors) >= 1


class TestTierManagerEdgeCases:
    def test_decay_with_future_timestamp_noop(self, tmp_path: Path) -> None:
        from mind_mem.memory_tiers import MemoryTier, TierManager

        mgr = TierManager(str(tmp_path / "tiers.db"))
        # Past timestamp normalised by _hours_since; "now" in the past
        # doesn't cause negative idle times.
        mgr._register_block("D-1", MemoryTier.SHARED)
        past = datetime.now(timezone.utc) - timedelta(days=365)
        demotions, evicted = mgr.run_decay_cycle(now=past)
        # No demotions/evictions when now < last_access.
        assert demotions == []
        assert evicted == []


class TestRecallSqlInjectionFlavour:
    def test_sql_quotes_in_query_do_not_crash(self, tmp_path: Path) -> None:
        """recall() uses parameterised queries — a ' quote in the query
        text must not escape out of the WHERE clause."""
        from mind_mem.init_workspace import init as init_workspace
        from mind_mem.recall import recall

        init_workspace(str(tmp_path))
        # Adversarial query with every SQL-y character
        queries = [
            "' OR 1=1 --",
            '"; DROP TABLE blocks; --',
            "\x00\x01\x02",  # control bytes
            "🔥" * 100,  # unicode flood
            "a" * 10000,  # huge query
        ]
        for q in queries:
            # Zero rows is fine; a crash is not.
            results = recall(str(tmp_path), q, limit=5)
            assert isinstance(results, list)


class TestOversizedMetadata:
    def test_huge_metadata_is_hashable(self) -> None:
        """metadata can be arbitrarily large; v3 hash must still produce
        a valid SHA-256 digest (no OOM, no truncation errors)."""
        from mind_mem.evidence_objects import _compute_evidence_hash_v3

        big_meta = {f"key_{i}": "x" * 1000 for i in range(100)}
        h = _compute_evidence_hash_v3(
            evidence_id="eid",
            timestamp_iso="2026-04-13T00:00:00Z",
            action="PROPOSE",
            actor="alice",
            target_block_id="D-1",
            payload_hash="p" * 64,
            previous_hash="0" * 64,
            target_file="",
            metadata=big_meta,
            confidence=0.5,
        )
        assert len(h) == 64  # SHA-256 hex
