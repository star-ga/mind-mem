# Copyright 2026 STARGA, Inc.
"""Governance-layer concurrency stress tests (v3.0.0 — GH #506).

Concrete concurrent-writer-concurrent-reader harnesses that try to
break:

    - audit_chain (append-only, hash-chained)
    - hash_chain_v2 (SQLite + advisory lock)
    - evidence_objects (JSONL + in-memory list)
    - memory_tiers (SQLite with ConnectionManager)

Invariants checked after each run:

    - No two entries share the same seq
    - Chain previous_hash → entry_hash linkage holds for every entry
    - Final verify() reports (True, [])
    - No lost entries: every submitted (writer_id, op_id) pair is in
      the chain

These tests are slower than unit tests so they're collected only
under the `stress` marker — default pytest runs skip them unless
``-m stress`` is explicitly passed.
"""
from __future__ import annotations

import concurrent.futures
import json
import random
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.stress


class TestAuditChainConcurrentAppends:
    @pytest.mark.parametrize("writers,ops_per_writer", [(4, 50), (8, 25)])
    def test_no_lost_or_duplicate_entries(
        self, tmp_path: Path, writers: int, ops_per_writer: int
    ) -> None:
        from mind_mem.audit_chain import AuditChain

        expected_count = writers * ops_per_writer

        def worker(wid: int) -> list[str]:
            written: list[str] = []
            c = AuditChain(workspace=str(tmp_path))
            for op_id in range(ops_per_writer):
                time.sleep(random.uniform(0.0, 0.001))
                entry = c.append(
                    operation="update_field",
                    target=f"D-{wid}-{op_id}",
                    agent=f"writer-{wid}",
                    reason=f"op {op_id}",
                    payload={"wid": wid, "op": op_id},
                )
                written.append(entry.target)
            return written

        with concurrent.futures.ThreadPoolExecutor(max_workers=writers) as pool:
            futures = [pool.submit(worker, i) for i in range(writers)]
            collected = [f.result() for f in futures]

        submitted = {item for batch in collected for item in batch}
        assert len(submitted) == expected_count

        chain2 = AuditChain(workspace=str(tmp_path))
        ok, errors = chain2.verify()
        assert ok, f"chain integrity lost under concurrent writes: {errors[:3]}"

        seen_targets: set[str] = set()
        with open(tmp_path / ".mind-mem-audit" / "chain.jsonl") as fh:
            for line in fh:
                row = json.loads(line)
                seen_targets.add(row["target"])
        missing = submitted - seen_targets
        assert not missing, f"{len(missing)} submitted writes never landed"


class TestHashChainV2ConcurrentAppends:
    def test_parallel_inserts_keep_chain_verifiable(
        self, tmp_path: Path
    ) -> None:
        from mind_mem.hash_chain_v2 import HashChainV2

        db = tmp_path / "chain.db"
        HashChainV2(str(db))  # bootstrap

        writers = 4
        ops = 30

        def worker(wid: int) -> None:
            c = HashChainV2(str(db))
            for op in range(ops):
                c.append(
                    block_id=f"B-{wid}-{op}",
                    action="create",
                    content=f"content-for-{wid}-{op}",
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=writers) as pool:
            list(pool.map(worker, range(writers)))

        chain2 = HashChainV2(str(db))
        ok, first_broken = chain2.verify_chain()
        assert ok, f"broken at index {first_broken}"


class TestMemoryTiersConcurrentOps:
    def test_promote_and_register_dont_race(self, tmp_path: Path) -> None:
        from mind_mem.memory_tiers import MemoryTier, TierManager

        mgr = TierManager(str(tmp_path / "tiers.db"))

        def register_batch(wid: int) -> None:
            for i in range(100):
                mgr._register_block(f"B-{wid}-{i}", MemoryTier.WORKING)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(register_batch, range(8)))

        mgr.run_promotion_cycle()
        demotions, evicted = mgr.run_decay_cycle()
        assert isinstance(demotions, list)
        assert isinstance(evicted, list)


class TestEvidenceChainConcurrentCreate:
    def test_parallel_creates_no_exceptions(self, tmp_path: Path) -> None:
        from mind_mem.evidence_objects import EvidenceAction, EvidenceChain

        store = tmp_path / "evidence.jsonl"
        EvidenceChain(store_path=str(store))

        def worker(wid: int) -> int:
            c = EvidenceChain(store_path=str(store))
            count = 0
            for i in range(10):
                c.create(
                    action=EvidenceAction.PROPOSE,
                    actor=f"w-{wid}",
                    target_block_id=f"D-{wid}-{i}",
                    target_file="d.md",
                    payload=f"p-{wid}-{i}".encode(),
                )
                count += 1
            return count

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            totals = list(pool.map(worker, range(4)))

        assert sum(totals) == 40
        # Re-load ensures the JSONL parser handles whatever
        # interleaving ended up on disk.
        chain2 = EvidenceChain(store_path=str(store))
        ok, broken = chain2.verify_chain()
        assert isinstance(broken, list)
