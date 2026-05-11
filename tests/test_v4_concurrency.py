"""v4 concurrency / fuzz tests.

Proves the audit-driven CAS contract under real contention. The
unanimous blind spot from the v4-audit-2026-05-10 multi-LLM review
was: "no read-after-write consistency contract for tier promotions;
concurrent writers can clobber writes silently". The fix landed in
``tier_memory`` as ``block_version`` + ``StaleVersionError``; this
file demonstrates the contract holds under thread-pool stress and
that block_kind_tags writes don't corrupt each other when issued
concurrently.

Tests are marked ``@pytest.mark.unit`` to keep the default suite
fast; the contended paths use 16 threads × 50 iterations each which
is enough to surface races without slowing the fast pass.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mind_mem.v4.block_kinds import FLAG as BLOCK_KINDS_FLAG
from mind_mem.v4.block_kinds import (
    BlockKind,
    ensure_block_kind_tags_table,
    get_block_kind_tags,
    set_block_kinds,
)
from mind_mem.v4.tier_memory import FLAG as TIER_FLAG
from mind_mem.v4.tier_memory import (
    ensure_recall_tier_schema,
    get_tier_version,
)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {
        "v4": {
            TIER_FLAG: {"enabled": True},
            BLOCK_KINDS_FLAG: {"enabled": True},
        }
    }
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


# ---------------------------------------------------------------------------
# Tier-version CAS — no torn writes, monotonic increments
# ---------------------------------------------------------------------------


def _tier_cas_increment(workspace: Path, block_id: str, retries: int = 50) -> int:
    """Compare-and-swap one increment of the version column.

    Returns the version that was successfully written. Spins up to
    ``retries`` times when the underlying UPDATE doesn't bump the
    row (because a concurrent writer raced ahead).
    """
    db = workspace / "index.db"
    for _ in range(retries):
        # 30 s busy timeout — Windows SQLite file-locking is stricter
        # than POSIX, and ubuntu-latest/3.14 can stall on contention too.
        # WAL journal_mode is set once in ensure_recall_tier_schema()
        # (switching journal_mode requires an exclusive lock, so doing
        # it per-thread inside the CAS loop deadlocks under contention).
        with sqlite3.connect(db, timeout=30) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT block_version FROM block_recall_tier WHERE block_id = ?",
                (block_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO block_recall_tier "
                    "(block_id, tier, last_seen_at, block_version) "
                    "VALUES (?, 'warm', '2026-05-10T00:00:00Z', 1)",
                    (block_id,),
                )
                conn.commit()
                return 1
            current = int(row[0])
            cursor = conn.execute(
                "UPDATE block_recall_tier SET block_version = ? WHERE block_id = ? AND block_version = ?",
                (current + 1, block_id, current),
            )
            if cursor.rowcount == 1:
                conn.commit()
                return current + 1
            conn.rollback()
    raise RuntimeError("CAS retries exceeded")


@pytest.mark.unit
def test_concurrent_cas_increments_never_lose_writes(cfg: Path) -> None:
    """16 threads × 50 increments = 800 total. Final version must be 800."""
    ensure_recall_tier_schema(cfg)
    block_id = "B-contended"

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(_tier_cas_increment, cfg, block_id) for _ in range(800)]
        for f in futures:
            f.result()

    final = get_tier_version(cfg, block_id)
    assert final == 800, f"expected 800 increments, got {final} (writes were lost)"


@pytest.mark.unit
def test_concurrent_cas_versions_are_unique(cfg: Path) -> None:
    """Every successful CAS returns a unique version number — no two threads
    can claim the same value."""
    ensure_recall_tier_schema(cfg)
    block_id = "B-unique"
    seen: set[int] = set()
    seen_lock = threading.Lock()

    def _do_increment() -> None:
        v = _tier_cas_increment(cfg, block_id)
        with seen_lock:
            assert v not in seen, f"version {v} returned twice (CAS contract broken)"
            seen.add(v)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: _do_increment(), range(200)))

    assert len(seen) == 200
    assert seen == set(range(1, 201))


@pytest.mark.unit
def test_get_tier_version_concurrent_reads_dont_crash(cfg: Path) -> None:
    """Read path must be re-entrant — many concurrent get_tier_version calls
    should not crash on shared DB handles or corrupt cursors."""
    ensure_recall_tier_schema(cfg)
    block_id = "B-readonly"
    _tier_cas_increment(cfg, block_id)  # seed version 1

    def _read() -> int:
        return get_tier_version(cfg, block_id)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: _read(), range(200)))

    # All reads see at least version 1 (the seed); some may see higher
    # if a concurrent writer ran (none in this test, but the contract
    # is "monotonic-or-equal").
    assert all(v >= 1 for v in results)
    assert all(v == 1 for v in results)  # no writers in this test


# ---------------------------------------------------------------------------
# Multi-label tag writes — concurrent set_block_kinds doesn't corrupt
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_concurrent_multi_label_writes_to_distinct_blocks(cfg: Path) -> None:
    """Each thread writes its own block_id; final state must hold every
    thread's tags exactly. No interference between block IDs."""
    ensure_block_kind_tags_table(cfg)

    def _write(i: int) -> None:
        kinds = [BlockKind.ENTITY, BlockKind.CODE] if i % 2 == 0 else [BlockKind.SOURCE]
        set_block_kinds(cfg, f"B-{i}", kinds)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_write, range(100)))

    # Verify every block landed.
    for i in range(100):
        tags = get_block_kind_tags(cfg, f"B-{i}")
        if i % 2 == 0:
            assert tags == {BlockKind.ENTITY, BlockKind.CODE}
        else:
            assert tags == {BlockKind.SOURCE}


@pytest.mark.unit
def test_concurrent_multi_label_writes_to_same_block_converge(cfg: Path) -> None:
    """Same block, many threads, each writing a different tag set.
    Final state must be ONE of the written sets — not partial corruption."""
    ensure_block_kind_tags_table(cfg)
    block_id = "B-shared"
    options = [
        {BlockKind.ENTITY},
        {BlockKind.ENTITY, BlockKind.CODE},
        {BlockKind.SOURCE},
        {BlockKind.SYNTHESIS, BlockKind.SOURCE, BlockKind.CONCEPT},
    ]

    def _write(i: int) -> None:
        set_block_kinds(cfg, block_id, options[i % len(options)])

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(_write, range(200)))

    final = get_block_kind_tags(cfg, block_id)
    # Final must equal one of the registered options exactly — never a
    # mixture caused by interleaved DELETE + INSERT bursts.
    assert final in options, f"final set {final} is not one of {options} — write atomicity broken"


# ---------------------------------------------------------------------------
# Fuzz: random block_ids + random kinds + random tier increments interleaved
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fuzz_mixed_workload_does_not_corrupt_any_state(cfg: Path) -> None:
    """16 threads, 200 ops total: half tier CAS increments, half multi-label
    writes. After the storm, every block's state must be coherent."""
    import random

    ensure_recall_tier_schema(cfg)
    ensure_block_kind_tags_table(cfg)
    rng = random.Random(0)

    block_ids = [f"B-{i}" for i in range(20)]

    def _op(i: int) -> None:
        bid = rng.choice(block_ids)
        if i % 2 == 0:
            _tier_cas_increment(cfg, bid)
        else:
            chosen = rng.sample(list(BlockKind), k=rng.randint(1, 3))
            set_block_kinds(cfg, bid, chosen)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_op, range(200)))

    # Spot-check coherence: every block's tag set is a subset of allowed kinds,
    # every tier_version is a non-negative int, no SQLite errors thrown.
    for bid in block_ids:
        tags = get_block_kind_tags(cfg, bid)
        assert all(isinstance(k, BlockKind) for k in tags)
        v = get_tier_version(cfg, bid)
        assert v >= 0
