"""v4 concurrency / fuzz tests.

Proves the audit-driven CAS contract under real contention. The
unanimous blind spot from the v4-audit-2026-05-10 multi-LLM review
was a missing read-after-write consistency contract for concurrent
writers. RA.0 deleted the recall-tier ladder that carried the CAS
column, so what remains under test here is that block_kind_tags
writes don't corrupt each other when issued concurrently.

Tests are marked ``@pytest.mark.unit`` to keep the default suite
fast; the contended paths use 16 threads × 50 iterations each which
is enough to surface races without slowing the fast pass.
"""

from __future__ import annotations

import json
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

# v4.0.8: file-level stress marker. 16-worker ThreadPoolExecutors with
# 200-800 iteration loops OOM the GitHub-hosted ubuntu runners. Local
# `make test` runs these for pre-release gating.
pytestmark = pytest.mark.stress


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = {
        "v4": {
            BLOCK_KINDS_FLAG: {"enabled": True},
        }
    }
    (tmp_path / "mind-mem.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_CONFIG", str(tmp_path / "mind-mem.json"))
    return tmp_path


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
# Fuzz: random block_ids + random kinds interleaved
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fuzz_mixed_workload_does_not_corrupt_any_state(cfg: Path) -> None:
    """16 threads, 200 multi-label writes. After the storm, every
    block's state must be coherent."""
    import random

    ensure_block_kind_tags_table(cfg)
    rng = random.Random(0)

    block_ids = [f"B-{i}" for i in range(20)]

    def _op(_i: int) -> None:
        bid = rng.choice(block_ids)
        chosen = rng.sample(list(BlockKind), k=rng.randint(1, 3))
        set_block_kinds(cfg, bid, chosen)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_op, range(200)))

    # Spot-check coherence: every block's tag set is a subset of the
    # allowed kinds, and no SQLite errors were thrown.
    for bid in block_ids:
        tags = get_block_kind_tags(cfg, bid)
        assert all(isinstance(k, BlockKind) for k in tags)
