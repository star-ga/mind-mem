"""resolve_conflict must NOT run vclock upserts when its UPDATE was a no-op.

Regression: the conditional UPDATE (... WHERE resolution IS NULL) had its
rowcount ignored, so a resolver that lost the race to a concurrent resolver
still ran the block_tier_vclock upserts with its own (stale) winner_version,
overwriting the winner's committed state. The fix returns None when
rowcount == 0. This drives that exact path deterministically by pointing the
second resolve at an already-resolved row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def federation_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "fed-on.json"
    cfg.write_text(json.dumps({"v4": {"federation": {"enabled": True}}}))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))


def test_lost_race_returns_none_and_skips_vclock(tmp_path: Path, federation_enabled, monkeypatch):
    from mind_mem.v4 import federation as fed

    td = str(tmp_path)
    block_id = "B-race-001"
    fed.record_agent_write(td, block_id, agent_id="alice")
    fed.record_agent_write(td, block_id, agent_id="alice")
    fed.record_agent_write(td, block_id, agent_id="bob")

    report = fed.detect_conflict(td, block_id)
    assert report is not None
    rowid = fed._find_open_conflict_rowid(td, block_id, report)
    assert rowid is not None

    # First resolve genuinely resolves the row + converges the vclock.
    first = fed.resolve_conflict(td, block_id, strategy=fed.MergeStrategy.HIGHER_VERSION)
    assert first is not None

    # Simulate a lost race: force a second resolve to target the SAME row,
    # which is now already resolved. detect/find are patched to return the
    # stale (pre-resolution) view so we reach the UPDATE — whose WHERE
    # resolution IS NULL now matches zero rows (rowcount == 0).
    monkeypatch.setattr(fed, "detect_conflict", lambda *a, **k: report)
    monkeypatch.setattr(fed, "_find_open_conflict_rowid", lambda *a, **k: rowid)

    second = fed.resolve_conflict(td, block_id, strategy=fed.MergeStrategy.HIGHER_VERSION)
    assert second is None  # guard fired: no overwrite of the winner's state
