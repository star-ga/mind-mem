"""Regression for issue #527: THREE_WAY_MERGE must bump the vclock.

Before the fix, `resolve_conflict(..., kind=THREE_WAY_MERGE)` recorded
the resolution in tier_conflict_log but never wrote the winner_version
to block_tier_vclock. The next detect_conflict call re-discovered the
same left vs right divergence — resolved conflicts recurred forever.

This test simulates the full loop: create conflict, resolve, detect
again -> must return None.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def federation_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same shape as tests/test_v4_federation_wire.py — write a
    mind-mem.json with v4.federation.enabled=true and point
    MIND_MEM_CONFIG at it."""
    cfg = tmp_path / "fed-on.json"
    cfg.write_text(json.dumps({"v4": {"federation": {"enabled": True}}}))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))


def test_three_way_merge_bumps_vclock(tmp_path: Path, federation_enabled):
    from mind_mem.v4 import federation as fed

    td = str(tmp_path)
    block_id = "B-issue527-001"

    # Two peers advance the same block at distinct versions.
    fed.record_agent_write(td, block_id, agent_id="alice")
    fed.record_agent_write(td, block_id, agent_id="alice")  # alice now at v2
    fed.record_agent_write(td, block_id, agent_id="bob")  # bob at v1

    # Detect conflict.
    report = fed.detect_conflict(td, block_id)
    assert report is not None, "should have detected the alice vs bob divergence"

    # Resolve via three-way merge with a constant merger.
    resolution = fed.resolve_conflict(
        td,
        block_id,
        strategy=fed.MergeStrategy.THREE_WAY_MERGE,
        merger=lambda r: b"merged-bytes",
    )
    assert resolution is not None
    assert resolution.winner_agent.startswith("merge:")
    # winner_version should be max(left, right) + 1 = max(2, 1) + 1 = 3.
    assert resolution.winner_version == 3

    # The critical assertion: detect_conflict must NOT re-discover the
    # same conflict on the next pass. Before the fix, the vclock wasn't
    # updated and this returned the same report again.
    re_report = fed.detect_conflict(td, block_id)
    assert (
        re_report is None
    ), "Issue #527: resolved THREE_WAY_MERGE conflict re-detected — winner_version was not persisted to block_tier_vclock"


def test_three_way_merge_audit_log_emits_hashes(tmp_path: Path, federation_enabled, caplog):
    """Issue #528: every three-way merge must log left/right/merged
    SHA-256 hashes so operators can audit anomalies in caller-supplied
    merge bytes."""
    import logging

    from mind_mem.v4 import federation as fed

    td = str(tmp_path)
    block_id = "B-issue528-001"

    fed.record_agent_write(td, block_id, agent_id="alice")
    fed.record_agent_write(td, block_id, agent_id="alice")  # alice now at v2
    fed.record_agent_write(td, block_id, agent_id="bob")  # bob at v1 — gap
    report = fed.detect_conflict(td, block_id)
    assert report is not None

    caplog.set_level(logging.INFO, logger="mind_mem.federation")
    fed.resolve_conflict(
        td,
        block_id,
        strategy=fed.MergeStrategy.THREE_WAY_MERGE,
        merger=lambda r: b"caller-supplied-bytes",
    )

    audit_records = [
        r for r in caplog.records if getattr(r, "message", "") == "three_way_merge_resolved" or r.getMessage() == "three_way_merge_resolved"
    ]
    assert audit_records, "three_way_merge_resolved audit log not emitted"
    rec = audit_records[0]
    # Required audit fields per #528.
    for field in (
        "block_id",
        "winner_agent",
        "winner_version",
        "left_payload_sha256",
        "right_payload_sha256",
        "merged_payload_sha256",
    ):
        assert hasattr(rec, field), f"audit log missing field {field}"
    # merged hash must reflect what the merger returned.
    import hashlib

    expected = hashlib.sha256(b"caller-supplied-bytes").hexdigest()
    assert rec.merged_payload_sha256 == expected
