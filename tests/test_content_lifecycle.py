# Copyright 2026 STARGA, Inc.
"""Content lifetimes through real recall, dream and tier-decay entry points."""

from __future__ import annotations

import copy
import json
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from mind_mem._recall_core import recall
from mind_mem.content_lifecycle import ContentLifecyclePolicy
from mind_mem.dream_cycle import pass_stale_detection
from mind_mem.init_workspace import init
from mind_mem.memory_tiers import DemotionReason, MemoryTier, TierManager
from mind_mem.validity_gate import apply_validity_gate

NOW = date(2026, 9, 14)
CFG = {"validity_gate": {"enabled": True, "content_categories": {"enabled": True, "ttl_days": {"infra": 2, "status": 1}}}}


def _workspace(tmp_path: Path, *, enabled: bool = True) -> str:
    workspace = tmp_path / "workspace"
    init(str(workspace))
    config_path = workspace / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if enabled:
        config["recall"].update(copy.deepcopy(CFG))
    config["recall"]["knee_cutoff"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    body = ""
    for seq, category, stamp in (
        (1, "status", "2026-09-12"),
        (2, "decision", "2020-01-01"),
        (3, "infra", "2026-09-14"),
        (4, "credential", "2020-01-01"),
    ):
        body += (
            f"[D-20260901-{seq:03d}]\nStatus: active\nStatement: orchid infrastructure policy fact {seq}\n"
            f"ContentCategory: {category}\nContentValidFrom: {stamp}\n\n"
        )
    (workspace / "decisions/DECISIONS.md").write_text(body, encoding="utf-8")
    return str(workspace)


@pytest.mark.parametrize("backend", ("scan", "sqlite"))
def test_actual_recall_uses_semantic_lifetime_and_pinned_clock(tmp_path: Path, monkeypatch, backend: str) -> None:
    workspace = _workspace(tmp_path)
    config_path = Path(workspace) / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["recall"]["backend"] = backend
    config_path.write_text(json.dumps(config), encoding="utf-8")
    if backend == "sqlite":
        from mind_mem.sqlite_index import build_index

        # Real nonmatching documents keep the query below half the corpus;
        # FTS5 otherwise clamps its IDF to tiny values rounded to zero.
        # Do not inject artificial scores into the production query path.
        corpus = Path(workspace) / "decisions/DECISIONS.md"
        with corpus.open("a") as handle:
            for seq in range(10, 20):
                handle.write(f"\n[D-20260901-{seq:03d}]\nStatus: active\nStatement: unrelated violet control {seq}\n")
        build_index(workspace)
    monkeypatch.setattr("mind_mem.scoring_instant._read_utc_today", lambda: pytest.fail("hidden clock"))
    before = recall(workspace, "orchid", limit=10, rerank=False, scoring_instant="2026-09-12")
    after = recall(workspace, "orchid", limit=10, rerank=False, scoring_instant=NOW)
    first = next(hit for hit in before if hit["_id"] == "D-20260901-001")
    expired = next(hit for hit in after if hit["_id"] == "D-20260901-001")
    assert first["validity"]["content_lifecycle"]["state"] == "current"
    assert expired["validity"]["content_lifecycle"]["state"] == "stale"
    assert expired["score"] < first["score"]
    assert after == recall(workspace, "orchid", limit=10, rerank=False, scoring_instant=NOW)
    durable = next(hit for hit in after if hit["_id"] == "D-20260901-002")
    assert durable["validity"]["content_lifecycle"]["state"] == "durable"
    assert not durable.get("_validity_demoted")


def test_cache_cannot_forge_renewal_or_access_itself_fresh(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    hit = {
        "_id": "D-20260901-001",
        "Status": "active",
        "score": 10.0,
        "ContentCategory": "decision",
        "ContentValidFrom": NOW.isoformat(),
        "access_count": 999999,
    }
    assert apply_validity_gate([hit], workspace, CFG, scoring_instant=NOW) == 1
    assert hit["validity"]["content_lifecycle"] == {
        "category": "status",
        "state": "stale",
        "valid_from": "2026-09-12",
        "age_days": 2,
        "ttl_days": 1,
    }


def test_dream_report_ignores_file_touch_and_does_not_mutate_facts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    corpus = Path(workspace) / "decisions/DECISIONS.md"
    original = corpus.read_bytes()
    os.utime(corpus, None)
    stale = pass_stale_detection(workspace, as_of=NOW)
    assert [block.block_id for block in stale] == ["D-20260901-001"]
    assert stale[0].days_stale == 2
    assert corpus.read_bytes() == original


def test_idle_tier_sweep_preserves_categories_but_manual_demotion_still_works(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    db_path = str(tmp_path / "tiers.db")
    manager = TierManager(db_path, workspace=workspace)
    try:
        for block_id in ("D-20260901-001", "D-20260901-002", "D-20260901-004", "LEGACY-1"):
            manager._register_block(block_id, MemoryTier.VERIFIED)
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE block_tiers SET updated_at='2020-01-01T00:00:00+00:00'")
        demotions, evictions = manager.run_decay_cycle(now=datetime(2026, 9, 14, tzinfo=timezone.utc))
        assert [item[0] for item in demotions] == ["LEGACY-1"]
        assert evictions == []
        assert manager.get_tier("D-20260901-002") == MemoryTier.VERIFIED
        assert manager.demote("D-20260901-002", MemoryTier.LONG_TERM, DemotionReason.MANUAL)
    finally:
        manager.close()


def test_credential_revocation_is_not_overridden_by_durability(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    corpus = Path(workspace) / "decisions/DECISIONS.md"
    corpus.write_text(
        corpus.read_text(encoding="utf-8").replace("[D-20260901-004]\nStatus: active", "[D-20260901-004]\nStatus: revoked"),
        encoding="utf-8",
    )
    hits = recall(workspace, "orchid", limit=10, rerank=False, scoring_instant=NOW)
    assert hits
    assert "D-20260901-004" not in {hit["_id"] for hit in hits}


def test_public_direct_fetch_withholds_revoked_credential_but_serves_decision(tmp_path: Path, monkeypatch) -> None:
    from mind_mem.mcp.tools import memory_ops

    workspace = _workspace(tmp_path)
    monkeypatch.setattr(memory_ops, "_workspace", lambda: workspace)
    active = json.loads(memory_ops.get_block("D-20260901-004"))
    assert active["found"] and active["block"]["ContentCategory"] == "credential"
    corpus = Path(workspace) / "decisions/DECISIONS.md"
    corpus.write_text(
        corpus.read_text(encoding="utf-8").replace("[D-20260901-004]\nStatus: active", "[D-20260901-004]\nStatus: revoked"),
        encoding="utf-8",
    )
    refused = json.loads(memory_ops.get_block("D-20260901-004"))
    assert refused["found"] is False and refused["withheld"] is True
    assert "block" not in refused
    assert json.loads(memory_ops.get_block("D-20260901-002"))["found"]


def test_current_corpus_renewal_changes_the_result_without_reindex(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    hit = {"_id": "D-20260901-001", "Status": "active", "score": 10.0}
    assert apply_validity_gate([copy.deepcopy(hit)], workspace, CFG, scoring_instant=NOW) == 1
    corpus = Path(workspace) / "decisions/DECISIONS.md"
    corpus.write_text(
        corpus.read_text(encoding="utf-8").replace("ContentValidFrom: 2026-09-12", "ContentValidFrom: 2026-09-14"), encoding="utf-8"
    )
    renewed = copy.deepcopy(hit)
    assert apply_validity_gate([renewed], workspace, CFG, scoring_instant=NOW) == 0
    assert renewed["validity"]["content_lifecycle"]["state"] == "current"


def test_disabled_policy_has_no_corpus_read_or_extra_annotations(monkeypatch) -> None:
    monkeypatch.setattr("mind_mem.content_lifecycle.live_content_blocks", lambda _: pytest.fail("policy-off corpus read"))
    hits = [{"_id": "D-1", "score": 2.0}]
    original = copy.deepcopy(hits)
    assert apply_validity_gate(hits, "missing-workspace", {}) == 0
    assert hits == original


@pytest.mark.parametrize(
    "ttl",
    (
        {},
        {"status": 1},
        {"status": 1, "infra": 2, "decision": 1},
        {"status": True, "infra": 2},
        {"status": 0, "infra": 2},
        {"status": 1.5, "infra": 2},
        {"status": 1, "infra": 36501},
    ),
)
def test_no_guessed_ttl_or_durable_auto_expiry(ttl) -> None:
    config = copy.deepcopy(CFG)
    config["validity_gate"]["content_categories"]["ttl_days"] = ttl
    with pytest.raises(ValueError):
        ContentLifecyclePolicy.from_recall_config(config)


@pytest.mark.parametrize(
    "stamp,state",
    (
        ("2026-09-13", "stale"),
        ("2026-09-14", "current"),
        ("2026-09-15", "future_date"),
        ("2026-W38-1", "invalid_date"),
        ("2026-02-31", "invalid_date"),
        ("", "invalid_date"),
        (42, "invalid_date"),
    ),
)
def test_exact_day_boundary_and_invalid_date_is_not_fresh(stamp, state) -> None:
    policy = ContentLifecyclePolicy.from_recall_config(CFG)
    assert policy is not None
    result = policy.evaluate({"ContentCategory": "status", "ContentValidFrom": stamp}, as_of=NOW)
    assert result is not None and result.state == state


def test_different_taxonomies_cannot_select_lifetime() -> None:
    policy = ContentLifecyclePolicy.from_recall_config(CFG)
    assert policy is not None
    assert policy.evaluate({"Kind": "entity", "Category": "status", "Date": "2020-01-01"}, as_of=NOW) is None
    result = policy.evaluate({"ContentCategory": "entity"}, as_of=NOW)
    assert result is not None and result.needs_review
