# Copyright 2026 STARGA, Inc.
"""The governed-head resolver must distinguish genesis from read failure."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from mind_mem.hash_chain_v2 import HashChainV2
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import public
from mind_mem.mcp.tools import recall as recall_tool
from mind_mem.prefetch import chain_head, chain_head_resolution, get_cache, reset_cache
from mind_mem.recall_attestation import (
    GENESIS_ANCHOR,
    INDEX_ANCHOR_UNRESOLVED,
    derive_recall_attestation_for_workspace,
    index_anchor_ledger_path,
    resolve_index_anchor,
)
from mind_mem.recall_cache import reset_singleton as reset_recall_cache
from mind_mem.served_ledger import read_served_runs


def _workspace(root: Path, statement: str = "retrieval anchor control") -> Path:
    for name in ("decisions", "tasks", "entities", "intelligence"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "DECISIONS.md").write_text(
        f"[D-ANCHOR-001]\nStatement: {statement}\nStatus: active\nDate: 2026-09-13\n\n",
        encoding="utf-8",
    )
    (root / "mind-mem.json").write_text(
        json.dumps(
            {
                "cache": {"enabled": True, "ttl_seconds": 3600, "anticipation": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )
    return root


def _corrupt(ws: Path) -> Path:
    path = Path(index_anchor_ledger_path(str(ws)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not sqlite")
    return path


def _append(ws: Path, content: str = "seed") -> None:
    HashChainV2(index_anchor_ledger_path(str(ws))).append("D-ANCHOR-001", "create", content)


def _call(ws: Path, *, reset: bool = True) -> dict:
    if reset:
        reset_recall_cache()
        reset_cache()
    with use_workspace(str(ws)):
        return json.loads(public.recall(query="retrieval anchor control", mode="bm25", scoring_instant="2026-09-13"))


def test_absent_and_empty_ledgers_are_resolved_genesis(tmp_path: Path) -> None:
    absent = _workspace(tmp_path / "absent")
    assert resolve_index_anchor(str(absent)).resolved is True
    assert resolve_index_anchor(str(absent)).anchor == GENESIS_ANCHOR

    empty = _workspace(tmp_path / "empty")
    HashChainV2(index_anchor_ledger_path(str(empty)))
    result = resolve_index_anchor(str(empty))
    assert result.resolved is True
    assert result.anchor == GENESIS_ANCHOR


def test_healthy_nonzero_head_is_resolved_and_legacy_api_matches(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "healthy")
    _append(ws)
    result = resolve_index_anchor(str(ws))
    assert result.resolved is True
    assert result.anchor != GENESIS_ANCHOR
    assert chain_head(str(ws)) == result.anchor


def test_corrupt_present_ledger_is_unresolved_not_genesis(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "corrupt")
    _corrupt(ws)
    result = resolve_index_anchor(str(ws))
    assert result.resolved is False
    assert result.anchor == INDEX_ANCHOR_UNRESOLVED
    assert chain_head_resolution(str(ws)).resolved is False
    assert chain_head(str(ws)) == INDEX_ANCHOR_UNRESOLVED


def test_dangling_ledger_symlink_is_unresolved_not_genesis(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "dangling")
    db_path = Path(index_anchor_ledger_path(str(ws)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    missing = db_path.parent / "missing-ledger.db"
    try:
        db_path.symlink_to(missing)
    except OSError as exc:
        pytest.skip(f"platform denies symlink creation: {exc}")

    result = resolve_index_anchor(str(ws))
    assert result.resolved is False
    assert result.anchor == INDEX_ANCHOR_UNRESOLVED
    assert "dangling" in (result.reason or "")


def test_dangling_memory_parent_symlink_is_unresolved_not_genesis(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "dangling-parent")
    memory = ws / "memory"
    missing = ws / "missing-memory"
    try:
        memory.symlink_to(missing, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"platform denies symlink creation: {exc}")

    result = resolve_index_anchor(str(ws))
    assert result.resolved is False
    assert result.anchor == INDEX_ANCHOR_UNRESOLVED
    assert "parent" in (result.reason or "")


def test_readable_memory_directory_symlink_preserves_healthy_head(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "readable-parent")
    real_memory = tmp_path / "real-memory"
    real_memory.mkdir()
    HashChainV2(str(real_memory / "hash_chain_v2.db")).append("D-ANCHOR-001", "create", "through directory symlink")
    try:
        (ws / "memory").symlink_to(real_memory, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"platform denies symlink creation: {exc}")

    result = resolve_index_anchor(str(ws))
    assert result.resolved is True
    assert result.anchor != GENESIS_ANCHOR


def test_readable_ledger_file_symlink_preserves_healthy_head(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "readable-file")
    real_db = tmp_path / "real-ledger.db"
    HashChainV2(str(real_db)).append("D-ANCHOR-001", "create", "through file symlink")
    db_path = Path(index_anchor_ledger_path(str(ws)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        db_path.symlink_to(real_db)
    except OSError as exc:
        pytest.skip(f"platform denies symlink creation: {exc}")

    result = resolve_index_anchor(str(ws))
    assert result.resolved is True
    assert result.anchor != GENESIS_ANCHOR


def test_valid_sqlite_with_malformed_schema_is_unresolved(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "malformed-schema")
    db_path = Path(index_anchor_ledger_path(str(ws)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE hash_chain (entry_hash TEXT)")
        conn.execute("INSERT INTO hash_chain(entry_hash) VALUES (?)", ("0" * 128,))

    result = resolve_index_anchor(str(ws))
    assert result.resolved is False
    assert result.anchor == INDEX_ANCHOR_UNRESOLVED
    assert "unreadable" in (result.reason or "")


def test_malformed_nonempty_head_is_unresolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "malformed-head")
    db_path = Path(index_anchor_ledger_path(str(ws)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"placeholder")

    class FakeLedger:
        def get_latest(self, n: int):
            return [SimpleNamespace(entry_hash="not-a-sha3-head")]

    monkeypatch.setattr("mind_mem.recall_attestation.HashChainV2.open_readonly", lambda unused: FakeLedger())
    result = resolve_index_anchor(str(ws))
    assert result.resolved is False
    assert result.anchor == INDEX_ANCHOR_UNRESOLVED
    assert "malformed" in (result.reason or "")


def test_path_stat_failure_is_unresolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "stat-failure")
    _corrupt(ws)
    db_path = index_anchor_ledger_path(str(ws))
    real_lstat = recall_tool.os.lstat

    def denied(path, *args, **kwargs):
        if str(path) == db_path:
            raise PermissionError("denied")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr("mind_mem.recall_attestation.os.lstat", denied)
    result = resolve_index_anchor(str(ws))
    assert result.resolved is False
    assert result.anchor == INDEX_ANCHOR_UNRESOLVED


def test_public_failure_is_unproven_and_writes_no_served_row(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "public-failure")
    _corrupt(ws)
    result = _call(ws)
    assert result["count"] >= 1
    attestation = result["attestation"]
    assert attestation["served_proof"] == "unproven"
    assert attestation["served_seq"] is None
    assert read_served_runs(str(ws)) == ()


def test_direct_recall_facade_does_not_launder_failure(tmp_path: Path) -> None:
    from mind_mem import recall as recall_facade

    ws = _workspace(tmp_path / "direct-failure")
    _corrupt(ws)
    reset_recall_cache()
    reset_cache()
    served = recall_facade.recall(str(ws), "retrieval anchor control")
    assert served
    assert served.attestation is not None
    assert served.attestation["served_proof"] == "unproven"
    assert read_served_runs(str(ws)) == ()


def test_public_unresolved_path_bypasses_read_and_write_of_ordinary_cache(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "ordinary-cache")
    _corrupt(ws)
    first = _call(ws)
    _workspace(ws, "different retrieval anchor control")
    second = _call(ws, reset=False)
    first_text = first["results"][0]["excerpt"]
    second_text = second["results"][0]["excerpt"]
    assert first_text != second_text
    assert first["attestation"]["served_proof"] == "unproven"
    assert second["attestation"]["served_proof"] == "unproven"


def test_public_recovery_returns_healthy_attestation_and_row(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "recovery")
    _corrupt(ws)
    failed = _call(ws)
    assert failed["attestation"]["served_proof"] == "unproven"

    Path(index_anchor_ledger_path(str(ws))).unlink()
    _append(ws, "recovered")
    healthy = _call(ws)
    anchor = resolve_index_anchor(str(ws)).anchor
    assert healthy["count"] >= 1
    assert anchor != GENESIS_ANCHOR
    assert healthy["attestation"]["index_anchor"] == anchor
    assert healthy["attestation"]["served_proof"] == "recorded"
    assert len(read_served_runs(str(ws))) == 1


def test_prefetch_unresolved_path_skips_anticipation_cache_and_row(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "prefetch-failure", "retrieval anchor control")
    _corrupt(ws)
    reset_cache()
    with use_workspace(str(ws)):
        result = json.loads(recall_tool.prefetch("retrieval anchor control", limit=2))
    assert result["count"] >= 1
    assert result["attestation"]["served_proof"] == "unproven"
    assert read_served_runs(str(ws)) == ()
    stats = get_cache().stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_direct_anticipation_cache_refuses_unresolved_sentinel(tmp_path: Path) -> None:
    from mind_mem.prefetch import AnticipationCache

    cache = AnticipationCache()
    hits = [{"_id": "D-ANCHOR-001", "Statement": "retrieval anchor control"}]
    assert cache.record(str(tmp_path), "recall", hits, head=INDEX_ANCHOR_UNRESOLVED) is None
    decision = cache.lookup(
        str(tmp_path),
        "retrieval anchor control",
        head=INDEX_ANCHOR_UNRESOLVED,
    )
    assert decision.serve_from_cache is False
    assert decision.head == INDEX_ANCHOR_UNRESOLVED
    assert decision.bundle_count == 0
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 0


def test_direct_anticipation_cache_refuses_failed_head_resolution(tmp_path: Path) -> None:
    from mind_mem.prefetch import AnticipationCache

    ws = _workspace(tmp_path / "direct-prefetch-failure")
    _corrupt(ws)
    cache = AnticipationCache()
    hits = [{"_id": "D-ANCHOR-001", "Statement": "retrieval anchor control"}]
    assert cache.record(str(ws), "recall", hits) is None
    decision = cache.lookup(str(ws), "retrieval anchor control")
    assert decision.serve_from_cache is False
    assert decision.head == INDEX_ANCHOR_UNRESOLVED
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 0


def test_convenience_attestation_does_not_launder_failure(tmp_path: Path) -> None:
    ws = _workspace(tmp_path / "convenience")
    _corrupt(ws)
    with pytest.raises(RuntimeError, match="governed head"):
        derive_recall_attestation_for_workspace(
            [{"_id": "D-ANCHOR-001", "_score": 1.0}],
            str(ws),
            vector_requested=False,
            vector_available=False,
            query="anchor",
        )
