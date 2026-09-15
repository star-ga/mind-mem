# Copyright 2026 STARGA, Inc.
"""Backfill must extract from the configured, admitted corpus of record."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.graph_ingest import backfill


def _workspace(tmp_path: Path, *, postgres: bool = False, lifecycle: bool = False) -> str:
    (tmp_path / "decisions").mkdir()
    config: dict = {"block_store": {"backend": "postgres" if postgres else "markdown"}}
    if lifecycle:
        config["recall"] = {
            "validity_gate": {
                "enabled": True,
                "content_categories": {"enabled": True, "ttl_days": {"infra": 2, "status": 1}},
            }
        }
    (tmp_path / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    return str(tmp_path)


def _extractor(seen: list[str]):
    def extract(text: str) -> list[dict]:
        seen.append(text)
        return [{"subject": "memory", "predicate": "depends_on", "object": "mind"}]

    return extract


def test_configured_backend_is_used_instead_of_markdown_shadow(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path, postgres=True)
    (tmp_path / "decisions/DECISIONS.md").write_text(
        "[D-20260915-001]\nStatus: active\nStatement: stale local shadow\n", encoding="utf-8"
    )
    calls = []

    class Store:
        def get_all(self, *, active_only):
            calls.append(active_only)
            return [
                {"_id": "D-20260915-001", "Status": "active", "Statement": "current database fact"},
                {"_id": "D-20260915-002", "Status": "quarantined", "Statement": "withheld database fact"},
            ]

    def configured_store(ws, *, config):
        assert ws == workspace
        assert config["block_store"]["backend"] == "postgres"
        return Store()

    monkeypatch.setattr("mind_mem.storage.get_block_store", configured_store)
    seen = []
    report = backfill(workspace, extract_fn=_extractor(seen))
    assert calls == [False]
    assert seen == ["current database fact"]
    assert report["blocks_scanned"] == report["edges_extracted"] == 1
    assert report["signals_written"] == 0
    assert not (tmp_path / "intelligence").exists()


def test_backend_failure_does_not_become_empty_success(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path, postgres=True)

    def unavailable(*args, **kwargs):
        raise OSError("configured store unavailable")

    monkeypatch.setattr("mind_mem.storage.get_block_store", unavailable)
    seen = []
    with pytest.raises(OSError, match="configured store unavailable"):
        backfill(workspace, extract_fn=_extractor(seen))
    assert seen == []


@pytest.mark.parametrize("status", ["pending", "quarantined", "unvalidated"])
def test_markdown_withheld_source_does_not_reach_extractor(tmp_path, status):
    workspace = _workspace(tmp_path)
    (tmp_path / "decisions/DECISIONS.md").write_text(
        "[D-20260915-001]\nStatus: active\nStatement: admitted source\n\n"
        f"[D-20260915-002]\nStatus: {status}\nStatement: withheld source\n",
        encoding="utf-8",
    )
    seen = []
    report = backfill(workspace, extract_fn=_extractor(seen))
    assert seen == ["admitted source"]
    assert report["blocks_scanned"] == report["edges_extracted"] == 1


def test_revoked_credential_does_not_reach_extractor(tmp_path):
    workspace = _workspace(tmp_path, lifecycle=True)
    (tmp_path / "decisions/DECISIONS.md").write_text(
        "[D-20260915-001]\nStatus: active\nStatement: admitted source\n\n"
        "[D-20260915-002]\nStatus: revoked\nContentCategory: credential\nStatement: withdrawn credential\n",
        encoding="utf-8",
    )
    seen = []
    report = backfill(workspace, extract_fn=_extractor(seen))
    assert seen == ["admitted source"]
    assert report["blocks_scanned"] == report["edges_extracted"] == 1


def test_release_decision_and_import_are_admitted_together(tmp_path):
    workspace = _workspace(tmp_path)
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory/IMPORTED.md").write_text(
        "[IMP-20260915-001]\nStatus: quarantined\nStatement: released import\n\n"
        "[IMP-20260915-002]\nStatus: quarantined\nStatement: unreleased import\n",
        encoding="utf-8",
    )
    (tmp_path / "decisions/DECISIONS.md").write_text(
        "[D-20260915-001]\nStatus: active\nReleases: IMP-20260915-001\nStatement: release decision\n",
        encoding="utf-8",
    )
    seen = []
    report = backfill(workspace, extract_fn=_extractor(seen))
    assert set(seen) == {"release decision", "released import"}
    assert report["blocks_scanned"] == 2


def test_duplicate_source_ids_cannot_mint_ambiguous_edges(tmp_path):
    workspace = _workspace(tmp_path)
    (tmp_path / "decisions/DECISIONS.md").write_text(
        "[D-20260915-001]\nStatus: active\nStatement: first ambiguous origin\n\n"
        "[D-20260915-002]\nStatus: active\nStatement: unique origin\n",
        encoding="utf-8",
    )
    (tmp_path / "decisions/extra.md").write_text(
        "[D-20260915-001]\nStatus: active\nStatement: second ambiguous origin\n",
        encoding="utf-8",
    )
    seen = []
    report = backfill(workspace, extract_fn=_extractor(seen))
    assert seen == ["unique origin"]
    assert report["blocks_scanned"] == 1


def test_cli_uses_admitted_sources_before_extraction(tmp_path, monkeypatch, capsys):
    from mind_mem.mm_cli import main

    workspace = _workspace(tmp_path)
    (tmp_path / "decisions/DECISIONS.md").write_text(
        "[D-20260915-001]\nStatus: active\nStatement: admitted CLI source\n\n"
        "[D-20260915-002]\nStatus: quarantined\nStatement: withheld CLI source\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    seen = []
    monkeypatch.setattr("mind_mem.graph_ingest._default_extract_fn", lambda ws: _extractor(seen))
    assert main(["graph-backfill", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["blocks_scanned"] == 1
    assert seen == ["admitted CLI source"]
    assert report["dry_run"] is True
    assert report["signals_written"] == 0
