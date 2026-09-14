"""Export reads the configured corpus, including released ingestion records.

Copyright 2026 STARGA, Inc.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.compliance.export import build_bundle, render_bundle


def _workspace(path: Path, *, backend: str = "markdown") -> str:
    path.mkdir()
    (path / "decisions").mkdir()
    (path / "mind-mem.json").write_text(
        json.dumps({"block_store": {"backend": backend}, "v4": {"compliance_export": {"enabled": True}}}),
        encoding="utf-8",
    )
    return str(path)


@pytest.mark.parametrize("fmt", ["jsonl", "markdown"])
def test_registered_ingestion_corpus_exports_only_released_records(tmp_path, fmt):
    from mind_mem.admissibility import RELEASE_FIELD

    ws = _workspace(tmp_path / "workspace")
    (Path(ws) / "decisions/DECISIONS.md").write_text(f"[D-EXPORT-1]\nStatus: active\n{RELEASE_FIELD}: IMP-EXPORT-1\n\n", encoding="utf-8")
    memory = Path(ws) / "memory"
    memory.mkdir()
    (memory / "IMPORTED.md").write_text(
        "[IMP-EXPORT-1]\nStatus: quarantined\nStatement: released-import-canary\n\n"
        "[IMP-EXPORT-2]\nStatus: quarantined\nStatement: unreleased-import-canary\n\n",
        encoding="utf-8",
    )
    (memory / "unregistered.md").write_text("[D-EXPORT-3]\nStatus: active\nStatement: outside-corpus-canary\n", encoding="utf-8")
    bundle = build_bundle(ws, fmt=fmt)
    payload = render_bundle(bundle)
    assert b"released-import-canary" in payload
    assert b"unreleased-import-canary" not in payload
    assert b"outside-corpus-canary" not in payload
    assert bundle.envelope["withheld_count"] == 1
    assert {record["source"] for record in bundle.records} == {"decisions/DECISIONS.md", "memory/IMPORTED.md"}


def test_configured_database_is_authority_even_with_markdown_shadow(tmp_path, monkeypatch):
    import mind_mem.storage as storage

    ws = _workspace(tmp_path / "workspace", backend="postgres")
    (Path(ws) / "decisions/DECISIONS.md").write_text("[D-EXPORT-1]\nStatus: active\nStatement: shadow-canary\n", encoding="utf-8")
    calls = []

    class Store:
        def get_all(self, *, active_only=False):
            calls.append(active_only)
            return [
                {"_id": "D-EXPORT-1", "Status": "active", "Statement": "database-canary", "_source_file": "decisions/DECISIONS.md"},
                {"_id": "D-EXPORT-2", "Status": "quarantined", "Statement": "withheld-database-canary"},
            ]

    monkeypatch.setattr(storage, "get_block_store", lambda workspace, config=None: Store())
    bundle = build_bundle(ws)
    payload = render_bundle(bundle)
    assert calls == [False]
    assert b"database-canary" in payload
    assert b"withheld-database-canary" not in payload
    assert b"shadow-canary" not in payload
    assert bundle.envelope["withheld_count"] == 1
    assert bundle.records[0]["source"] == "decisions/DECISIONS.md"


@pytest.mark.parametrize("backend", ["markdown", "encrypted"])
def test_export_does_not_read_symlinked_corpus_sources(tmp_path, monkeypatch, backend):
    from mind_mem import block_parser

    ws = _workspace(tmp_path / "workspace", backend=backend)
    outside = tmp_path / "outside.md"
    outside.write_text("[D-EXPORT-1]\nStatus: active\nStatement: outside-file-canary\n", encoding="utf-8")
    try:
        (Path(ws) / "decisions/DECISIONS.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not available on this platform")
    (Path(ws) / "decisions/LOCAL.md").write_text("[D-EXPORT-2]\nStatus: active\nStatement: local-positive-control\n", encoding="utf-8")
    parsed_paths = []
    original = block_parser.parse_file

    def record_parse(path):
        parsed_paths.append(Path(path))
        return original(path)

    monkeypatch.setattr(block_parser, "parse_file", record_parse)
    payload = render_bundle(build_bundle(ws))
    assert b"local-positive-control" in payload
    assert b"outside-file-canary" not in payload
    assert Path(ws) / "decisions/DECISIONS.md" not in parsed_paths


def test_required_write_provenance_does_not_hide_historical_export(tmp_path):
    ws = _workspace(tmp_path / "workspace")
    config_path = Path(ws) / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["v4"]["provenance"] = {"enabled": True, "policy": "required"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (Path(ws) / "decisions/DECISIONS.md").write_text(
        "[D-EXPORT-1]\nStatus: active\nStatement: historical-unattributed-fact\n", encoding="utf-8"
    )
    assert b"historical-unattributed-fact" in render_bundle(build_bundle(ws))


def test_encrypted_export_decrypts_and_missing_key_refuses(tmp_path, monkeypatch):
    from mind_mem.block_store_encrypted import encrypt_workspace
    from mind_mem.encryption import has_magic

    ws = _workspace(tmp_path / "workspace", backend="encrypted")
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", "export-fixture-passphrase")
    path = Path(ws) / "decisions/DECISIONS.md"
    path.write_text("[D-EXPORT-1]\nStatus: active\nStatement: encrypted-positive-control\n", encoding="utf-8")
    before = render_bundle(build_bundle(ws))
    assert encrypt_workspace(ws)["encrypted"] >= 1
    assert has_magic(path.read_bytes())
    assert render_bundle(build_bundle(ws)) == before
    monkeypatch.delenv("MIND_MEM_ENCRYPTION_PASSPHRASE")
    with pytest.raises(ValueError, match="MIND_MEM_ENCRYPTION_PASSPHRASE"):
        build_bundle(ws)


def test_database_failure_does_not_fall_back_to_markdown_export(tmp_path, monkeypatch):
    import mind_mem.storage as storage

    ws = _workspace(tmp_path / "workspace", backend="postgres")
    (Path(ws) / "decisions/DECISIONS.md").write_text("[D-EXPORT-1]\nStatus: active\nStatement: wrong-backend-canary\n", encoding="utf-8")

    class Store:
        def get_all(self, *, active_only=False):
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(storage, "get_block_store", lambda workspace, config=None: Store())
    with pytest.raises(RuntimeError, match="database unavailable"):
        build_bundle(ws)
