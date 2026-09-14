"""MCP export must use the same configured, admitted corpus as other exports.

Copyright 2026 STARGA, Inc.
"""

import json
from pathlib import Path

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.memory_ops import export_memory


def workspace(tmp_path, monkeypatch, backend="markdown"):
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "decisions").mkdir()
    (ws / "memory").mkdir()
    (ws / "mind-mem.json").write_text(json.dumps({"block_store": {"backend": backend}}), encoding="utf-8")
    return ws


def test_mcp_export_includes_released_imports_and_counts_withheld(tmp_path, monkeypatch):
    from mind_mem.admissibility import RELEASE_FIELD

    ws = workspace(tmp_path, monkeypatch)
    (ws / "decisions/DECISIONS.md").write_text(f"[D-EXPORT-1]\nStatus: active\n{RELEASE_FIELD}: IMP-EXPORT-1\n\n", encoding="utf-8")
    (ws / "memory/IMPORTED.md").write_text(
        "[IMP-EXPORT-1]\nStatus: quarantined\nStatement: released-canary\n\n"
        "[IMP-EXPORT-2]\nStatus: quarantined\nStatement: withheld-canary\n\n",
        encoding="utf-8",
    )
    (ws / "memory/unregistered.md").write_text("[D-EXPORT-3]\nStatus: active\nStatement: unregistered-canary\n", encoding="utf-8")
    with use_workspace(str(ws)):
        payload = json.loads(export_memory())
    assert payload["withheld_count"] == 1
    assert "released-canary" in payload["data"]
    assert "withheld-canary" not in payload["data"]
    assert "unregistered-canary" not in payload["data"]


def test_mcp_export_confines_source_before_open(tmp_path, monkeypatch):
    import mind_mem.mcp.tools.memory_ops as memory_ops
    from mind_mem import block_parser

    ws = workspace(tmp_path, monkeypatch)
    outside = tmp_path / "outside.md"
    outside.write_text("[D-EXPORT-1]\nStatus: active\nStatement: outside-canary\n", encoding="utf-8")
    link = ws / "decisions/DECISIONS.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    (ws / "decisions/LOCAL.md").write_text("[D-EXPORT-2]\nStatus: active\nStatement: local-canary\n", encoding="utf-8")
    parsed = []
    original = block_parser.parse_file

    def parse(path):
        parsed.append(Path(path))
        return original(path)

    monkeypatch.setattr(block_parser, "parse_file", parse)
    monkeypatch.setattr(memory_ops, "parse_file", parse)
    with use_workspace(str(ws)):
        payload = json.loads(export_memory())
    assert "local-canary" in payload["data"]
    assert "outside-canary" not in payload["data"]
    assert link not in parsed


def test_mcp_export_decrypts_registered_source(tmp_path, monkeypatch):
    from mind_mem.block_store_encrypted import encrypt_workspace
    from mind_mem.encryption import has_magic

    ws = workspace(tmp_path, monkeypatch, "encrypted")
    monkeypatch.setenv("MIND_MEM_ENCRYPTION_PASSPHRASE", "export-test-passphrase")
    source = ws / "decisions/DECISIONS.md"
    source.write_text("[D-EXPORT-1]\nStatus: active\nStatement: encrypted-canary\n", encoding="utf-8")
    encrypt_workspace(str(ws))
    assert has_magic(source.read_bytes())
    with use_workspace(str(ws)):
        payload = json.loads(export_memory())
    assert "encrypted-canary" in payload["data"]
    assert payload["block_count"] == 1


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "5"])
def test_mcp_export_rejects_invalid_cap(tmp_path, monkeypatch, cap):
    ws = workspace(tmp_path, monkeypatch)
    with use_workspace(str(ws)):
        payload = json.loads(export_memory(max_blocks=cap))
    assert "error" in payload
    assert "data" not in payload
