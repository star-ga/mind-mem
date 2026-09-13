"""The captured empty policy must not fall back to a later backend choice."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from mind_mem import _recall_core
from mind_mem._recall_core import main
from mind_mem.mcp.infra import config as config_module
from mind_mem.recall import _CONFIG_HASH_UNRESOLVED, capture_policy_snapshot
from mind_mem.request_context import RequestContext, bind_request_context


def test_empty_bound_config_survives_backend_probe_after_disk_mutation(tmp_path: Path) -> None:
    """A valid empty snapshot means the default Markdown route, even if disk changes to Postgres."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = workspace / "mind-mem.json"
    config_path.write_text(json.dumps({}), encoding="utf-8", newline="\n")
    context = RequestContext(workspace=workspace, config={})

    # This is the mutation: the request has captured {}, then the mutable workspace selects PG.
    config_path.write_text(
        json.dumps({"block_store": {"backend": "postgres", "dsn": "postgresql://user@host/db"}}),
        encoding="utf-8",
        newline="\n",
    )
    with bind_request_context(context):
        backend = _recall_core._load_backend(str(workspace))

    assert backend is None, "the backend probe reread disk instead of honoring the empty snapshot"
    assert context.reads >= 1, "the real _load_backend path did not consume the bound context"


def test_unreadable_capture_returns_unresolved_hash(monkeypatch, tmp_path: Path) -> None:
    """A failed config read cannot be represented as the hash of an empty fallback."""
    workspace = tmp_path / "unreadable"
    workspace.mkdir()

    def fail_loader(_workspace: str):
        raise OSError("synthetic config read failure")

    monkeypatch.setattr(config_module, "_load_config", fail_loader)
    config, config_hash, anchor = capture_policy_snapshot(str(workspace))

    assert config is None
    assert config_hash == _CONFIG_HASH_UNRESOLVED
    assert anchor == ""


def test_cli_snapshot_keeps_later_engine_reads_on_captured_config(tmp_path: Path, monkeypatch, capsys) -> None:
    """The real CLI engine stays on A after its first read changes disk to B."""
    workspace = tmp_path / "cli"
    (workspace / "decisions").mkdir(parents=True)
    (workspace / "decisions" / "DECISIONS.md").write_text(
        "[D-RA1-CLI-001]\nStatement: deterministic compiler context\nStatus: active\nDate: 2026-01-01\n\n",
        encoding="utf-8",
        newline="\n",
    )
    config_a = {"cache": {"enabled": False}, "extraction": {"model": "cli-a"}}
    config_b = {"cache": {"enabled": False}, "extraction": {"model": "cli-b"}}
    config_path = workspace / "mind-mem.json"
    config_path.write_text(json.dumps(config_a), encoding="utf-8", newline="\n")
    seen: list[str | None] = []
    changed = False
    real_get_config = _recall_core._get_config

    def observe_and_change(ws: str):
        nonlocal changed
        config = real_get_config(ws)
        seen.append(config.get("extraction", {}).get("model"))
        if not changed:
            changed = True
            config_path.write_text(json.dumps(config_b), encoding="utf-8", newline="\n")
            now = os.path.getmtime(config_path)
            os.utime(config_path, (now + 2, now + 2))
        return config

    monkeypatch.setattr(_recall_core, "_get_config", observe_and_change)
    monkeypatch.setattr(sys, "argv", ["recall", "--query", "deterministic", "--workspace", str(workspace), "--backend", "scan", "--json"])
    main()
    json.loads(capsys.readouterr().out)

    assert changed and seen, "the real CLI retrieval path did not execute the mutation hook"
    assert all(model == "cli-a" for model in seen), seen
