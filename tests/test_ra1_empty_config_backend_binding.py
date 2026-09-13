"""The captured empty policy must not fall back to a later backend choice."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

from mind_mem import _recall_core
from mind_mem import recall as public_recall
from mind_mem._recall_core import main
from mind_mem.mcp.infra import config as config_module
from mind_mem.mcp.tools import recall as recall_tool
from mind_mem.pipeline_hash import current_pipeline_hash
from mind_mem.recall import _CONFIG_HASH_UNRESOLVED, capture_policy_snapshot
from mind_mem.request_context import RequestContext, active_request_context, bind_request_context
from mind_mem.served_ledger import read_served_runs, row_hash


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


def test_raised_config_loader_is_unproven_without_a_ledger_row(monkeypatch, tmp_path: Path) -> None:
    """A loader exception is distinct from the normal malformed-file fallback.

    The historical config loader still handles malformed JSON with its built-in
    defaults.  A loader that raises before returning a mapping has no snapshot,
    so recall may answer but must expose an unproven marker and write no row.
    """
    workspace = tmp_path / "unproven"
    (workspace / "decisions").mkdir(parents=True)
    (workspace / "decisions" / "DECISIONS.md").write_text(
        "[D-RA1-BAD-CONFIG]\nStatement: deterministic compiler\nStatus: active\nDate: 2026-01-01\n\n",
        encoding="utf-8",
        newline="\n",
    )

    def fail_loader(_workspace: str):
        raise OSError("synthetic loader failure")

    monkeypatch.setattr(config_module, "_load_config", fail_loader)
    served = public_recall.recall(str(workspace), "deterministic")

    assert served, "the historical fail-soft retrieval path should still answer"
    assert served.attestation is not None
    assert served.attestation["served_proof"] == "unproven"
    assert served.attestation["served_row_hash"] is None
    assert "config hash could not be resolved" in served.attestation["ledger_error"]
    assert read_served_runs(workspace) == ()


def test_prefetch_context_uses_the_serialized_resolved_scoring_instant(monkeypatch, tmp_path: Path) -> None:
    """The bound prefetch context and real fan-out receive one ISO date."""
    workspace = tmp_path / "prefetch"
    workspace.mkdir()
    seen: list[tuple[object, object]] = []
    from mind_mem import recall as recall_module

    def observe_context(_workspace: str, _signals: list[str], **kwargs):
        context = active_request_context()
        seen.append((kwargs["scoring_instant"], None if context is None else context.scoring_instant))
        return []

    monkeypatch.setattr(recall_tool, "_workspace", lambda: str(workspace))
    monkeypatch.setattr(recall_tool, "resolve_scoring_instant", lambda _value: date(2026, 9, 13))
    monkeypatch.setattr(recall_module, "prefetch_context", observe_context)

    json.loads(recall_tool.prefetch.__wrapped__("scoring", limit=1))

    assert seen == [(date(2026, 9, 13), "2026-09-13")]


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
    hash_a = current_pipeline_hash(str(workspace))
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
    payload = json.loads(capsys.readouterr().out)

    assert changed and seen, "the real CLI retrieval path did not execute the mutation hook"
    assert all(model == "cli-a" for model in seen), seen
    assert isinstance(payload, list) and payload, "the CLI control must return a real retrieval result"
    rows = read_served_runs(workspace)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row.pipeline_hash == hash_a, (row, hash_a)
    assert row_hash(row), "the final recorded row must have a concrete hash"
