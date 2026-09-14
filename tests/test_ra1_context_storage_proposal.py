"""Source-bound controls for the e376 serving-context proposal.

These controls mutate only temporary workspaces.  They exercise the public
direct Python and MCP prefetch doors with a real BM25 retrieval, while the
spies observe which config the real engine loaded.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from mind_mem import _recall_core as core
from mind_mem import recall as public_recall
from mind_mem.mcp.tools import recall as recall_tool
from mind_mem.pipeline_hash import current_pipeline_hash
from mind_mem.served_ledger import read_served_runs

_QUERY = "deterministic compiler"


def _seed(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    (root / "decisions").mkdir(parents=True)
    for name in ("tasks", "entities", "intelligence"):
        (root / name).mkdir()
    (root / "decisions" / "DECISIONS.md").write_text(
        "[D-CTX-001]\nStatement: deterministic compiler retrieval context\nStatus: active\nDate: 2026-01-01\n\n",
        encoding="utf-8",
        newline="\n",
    )
    config_a: dict[str, Any] = {
        "cache": {"enabled": False},
        "extraction": {"backend": "unknown", "model": "capture-a"},
        "recall": {"vector_enabled": False},
        "served_ledger": {"enabled": True},
    }
    config_b = copy.deepcopy(config_a)
    config_b["extraction"]["model"] = "capture-b"
    (root / "mind-mem.json").write_text(json.dumps(config_a), encoding="utf-8", newline="\n")
    return config_a, config_b


def test_direct_python_door_binds_hash_to_engine_snapshot(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "direct"
    config_a, config_b = _seed(workspace)
    config_path = workspace / "mind-mem.json"
    hash_a = current_pipeline_hash(str(workspace))
    from mind_mem.mcp.infra import config as config_module

    real_loader = config_module._load_config

    def capture_then_flip(ws: str) -> dict[str, Any]:
        loaded = real_loader(ws)
        config_path.write_text(json.dumps(config_b), encoding="utf-8", newline="\n")
        return loaded

    monkeypatch.setattr(config_module, "_load_config", capture_then_flip)

    result = public_recall.recall(str(workspace), _QUERY)
    rows = read_served_runs(str(workspace))

    assert result, "the control must execute a real retrieval"
    assert len(rows) == 1
    assert rows[0].pipeline_hash == hash_a
    assert result.attestation is not None
    assert result.attestation["config_hash"] == hash_a


def test_public_prefetch_binds_worker_engine_to_pre_retrieval_snapshot(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "prefetch"
    config_a, config_b = _seed(workspace)
    config_path = workspace / "mind-mem.json"
    hash_a = current_pipeline_hash(str(workspace))
    seen: list[str | None] = []
    during_prefetch = False

    real_prefetch_context = public_recall.prefetch_context

    def flip_for_real_fanout(workspace_name: str, signals: list[str], **kwargs: Any):
        nonlocal during_prefetch
        config_path.write_text(json.dumps(config_b), encoding="utf-8", newline="\n")
        during_prefetch = True
        try:
            return real_prefetch_context(workspace_name, signals, **kwargs)
        finally:
            during_prefetch = False
            config_path.write_text(json.dumps(config_a), encoding="utf-8", newline="\n")

    real_get_config = core._get_config

    def observe_engine_config(workspace_name: str) -> dict[str, Any]:
        config = real_get_config(workspace_name)
        if during_prefetch:
            seen.append(config.get("extraction", {}).get("model"))
        return config

    monkeypatch.setattr(public_recall, "prefetch_context", flip_for_real_fanout)
    monkeypatch.setattr(core, "_get_config", observe_engine_config)
    monkeypatch.setattr(recall_tool, "_workspace", lambda: str(workspace))

    payload = json.loads(recall_tool.prefetch.__wrapped__(_QUERY, limit=1))
    rows = read_served_runs(str(workspace))

    assert payload.get("count", 0) > 0, payload
    assert seen and all(model == "capture-a" for model in seen), seen
    assert len(rows) == 1
    assert rows[0].pipeline_hash == hash_a
    assert payload["attestation"]["config_hash"] == hash_a
