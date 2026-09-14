"""Namespace-selected MCP direct reads remain ACL- and admission-bound."""

from __future__ import annotations

import json
from pathlib import Path

from mind_mem._recall_core import recall
from mind_mem.audit_context import bind_current_agent
from mind_mem.block_store import MarkdownBlockStore
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.memory_ops import get_block


def _block(path: Path, block_id: str, statement: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{block_id}]\nType: Decision\nStatement: {statement}\nStatus: active\n\n")


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "namespace-direct-get"
    init(str(ws))
    _block(ws / "shared/decisions/DECISIONS.md", "D-SHARED-1", "shared direct-only evidence")
    _block(ws / "agents/alice/decisions/DECISIONS.md", "D-ALICE-1", "alice direct-only evidence")
    _block(ws / "agents/bob/decisions/DECISIONS.md", "D-BOB-1", "bob private evidence")
    config = json.loads((ws / "mind-mem.json").read_text(encoding="utf-8"))
    config["recall"]["namespace_properties"] = {
        "shared": {"reachability": "direct-only", "floor": "none"},
        "agents/alice": {"reachability": "direct-only", "floor": "none"},
    }
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    (ws / "mind-mem-acl.json").write_text(
        json.dumps(
            {
                "default_policy": "read",
                "agents": {
                    "alice": {
                        "namespaces": ["shared", "agents/alice"],
                        "read": ["shared", "agents/alice"],
                        "write": ["agents/alice"],
                    },
                    "bob": {
                        "namespaces": ["shared", "agents/bob"],
                        "read": ["shared", "agents/bob"],
                        "write": ["agents/bob"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return ws


def _get(ws: Path, block_id: str, namespace: str, agent: str) -> dict:
    import os

    os.environ["MIND_MEM_WORKSPACE"] = str(ws)
    with bind_current_agent(agent), use_workspace(str(ws)):
        return json.loads(get_block(block_id, namespace=namespace))


def test_authenticated_agent_reads_shared_and_own_direct_only_blocks_but_search_stays_absent(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    shared = _get(ws, "D-SHARED-1", "shared", "alice")
    own = _get(ws, "D-ALICE-1", "agents/alice", "alice")
    assert shared["found"] is True and shared["block"]["_id"] == "D-SHARED-1"
    assert own["found"] is True and own["block"]["_id"] == "D-ALICE-1"
    assert recall(str(ws), "direct-only evidence", agent_id="alice", limit=10, rerank=False) == []


def test_agent_acl_and_selector_validation_refuse_private_or_forged_reads(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    denied = _get(ws, "D-BOB-1", "agents/bob", "alice")
    traversal = _get(ws, "D-ALICE-1", "agents/../shared", "alice")
    forged = _get(ws, "D-ALICE-1", "agents/bob", "alice")
    assert denied["error"] == "namespace access denied"
    assert traversal["error"] == "invalid namespace selector"
    assert forged["error"] == "namespace access denied"


def test_selected_namespace_duplicate_ids_are_explicitly_ambiguous(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _block(ws / "shared/tasks/TASKS.md", "D-SHARED-1", "duplicate shared ID")
    result = _get(ws, "D-SHARED-1", "shared", "alice")
    assert result["found"] is False
    assert result["ambiguous"] is True
    assert "ambiguous" in result["error"].lower()


def test_legacy_id_only_root_resolution_remains_available(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _block(ws / "decisions/DECISIONS.md", "D-LEGACY-1", "legacy root direct read")
    import os

    os.environ["MIND_MEM_WORKSPACE"] = str(ws)
    with use_workspace(str(ws)):
        result = json.loads(get_block("D-LEGACY-1"))
    assert result["found"] is True
    assert result["block"]["_id"] == "D-LEGACY-1"
    assert MarkdownBlockStore(str(ws)).get_by_id("D-LEGACY-1") is not None
