"""M2: direct reads and recall for explicitly declared custom roots."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem._recall_core import recall
from mind_mem.audit_context import bind_current_agent
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.memory_ops import get_block


def _block(path: Path, block_id: str, statement: str, status: str = "active", releases: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    release_line = f"Releases: {releases}\n" if releases else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{block_id}]\nType: Decision\nStatement: {statement}\n{release_line}Status: {status}\n\n")


def _workspace(tmp_path: Path, reachability: str = "direct-only") -> Path:
    ws = tmp_path / "custom-namespace"
    init(str(ws))
    _block(ws / "shared/decisions/DECISIONS.md", "D-SAME-1", "shared namespace statement")
    _block(ws / "custom/decisions/DECISIONS.md", "D-CUSTOM-1", "custom namespace statement")
    _block(ws / "custom/decisions/DECISIONS.md", "D-SAME-1", "custom namespace duplicate")
    config = json.loads((ws / "mind-mem.json").read_text(encoding="utf-8"))
    config["recall"]["namespace_properties"]["custom"] = {
        "reachability": reachability,
        "floor": "none",
    }
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    (ws / "mind-mem-acl.json").write_text(
        json.dumps(
            {
                "default_policy": "read",
                "agents": {
                    "alice": {"namespaces": ["shared", "custom"], "read": ["shared", "custom"], "write": []},
                    "bob": {"namespaces": ["shared"], "read": ["shared"], "write": []},
                },
            }
        ),
        encoding="utf-8",
    )
    return ws


def _get(workspace: Path, block_id: str, namespace: str, agent: str | None) -> dict:
    previous = os.environ.get("MIND_MEM_WORKSPACE")
    os.environ["MIND_MEM_WORKSPACE"] = str(workspace)
    try:
        with bind_current_agent(agent), use_workspace(str(workspace)):
            return json.loads(get_block(block_id, namespace=namespace))
    finally:
        if previous is None:
            os.environ.pop("MIND_MEM_WORKSPACE", None)
        else:
            os.environ["MIND_MEM_WORKSPACE"] = previous


def test_declared_direct_only_custom_root_is_directly_readable_but_not_searchable(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, "direct-only")
    direct = _get(ws, "D-CUSTOM-1", "custom", "alice")
    assert direct["found"] is True
    assert direct["block"]["Statement"] == "custom namespace statement"
    hits = recall(str(ws), "custom namespace statement", agent_id="alice", limit=10, rerank=False)
    assert all(hit["_id"] != "D-CUSTOM-1" for hit in hits)


def test_declared_searchable_custom_root_round_trips_through_recall_and_direct_get(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, "searchable")
    direct = _get(ws, "D-CUSTOM-1", "custom", "alice")
    hits = recall(str(ws), "custom namespace statement", agent_id="alice", limit=10, rerank=False)
    assert direct["found"] is True
    assert [(hit["_id"], float(hit["score"])) for hit in hits if hit["_id"] == "D-CUSTOM-1"]
    assert all(float(hit["score"]) > 0.0 for hit in hits if hit["_id"] == "D-CUSTOM-1")


def test_custom_selector_honors_agent_acl_and_does_not_borrow_same_id(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, "searchable")
    own = _get(ws, "D-SAME-1", "custom", "alice")
    shared = _get(ws, "D-SAME-1", "shared", "alice")
    denied = _get(ws, "D-CUSTOM-1", "custom", "bob")
    assert own["found"] is True and own["block"]["Statement"] == "custom namespace duplicate"
    assert shared["found"] is True and shared["block"]["Statement"] == "shared namespace statement"
    assert denied["error"] == "namespace access denied"


def test_custom_namespace_release_is_source_bound(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, "direct-only")
    imported = "IMP-CUSTOM-1"
    _block(ws / "custom/memory/IMPORTED.md", imported, "custom imported", status="quarantined")
    _block(ws / "decisions/DECISIONS.md", "D-ROOT-RELEASE", "root release", releases=imported)
    refused = _get(ws, imported, "custom", "alice")
    assert refused["found"] is False and refused["withheld"] is True
    _block(ws / "custom/decisions/DECISIONS.md", "D-CUSTOM-RELEASE", "custom release", releases=imported)
    admitted = _get(ws, imported, "custom", "alice")
    assert admitted["found"] is True and admitted["block"]["_id"] == imported


def test_custom_selector_requires_exact_declaration_and_rejects_traversal(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, "searchable")
    assert _get(ws, "D-CUSTOM-1", "undeclared", "alice")["error"] == "invalid namespace selector"
    assert _get(ws, "D-CUSTOM-1", "custom/../shared", "alice")["error"] == "invalid namespace selector"


def test_custom_symlink_root_is_refused_and_not_recalled(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, "searchable")
    outside = tmp_path / "outside"
    _block(outside / "decisions/DECISIONS.md", "D-ESCAPE-1", "custom escape statement")
    import shutil

    shutil.rmtree(ws / "custom")
    try:
        os.symlink(outside, ws / "custom", target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    assert _get(ws, "D-ESCAPE-1", "custom", "alice")["error"] == "namespace access denied"
    hits = recall(str(ws), "custom escape statement", agent_id="alice", limit=10, rerank=False)
    assert all(hit["_id"] != "D-ESCAPE-1" for hit in hits)
