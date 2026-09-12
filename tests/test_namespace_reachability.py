"""M2: namespace direct-get versus production recall reachability.

These fixtures use the namespace layout documented by ``NamespaceManager``
(``shared/<corpus-file>`` and ``agents/<id>/<corpus-file>``).  The calls go
through ``_recall_core.recall(..., agent_id=...)`` because that is the current
caller carrying agent identity; the public MCP wrapper does not expose it.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from mind_mem._recall_core import recall
from mind_mem.block_store import MarkdownBlockStore
from mind_mem.init_workspace import init
from mind_mem.namespaces import NamespaceManager


def _block(path: Path, block_id: str, statement: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"[{block_id}]\nType: Decision\nStatement: {statement}\nStatus: active\n\n",
        encoding="utf-8",
    )


@pytest.fixture()
def namespaced_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    init(str(ws))

    # The documented NamespaceManager layout, with intentionally disjoint
    # terms so a hit identifies both namespace and query routing.
    _block(ws / "shared/decisions/DECISIONS.md", "SHARED-1", "shared-only aurora fact")
    _block(ws / "agents/alice/decisions/DECISIONS.md", "ALICE-1", "alice-only nebula fact")
    _block(ws / "agents/bob/decisions/DECISIONS.md", "BOB-1", "bob-only comet fact")
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


def _ids_and_scores(workspace: Path, agent: str, query: str) -> list[tuple[str, float]]:
    """Run the production core entry point and expose rankable evidence."""
    hits = recall(str(workspace), query, limit=10, agent_id=agent, rerank=False)
    return [(str(hit["_id"]), float(hit["score"])) for hit in hits if hit.get("_id")]


def test_known_shared_block_direct_get_and_search_reachability(namespaced_workspace: Path) -> None:
    """An ACL-allowed shared block must be searchable by the allowed agent."""
    workspace = namespaced_workspace
    shared = MarkdownBlockStore(str(workspace / "shared"))
    manager = NamespaceManager(str(workspace), agent_id="alice")

    direct = shared.get_by_id("SHARED-1")
    assert direct is not None
    assert manager.can_read("shared/decisions/DECISIONS.md")

    hits = _ids_and_scores(workspace, "alice", "aurora")
    assert hits, "allowed shared block disappeared from production recall"
    assert hits[0][0] == "SHARED-1"
    assert hits[0][1] > 0.0


def test_private_blocks_are_reachable_only_by_their_agent(namespaced_workspace: Path) -> None:
    workspace = namespaced_workspace
    alice_store = MarkdownBlockStore(str(workspace / "agents/alice"))
    bob_store = MarkdownBlockStore(str(workspace / "agents/bob"))
    assert alice_store.get_by_id("ALICE-1") is not None
    assert bob_store.get_by_id("BOB-1") is not None

    alice_hits = _ids_and_scores(workspace, "alice", "nebula")
    bob_hits = _ids_and_scores(workspace, "bob", "nebula")
    assert alice_hits and alice_hits[0][0] == "ALICE-1"
    assert alice_hits[0][1] > 0.0
    assert all(block_id != "ALICE-1" for block_id, _ in bob_hits)


def test_same_query_reports_individual_scores_and_stable_ranks(namespaced_workspace: Path) -> None:
    workspace = namespaced_workspace
    first = _ids_and_scores(workspace, "alice", "nebula")
    second = _ids_and_scores(workspace, "alice", "nebula")
    assert first == second
    assert first
    assert all(score == score and score > 0.0 for _, score in first)
    # The helper emits result order as rank order; no score is fabricated by
    # the test, and a score/routing regression changes this evidence directly.
    assert [block_id for block_id, _ in first] == ["ALICE-1"]


def test_acl_denial_removes_other_agent_private_block(namespaced_workspace: Path) -> None:
    """A real ACL mutation must change the production result, not just direct get."""
    workspace = namespaced_workspace
    acl = json.loads((workspace / "mind-mem-acl.json").read_text(encoding="utf-8"))
    acl["agents"]["bob"]["read"] = ["shared"]
    acl["agents"]["bob"]["namespaces"] = ["shared"]
    (workspace / "mind-mem-acl.json").write_text(json.dumps(acl), encoding="utf-8")

    assert MarkdownBlockStore(str(workspace / "agents/bob")).get_by_id("BOB-1") is not None
    denied = _ids_and_scores(workspace, "bob", "comet")
    assert all(block_id != "BOB-1" for block_id, _ in denied)


def test_shared_acl_denial_removes_shared_hit(namespaced_workspace: Path) -> None:
    workspace = namespaced_workspace
    acl = json.loads((workspace / "mind-mem-acl.json").read_text(encoding="utf-8"))
    acl["agents"]["alice"]["read"] = ["agents/alice"]
    acl["agents"]["alice"]["namespaces"] = ["agents/alice"]
    (workspace / "mind-mem-acl.json").write_text(json.dumps(acl), encoding="utf-8")

    denied = _ids_and_scores(workspace, "alice", "aurora")
    assert all(block_id != "SHARED-1" for block_id, _ in denied)


def test_unknown_agent_uses_documented_shared_default(namespaced_workspace: Path) -> None:
    hits = _ids_and_scores(namespaced_workspace, "unlisted-agent", "aurora")
    assert hits and hits[0][0] == "SHARED-1"
    assert hits[0][1] > 0.0


def test_shared_symlink_escape_is_not_recalled(namespaced_workspace: Path, tmp_path: Path) -> None:
    workspace = namespaced_workspace
    outside = tmp_path / "outside"
    outside.mkdir()
    _block(outside / "DECISIONS.md", "ESCAPE-1", "outside-only aurora fact")

    shutil.rmtree(workspace / "shared/decisions")
    os.symlink(outside, workspace / "shared/decisions")

    # The logical path remains ACL-readable and a raw store follows the link;
    # recall must enforce realpath confinement before parsing it.
    assert NamespaceManager(str(workspace), agent_id="alice").can_read(
        "shared/decisions/DECISIONS.md"
    )
    assert MarkdownBlockStore(str(workspace / "shared")).get_by_id("ESCAPE-1") is not None
    hits = _ids_and_scores(workspace, "alice", "aurora")
    assert all(block_id != "ESCAPE-1" for block_id, _ in hits)
