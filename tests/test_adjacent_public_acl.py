"""ACL controls for public retrieval doors beside the ranked recall tool."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mind_mem.mcp.infra.acl as acl
import mind_mem.mcp.tools.recall as recall_tools
import mind_mem.sqlite_index as sqlite_index
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace


def _write_block(path: Path, block_id: str, statement: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"[{block_id}]\nType: Decision\nStatement: {statement}\nStatus: active\n\n",
        encoding="utf-8",
    )


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "adjacent-public-acl"
    init(str(ws))
    for path, block_id, text in (
        ("shared/decisions/DECISIONS.md", "D-SHARED", "shared aurora evidence"),
        ("agents/alice/decisions/DECISIONS.md", "D-ALICE", "alice aurora evidence"),
        ("agents/bob/decisions/DECISIONS.md", "D-BOB", "bob aurora evidence"),
    ):
        _write_block(ws / path, block_id, text)
    config = json.loads((ws / "mind-mem.json").read_text(encoding="utf-8"))
    config.setdefault("recall", {})["backend"] = "sqlite"
    (ws / "mind-mem.json").write_text(json.dumps(config), encoding="utf-8")
    (ws / "mind-mem-acl.json").write_text(
        json.dumps(
            {
                "default_policy": "read",
                "agents": {
                    "alice": {"read": ["shared", "agents/alice"], "namespaces": ["shared", "agents/alice"]},
                    "bob": {"read": ["shared", "agents/bob"], "namespaces": ["shared", "agents/bob"]},
                },
            }
        ),
        encoding="utf-8",
    )
    manager = sqlite_index._get_conn_manager(str(ws))
    with manager.write_lock:
        connection = manager.get_write_connection()
        connection.row_factory = sqlite_index.sqlite3.Row
        sqlite_index._init_schema(connection)
        for relative, block_id in (
            ("shared/decisions/DECISIONS.md", "D-SHARED"),
            ("agents/alice/decisions/DECISIONS.md", "D-ALICE"),
            ("agents/bob/decisions/DECISIONS.md", "D-BOB"),
        ):
            sqlite_index._index_file(connection, str(ws), relative, relative, {block_id}, force=True)
        connection.commit()
    return ws


def _token(monkeypatch: pytest.MonkeyPatch, subject: str) -> None:
    monkeypatch.setattr(acl, "get_access_token", lambda: SimpleNamespace(claims={"sub": subject}))


def test_persona_walkthrough_guardrails_and_chat_keep_verified_principal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every adjacent public door reaches the ACL-bound ranked core."""
    from mind_mem.mcp.tools import chat as chat_tools
    from mind_mem.mcp.tools import guardrails, walkthrough_persona

    ws = _workspace(tmp_path)
    _token(monkeypatch, "alice")
    with use_workspace(str(ws)):
        persona = json.loads(walkthrough_persona.recall_with_persona.__wrapped__("aurora", persona="brief", limit=10))
        walkthrough = json.loads(walkthrough_persona.compile_truth_walkthrough.__wrapped__("aurora", limit=10))
        guarded = json.loads(guardrails.recall_with_guardrails.__wrapped__("aurora", limit=10))
        chatted = json.loads(chat_tools.chat_with_memory.__wrapped__("aurora", limit=10))

    for envelope in (persona, walkthrough, guarded):
        rendered = json.dumps(envelope)
        assert "D-ALICE" in rendered and "D-SHARED" in rendered
        assert "D-BOB" not in rendered
    assert "D-ALICE" in {str(item.get("block_id")) for item in chatted.get("evidence", [])}
    assert "D-BOB" not in json.dumps(chatted)
    assert chatted["grounded"] is True
    assert chatted["rejected"] is False


def test_chat_injected_recall_is_filtered_before_generator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An extension recall function cannot place a private excerpt in a prompt."""
    from mind_mem.chat_memory import chat_with_memory

    ws = _workspace(tmp_path)
    _token(monkeypatch, "alice")
    seen_prompts: list[str] = []

    def generator(request: object) -> str:
        seen_prompts.append(str(getattr(request, "prompt", "")))
        return "private [[D-BOB]]"

    with use_workspace(str(ws)):
        result = chat_with_memory(
            str(ws),
            "aurora",
            recall_fn=lambda _ws, _question, _limit: [
                {"_id": "D-BOB", "excerpt": "private bob evidence", "file": "agents/bob/decisions/DECISIONS.md"}
            ],
            generator=generator,
            agent_id="alice",
            on_invalid="reject",
        )
    assert result.no_record is True
    assert result.rejected is False
    assert seen_prompts == []


def test_similar_public_door_filters_seed_and_cooccurrence_neighbors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The ID-only co-occurrence index cannot disclose another namespace."""
    from mind_mem.block_metadata import BlockMetadataManager, block_meta_db_path
    from mind_mem.mcp.tools import public as public_tools

    ws = _workspace(tmp_path)
    BlockMetadataManager(block_meta_db_path(str(ws))).record_access(["D-SHARED", "D-ALICE", "D-BOB"])
    _token(monkeypatch, "alice")
    with use_workspace(str(ws)):
        shared = json.loads(recall_tools.find_similar.__wrapped__("D-SHARED", limit=10))
        private_seed = json.loads(recall_tools.find_similar.__wrapped__("D-BOB", limit=10))
        consolidated = json.loads(public_tools.recall.__wrapped__("aurora", mode="similar", block_id="D-SHARED", limit=10))

    shared_ids = set(shared.get("similar", []))
    assert "D-ALICE" in shared_ids
    assert "D-BOB" not in shared_ids
    assert private_seed.get("similar") == []
    assert "D-ALICE" in set(consolidated.get("similar", []))
    assert "D-BOB" not in set(consolidated.get("similar", []))


def test_kind_similarity_filters_private_partition_neighbors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The opt-in kind partition applies the same seed/neighbor ACL."""
    from mind_mem.v4 import feature_flags, hnsw_kind_index

    ws = _workspace(tmp_path)
    monkeypatch.setattr(feature_flags, "is_enabled_quiet", lambda _flag: True)
    monkeypatch.setattr(hnsw_kind_index, "get_block_embedding", lambda _ws, _bid: [1.0, 0.0])
    monkeypatch.setattr(
        hnsw_kind_index,
        "knn_by_kind",
        lambda _ws, _kind, _query, k: [("D-SHARED", 0.0), ("D-ALICE", 0.1), ("D-BOB", 0.2)],
    )
    _token(monkeypatch, "alice")
    with use_workspace(str(ws)):
        allowed = json.loads(recall_tools.find_similar.__wrapped__("D-SHARED", limit=10, kind="decision"))
        denied = json.loads(recall_tools.find_similar.__wrapped__("D-BOB", limit=10, kind="decision"))
    assert [row["block_id"] for row in allowed["similar"]] == ["D-ALICE"]
    assert denied["similar"] == []


def test_similarity_enumerates_an_explicitly_acl_granted_other_agent_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An ACL grant to another agent is included by the namespace resolver."""
    ws = _workspace(tmp_path)
    acl_config = json.loads((ws / "mind-mem-acl.json").read_text(encoding="utf-8"))
    acl_config["agents"]["alice"]["read"].append("agents/bob")
    (ws / "mind-mem-acl.json").write_text(json.dumps(acl_config), encoding="utf-8")
    _token(monkeypatch, "alice")
    with use_workspace(str(ws)):
        servable = recall_tools._servable_block_ids(str(ws), "alice")
    assert servable is not None
    assert {"D-SHARED", "D-ALICE", "D-BOB"}.issubset(servable)


def test_indexed_metadata_cannot_claim_postgres_authority_over_acl(tmp_path: Path) -> None:
    """A source marker cannot override the authenticated namespace ACL."""
    from mind_mem._recall_core import _indexed_hit_is_readable
    from mind_mem.namespaces import NamespaceManager

    ws = _workspace(tmp_path)
    hit = {
        "_id": "D-ALICE",
        "_source_file": "agents/alice/decisions/DECISIONS.md",
        "file": "agents/alice/decisions/DECISIONS.md",
        "_source_authority": "configured_postgres",
    }
    manager = NamespaceManager(str(ws), agent_id="alice")
    assert _indexed_hit_is_readable(str(ws), hit, manager)
    hit["file"] = "agents/bob/decisions/DECISIONS.md"
    assert not _indexed_hit_is_readable(str(ws), hit, manager)
