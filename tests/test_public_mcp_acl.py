"""Public MCP recall binds verified transport principals before retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mind_mem.hybrid_recall as hybrid_recall
import mind_mem.mcp.infra.acl as acl
import mind_mem.mcp.tools.public as public_tools
import mind_mem.mcp.tools.recall as recall_tools
import mind_mem.recall_cache as recall_cache
import mind_mem.sqlite_index as sqlite_index
from mind_mem.audit_context import bind_audit_context, bind_current_agent, context_from_headers
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace


def _write_block(path: Path, block_id: str, statement: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"[{block_id}]\nType: Decision\nStatement: {statement}\nStatus: active\n\n",
        encoding="utf-8",
    )


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "public-mcp-acl"
    init(str(ws))
    for path, block_id, text in (
        ("shared/decisions/DECISIONS.md", "D-SHARED", "shared aurora evidence"),
        ("agents/alice/decisions/DECISIONS.md", "D-ALICE", "alice aurora evidence"),
        ("agents/bob/decisions/DECISIONS.md", "D-BOB", "bob aurora evidence"),
    ):
        _write_block(ws / path, block_id, text)
    config = json.loads((ws / "mind-mem.json").read_text(encoding="utf-8"))
    config.setdefault("recall", {})["backend"] = "sqlite"
    config.setdefault("cache", {})["enabled"] = True
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


def _public_recall(ws: Path, token_subject: str, *, query: str = "aurora") -> list[str]:
    old = acl.get_access_token
    acl.get_access_token = lambda: SimpleNamespace(claims={"sub": token_subject})  # type: ignore[assignment]
    try:
        with use_workspace(str(ws)):
            payload = json.loads(recall_tools.recall.__wrapped__(query, backend="bm25", limit=10))
        return [str(hit["_id"]) for hit in payload.get("results", [])]
    finally:
        acl.get_access_token = old


def test_verified_token_principal_controls_cold_warm_public_recall_and_revoke(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    alice_cold = _public_recall(ws, "alice")
    bob_warm = _public_recall(ws, "bob")
    alice_warm = _public_recall(ws, "alice")
    assert set(alice_cold) == {"D-SHARED", "D-ALICE"}
    assert set(bob_warm) == {"D-SHARED", "D-BOB"}
    assert alice_warm == alice_cold

    policy = json.loads((ws / "mind-mem-acl.json").read_text(encoding="utf-8"))
    policy["agents"]["alice"]["read"] = ["shared"]
    policy["agents"]["alice"]["namespaces"] = ["shared"]
    (ws / "mind-mem-acl.json").write_text(json.dumps(policy), encoding="utf-8")
    assert _public_recall(ws, "alice") == ["D-SHARED"]


def test_forged_actor_claim_cannot_change_verified_token_scope(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    old = acl.get_access_token
    acl.get_access_token = lambda: SimpleNamespace(claims={"sub": "alice"})  # type: ignore[assignment]
    try:
        with bind_current_agent("alice"), use_workspace(str(ws)):
            with bind_audit_context(context_from_headers(lambda key: {"x-mindmem-actor": "bob"}.get(key), transport="stdio")):
                payload = json.loads(recall_tools.recall.__wrapped__("aurora", backend="bm25", limit=10))
        ids = [str(hit["_id"]) for hit in payload.get("results", [])]
    finally:
        acl.get_access_token = old
    assert "D-ALICE" in ids
    assert "D-BOB" not in ids


def test_bound_principal_bypasses_shared_recall_cache_and_anticipation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    called = {"cache": 0, "anticipation": 0}

    def _cache(*args, **kwargs):
        called["cache"] += 1
        raise AssertionError("bound principals must not consume shared recall cache")

    def _anticipation(*args, **kwargs):
        called["anticipation"] += 1
        raise AssertionError("bound principals must not consume shared anticipation bundles")

    monkeypatch.setattr(recall_cache, "cached_recall", _cache)
    monkeypatch.setattr(recall_tools, "_anticipation_envelope", _anticipation)
    assert set(_public_recall(ws, "alice")) == {"D-SHARED", "D-ALICE"}
    assert called == {"cache": 0, "anticipation": 0}


def test_axis_and_pack_use_the_same_verified_principal(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    old = acl.get_access_token
    acl.get_access_token = lambda: SimpleNamespace(claims={"sub": "alice"})  # type: ignore[assignment]
    try:
        with use_workspace(str(ws)):
            axis = json.loads(recall_tools.recall_with_axis.__wrapped__("aurora", axes="lexical", limit=10))
            packed = json.loads(recall_tools.pack_recall_budget.__wrapped__("aurora", max_tokens=2000, limit=10))
    finally:
        acl.get_access_token = old
    axis_ids = {str(hit["_id"]) for hit in axis.get("results", []) if isinstance(hit, dict) and "_id" in hit}
    assert axis_ids == {"D-ALICE", "D-SHARED"}
    assert "D-BOB" not in axis_ids
    assert "D-ALICE" in {str(hit["_id"]) for hit in packed.get("included", [])}
    assert "D-BOB" not in {str(hit["_id"]) for hit in packed.get("included", [])}


def test_consolidated_public_recall_and_hybrid_never_cross_principal(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    old = acl.get_access_token
    try:
        acl.get_access_token = lambda: SimpleNamespace(claims={"sub": "alice"})  # type: ignore[assignment]
        with use_workspace(str(ws)):
            public_result = json.loads(public_tools.recall.__wrapped__("aurora", mode="bm25", limit=10))
            hybrid_result = json.loads(recall_tools.hybrid_search.__wrapped__("aurora", limit=10))
    finally:
        acl.get_access_token = old
    for envelope in (public_result, hybrid_result):
        ids = {str(hit["_id"]) for hit in envelope.get("results", [])}
        assert "D-ALICE" in ids and "D-SHARED" in ids
        assert "D-BOB" not in ids


def test_post_retrieval_expansion_is_filtered_before_public_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    original = hybrid_recall.HybridBackend._maybe_entity_prefetch

    def append_private(self, query, workspace, result, **kwargs):
        expanded = list(result)
        expanded.append(
            {
                "_id": "D-BOB",
                "file": "agents/bob/decisions/DECISIONS.md",
                "excerpt": "private expansion",
                "score": 99.0,
            }
        )
        return expanded

    monkeypatch.setattr(hybrid_recall.HybridBackend, "_maybe_entity_prefetch", append_private)
    try:
        assert set(_public_recall(ws, "alice")) == {"D-SHARED", "D-ALICE"}
    finally:
        monkeypatch.setattr(hybrid_recall.HybridBackend, "_maybe_entity_prefetch", original)


def test_registered_public_callable_binds_verified_token_principal(tmp_path: Path) -> None:
    from fastmcp.server.auth import AccessToken

    ws = _workspace(tmp_path)
    old = acl.get_access_token
    acl.get_access_token = lambda: AccessToken(
        token="fixture-token",
        client_id="fixture-client",
        scopes=["user"],
        claims={"sub": "alice"},
    )
    try:
        with use_workspace(str(ws)):
            payload = json.loads(public_tools.recall("aurora", mode="bm25", limit=10))
    finally:
        acl.get_access_token = old
    ids = {str(hit["_id"]) for hit in payload.get("results", [])}
    assert ids == {"D-SHARED", "D-ALICE"}


def test_authenticated_agent_id_rejects_invalid_verified_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acl, "get_access_token", lambda: SimpleNamespace(claims={"sub": "../bob"}))
    with pytest.raises(Exception):
        acl.authenticated_agent_id()
