"""Consolidation previews must use the caller's canonical admitted corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp.server.auth import AccessToken

import mind_mem.mcp.infra.acl as acl
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.consolidation import plan_consolidation, project_profile, propagate_staleness
from mind_mem.sqlite_index import _connect, _index_file, _init_schema


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    ws = tmp_path / "consolidation"
    init(str(ws))
    files = {
        "shared/decisions/DECISIONS.md": ("D-SHARED", "shared"),
        "agents/alice/decisions/DECISIONS.md": ("D-ALICE", "alice"),
        "agents/bob/decisions/DECISIONS.md": ("D-BOB", "bob"),
    }
    for relative, (block_id, owner) in files.items():
        path = ws / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"[{block_id}]\nStatement: aurora memory evidence for {owner}\nStatus: active\nDate: 2020-01-01\n\n",
            encoding="utf-8",
        )
    config_path = ws / "mind-mem.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("v4", {})["granularity_align"] = {"enabled": True, "min_similarity": 0.0}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (ws / "mind-mem-acl.json").write_text(
        json.dumps({"default_policy": "deny", "agents": {who: {"read": ["shared", f"agents/{who}"]} for who in ("alice", "bob")}}),
        encoding="utf-8",
    )
    conn = _connect(str(ws))
    try:
        _init_schema(conn)
        for relative, (block_id, _) in files.items():
            _index_file(conn, str(ws), relative, relative, {block_id}, force=True)
        conn.commit()
    finally:
        conn.close()
    return ws


def _bind_principal(monkeypatch: pytest.MonkeyPatch, actor: str) -> None:
    monkeypatch.setattr(
        acl,
        "get_access_token",
        lambda: AccessToken(token="fixture", client_id="fixture", scopes=["user"], claims={"sub": actor}),
    )


def _preview(ws: Path, monkeypatch: pytest.MonkeyPatch, actor: str, *, maturity: bool = False) -> dict:
    _bind_principal(monkeypatch, actor)
    with use_workspace(str(ws)):
        return json.loads(plan_consolidation(importance_threshold=1.0, stale_days=0, maturity_gate=maturity, min_maturity=0.0))


@pytest.mark.parametrize("actor,other", [("alice", "bob"), ("bob", "alice")])
@pytest.mark.parametrize("maturity", [False, True])
def test_registered_preview_keeps_private_ids_and_merge_text_isolated(corpus, monkeypatch, actor, other, maturity):
    result = _preview(corpus, monkeypatch, actor, maturity=maturity)
    assert set(result["plan"]["mark"]) == {"D-SHARED", f"D-{actor.upper()}"}
    assert result["granularity_align"]["candidates"], "authorized merge candidates must still be produced"
    assert f"D-{other.upper()}" not in json.dumps(result)
    assert f"evidence for {other}" not in json.dumps(result)


def test_preview_rebuilds_content_from_corpus_after_index_tampering(corpus, monkeypatch):
    conn = _connect(str(corpus))
    try:
        conn.execute("UPDATE blocks SET json_blob = ?", (json.dumps({"Statement": "PRIVATE INDEX CANARY", "Maturity": 1.0}),))
        conn.commit()
    finally:
        conn.close()
    result = _preview(corpus, monkeypatch, "alice")
    assert result["granularity_align"]["candidates"]
    assert "aurora memory evidence" in json.dumps(result)
    assert "PRIVATE INDEX CANARY" not in json.dumps(result)


def test_preview_withholds_live_revocation_even_with_stale_active_index(corpus, monkeypatch):
    path = corpus / "agents/alice/decisions/DECISIONS.md"
    path.write_text(path.read_text(encoding="utf-8").replace("Status: active", "Status: quarantined"), encoding="utf-8")
    result = _preview(corpus, monkeypatch, "alice", maturity=True)
    assert result["plan"]["mark"] == ["D-SHARED"]
    assert "D-ALICE" not in json.dumps(result)
    assert "D-BOB" not in json.dumps(result)


def test_denied_principal_does_not_fall_back_to_global_index(corpus, monkeypatch):
    policy_path = corpus / "mind-mem-acl.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["agents"]["outsider"] = {"read": []}
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    result = _preview(corpus, monkeypatch, "outsider", maturity=True)
    assert result["plan"]["total"] == 0
    assert result["granularity_align"]["scanned_blocks"] == 0
    assert all(block_id not in json.dumps(result) for block_id in ("D-SHARED", "D-ALICE", "D-BOB"))


def test_staleness_expansion_does_not_disclose_private_neighbors(corpus, monkeypatch):
    conn = _connect(str(corpus))
    try:
        conn.executemany("INSERT INTO xref_edges (src, dst) VALUES (?, ?)", [("D-SHARED", "D-ALICE"), ("D-SHARED", "D-BOB")])
        conn.commit()
    finally:
        conn.close()
    _bind_principal(monkeypatch, "alice")
    with use_workspace(str(corpus)):
        result = json.loads(propagate_staleness("D-SHARED"))
        denied = json.loads(propagate_staleness("D-BOB"))
    assert set(result["scores"]) == {"D-SHARED", "D-ALICE"}
    assert "D-BOB" not in json.dumps(result)
    assert denied["seed"] == [] and denied["scores"] == {}


def test_project_profile_uses_only_canonical_authorized_sources(corpus, monkeypatch):
    _bind_principal(monkeypatch, "alice")
    with use_workspace(str(corpus)):
        result = json.loads(project_profile())
    assert result["total_blocks"] == 2
    assert set(result["top_files"]) == {"shared/decisions/DECISIONS.md", "agents/alice/decisions/DECISIONS.md"}
    assert "bob" not in json.dumps(result)
    assert "aurora" in result["top_concepts"]


def test_explicit_default_deny_withholds_unknown_principal_but_read_default_still_works(corpus, monkeypatch):
    denied = _preview(corpus, monkeypatch, "outsider")
    assert denied["plan"]["total"] == 0
    policy_path = corpus / "mind-mem-acl.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["default_policy"] = "read"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    allowed = _preview(corpus, monkeypatch, "outsider")
    assert allowed["plan"]["mark"] == ["D-SHARED"]
    assert set(_preview(corpus, monkeypatch, "alice")["plan"]["mark"]) == {"D-SHARED", "D-ALICE"}


def test_preview_does_not_follow_shared_symlink_into_private_corpus(corpus, monkeypatch):
    shared = corpus / "shared/decisions/DECISIONS.md"
    shared.unlink()
    try:
        shared.symlink_to(corpus / "agents/bob/decisions/DECISIONS.md")
    except (OSError, NotImplementedError):
        pytest.skip("host does not support symlink controls")
    result = _preview(corpus, monkeypatch, "alice")
    assert result["plan"]["mark"] == ["D-ALICE"]
    assert "D-BOB" not in json.dumps(result)
    assert result["granularity_align"]["scanned_blocks"] == 1
