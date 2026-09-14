"""Governed RA.4 entity equivalence controls.

These tests exercise the registered MCP doors and a real SQLite transaction.
The merge view must preserve raw source rows; it is an approved SAME_AS
neighborhood, not an ID-rewriting migration.
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest
from fastmcp.server.auth import AccessToken

import mind_mem.mcp.infra.acl as acl
from mind_mem.knowledge_graph import KnowledgeGraph, default_db_path
from mind_mem.mcp.tools.graph import (
    approve_entity_merge,
    graph_add_edge,
    graph_query,
    propose_entity_merge,
    reverse_entity_merge,
)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "decisions").mkdir(parents=True)
    (workspace / "decisions" / "DECISIONS.md").write_text("# decisions\n", encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(workspace))
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    return str(workspace)


def _admin(monkeypatch):
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")


def _seed(ws: str, monkeypatch) -> None:
    _admin(monkeypatch)
    assert "error" not in json.loads(graph_add_edge("loser", "depends_on", "target", "SRC-L"))
    assert "error" not in json.loads(graph_add_edge("winner", "supports", "anchor", "SRC-W"))
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
    with KnowledgeGraph(default_db_path(ws)) as kg:
        kg.entities.add_alias("loser-alias", "loser")


def _rows(ws: str):
    with KnowledgeGraph(default_db_path(ws)) as kg:
        return [
            tuple(row)
            for row in kg._conn.execute(
                "SELECT subject, predicate, object, source_block_id FROM edges ORDER BY subject, predicate, object, source_block_id"
            ).fetchall()
        ]


def test_merge_stages_is_immutable_and_user_cannot_approve(ws, monkeypatch):
    _seed(ws, monkeypatch)
    before = _rows(ws)
    staged = json.loads(propose_entity_merge("winner", "loser", "same record by review"))
    assert staged["status"] == "staged"
    assert _rows(ws) == before
    denied = json.loads(approve_entity_merge(staged["proposal_id"]))
    assert "admin scope" in denied["error"]
    assert _rows(ws) == before


def test_approved_merge_reads_both_preserved_neighborhoods_and_is_idempotent(ws, monkeypatch):
    _seed(ws, monkeypatch)
    from mind_mem.v4 import feature_flags

    monkeypatch.setattr(feature_flags, "is_enabled", lambda flag: flag == "entity_observations")
    with KnowledgeGraph(default_db_path(ws)) as kg:
        kg.entities.add_observation("loser", "loser source fact")
        before_observations = kg.entities.observations("loser")
    staged = json.loads(propose_entity_merge("winner", "loser", "same record by review"))
    _admin(monkeypatch)
    approved = json.loads(approve_entity_merge(staged["proposal_id"]))
    assert approved["status"] == "applied"
    raw = _rows(ws)
    assert ("loser", "depends_on", "target", "SRC-L") in raw
    assert ("winner", "supports", "anchor", "SRC-W") in raw
    assert ("winner", "same_as", "loser", staged["proposal_id"]) in raw
    with KnowledgeGraph(default_db_path(ws)) as kg:
        assert kg.entities.observations("loser") == before_observations
    component_query = json.loads(graph_query("loser-alias", resolve_same_as=True))
    assert any(item["entity"] == "anchor" for item in component_query["neighbors"])
    winner_query = json.loads(graph_query("winner", resolve_same_as=True))
    assert any(item["entity"] == "target" for item in winner_query["neighbors"])
    again = json.loads(approve_entity_merge(staged["proposal_id"]))
    assert again["status"] == "applied"
    assert _rows(ws) == raw


def test_reverse_retracts_only_same_as_and_restores_raw_view(ws, monkeypatch):
    _seed(ws, monkeypatch)
    staged = json.loads(propose_entity_merge("winner", "loser", "same record by review"))
    _admin(monkeypatch)
    approve_entity_merge(staged["proposal_id"])
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
    denied = json.loads(reverse_entity_merge(staged["proposal_id"]))
    assert "admin scope" in denied["error"]
    _admin(monkeypatch)
    reversed_out = json.loads(reverse_entity_merge(staged["proposal_id"]))
    assert reversed_out["status"] == "reversed"
    assert ("loser", "depends_on", "target", "SRC-L") in _rows(ws)
    assert not any(row[1] == "same_as" for row in _rows(ws))
    with KnowledgeGraph(default_db_path(ws)) as kg:
        assert kg.entities.lookup("loser-alias") == "loser"
        assert kg.same_as_component("winner") == ("winner",)


def test_missing_self_and_existing_equivalence_are_refused(ws, monkeypatch):
    _seed(ws, monkeypatch)
    assert "distinct" in json.loads(propose_entity_merge("winner", "winner", "self link"))["error"]
    assert "existing entities" in json.loads(propose_entity_merge("winner", "missing", "missing target"))["error"]
    first = json.loads(propose_entity_merge("winner", "loser", "same record by review"))
    _admin(monkeypatch)
    assert json.loads(approve_entity_merge(first["proposal_id"]))["status"] == "applied"
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
    reverse_direction = json.loads(propose_entity_merge("loser", "winner", "reverse duplicate"))
    _admin(monkeypatch)
    conflict = json.loads(approve_entity_merge(reverse_direction["proposal_id"]))
    assert "conflicts" in conflict["error"]


def test_database_failure_rolls_back_edge_and_lineage(ws, monkeypatch):
    _seed(ws, monkeypatch)
    staged = json.loads(propose_entity_merge("winner", "loser", "same record by review"))
    with KnowledgeGraph(default_db_path(ws)) as kg:
        kg._conn.execute(
            "CREATE TRIGGER fail_ra4_merge BEFORE INSERT ON edges "
            "WHEN NEW.predicate = 'same_as' BEGIN SELECT RAISE(ABORT, 'injected rollback'); END"
        )
        kg._conn.commit()
    _admin(monkeypatch)
    failed = json.loads(approve_entity_merge(staged["proposal_id"]))
    assert failed["error"] == "database backend error"
    with KnowledgeGraph(default_db_path(ws)) as kg:
        assert kg.get_entity_merge_proposal(staged["proposal_id"]).status == "staged"
        assert (
            kg._conn.execute("SELECT COUNT(*) FROM entity_merge_lineage WHERE proposal_id = ?", (staged["proposal_id"],)).fetchone()[0] == 0
        )
        kg._conn.execute("DROP TRIGGER fail_ra4_merge")
        kg._conn.commit()
    assert not any(row[1] == "same_as" for row in _rows(ws))


def test_generic_same_as_edge_door_is_refused(ws, monkeypatch):
    _admin(monkeypatch)
    out = json.loads(graph_add_edge("winner", "same_as", "loser", "UNSAFE"))
    assert "propose_entity_merge" in out["error"]
    assert not os.path.exists(default_db_path(ws))


def test_connected_component_and_forged_proposal_refuse_with_access_tokens(ws, monkeypatch):
    """Approval binds both the verified caller and persisted proposal identity."""
    with KnowledgeGraph(default_db_path(ws)) as kg:
        for entity in ("a", "b", "c"):
            kg.entities.resolve(entity)

    current = {"scope": "user"}

    def token():
        return AccessToken(
            token="fixture-entity-merge",
            client_id="fixture-client",
            scopes=[current["scope"]],
            claims={"sub": "fixture-admin" if current["scope"] == "admin" else "fixture-user"},
        )

    monkeypatch.setattr(acl, "get_access_token", token)
    padded = json.loads(propose_entity_merge("a", "b", "a b c d e   "))
    assert "non-whitespace" in padded["error"]
    for winner, loser in (("a", "b"), ("b", "c")):
        staged = json.loads(propose_entity_merge(winner, loser, "reviewed entity relation"))
        assert staged["status"] == "staged"
        current["scope"] = "admin"
        assert json.loads(approve_entity_merge(staged["proposal_id"]))["status"] == "applied"
        current["scope"] = "user"

    cycle = json.loads(propose_entity_merge("c", "a", "reviewed cycle relation"))
    assert cycle["status"] == "staged"
    current["scope"] = "admin"
    cycle_result = json.loads(approve_entity_merge(cycle["proposal_id"]))
    assert "cycle" in cycle_result["error"]

    db = default_db_path(ws)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO entity_merge_proposals (proposal_id, winner_id, loser_id, rationale, status, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            ("EMP-FORGED", "a", "c", "forged persisted relation", "staged", "{}"),
        )
        conn.commit()
    forged = json.loads(approve_entity_merge("EMP-FORGED"))
    assert "identity" in forged["error"]
    with KnowledgeGraph(db) as kg:
        assert kg._conn.execute("SELECT COUNT(*) FROM edges WHERE predicate = 'same_as'").fetchone()[0] == 2


def _install_entity_merge_token(monkeypatch):
    current = {"scope": "user"}

    def token():
        return AccessToken(
            token="fixture-entity-merge-integrity",
            client_id="fixture-client",
            scopes=[current["scope"]],
            claims={"sub": "fixture-admin"},
        )

    monkeypatch.setattr(acl, "get_access_token", token)
    return current


def test_applied_merge_missing_edge_refuses_idempotent_approval(ws, monkeypatch):
    """An applied proposal cannot hide a deleted SAME_AS edge."""
    current = _install_entity_merge_token(monkeypatch)
    with KnowledgeGraph(default_db_path(ws)) as kg:
        kg.entities.resolve("winner")
        kg.entities.resolve("loser")
    staged = json.loads(propose_entity_merge("winner", "loser", "reviewed relation identity"))
    current["scope"] = "admin"
    assert json.loads(approve_entity_merge(staged["proposal_id"]))["status"] == "applied"
    with KnowledgeGraph(default_db_path(ws)) as kg:
        kg._conn.execute(
            "DELETE FROM edges WHERE predicate = 'same_as' AND source_block_id = ?",
            (staged["proposal_id"],),
        )
        kg._conn.commit()
    refused = json.loads(approve_entity_merge(staged["proposal_id"]))
    assert "missing or changed" in refused["error"]
    with KnowledgeGraph(default_db_path(ws)) as kg:
        assert kg._conn.execute("SELECT COUNT(*) FROM edges WHERE predicate = 'same_as'").fetchone()[0] == 0
        assert kg.get_entity_merge_proposal(staged["proposal_id"]).status == "applied"


def test_applied_merge_mutated_lineage_refuses_idempotent_approval(ws, monkeypatch):
    """An applied proposal cannot approve a lineage with changed endpoints."""
    current = _install_entity_merge_token(monkeypatch)
    with KnowledgeGraph(default_db_path(ws)) as kg:
        kg.entities.resolve("winner")
        kg.entities.resolve("loser")
    staged = json.loads(propose_entity_merge("winner", "loser", "reviewed relation identity"))
    current["scope"] = "admin"
    assert json.loads(approve_entity_merge(staged["proposal_id"]))["status"] == "applied"
    with KnowledgeGraph(default_db_path(ws)) as kg:
        kg._conn.execute(
            "UPDATE entity_merge_lineage SET loser_id = 'forged' WHERE proposal_id = ?",
            (staged["proposal_id"],),
        )
        kg._conn.commit()
    refused = json.loads(approve_entity_merge(staged["proposal_id"]))
    assert "endpoints do not match" in refused["error"]
    with KnowledgeGraph(default_db_path(ws)) as kg:
        assert kg._conn.execute("SELECT COUNT(*) FROM edges WHERE predicate = 'same_as'").fetchone()[0] == 1


def test_graph_query_expands_equivalence_at_each_bounded_hop(ws, monkeypatch):
    """A SAME_AS component supplies edges at the current BFS level."""
    with KnowledgeGraph(default_db_path(ws)) as kg:
        for entity in ("a", "b", "c", "d"):
            kg.entities.resolve(entity)
        # SAME_AS is deliberately not a generic edge-door write.  This is a
        # small persisted read fixture for the approved-edge representation.
        for subject, predicate, object_, source in (
            ("a", "related_to", "b", "SRC-AB"),
            ("b", "same_as", "c", "EMP-BC"),
            ("c", "related_to", "d", "SRC-CD"),
        ):
            kg._conn.execute(
                "INSERT INTO edges(subject, predicate, object, source_block_id, confidence, metadata) VALUES (?, ?, ?, ?, 1.0, '{}')",
                (subject, predicate, object_, source),
            )
        kg._conn.commit()

    monkeypatch.setattr(
        acl,
        "get_access_token",
        lambda: AccessToken(
            token="fixture-graph-query",
            client_id="fixture-client",
            scopes=["user"],
            claims={"sub": "fixture-user"},
        ),
    )
    resolved = json.loads(graph_query("a", depth=2, resolve_same_as=True))
    assert [(item["entity"], item["hop"], item["predicate"]) for item in resolved["neighbors"]] == [
        ("b", 1, "related_to"),
        ("d", 2, "related_to"),
    ]
    plain = json.loads(graph_query("a", depth=2, resolve_same_as=False))
    assert [item["entity"] for item in plain["neighbors"]] == ["b", "c"]
