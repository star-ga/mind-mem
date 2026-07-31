"""MCP-surface tests for the HITL typed-edge flow + provenance (v4.4.0 Finding 1)
and the entity-observation tools (Finding 4).

These lock the audit remediation: the propose→approve→reject→list proposal flow
is reachable as real MCP tools, the source-of-truth ``edges`` table is never
written without an explicit operator approval, direct admin writes are stamped
with a distinguishing provenance marker, and a default user-scope caller cannot
reach either direct-commit path.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.knowledge_graph import (
    EDGE_ORIGIN_DIRECT_ADMIN,
    EDGE_ORIGIN_HITL_APPROVED,
    KnowledgeGraph,
    default_db_path,
)
from mind_mem.mcp.tools.graph import (
    approve_edge,
    entity_add_observation,
    entity_observations,
    graph_add_edge,
    list_edge_proposals,
    propose_edge,
    reject_edge,
)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace with the MCP env pointed at it. Scope defaults to user."""
    d = tmp_path / "ws"
    d.mkdir()
    # _check_workspace requires the Markdown-backend decisions/ dir.
    dec = d / "decisions"
    dec.mkdir()
    (dec / "DECISIONS.md").write_text("# DECISIONS\n", encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(d))
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)  # default: user
    return str(d)


def _admin(monkeypatch):
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")


def _user(monkeypatch):
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)


def _edges(ws: str):
    with KnowledgeGraph(default_db_path(ws)) as kg:
        return kg._query_edges()  # all edges


# ---------------------------------------------------------------------------
# Finding 1 (i) — propose→approve is the MCP path; un-approved does NOT commit
# ---------------------------------------------------------------------------


def test_propose_stages_without_committing(ws, monkeypatch):
    _user(monkeypatch)  # propose is user-scope
    out = json.loads(propose_edge("STARGA", "supports", "MIND", "D-20260101-001", 0.8))
    assert out["status"] == "staged"
    assert out["proposal_id"].startswith("EP-")
    # Source-of-truth graph untouched: no edge committed by a proposal.
    assert _edges(ws) == []


def test_approve_is_sole_committer(ws, monkeypatch):
    _user(monkeypatch)
    prop = json.loads(propose_edge("STARGA", "supports", "MIND", "D-20260101-001", 0.8))
    pid = prop["proposal_id"]
    assert _edges(ws) == []  # still nothing before approval

    _admin(monkeypatch)  # approval is an explicit operator (admin) signal
    approved = json.loads(approve_edge(pid))
    assert approved["approved"] == pid
    edges = _edges(ws)
    assert len(edges) == 1
    # Approved edges are stamped as HITL-reviewed, distinct from direct writes.
    assert edges[0].metadata.get("origin") == EDGE_ORIGIN_HITL_APPROVED


def test_reject_never_writes(ws, monkeypatch):
    _user(monkeypatch)
    pid = json.loads(propose_edge("A", "contradicts", "B", "D-20260101-002"))["proposal_id"]
    _admin(monkeypatch)
    out = json.loads(reject_edge(pid))
    assert out["status"] == "rejected"
    assert _edges(ws) == []


def test_list_edge_proposals_read_at_user_scope(ws, monkeypatch):
    _user(monkeypatch)
    propose_edge("A", "supports", "B", "D-1")
    propose_edge("C", "refines", "D", "D-2")
    out = json.loads(list_edge_proposals(status="staged"))
    assert out["count"] == 2
    assert {p["predicate"] for p in out["proposals"]} == {"supports", "refines"}


# ---------------------------------------------------------------------------
# Finding 1 (ii) — direct admin graph_add_edge stamps the provenance marker
# ---------------------------------------------------------------------------


def test_direct_admin_edge_stamps_origin(ws, monkeypatch):
    _admin(monkeypatch)
    out = json.loads(graph_add_edge("STARGA", "depends_on", "mindc", "D-20260101-001"))
    assert out["metadata"]["origin"] == EDGE_ORIGIN_DIRECT_ADMIN
    edges = _edges(ws)
    assert len(edges) == 1
    assert edges[0].metadata.get("origin") == EDGE_ORIGIN_DIRECT_ADMIN


def test_direct_and_hitl_edges_are_distinguishable(ws, monkeypatch):
    _admin(monkeypatch)
    graph_add_edge("X", "supports", "Y", "D-1")
    _user(monkeypatch)
    pid = json.loads(propose_edge("P", "supports", "Q", "D-2"))["proposal_id"]
    _admin(monkeypatch)
    approve_edge(pid)
    origins = sorted(e.metadata.get("origin") for e in _edges(ws))
    assert origins == [EDGE_ORIGIN_DIRECT_ADMIN, EDGE_ORIGIN_HITL_APPROVED]


# ---------------------------------------------------------------------------
# Finding 1 (iii) — a default user-scope caller cannot reach a direct-commit path
# ---------------------------------------------------------------------------


def test_user_cannot_direct_add_edge(ws, monkeypatch):
    _user(monkeypatch)
    out = json.loads(graph_add_edge("STARGA", "depends_on", "mindc", "D-1"))
    assert "admin scope" in out["error"]
    # No graph db write occurred.
    import os

    assert not os.path.isfile(default_db_path(ws))


def test_user_cannot_approve(ws, monkeypatch):
    _user(monkeypatch)
    pid = json.loads(propose_edge("A", "supports", "B", "D-1"))["proposal_id"]
    # User attempts to commit via approve — denied by ACL, nothing committed.
    out = json.loads(approve_edge(pid))
    assert "admin scope" in out["error"]
    assert _edges(ws) == []


def test_user_cannot_reject(ws, monkeypatch):
    _user(monkeypatch)
    pid = json.loads(propose_edge("A", "supports", "B", "D-1"))["proposal_id"]
    out = json.loads(reject_edge(pid))
    assert "admin scope" in out["error"]


# ---------------------------------------------------------------------------
# Finding 4 — entity observation MCP tools (flag-gated, ACL-gated)
# ---------------------------------------------------------------------------


@pytest.fixture
def obs_enabled(monkeypatch):
    """Force the v4 entity_observations flag ON for the observation tools."""
    from mind_mem.v4 import feature_flags

    monkeypatch.setattr(feature_flags, "is_enabled", lambda flag: flag == "entity_observations")


def test_entity_add_and_read_observation(ws, monkeypatch, obs_enabled):
    _admin(monkeypatch)  # writing an observation mutates the registry
    out = json.loads(entity_add_observation("STARGA", "ships the MIND compiler"))
    assert out["observations"] == ["ships the MIND compiler"]

    _user(monkeypatch)  # reading is user-scope
    read = json.loads(entity_observations("STARGA"))
    assert read["observations"] == ["ships the MIND compiler"]
    assert read["count"] == 1


def test_entity_observations_unknown_entity_empty(ws, monkeypatch, obs_enabled):
    _user(monkeypatch)
    out = json.loads(entity_observations("never-seen"))
    assert out["observations"] == []
    assert out["count"] == 0


def test_user_cannot_add_observation(ws, monkeypatch, obs_enabled):
    _user(monkeypatch)
    out = json.loads(entity_add_observation("STARGA", "fact"))
    assert "admin scope" in out["error"]


def test_observation_flag_disabled_errors(ws, monkeypatch):
    from mind_mem.v4 import feature_flags

    monkeypatch.setattr(feature_flags, "is_enabled", lambda flag: False)
    _admin(monkeypatch)
    out = json.loads(entity_add_observation("STARGA", "fact"))
    assert "entity_observations" in out["error"]
