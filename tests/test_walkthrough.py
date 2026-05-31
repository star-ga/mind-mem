"""Tests for the v3.9 dependency-ordered walkthrough."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from mind_mem.walkthrough import (
    _date_key,
    _role_for,
    _topo_sort,
    compile_walkthrough,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)
    (ws / "intelligence" / "state").mkdir(parents=True)
    (ws / "memory").mkdir()
    config = {
        "version": "3.9.0",
        "workspace_path": str(ws),
        "block_store": {"backend": "markdown"},
    }
    (ws / "mind-mem.json").write_text(json.dumps(config))

    decisions = """[D-20260101-001]
Date: 2026-01-01
Status: active
Subject: Authentication foundation chosen
Statement: Adopt OAuth2 with PKCE for the auth subsystem rewrite.

[D-20260201-001]
Date: 2026-02-01
Status: active
Subject: JWT signing key rotation policy
Statement: Authentication keys rotate every 30 days; old keys remain valid for 7 days during overlap.

[D-20260301-001]
Date: 2026-03-01
Status: active
Subject: MFA rollout schedule
Statement: Authentication MFA gates rolled out per cohort starting March 2026.
"""
    (ws / "decisions" / "DECISIONS.md").write_text(decisions)
    return str(ws)


def _seed_co_retrieval(workspace: str, edges: list[tuple[str, str, float]]) -> None:
    """Write a co_retrieval row per edge into the workspace's
    retrieval_graph.db (creating the schema if absent)."""
    db_path = Path(workspace) / "intelligence" / "state" / "retrieval_graph.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS co_retrieval (
                mem1_id TEXT NOT NULL,
                mem2_id TEXT NOT NULL,
                weight REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (mem1_id, mem2_id)
            );
            """
        )
        for a, b, w in edges:
            conn.execute(
                "INSERT OR REPLACE INTO co_retrieval (mem1_id, mem2_id, weight) VALUES (?, ?, ?)",
                (a, b, w),
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


class TestDateKey:
    def test_extracts_yyyymmdd(self) -> None:
        assert _date_key("D-20260101-001") == 20260101

    def test_no_match_returns_zero(self) -> None:
        assert _date_key("WEIRD-NOPE") == 0
        assert _date_key("") == 0

    def test_invalid_date_returns_zero(self) -> None:
        # Regex requires exactly 8 digits — "X-9999-001" doesn't match.
        assert _date_key("X-9999-001") == 0


class TestRoleFor:
    def test_single_step(self) -> None:
        assert _role_for(0, 1) == "context"

    def test_first_third_is_foundation(self) -> None:
        assert _role_for(0, 10) == "foundation"
        assert _role_for(2, 10) == "foundation"  # idx 2 / (10-1) = 0.222

    def test_last_third_is_current(self) -> None:
        assert _role_for(9, 10) == "current"
        assert _role_for(7, 10) == "current"  # idx 7 / 9 = 0.778

    def test_middle_is_context(self) -> None:
        assert _role_for(4, 10) == "context"
        assert _role_for(5, 10) == "context"


class TestTopoSort:
    def test_empty_is_empty(self) -> None:
        assert _topo_sort([], []) == []

    def test_no_edges_keeps_lex_order(self) -> None:
        assert _topo_sort(["c", "a", "b"], []) == ["a", "b", "c"]

    def test_simple_chain(self) -> None:
        # a -> b -> c
        assert _topo_sort(["a", "b", "c"], [("a", "b"), ("b", "c")]) == ["a", "b", "c"]

    def test_chain_with_extra_node(self) -> None:
        # a -> b, c isolated
        out = _topo_sort(["a", "b", "c"], [("a", "b")])
        assert out.index("a") < out.index("b")

    def test_cycle_broken(self) -> None:
        # a -> b -> a is a cycle. _topo_sort must still emit both nodes.
        out = _topo_sort(["a", "b"], [("a", "b"), ("b", "a")])
        assert sorted(out) == ["a", "b"]

    def test_ignores_self_loop(self) -> None:
        out = _topo_sort(["a", "b"], [("a", "a"), ("a", "b")])
        assert out == ["a", "b"]

    def test_ignores_unknown_nodes(self) -> None:
        out = _topo_sort(["a"], [("a", "ghost"), ("ghost", "a")])
        assert out == ["a"]

    def test_deterministic(self) -> None:
        nodes = ["alpha", "bravo", "charlie", "delta"]
        edges = [("alpha", "charlie"), ("bravo", "delta")]
        a = _topo_sort(nodes, edges)
        b = _topo_sort(nodes, edges)
        assert a == b


# ---------------------------------------------------------------------------
# compile_walkthrough — workspace integration
# ---------------------------------------------------------------------------


class TestCompileWalkthrough:
    def test_empty_workspace_rejected(self) -> None:
        with pytest.raises(ValueError, match="workspace"):
            compile_walkthrough("", "auth")

    def test_empty_topic_rejected(self, workspace: str) -> None:
        with pytest.raises(ValueError, match="topic"):
            compile_walkthrough(workspace, "")

    def test_limit_out_of_range(self, workspace: str) -> None:
        with pytest.raises(ValueError, match="limit"):
            compile_walkthrough(workspace, "x", limit=0)
        with pytest.raises(ValueError, match="limit"):
            compile_walkthrough(workspace, "x", limit=99999)

    def test_no_results_returns_empty(self, workspace: str) -> None:
        out = compile_walkthrough(workspace, "this-topic-matches-nothing-zzzzz")
        assert out == []

    def test_results_in_chronological_order(self, workspace: str) -> None:
        # recall may not return every fixture block depending on score
        # cutoff, so the test only asserts that whichever blocks DO
        # come back are emitted in chronological order.
        out = compile_walkthrough(workspace, "authentication")
        assert len(out) >= 2
        steps = [s["block_id"] for s in out]
        # Strip the date prefix and check it's monotonically non-decreasing.
        from mind_mem.walkthrough import _date_key

        dates = [_date_key(b) for b in steps]
        assert dates == sorted(dates), f"steps out of order: {steps}"

    def test_step_numbers_are_one_based(self, workspace: str) -> None:
        out = compile_walkthrough(workspace, "authentication")
        assert out[0]["step"] == 1
        for i in range(len(out)):
            assert out[i]["step"] == i + 1

    def test_steps_have_role_score_subject(self, workspace: str) -> None:
        out = compile_walkthrough(workspace, "authentication")
        assert all({"step", "block_id", "role", "subject"} <= set(s.keys()) for s in out)
        assert all(s["role"] in {"foundation", "context", "current"} for s in out)

    def test_first_step_role_is_foundation(self, workspace: str) -> None:
        out = compile_walkthrough(workspace, "authentication")
        if len(out) >= 2:
            assert out[0]["role"] == "foundation"

    def test_last_step_role_is_current(self, workspace: str) -> None:
        out = compile_walkthrough(workspace, "authentication")
        if len(out) >= 2:
            assert out[-1]["role"] == "current"

    def test_co_retrieval_edges_are_consumed(self, workspace: str) -> None:
        # Add a co-retrieval edge — the walkthrough must still emit a
        # valid sequence (not crash) with chronological order preserved.
        _seed_co_retrieval(workspace, [("D-20260101-001", "D-20260301-001", 0.9)])
        out = compile_walkthrough(workspace, "authentication")
        assert len(out) >= 2
        from mind_mem.walkthrough import _date_key

        dates = [_date_key(s["block_id"]) for s in out]
        assert dates == sorted(dates)

    def test_no_db_no_crash(self, workspace: str) -> None:
        # Removing the directory should not cause the walkthrough to fail.
        db = Path(workspace) / "intelligence" / "state" / "retrieval_graph.db"
        if db.exists():
            db.unlink()
        out = compile_walkthrough(workspace, "authentication")
        assert isinstance(out, list)

    def test_deterministic_across_calls(self, workspace: str) -> None:
        a = compile_walkthrough(workspace, "authentication")
        b = compile_walkthrough(workspace, "authentication")
        assert [s["block_id"] for s in a] == [s["block_id"] for s in b]


# ---------------------------------------------------------------------------
# Light integration: compile_walkthrough feeds into apply_persona
# ---------------------------------------------------------------------------


class TestWalkthroughIntegration:
    def test_walkthrough_steps_can_be_persona_brief(self, workspace: str) -> None:
        from mind_mem.personas import apply_persona

        steps = compile_walkthrough(workspace, "authentication")
        # Each step is a dict with 'block_id'/'subject'/'score'; brief
        # projection still produces a known shape.
        # Adapt steps to look like blocks (id key) for apply_persona.
        adapted: list[dict[str, Any]] = [{"id": s["block_id"], "Subject": s["subject"], "score": s["score"]} for s in steps]
        out = apply_persona(adapted, "brief")
        assert all(set(b.keys()) == {"id", "score", "subject"} for b in out)
