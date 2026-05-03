"""Tests for the v3.10 MCP walkthrough + persona wrapper tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)
    (ws / "intelligence" / "state").mkdir(parents=True)
    (ws / "memory").mkdir()
    (ws / "tasks").mkdir()
    (ws / "entities").mkdir()
    (ws / "mind-mem.json").write_text(
        json.dumps(
            {
                "version": "3.10.0",
                "workspace_path": str(ws),
                "block_store": {"backend": "markdown"},
            }
        )
    )

    decisions = """[D-20260101-001]
Date: 2026-01-01
Status: active
Subject: Authentication foundation chosen
Statement: Adopt OAuth2 with PKCE for the auth subsystem rewrite.

[D-20260201-001]
Date: 2026-02-01
Status: active
Subject: JWT signing key rotation policy
Statement: Authentication keys rotate every 30 days.

[D-20260301-001]
Date: 2026-03-01
Status: active
Subject: MFA rollout schedule
Statement: Authentication MFA gates rolled out per cohort.
"""
    (ws / "decisions" / "DECISIONS.md").write_text(decisions)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    return str(ws)


# ---------------------------------------------------------------------------
# compile_truth_walkthrough
# ---------------------------------------------------------------------------


class TestCompileTruthWalkthrough:
    def test_topic_required(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import compile_truth_walkthrough

        out = json.loads(compile_truth_walkthrough(""))
        assert "error" in out and "topic" in out["error"]

    def test_topic_too_long(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import compile_truth_walkthrough

        out = json.loads(compile_truth_walkthrough("x" * 5000))
        assert "error" in out and "≤4096" in out["error"]

    def test_limit_out_of_range(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import compile_truth_walkthrough

        out = json.loads(compile_truth_walkthrough("auth", limit=0))
        assert "error" in out and "limit" in out["error"]
        out = json.loads(compile_truth_walkthrough("auth", limit=999))
        assert "error" in out and "limit" in out["error"]

    def test_returns_envelope_with_steps(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import compile_truth_walkthrough

        out = json.loads(compile_truth_walkthrough("authentication"))
        assert "topic" in out and out["topic"] == "authentication"
        assert "count" in out
        assert "steps" in out
        assert isinstance(out["steps"], list)

    def test_steps_have_chronological_order(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import compile_truth_walkthrough
        from mind_mem.walkthrough import _date_key

        out = json.loads(compile_truth_walkthrough("authentication"))
        if len(out["steps"]) >= 2:
            dates = [_date_key(s["block_id"]) for s in out["steps"]]
            assert dates == sorted(dates)

    def test_no_match_empty_steps(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import compile_truth_walkthrough

        out = json.loads(compile_truth_walkthrough("nope-not-real-zzz-9999"))
        assert out["count"] == 0
        assert out["steps"] == []


# ---------------------------------------------------------------------------
# recall_with_persona
# ---------------------------------------------------------------------------


class TestRecallWithPersona:
    def test_query_required(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import recall_with_persona

        out = json.loads(recall_with_persona(""))
        assert "error" in out and "query" in out["error"]

    def test_unknown_persona_rejected(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import recall_with_persona

        out = json.loads(recall_with_persona("auth", persona="bogus"))
        assert "error" in out and "persona" in out["error"]

    def test_limit_out_of_range(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import recall_with_persona

        out = json.loads(recall_with_persona("auth", limit=0))
        assert "error" in out and "limit" in out["error"]
        out = json.loads(recall_with_persona("auth", limit=99999))
        assert "error" in out and "limit" in out["error"]

    def test_brief_persona_returns_compact_shape(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import recall_with_persona

        out = json.loads(recall_with_persona("authentication", persona="brief"))
        assert out["persona"] == "brief"
        assert "results" in out
        for r in out["results"]:
            # brief = {id, score, subject} only
            assert set(r.keys()) == {"id", "score", "subject"}

    def test_detailed_persona_default(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import recall_with_persona

        out = json.loads(recall_with_persona("authentication"))
        assert out["persona"] == "detailed"

    def test_technical_persona_promotes_governance(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import recall_with_persona

        out = json.loads(recall_with_persona("authentication", persona="technical"))
        assert out["persona"] == "technical"
        # technical is identity + governance promotion; results may be empty
        # for synthetic data, but the persona key still reports correctly.
        assert "results" in out

    def test_envelope_keys(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import recall_with_persona

        out = json.loads(recall_with_persona("authentication", persona="brief"))
        assert {"persona", "query", "count", "results"} <= set(out.keys())
        assert out["query"] == "authentication"

    def test_count_matches_results_length(self, workspace: str) -> None:
        from mind_mem.mcp.tools.walkthrough_persona import recall_with_persona

        out = json.loads(recall_with_persona("authentication", persona="brief"))
        assert out["count"] == len(out["results"])


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_calls_mcp_tool_for_each(self) -> None:
        from mind_mem.mcp.tools import walkthrough_persona

        registered: list[object] = []

        class FakeMCP:
            def tool(self, fn: object) -> object:
                registered.append(fn)
                return fn

        walkthrough_persona.register(FakeMCP())
        assert walkthrough_persona.compile_truth_walkthrough in registered
        assert walkthrough_persona.recall_with_persona in registered
