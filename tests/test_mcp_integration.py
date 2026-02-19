"""Integration tests for MCP server tools."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def _make_workspace(tmp_path):
    """Create a minimal workspace for testing."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    for d in ["decisions", "tasks", "entities", "intelligence", "intelligence/proposed",
              "intelligence/applied", "intelligence/state/snapshots", "memory", "summaries/weekly",
              "summaries/daily", "maintenance/weeklog", "categories"]:
        (ws / d).mkdir(parents=True, exist_ok=True)

    cfg = {
        "version": "1.0.5",
        "workspace_path": str(ws),
        "auto_capture": False,
        "auto_recall": False,
        "governance_mode": "detect_only",
        "recall": {"backend": "scan"},
        "proposal_budget": {"per_run": 3, "per_day": 6, "backlog_limit": 30},
    }
    (ws / "mind-mem.json").write_text(json.dumps(cfg))

    # Add a decision for recall to find
    dec = ws / "decisions" / "DECISIONS.md"
    dec.write_text("""[D-20260218-001]
Date: 2026-02-18
Status: active
Statement: Use PostgreSQL for the primary database
Tags: database, infrastructure
Rationale: Better JSON support than MySQL
""")

    # Add signals file
    (ws / "intelligence" / "SIGNALS.md").write_text("")
    (ws / "intelligence" / "CONTRADICTIONS.md").write_text("")
    (ws / "intelligence" / "DRIFT.md").write_text("")
    (ws / "intelligence" / "IMPACT.md").write_text("")
    (ws / "intelligence" / "BRIEFINGS.md").write_text("")
    (ws / "intelligence" / "AUDIT.md").write_text("")
    (ws / "intelligence" / "SCAN_LOG.md").write_text("")

    # Add empty corpus files expected by recall engine
    for fname in ["entities/projects.md", "entities/people.md",
                  "entities/tools.md", "entities/incidents.md"]:
        path = ws / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {os.path.basename(fname)}\n")

    # Add tasks file
    (ws / "tasks" / "TASKS.md").write_text("# Tasks\n")

    return ws


class TestRecallTool:
    """Tests for the recall MCP tool logic."""

    def test_recall_finds_decision(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from recall import recall
        results = recall(str(ws), "PostgreSQL database", limit=5)
        assert len(results) > 0
        assert any("PostgreSQL" in str(r) for r in results)

    def test_recall_empty_query(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from recall import recall
        results = recall(str(ws), "", limit=5)
        # Empty query returns empty (no tokens after tokenization)
        assert isinstance(results, list)

    def test_recall_no_match(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from recall import recall
        results = recall(str(ws), "quantum computing spaceship", limit=5)
        assert isinstance(results, list)


class TestIntentClassify:
    """Tests for intent classification."""

    def test_temporal_intent(self):
        from intent_router import IntentRouter
        router = IntentRouter()
        result = router.classify("When did we decide on PostgreSQL?")
        assert result.intent == "WHEN"

    def test_entity_intent(self):
        from intent_router import IntentRouter
        router = IntentRouter()
        result = router.classify("What is PostgreSQL used for?")
        assert result.intent is not None

    def test_verify_intent(self):
        from intent_router import IntentRouter
        router = IntentRouter()
        result = router.classify("Did we ever use MySQL?")
        assert result.intent is not None


class TestIndexStats:
    """Tests for index stats functionality."""

    def test_status_on_empty_workspace(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from sqlite_index import index_status
        stats = index_status(str(ws))
        # Should not crash on fresh workspace (no index built yet)
        assert stats is not None
        assert isinstance(stats, dict)
        assert stats["exists"] is False

    def test_build_and_query_index(self, tmp_path):
        ws = _make_workspace(tmp_path)
        from sqlite_index import build_index, query_index
        build_index(str(ws), incremental=False)
        results = query_index(str(ws), "PostgreSQL", limit=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert any("PostgreSQL" in str(r) for r in results)
