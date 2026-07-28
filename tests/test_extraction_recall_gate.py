"""Read-path extraction gate + feedback anchoring + graph-edge ACL.

``extraction.enabled: true`` used to trigger per-recall enrichment
calls whose output was attached to results and then discarded by the
caller — a pure latency tax on every query. Enrichment on the read
path is now additionally gated behind ``extraction.enrich_on_recall``
(default false) so the extraction budget funds the write-path
backfill instead.
"""

from __future__ import annotations

import json
import os

import mind_mem.llm_extractor as llm
from mind_mem.extraction_feedback import ExtractionFeedback, default_feedback_path


def _write_config(tmpdir: str, extraction: dict) -> None:
    with open(os.path.join(tmpdir, "mind-mem.json"), "w") as f:
        json.dump({"extraction": extraction}, f)


class TestEnrichOnRecallGate:
    def test_enabled_alone_does_not_enrich(self, tmp_path, monkeypatch) -> None:
        ws = str(tmp_path)
        _write_config(ws, {"enabled": True, "model": "m", "backend": "ollama"})
        calls: list[dict] = []
        monkeypatch.setattr(llm, "enrich_block", lambda block, **kw: calls.append(block) or block)
        results = [{"_id": "A", "excerpt": "text", "score": 1.0}]
        out = llm.enrich_results(results, workspace=ws)
        assert out is results
        assert calls == []

    def test_enrich_on_recall_true_enriches(self, tmp_path, monkeypatch) -> None:
        ws = str(tmp_path)
        _write_config(ws, {"enabled": True, "enrich_on_recall": True, "model": "m", "backend": "ollama"})
        calls: list[dict] = []
        monkeypatch.setattr(llm, "enrich_block", lambda block, **kw: calls.append(block) or block)
        results = [{"_id": "A", "excerpt": "text", "score": 1.0}]
        llm.enrich_results(results, workspace=ws)
        assert len(calls) == 1

    def test_enrich_on_recall_without_enabled_stays_off(self, tmp_path, monkeypatch) -> None:
        ws = str(tmp_path)
        _write_config(ws, {"enabled": False, "enrich_on_recall": True})
        calls: list[dict] = []
        monkeypatch.setattr(llm, "enrich_block", lambda block, **kw: calls.append(block) or block)
        results = [{"_id": "A", "excerpt": "text", "score": 1.0}]
        out = llm.enrich_results(results, workspace=ws)
        assert out is results
        assert calls == []

    def test_default_config_has_gate_off(self, tmp_path) -> None:
        config = llm.load_config(str(tmp_path))
        assert config["enrich_on_recall"] is False


class TestFeedbackPathAnchoring:
    def test_default_path_anchors_to_workspace(self, tmp_path) -> None:
        ws = str(tmp_path)
        fb = ExtractionFeedback(workspace=ws)
        assert fb.path == os.path.join(ws, ".mind-mem", "extraction-feedback.json")
        fb.record(model="m", operation="entities", input_length=10, output_count=1, latency_ms=5.0)
        fb.flush()
        assert os.path.isfile(fb.path)

    def test_env_workspace_anchors_default(self, tmp_path, monkeypatch) -> None:
        ws = str(tmp_path)
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        fb = ExtractionFeedback()
        assert fb.path == default_feedback_path(ws)

    def test_explicit_path_wins(self, tmp_path) -> None:
        p = str(tmp_path / "custom.json")
        fb = ExtractionFeedback(path=p, workspace=str(tmp_path))
        assert fb.path == p


class TestGraphEdgeAcl:
    """Every graph mutation routes through HITL — direct edge writes
    now require the admin scope."""

    def test_graph_add_edge_is_admin(self) -> None:
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "graph_add_edge" in ADMIN_TOOLS
        assert "graph_add_edge" not in USER_TOOLS

    def test_user_scope_denied(self) -> None:
        from mind_mem.mcp.infra.acl import check_tool_acl

        denied = check_tool_acl("graph_add_edge", "user")
        assert denied is not None
        assert "admin scope" in json.loads(denied)["error"]

    def test_admin_scope_allowed(self) -> None:
        from mind_mem.mcp.infra.acl import check_tool_acl

        assert check_tool_acl("graph_add_edge", "admin") is None

    def test_read_only_graph_tools_stay_user(self) -> None:
        from mind_mem.mcp.infra.acl import USER_TOOLS

        assert "graph_query" in USER_TOOLS
        assert "graph_stats" in USER_TOOLS


class TestConsolidatedGraphDispatcherAcl:
    """Confused-deputy guard: the consolidated ``graph()`` dispatcher
    calls the edge writer via ``__wrapped__``, which strips the
    decorator that enforces ACL — so the ``add_edge`` branch must
    enforce the ``graph_add_edge`` capability itself. Otherwise any
    future reclassification of the "graph" dispatcher name would
    silently re-open an unreviewed user-scope graph write."""

    @staticmethod
    def _dispatch(workspace: str, monkeypatch, scope_env: str | None = None) -> str:
        from mind_mem.mcp.tools.public import graph

        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
        if scope_env is None:
            monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
        else:
            monkeypatch.setenv("MIND_MEM_SCOPE", scope_env)
        return graph.__wrapped__(  # type: ignore[attr-defined]
            action="add_edge",
            subject="starga",
            predicate="depends_on",
            object="mindc",
            source_block_id="D-20260101-001",
        )

    def test_user_scope_denied_before_writer(self, workspace, monkeypatch) -> None:
        import os

        from mind_mem.knowledge_graph import default_db_path

        out = json.loads(self._dispatch(workspace, monkeypatch))
        assert "admin scope" in out["error"]
        # Denied before the writer ran — no graph database created.
        assert not os.path.isfile(default_db_path(workspace))

    def test_admin_scope_reaches_writer(self, workspace, monkeypatch) -> None:
        from mind_mem.knowledge_graph import KnowledgeGraph, default_db_path

        out = json.loads(self._dispatch(workspace, monkeypatch, scope_env="admin"))
        assert out.get("subject") == "starga"
        with KnowledgeGraph(default_db_path(workspace)) as kg:
            assert kg.stats().edges == 1

    def test_deny_sentinel_denied(self, workspace, monkeypatch) -> None:
        from mind_mem.mcp.infra import acl as _acl

        monkeypatch.setattr(_acl, "_get_request_scope", lambda: "deny")
        out = json.loads(self._dispatch(workspace, monkeypatch, scope_env="admin"))
        assert "Permission denied" in out["error"]

    def test_read_actions_stay_open_at_user_scope(self, workspace, monkeypatch) -> None:
        from mind_mem.mcp.tools.public import graph

        monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
        monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
        out = json.loads(graph.__wrapped__(action="stats"))  # type: ignore[attr-defined]
        assert "error" not in out
