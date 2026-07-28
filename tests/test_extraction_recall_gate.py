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
