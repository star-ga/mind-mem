"""Tests for the v3.9.0 pipeline-hash MCP tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.mcp.infra.rate_limit import _rate_limiters, _rate_limiters_lock


@pytest.fixture(autouse=True)
def _reset_mcp_rate_limiters():
    """Clear the process-global MCP rate-limiter cache between tests."""
    with _rate_limiters_lock:
        _rate_limiters.clear()
    yield


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    ws = tmp_path / "ws"
    (ws / "decisions").mkdir(parents=True)
    (ws / "memory").mkdir()
    (ws / "tasks").mkdir()
    (ws / "intelligence" / "state").mkdir(parents=True)
    (ws / "mind-mem.json").write_text(
        json.dumps(
            {
                "version": "3.10.0",
                "workspace_path": str(ws),
                "block_store": {"backend": "markdown"},
            }
        )
    )
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(ws))
    return str(ws)


def _seed_dirty_block(workspace: str) -> str:
    block_id = "D-20260503-500"
    (Path(workspace) / "decisions" / "DECISIONS.md").write_text(f"[{block_id}]\nStatus: active\nStatement: stale block, no TransformHash\n")
    return block_id


# ---------------------------------------------------------------------------
# pipeline_status
# ---------------------------------------------------------------------------


class TestPipelineStatus:
    def test_envelope_keys(self, workspace: str) -> None:
        from mind_mem.mcp.tools.pipeline import pipeline_status

        out = json.loads(pipeline_status())
        assert {"current_hash", "inputs", "dirty_count", "dirty_ids", "truncated"} <= set(out.keys())
        assert isinstance(out["current_hash"], str)
        assert len(out["current_hash"]) == 64

    def test_inputs_carry_pipeline_metadata(self, workspace: str) -> None:
        from mind_mem.mcp.tools.pipeline import pipeline_status

        out = json.loads(pipeline_status())
        inputs = out["inputs"]
        assert {
            "package_version",
            "backend",
            "model",
            "extractor_source_sha256",
            "prompt_template_sha256",
        } <= set(inputs.keys())

    def test_dirty_count_zero_on_empty_workspace(self, workspace: str) -> None:
        from mind_mem.mcp.tools.pipeline import pipeline_status

        out = json.loads(pipeline_status())
        assert out["dirty_count"] == 0
        assert out["dirty_ids"] == []
        assert out["truncated"] is False

    def test_dirty_count_after_seeding(self, workspace: str) -> None:
        from mind_mem.mcp.tools.pipeline import pipeline_status

        block_id = _seed_dirty_block(workspace)
        out = json.loads(pipeline_status())
        assert out["dirty_count"] >= 1
        assert block_id in out["dirty_ids"]


# ---------------------------------------------------------------------------
# reindex_dirty
# ---------------------------------------------------------------------------


class TestReindexDirty:
    def test_dry_run_lists_without_writing(self, workspace: str) -> None:
        from mind_mem.mcp.tools.pipeline import reindex_dirty

        block_id = _seed_dirty_block(workspace)
        out = json.loads(reindex_dirty(dry_run=True))
        assert out["dry_run"] is True
        assert out["processed"] == 0
        assert block_id in out["ids"]
        # File should still lack TransformHash.
        text = (Path(workspace) / "decisions" / "DECISIONS.md").read_text()
        assert "TransformHash:" not in text

    def test_processes_and_stamps(self, workspace: str) -> None:
        from mind_mem.mcp.tools.pipeline import reindex_dirty

        _seed_dirty_block(workspace)
        out = json.loads(reindex_dirty())
        assert out["dry_run"] is False
        assert out["processed"] == 1
        text = (Path(workspace) / "decisions" / "DECISIONS.md").read_text()
        assert "TransformHash:" in text

    def test_after_processing_status_clean(self, workspace: str) -> None:
        from mind_mem.mcp.tools.pipeline import pipeline_status, reindex_dirty

        _seed_dirty_block(workspace)
        json.loads(reindex_dirty())
        status = json.loads(pipeline_status())
        # The seeded block should no longer appear.
        assert "D-20260503-500" not in status["dirty_ids"]

    def test_negative_limit_rejected(self, workspace: str) -> None:
        from mind_mem.mcp.tools.pipeline import reindex_dirty

        out = json.loads(reindex_dirty(limit=-1))
        assert "error" in out and "limit" in out["error"]

    def test_zero_limit_means_all(self, workspace: str) -> None:
        from mind_mem.mcp.tools.pipeline import reindex_dirty

        # Seed multiple dirty blocks.
        (Path(workspace) / "decisions" / "DECISIONS.md").write_text(
            "[D-20260503-600]\nStatus: active\nStatement: a\n\n"
            "[D-20260503-601]\nStatus: active\nStatement: b\n\n"
            "[D-20260503-602]\nStatus: active\nStatement: c\n"
        )
        out = json.loads(reindex_dirty(limit=0, dry_run=True))
        assert len(out["ids"]) >= 3


class TestRegister:
    def test_register_wires_both(self) -> None:
        from mind_mem.mcp.tools import pipeline as pipeline_module

        registered: list[object] = []

        class FakeMCP:
            def tool(self, fn: object) -> object:
                registered.append(fn)
                return fn

        pipeline_module.register(FakeMCP())
        assert pipeline_module.pipeline_status in registered
        assert pipeline_module.reindex_dirty in registered
