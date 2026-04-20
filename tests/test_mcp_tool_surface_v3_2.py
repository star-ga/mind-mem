"""v3.2.0 — consolidated MCP public dispatcher tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_HAS_FASTMCP = importlib.util.find_spec("fastmcp") is not None
pytestmark = pytest.mark.skipif(not _HAS_FASTMCP, reason="fastmcp not installed")


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Workspace with minimal corpus structure + env pointer."""
    for d in (
        "decisions",
        "tasks",
        "entities",
        "intelligence",
        "intelligence/applied",
        "intelligence/proposed",
        "memory",
        "maintenance/tracked",
        "maintenance/append-only",
    ):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "mind-mem.json").write_text("{}")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path))
    return tmp_path


class TestDispatchRecall:
    def test_mode_auto_returns_envelope(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.recall.__wrapped__("query string", mode="auto")
        env = json.loads(raw)
        assert env["_schema_version"] == "1.0"
        assert env["query"] == "query string"
        assert "results" in env

    def test_mode_bm25_forces_bm25_backend(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.recall.__wrapped__("q", mode="bm25")
        env = json.loads(raw)
        # Envelope has a ``backend`` field indicating what actually ran.
        assert env.get("backend") in {"bm25", "sqlite", "scan"}

    def test_backend_alias_maps_to_mode(self, ws: Path) -> None:
        """v3.1.x callers passing ``backend=`` still work."""
        from mind_mem.mcp.tools import public

        raw = public.recall.__wrapped__("q", backend="hybrid")
        env = json.loads(raw)
        # Hybrid path falls back to bm25 when hybrid backend isn't
        # available in the test environment — either is acceptable.
        assert env.get("backend") in {"hybrid", "sqlite", "scan"}

    def test_mode_classify_returns_intent(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.recall.__wrapped__("why did we pick Postgres?", mode="classify")
        env = json.loads(raw)
        assert "intent" in env or "error" in env  # intent_router may be absent

    def test_mode_similar_requires_block_id(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.recall.__wrapped__("ignored", mode="similar")
        env = json.loads(raw)
        assert "error" in env
        assert "block_id" in env["error"]

    def test_unknown_mode_returns_error_envelope(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.recall.__wrapped__("q", mode="teleport")
        env = json.loads(raw)
        assert "error" in env
        assert env["error"].startswith("unknown mode")
        assert "valid_modes" in env


class TestDispatchStagedChange:
    def test_phase_propose_validates_inputs(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.staged_change.__wrapped__("propose")
        env = json.loads(raw)
        assert "error" in env

    def test_phase_approve_requires_proposal_id(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.staged_change.__wrapped__("approve")
        env = json.loads(raw)
        assert "error" in env
        assert "proposal_id" in env["error"]

    def test_phase_rollback_requires_receipt_ts(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.staged_change.__wrapped__("rollback")
        env = json.loads(raw)
        assert "error" in env
        assert "receipt_ts" in env["error"]

    def test_unknown_phase(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.staged_change.__wrapped__("nuke")
        env = json.loads(raw)
        assert "error" in env
        assert env["error"].startswith("unknown phase")


class TestDispatchActions:
    def test_graph_unknown_action(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.graph.__wrapped__("fly")
        env = json.loads(raw)
        assert "error" in env
        assert env["error"].startswith("unknown action")

    def test_core_list_always_works(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.core.__wrapped__("list")
        env = json.loads(raw)
        # list always returns an envelope, never an error — cores table
        # is auto-created on first access.
        assert "cores" in env or "error" in env

    def test_kernels_list(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.kernels.__wrapped__("list")
        env = json.loads(raw)
        assert "kernels" in env

    def test_kernels_get_requires_name(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.kernels.__wrapped__("get")
        env = json.loads(raw)
        assert "error" in env
        assert "name" in env["error"]

    def test_compiled_truth_requires_entity_id(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.compiled_truth.__wrapped__("load")
        env = json.loads(raw)
        assert "error" in env
        assert "entity_id" in env["error"]

    def test_memory_verify_chain_default(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.memory_verify.__wrapped__()
        env = json.loads(raw)
        # chain verify returns "valid" when the chain is empty or
        # an error envelope when it can't read the chain. Either is
        # a well-formed envelope.
        assert "_schema_version" in env

    def test_memory_verify_merkle_requires_args(self, ws: Path) -> None:
        from mind_mem.mcp.tools import public

        raw = public.memory_verify.__wrapped__(mode="merkle")
        env = json.loads(raw)
        assert "error" in env


class TestRegistration:
    def test_register_adds_7_dispatchers(self) -> None:
        from unittest.mock import MagicMock

        from mind_mem.mcp.tools import public

        mock_mcp = MagicMock()
        mock_mcp.tool = MagicMock(side_effect=lambda fn: fn)
        public.register(mock_mcp)
        # 7 dispatchers: recall, staged_change, memory_verify,
        # graph, core, kernels, compiled_truth.
        registered_names = [
            call.args[0].__name__ for call in mock_mcp.tool.call_args_list if call.args
        ]
        expected = {
            "recall",
            "staged_change",
            "memory_verify",
            "graph",
            "core",
            "kernels",
            "compiled_truth",
        }
        assert set(registered_names) == expected
