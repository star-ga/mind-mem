"""v4.0 prep — gRPC wire protocol (tests for the grpcio-free handlers)."""

from __future__ import annotations

import json
from unittest.mock import patch

from mind_mem.api.grpc_server import (
    GovernanceRequest,
    RecallRequest,
    handle_governance,
    handle_health,
    handle_recall,
)


class TestRecallHandler:
    def test_dispatches_to_recall_impl(self) -> None:
        req = RecallRequest(query="hello", limit=5)
        with patch("mind_mem.mcp.tools.recall._recall_impl", return_value='{"results": []}') as spy:
            resp = handle_recall(req)
        spy.assert_called_once_with(
            query="hello",
            limit=5,
            active_only=False,
            backend="auto",
            format="blocks",
        )
        assert resp.payload == '{"results": []}'
        assert resp.took_ms >= 0

    def test_bundle_format_passes_through(self) -> None:
        req = RecallRequest(query="q", format="bundle")
        with patch("mind_mem.mcp.tools.recall._recall_impl", return_value='{"query":"q","facts":[]}') as spy:
            handle_recall(req)
        kwargs = spy.call_args.kwargs
        assert kwargs["format"] == "bundle"


class TestGovernanceHandler:
    def test_unknown_op_returns_error(self) -> None:
        resp = handle_governance(GovernanceRequest(operation="nope"))
        assert resp.ok is False
        assert "unknown operation" in (resp.error or "")

    def test_propose_dispatches_to_tool(self) -> None:
        with patch(
            "mind_mem.mcp.tools.governance.propose_update",
            return_value='{"proposal_id": "P-1"}',
        ):
            resp = handle_governance(
                GovernanceRequest(
                    operation="propose",
                    args={"block_type": "decision", "statement": "x"},
                )
            )
        assert resp.ok is True
        assert json.loads(resp.payload)["proposal_id"] == "P-1"

    def test_tool_exception_returns_error(self) -> None:
        with patch(
            "mind_mem.mcp.tools.governance.approve_apply",
            side_effect=RuntimeError("boom"),
        ):
            resp = handle_governance(GovernanceRequest(operation="approve", args={"proposal_id": "P-1"}))
        assert resp.ok is False
        assert "boom" in (resp.error or "")

    def test_error_payload_marks_ok_false(self) -> None:
        with patch(
            "mind_mem.mcp.tools.governance.scan",
            return_value='{"error": "workspace missing"}',
        ):
            resp = handle_governance(GovernanceRequest(operation="scan"))
        assert resp.ok is False


class TestHealthHandler:
    def test_health_returns_schema(self) -> None:
        with patch("mind_mem.mcp.infra.workspace._workspace", return_value="/tmp/ws"):
            resp = handle_health()
        assert resp.status == "ok"
        assert resp.workspace == "/tmp/ws"
        assert resp.schema_version
