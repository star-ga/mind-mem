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
            # The gRPC surface forwards the recall scoring instant. None means
            # "resolve at the boundary", which is the documented default; the
            # point of asserting it here is that the parameter is threaded at
            # all rather than dropped on this transport.
            scoring_instant=None,
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


class TestTenantIsNotSilentlyDropped:
    """``tenant_id`` is on the wire shape but this transport cannot route it.

    The module never enters ``use_workspace``, so a scoped request would be
    answered from the process-wide workspace — and answered with ``ok``.
    Both handlers must refuse instead of dropping the field.
    """

    def test_recall_with_tenant_raises_before_touching_the_corpus(self) -> None:
        import pytest

        with patch("mind_mem.mcp.tools.recall._recall_impl", return_value='{"results": []}') as spy:
            with pytest.raises(ValueError, match="tenant_id is not supported"):
                handle_recall(RecallRequest(query="hello", tenant_id="acme"))
        spy.assert_not_called()

    def test_recall_without_tenant_still_works(self) -> None:
        with patch("mind_mem.mcp.tools.recall._recall_impl", return_value='{"results": []}'):
            resp = handle_recall(RecallRequest(query="hello"))
        assert resp.payload == '{"results": []}'

    def test_governance_with_tenant_is_refused_without_dispatching(self) -> None:
        with patch("mind_mem.mcp.tools.governance.approve_apply", return_value='{"applied": true}') as spy:
            resp = handle_governance(GovernanceRequest(operation="approve", args={"proposal_id": "P-1"}, tenant_id="acme"))
        spy.assert_not_called()
        assert resp.ok is False
        assert "tenant_id is not supported" in (resp.error or "")


class TestGeneratedServicerRegistration:
    def test_missing_generated_package_is_logged_not_swallowed(self) -> None:
        from mind_mem.api import grpc_server

        with patch.object(grpc_server, "_log") as log:
            registered = grpc_server._register_generated_services(object(), object())

        assert registered is False
        log.warning.assert_called_once()
        assert log.warning.call_args.args[0] == "grpc_generated_missing"

    def test_present_generated_package_is_registered(self) -> None:
        import sys
        import types

        from mind_mem.api import grpc_server

        calls: list[tuple[object, object]] = []
        stub = types.ModuleType("mind_mem.api.grpc_generated")
        stub.register = lambda server, servicer: calls.append((server, servicer))  # type: ignore[attr-defined]
        server, servicer = object(), object()
        with patch.dict(sys.modules, {"mind_mem.api.grpc_generated": stub}):
            registered = grpc_server._register_generated_services(server, servicer)

        assert registered is True
        assert calls == [(server, servicer)]


class TestInsecureBindIsRefused:
    """The gRPC surface fails closed on a routable bind, like REST does.

    REST refuses to bind a network port without authentication
    (``_enforce_fail_closed``). This transport has *no* authentication to
    configure at all, no TLS, and drives governance mutations
    (approve/rollback) — so the same mistake was strictly worse here, and
    the only thing standing against it was a comment. An operator who
    sets ``MIND_MEM_GRPC_HOST=0.0.0.0`` must now say plainly that they
    mean it.
    """

    def test_loopback_binds_are_permitted(self, monkeypatch) -> None:
        from mind_mem.api.grpc_server import _enforce_grpc_bind

        monkeypatch.delenv("MIND_MEM_GRPC_ALLOW_INSECURE_BIND", raising=False)
        for host in ("127.0.0.1", "localhost", "::1", "[::1]"):
            _enforce_grpc_bind(host)  # must not raise

    def test_routable_bind_is_refused(self, monkeypatch) -> None:
        import pytest as _pytest

        from mind_mem.api.grpc_server import _enforce_grpc_bind

        monkeypatch.delenv("MIND_MEM_GRPC_ALLOW_INSECURE_BIND", raising=False)
        for host in ("0.0.0.0", "::", "10.0.0.5", "203.0.113.9"):
            with _pytest.raises(RuntimeError, match="MIND_MEM_GRPC_ALLOW_INSECURE_BIND"):
                _enforce_grpc_bind(host)

    def test_explicit_acknowledgement_permits_it(self, monkeypatch) -> None:
        from mind_mem.api.grpc_server import _enforce_grpc_bind

        monkeypatch.setenv("MIND_MEM_GRPC_ALLOW_INSECURE_BIND", "true")
        _enforce_grpc_bind("0.0.0.0")  # must not raise

    def test_serve_enforces_before_binding(self, monkeypatch) -> None:
        """The gate must run before the port is opened, not after.

        A check that fires after ``add_insecure_port`` would leave the
        socket listening for the lifetime of the traceback.

        ``grpc`` is stubbed for the duration: with the real package this
        test would otherwise reach ``server.wait_for_termination()`` and
        hang forever if the gate ever regressed — the failure mode a
        regression test must not have.
        """
        import sys
        import types

        import pytest as _pytest

        from mind_mem.api import grpc_server

        events: list[str] = []

        class _Server:
            def add_insecure_port(self, target):
                events.append(f"bind:{target}")

            def start(self):
                events.append("start")

            def wait_for_termination(self):  # pragma: no cover - must never run
                raise AssertionError("serve() bound a routable port despite the gate")

        stub = types.ModuleType("grpc")
        stub.server = lambda *a, **k: _Server()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "grpc", stub)
        monkeypatch.delenv("MIND_MEM_GRPC_ALLOW_INSECURE_BIND", raising=False)
        monkeypatch.setenv("MIND_MEM_GRPC_HOST", "0.0.0.0")

        with _pytest.raises(RuntimeError, match="MIND_MEM_GRPC_ALLOW_INSECURE_BIND"):
            grpc_server.serve(50099)
        assert not events, f"serve() reached {events} before refusing the bind"
