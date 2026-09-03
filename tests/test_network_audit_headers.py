"""Audit headers propagate end-to-end — roadmap v4.0.0 Group D (RM-2291).

``X-MindMem-Request-Id`` / ``X-MindMem-Actor`` / ``X-MindMem-Purpose`` were
parsed by the REST middleware, written onto ``request.state`` and echoed on
the response — and then nothing anywhere read them back. The values died at
the middleware, so "propagated end-to-end" was untrue even inside REST, and
the gRPC leg parsed nothing at all.

What is asserted here is the *reading* half, at each place the values now
land:

1. the governed block a REST proposal writes carries ``ActorId`` and
   ``Purpose`` — and ``ActorId`` is the identity the server AUTHENTICATED,
   never the actor the caller claimed, because a provenance field filled
   from an unverified claim reads like a fact;
2. the structured-log context carries the correlation id, so every log line
   emitted while serving the request is stitchable to it;
3. the federation client stamps the same three headers on its outbound
   request, so one correlation id survives a peer hop;
4. the gRPC servicer binds the same context from invocation metadata,
   and — deliberately — never fills ``agent_authenticated`` there, because
   that transport authenticates nobody;
5. the authenticated identity is in scope for ``governance_gate``'s
   ``_current_agent()`` fallback at the moment the endpoint calls into the
   governance layer, which it was not before (it read ``"system"``).

Every negative assertion below is paired with a positive control that the
value exists and the probe can see it.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Generator, Iterator

import pytest

from mind_mem import audit_context as ac

# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


class TestSanitiser:
    def test_crlf_and_control_bytes_are_stripped(self) -> None:
        assert ac.sanitize_header_value("ok\r\nX-Injected: 1") == "okX-Injected: 1"
        assert ac.sanitize_header_value("a\x00b\x1fc\x7fd") == "abcd"

    def test_value_is_length_bounded(self) -> None:
        assert len(ac.sanitize_header_value("x" * 5000)) == ac.MAX_FIELD_LEN

    def test_a_value_that_sanitises_to_nothing_takes_the_default(self) -> None:
        """Control characters only is an absent value, not a present empty one."""
        assert ac.sanitize_header_value("\r\n\x00", default="fallback") == "fallback"
        # Positive control: a value that survives sanitising is kept.
        assert ac.sanitize_header_value("\r\nreal", default="fallback") == "real"


class TestContextSemantics:
    def test_missing_request_id_is_minted_and_bounded(self) -> None:
        ctx = ac.context_from_headers({}.get, transport="rest")
        assert ctx.request_id
        assert len(ctx.request_id) <= ac.MAX_REQUEST_ID_LEN
        # Positive control: a supplied id is used, not replaced.
        supplied = ac.context_from_headers({"x-mindmem-request-id": "abc-123"}.get, transport="rest")
        assert supplied.request_id == "abc-123"

    def test_headers_never_fill_the_authenticated_identity(self) -> None:
        ctx = ac.context_from_headers({"x-mindmem-actor": "eve"}.get, transport="rest")
        assert ctx.actor_claimed == "eve"
        assert ctx.agent_authenticated is None
        # Positive control: the field is settable — it is empty because
        # the header parser refuses to fill it, not because it is inert.
        ctx.agent_authenticated = "alice"
        assert ctx.agent_authenticated == "alice"

    def test_outbound_headers_prefer_the_authenticated_identity(self) -> None:
        ctx = ac.AuditContext(request_id="rid", actor_claimed="eve", purpose="why")
        assert ctx.outbound_headers()[ac.HEADER_ACTOR] == "eve"
        ctx.agent_authenticated = "alice"
        assert ctx.outbound_headers()[ac.HEADER_ACTOR] == "alice"

    def test_recording_an_identity_outside_a_request_reports_that_it_landed_nowhere(self) -> None:
        assert ac.current_audit_context() is None
        assert ac.record_authenticated_agent("alice") is False
        # Positive control: inside a bound context it does land.
        with ac.bind_audit_context(ac.AuditContext(request_id="rid")) as ctx:
            assert ac.record_authenticated_agent("alice") is True
            assert ctx.agent_authenticated == "alice"

    def test_binding_is_unwound_on_the_way_out(self) -> None:
        with ac.bind_audit_context(ac.AuditContext(request_id="rid")):
            assert ac.current_audit_context() is not None
        assert ac.current_audit_context() is None

    def test_no_context_means_no_outbound_headers(self) -> None:
        assert ac.outbound_audit_headers() == {}
        with ac.bind_audit_context(ac.AuditContext(request_id="rid")):
            assert ac.outbound_audit_headers() == {ac.HEADER_REQUEST_ID: "rid"}


# ---------------------------------------------------------------------------
# Leg 3: the outbound federation hop
# ---------------------------------------------------------------------------


class TestFederationClientPropagatesTheHeaders:
    def _captured_headers(self) -> dict[str, str]:
        from mind_mem.v4.federation_client import FederationClient

        seen: dict[str, str] = {}

        class _FakeResponse:
            def read(self, _n: int) -> bytes:
                return json.dumps({"version_vector": {"a": 1}}).encode("utf-8")

            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, *exc: Any) -> None:
                return None

        class _FakeOpener:
            def open(self, req: Any, timeout: float | None = None) -> _FakeResponse:
                seen.update({k.lower(): v for k, v in req.header_items()})
                return _FakeResponse()

        client = FederationClient("http://peer.local:8765", token="t")
        client._opener = _FakeOpener()  # type: ignore[assignment]
        client.get_vclock("block-1")
        return seen

    def test_headers_are_stamped_on_the_outbound_request(self) -> None:
        ctx = ac.AuditContext(request_id="rid-42", actor_claimed="eve", purpose="sync")
        ctx.agent_authenticated = "alice"
        with ac.bind_audit_context(ctx):
            seen = self._captured_headers()
        assert seen["x-mindmem-request-id"] == "rid-42"
        assert seen["x-mindmem-actor"] == "alice"
        assert seen["x-mindmem-purpose"] == "sync"

    def test_no_bound_context_leaves_the_request_exactly_as_it_was(self) -> None:
        """The off path must add nothing — this client is also used standalone."""
        seen = self._captured_headers()
        assert "x-mindmem-request-id" not in seen
        assert "x-mindmem-actor" not in seen
        assert "x-mindmem-purpose" not in seen
        # Positive control: the probe does see headers when there are any.
        assert seen["x-mindmem-token"] == "t"


# ---------------------------------------------------------------------------
# Leg 4: gRPC invocation metadata
# ---------------------------------------------------------------------------


class _FakeServicerContext:
    def __init__(self, metadata: list[tuple[str, Any]] | None) -> None:
        self._metadata = metadata

    def invocation_metadata(self) -> list[tuple[str, Any]] | None:
        return self._metadata


class TestGrpcLeg:
    def test_metadata_becomes_a_bound_context(self) -> None:
        pytest.importorskip("fastapi", reason="mind_mem.api imports the REST app at package import")
        from mind_mem.api.grpc_server import bind_call_context

        md = [("x-mindmem-request-id", "rid-9"), ("x-mindmem-actor", "peer-a"), ("x-mindmem-purpose", "replicate")]
        with bind_call_context(_FakeServicerContext(md)) as ctx:
            assert ctx is not None
            assert ac.current_audit_context() is ctx
            assert ctx.request_id == "rid-9"
            assert ctx.actor_claimed == "peer-a"
            assert ctx.purpose == "replicate"
            assert ctx.transport == "grpc"

    def test_an_unauthenticated_transport_never_claims_an_authenticated_identity(self) -> None:
        pytest.importorskip("fastapi", reason="mind_mem.api imports the REST app at package import")
        from mind_mem.api.grpc_server import audit_context_from_metadata

        ctx = audit_context_from_metadata([("x-mindmem-actor", "peer-a")])
        assert ctx.actor_claimed == "peer-a"
        assert ctx.agent_authenticated is None

    def test_binary_metadata_is_ignored_rather_than_coerced(self) -> None:
        pytest.importorskip("fastapi", reason="mind_mem.api imports the REST app at package import")
        from mind_mem.api.grpc_server import audit_context_from_metadata

        ctx = audit_context_from_metadata([("x-mindmem-actor-bin", b"\x00\x01"), ("x-mindmem-actor", "peer-a")])
        assert ctx.actor_claimed == "peer-a"

    def test_a_context_without_metadata_binds_nothing(self) -> None:
        pytest.importorskip("fastapi", reason="mind_mem.api imports the REST app at package import")
        from mind_mem.api.grpc_server import bind_call_context

        with bind_call_context(object()) as ctx:
            assert ctx is None
            assert ac.current_audit_context() is None

    def test_the_servicer_binds_the_context_around_a_real_handler(self) -> None:
        """The servicer is the mount point generated code calls, so wire it there."""
        pytest.importorskip("grpc", reason="grpcio is not a mind-mem dependency")
        from mind_mem.api.grpc_server import _build_servicer

        servicer = _build_servicer()
        seen: dict[str, Any] = {}

        import mind_mem.api.grpc_server as grpc_server

        original = grpc_server.handle_health

        def _probe() -> Any:
            seen["ctx"] = ac.current_audit_context()
            return original()

        grpc_server.handle_health = _probe  # type: ignore[assignment]
        try:
            servicer.Health({}, _FakeServicerContext([("x-mindmem-request-id", "rid-77")]))
        finally:
            grpc_server.handle_health = original  # type: ignore[assignment]
        assert seen["ctx"] is not None
        assert seen["ctx"].request_id == "rid-77"


# ---------------------------------------------------------------------------
# Legs 1, 2 and 5: the REST request
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed; skipping REST audit-header tests")

from fastapi.testclient import TestClient  # noqa: E402

from mind_mem.api.rest import create_app  # noqa: E402

_ADMIN_TOKEN = "admin-token-for-audit-header-tests"


@pytest.fixture()
def workspace(tmp_path: Path) -> str:
    from mind_mem.init_workspace import init as init_workspace

    init_workspace(str(tmp_path))
    return str(tmp_path)


@pytest.fixture()
def admin_client(workspace: str, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    monkeypatch.setenv("MIND_MEM_ADMIN_TOKEN", _ADMIN_TOKEN)
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    # The MCP tool layer has its own scope gate below the REST admin gate.
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    app = create_app(workspace)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


@pytest.fixture()
def unauthenticated_client(workspace: str, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """The loopback opt-in deployment: nothing identifies the caller."""
    monkeypatch.setenv("MIND_MEM_WORKSPACE", workspace)
    monkeypatch.delenv("MIND_MEM_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)
    monkeypatch.delenv("MIND_MEM_API_KEY_DB", raising=False)
    monkeypatch.setenv("MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST", "1")
    monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
    app = create_app(workspace)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {_ADMIN_TOKEN}"}


_PROPOSAL = {
    "block_type": "decision",
    "statement": "Adopt a TLS 1.3 floor on every federation transport hop.",
    "rationale": "network hardening rationale long enough for the decision gate",
    "confidence": "high",
}


class TestResponseEcho:
    def test_request_id_is_always_present_and_actor_only_when_sent(self, admin_client: TestClient) -> None:
        bare = admin_client.get("/v1/health")
        assert bare.headers.get("X-MindMem-Request-Id")
        assert "X-MindMem-Actor" not in bare.headers

        # Positive control: the same endpoint does echo an actor that was sent.
        with_actor = admin_client.get("/v1/health", headers={"X-MindMem-Actor": "alice"})
        assert with_actor.headers["X-MindMem-Actor"] == "alice"

    def test_a_supplied_request_id_is_echoed_back(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/v1/health", headers={"X-MindMem-Request-Id": "trace-7"})
        assert resp.headers["X-MindMem-Request-Id"] == "trace-7"

    def test_header_injection_is_stripped_before_the_echo(self, admin_client: TestClient) -> None:
        resp = admin_client.get("/v1/health", headers={"X-MindMem-Actor": "alice"})
        assert resp.headers["X-MindMem-Actor"] == "alice"
        # An actor carrying a CR/LF must not split the response headers.
        resp = admin_client.get("/v1/health", headers={"X-MindMem-Actor": "alice\rX-Evil: 1"})
        assert resp.headers["X-MindMem-Actor"] == "aliceX-Evil: 1"
        assert "X-Evil" not in resp.headers


class TestProvenanceReachesTheGovernedBlock:
    """Leg 1 — the header stops being decoration and becomes a recorded field."""

    def _signals(self, workspace: str) -> str:
        return Path(workspace, "intelligence", "SIGNALS.md").read_text(encoding="utf-8")

    def test_purpose_header_is_written_into_the_proposal(self, admin_client: TestClient, workspace: str) -> None:
        before = self._signals(workspace)
        assert "Purpose: harden-network" not in before  # negative assertion, pre-state

        resp = admin_client.post(
            "/v1/propose_update",
            json=_PROPOSAL,
            headers={**_auth(), "X-MindMem-Purpose": "harden-network"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json().get("written") == 1, resp.text  # positive control: a block was written
        assert "Purpose: harden-network" in self._signals(workspace)

    def test_actor_recorded_is_the_authenticated_one_not_the_claimed_one(self, admin_client: TestClient, workspace: str) -> None:
        resp = admin_client.post(
            "/v1/propose_update",
            json=_PROPOSAL,
            headers={**_auth(), "X-MindMem-Actor": "eve-the-impostor"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json().get("written") == 1, resp.text
        signals = self._signals(workspace)
        # Positive control first: an ActorId WAS recorded, so the absence
        # of the claim below is a rejection and not a missing field.
        assert "ActorId: admin" in signals
        assert "eve-the-impostor" not in signals

    def test_no_headers_writes_the_block_it_always_wrote(self, admin_client: TestClient, workspace: str) -> None:
        """No headers still records the identity — that part is not a header."""
        resp = admin_client.post("/v1/propose_update", json=_PROPOSAL, headers=_auth())
        assert resp.status_code == 200, resp.text
        signals = self._signals(workspace)
        assert "Purpose:" not in signals
        assert "ActorId: admin" in signals

    def test_unauthenticated_deployment_records_no_identity_at_all(self, unauthenticated_client: TestClient, workspace: str) -> None:
        """ "anonymous" is the ABSENCE of an identity and must not be stamped.

        Writing ``ActorId: anonymous`` would read like a fact about who
        called. The loopback opt-in deployment keeps writing the block it
        wrote before this change.
        """
        resp = unauthenticated_client.post("/v1/propose_update", json=_PROPOSAL)
        assert resp.status_code == 200, resp.text
        assert resp.json().get("written") == 1, resp.text
        signals = self._signals(workspace)
        assert "ActorId:" not in signals

    def test_unauthenticated_deployment_falls_back_to_the_claimed_actor(self, unauthenticated_client: TestClient, workspace: str) -> None:
        """POSITIVE CONTROL for the test above: the field is reachable here.

        With nothing to authenticate against, the caller's own claim is all
        that exists — and it is recorded as such, into a field whose
        contract has always been "caller-declared".
        """
        resp = unauthenticated_client.post(
            "/v1/propose_update",
            json=_PROPOSAL,
            headers={"X-MindMem-Actor": "local-cli"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json().get("written") == 1, resp.text
        assert "ActorId: local-cli" in self._signals(workspace)


class TestAuthenticatedIdentityIsInScopeForTheGate:
    """Leg 5 — the identity survives the threadpool boundary into the endpoint.

    ``governance_gate._current_agent()`` is the fallback an admission takes
    when its caller passes no explicit actor. Before this change it read
    ``"system"`` for every REST-driven call, because ``_require_auth`` set
    the ContextVar in a threadpool worker whose context the endpoint never
    sees. The probe below reads it at the exact boundary where the REST
    endpoint hands control to the governance layer.
    """

    def test_current_agent_is_the_authenticated_identity_at_the_governance_call(
        self, admin_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mind_mem import governance_gate
        from mind_mem.mcp.tools import governance as governance_tools

        seen: dict[str, str] = {}

        def _probe(**kwargs: Any) -> str:
            seen["agent"] = governance_gate._current_agent()
            return json.dumps({"status": "proposed", "written": 0})

        monkeypatch.setattr(governance_tools, "propose_update", _probe)
        resp = admin_client.post("/v1/propose_update", json=_PROPOSAL, headers=_auth())
        assert resp.status_code == 200, resp.text
        assert seen["agent"] == "admin"

    def test_outside_a_request_the_fallback_is_unchanged(self) -> None:
        """POSITIVE CONTROL for the assertion above: 'admin' is not the default."""
        from mind_mem import governance_gate

        assert governance_gate._current_agent() in ("system", "anonymous")


class TestLogCorrelation:
    """Leg 2 — the request id reaches every structured log line.

    Gated behind ``v4.logging_context`` like the rest of that surface, so
    the flag-OFF path is asserted too: with the filter unarmed nothing is
    pushed and log records carry no ``ctx``.
    """

    @contextlib.contextmanager
    def _armed(self, workspace: str, monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> Iterator[None]:
        """Arm ``v4.logging_context`` for the block, and DISARM on the way out.

        The filter is installed on a process-wide handler, so a test that
        arms it and only unsets ``MIND_MEM_CONFIG`` leaves it armed for
        every later test in the process: ``_config_path()`` then falls back
        to ``$MIND_MEM_WORKSPACE/mind-mem.json``, which is still the armed
        file. Teardown therefore writes the flag back OFF and asserts the
        disarm actually took, rather than trusting an env restore.
        """
        from mind_mem import observability

        cfg = Path(workspace, "mind-mem.json")
        original = cfg.read_text(encoding="utf-8") if cfg.exists() else None

        def _write(flag: bool) -> None:
            body = json.loads(original) if original else {}
            body["v4"] = {"logging_context": {"enabled": flag}}
            cfg.write_text(json.dumps(body), encoding="utf-8")

        _write(enabled)
        monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))
        assert observability.sync_log_context() is enabled
        try:
            yield
        finally:
            _write(False)
            os.environ["MIND_MEM_CONFIG"] = str(cfg)
            assert observability.sync_log_context() is False
            if original is not None:
                cfg.write_text(original, encoding="utf-8")

    def test_request_id_is_bound_for_the_duration_of_the_request(
        self, admin_client: TestClient, workspace: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mind_mem.mcp.tools import governance as governance_tools
        from mind_mem.v4.logging_context import current_context

        with self._armed(workspace, monkeypatch, enabled=True):
            seen: dict[str, Any] = {}

            def _probe(**kwargs: Any) -> str:
                seen["ctx"] = current_context()
                return json.dumps({"status": "proposed", "written": 0})

            monkeypatch.setattr(governance_tools, "propose_update", _probe)
            resp = admin_client.post(
                "/v1/propose_update",
                json=_PROPOSAL,
                headers={**_auth(), "X-MindMem-Request-Id": "trace-99", "X-MindMem-Purpose": "audit"},
            )
            assert resp.status_code == 200, resp.text
            assert seen["ctx"]["correlation_id"] == "trace-99"
            assert seen["ctx"]["request_id"] == "trace-99"
            assert seen["ctx"]["purpose"] == "audit"
            assert seen["ctx"]["agent"] == "admin"
            assert seen["ctx"]["transport"] == "rest"

    def test_flag_off_pushes_nothing(self, admin_client: TestClient, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The off path must cost nothing and change nothing."""
        from mind_mem.mcp.tools import governance as governance_tools
        from mind_mem.v4.logging_context import current_context

        with self._armed(workspace, monkeypatch, enabled=False):
            seen: dict[str, Any] = {}

            def _probe(**kwargs: Any) -> str:
                seen["ctx"] = current_context()
                return json.dumps({"status": "proposed", "written": 0})

            monkeypatch.setattr(governance_tools, "propose_update", _probe)
            resp = admin_client.post(
                "/v1/propose_update",
                json=_PROPOSAL,
                headers={**_auth(), "X-MindMem-Request-Id": "trace-99"},
            )
            assert resp.status_code == 200, resp.text
            assert seen["ctx"] == {}
