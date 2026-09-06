# Copyright 2026 STARGA, Inc.
"""Audit headers on the stdlib HTTP transport — roadmap v4.0.0 Group D.

``X-MindMem-Request-Id`` / ``X-MindMem-Actor`` / ``X-MindMem-Purpose``
already travelled the REST and gRPC legs
(``tests/test_network_audit_headers.py``). The stdlib transport in
:mod:`mind_mem.http_transport` — the one ``mm serve`` actually starts —
parsed none of them: ``grep -c audit_context src/mind_mem/http_transport.py``
answered ``0`` before this slice. A request served there was correlatable
with nothing, and the evidence-chain row a ``DELETE`` left named the door
credential and no request.

Four legs are asserted here, each in both directions, because the whole
point of the item is that the headers are OPTIONAL:

1. **inbound + echo** — a caller's request id comes back on the response;
   a caller who sends none still gets a server-minted one to correlate on,
   and gets *no* actor/purpose echo invented for them;
2. **sanitisation** — the values are echoed into response headers and into
   log lines, so a raw CR/LF in one must not forge either. Driven over a
   raw socket, because ``http.client`` refuses to *send* a split header
   and a test that cannot send the attack proves nothing about the defence;
3. **structured-log context** — the correlation id and the claims are
   bound for the duration of the request, gated behind
   ``v4.logging_context`` exactly like the REST leg, with the flag-OFF
   path asserted to push nothing;
4. **audit trail** — a governed ``DELETE`` records the request that asked
   for it in its chain-entry metadata, and a request that sent no audit
   headers records byte-for-byte what it recorded before (``metadata=None``
   → the gate's ``metadata or {}``).

The claimed actor is checked never to become the recorded one: it is an
unauthenticated string from the wire, and a provenance field filled from a
claim reads like a fact.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any, Iterator

import pytest

from mind_mem import audit_context as ac
from mind_mem import http_transport
from mind_mem.governance_gate import evict_gate
from mind_mem.http_transport import (
    _MEMORY_ID_PREFIX,
    HTTP_TOKEN_ACTOR_PREFIX,
    PATH_MEMORIES,
    PATH_STATUS,
    _token_actor,
    serve_http,
)
from mind_mem.protection import AUTH_HEADER

TOKEN = "an-operator-token-for-the-audit-header-slice"
SEED_ID = "D-20260904-001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(autouse=True)
def _no_ambient_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server authenticates the token the test hands it, not the shell's."""
    monkeypatch.delenv("MIND_MEM_TOKENS", raising=False)
    monkeypatch.delenv("MIND_MEM_TOKEN", raising=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[str]:
    ws = tmp_path / "ws"
    for sub in ("memory", "decisions", "tasks", "entities", "intelligence"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    (ws / "mind-mem.json").write_text(
        json.dumps({"workspace_path": str(ws), "block_store": {"backend": "markdown"}}),
        encoding="utf-8",
    )
    with open(ws / "decisions" / "DECISIONS.md", "a", encoding="utf-8") as handle:
        handle.write(f"[{SEED_ID}]\nStatement: seed for the audit-header slice\nDate: 2026-09-04\nStatus: active\n\n---\n\n")
    try:
        yield str(ws)
    finally:
        evict_gate(str(ws))


@contextlib.contextmanager
def _serve(workspace: str, *, token: str | None) -> Iterator[int]:
    port = _free_port()
    _thread, stop = serve_http(
        workspace=workspace,
        port=port,
        host="127.0.0.1",
        token=token,
        allow_unauthenticated_localhost=token is None,
    )
    try:
        yield port
    finally:
        stop()


def _raw(port: int, request: bytes) -> bytes:
    """Send *request* verbatim and return the whole response, headers included.

    Deliberately not ``http.client``: it raises ``ValueError`` on a header
    value carrying CR/LF, so it cannot transmit the injection this suite
    exists to refuse.
    """
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(request)
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def _get(
    port: int,
    path: str = PATH_STATUS,
    *,
    token: str | None = None,
    audit: dict[str, str] | None = None,
    method: str = "GET",
) -> tuple[int, dict[str, str], bytes]:
    """One request; returns (status, response headers lowercased, body)."""
    lines = [f"{method} {path} HTTP/1.1", "Host: 127.0.0.1", "Connection: close"]
    if token is not None:
        lines.append(f"{AUTH_HEADER}: {token}")
    for name, value in (audit or {}).items():
        lines.append(f"{name}: {value}")
    raw = _raw(port, ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8"))
    head, _, body = raw.partition(b"\r\n\r\n")
    head_lines = head.decode("latin-1").split("\r\n")
    status = int(head_lines[0].split(" ")[1])
    headers: dict[str, str] = {}
    for line in head_lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return status, headers, body


def _records(ws: str) -> list[dict[str, Any]]:
    path = os.path.join(ws, "memory", "evidence_chain.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _delete_rows(ws: str) -> list[dict[str, Any]]:
    """Every DELETE-verb chain row, both phases (``delete_phase`` selects it)."""
    return [r for r in _records(ws) if (r.get("metadata") or {}).get("delete_phase")]


# ---------------------------------------------------------------------------
# Leg 1 — inbound acceptance and response echo
# ---------------------------------------------------------------------------


class TestInboundAndEcho:
    def test_a_caller_supplied_request_id_comes_back(self, workspace: str) -> None:
        with _serve(workspace, token=None) as port:
            status, headers, _ = _get(port, audit={ac.HEADER_REQUEST_ID: "trace-http-1"})
        assert status == 200
        assert headers["x-mindmem-request-id"] == "trace-http-1"

    def test_actor_and_purpose_are_echoed_when_sent(self, workspace: str) -> None:
        with _serve(workspace, token=None) as port:
            status, headers, _ = _get(
                port,
                audit={ac.HEADER_ACTOR: "agent-7", ac.HEADER_PURPOSE: "quarterly audit"},
            )
        assert status == 200
        assert headers["x-mindmem-actor"] == "agent-7"
        assert headers["x-mindmem-purpose"] == "quarterly audit"

    def test_absent_request_id_is_minted_as_a_uuid4(self, workspace: str) -> None:
        with _serve(workspace, token=None) as port:
            status, headers, _ = _get(port)
        assert status == 200
        minted = headers["x-mindmem-request-id"]
        assert uuid.UUID(minted).version == 4, minted

    def test_absent_actor_and_purpose_echo_nothing(self, workspace: str) -> None:
        """Header absence reads as 'unattributed'; a synthetic value would lie."""
        with _serve(workspace, token=None) as port:
            _status, headers, _ = _get(port)
        assert "x-mindmem-actor" not in headers
        assert "x-mindmem-purpose" not in headers

    def test_two_requests_get_two_different_minted_ids(self, workspace: str) -> None:
        """POSITIVE CONTROL: the id is per-request, not a per-process constant."""
        with _serve(workspace, token=None) as port:
            _s1, first, _b1 = _get(port)
            _s2, second, _b2 = _get(port)
        assert first["x-mindmem-request-id"] != second["x-mindmem-request-id"]

    def test_a_refused_request_still_carries_a_correlation_id(self, workspace: str) -> None:
        """401 is the response an operator most wants to correlate."""
        with _serve(workspace, token=TOKEN) as port:
            status, headers, _ = _get(port, token=None, audit={ac.HEADER_REQUEST_ID: "trace-401"})
        assert status == 401
        assert headers["x-mindmem-request-id"] == "trace-401"

    def test_a_404_still_carries_a_correlation_id(self, workspace: str) -> None:
        with _serve(workspace, token=None) as port:
            status, headers, _ = _get(port, "/no-such-route", audit={ac.HEADER_REQUEST_ID: "trace-404"})
        assert status == 404
        assert headers["x-mindmem-request-id"] == "trace-404"

    def test_the_body_is_unchanged_by_the_headers(self, workspace: str) -> None:
        """The three headers are attribution, not input: same request, same answer."""
        with _serve(workspace, token=None) as port:
            _s1, _h1, bare = _get(port, PATH_MEMORIES)
            _s2, _h2, attributed = _get(
                port,
                PATH_MEMORIES,
                audit={
                    ac.HEADER_REQUEST_ID: "trace-body",
                    ac.HEADER_ACTOR: "agent-7",
                    ac.HEADER_PURPOSE: "diff check",
                },
            )
        assert bare == attributed


# ---------------------------------------------------------------------------
# Leg 2 — sanitisation (the v4.0.11 CRLF class)
# ---------------------------------------------------------------------------


class TestSanitisation:
    def test_a_folded_request_id_cannot_forge_a_response_header(self, workspace: str) -> None:
        """RFC 7230 obs-fold is how a bare newline reaches this handler at all.

        ``http.server`` will not hand a handler a header value containing a
        raw CR or LF by any other route: a continuation line is the one
        shape its parser accepts and preserves, keeping the ``\\n`` inside
        the value. Echoed unsanitised, that newline ends the
        ``X-MindMem-Request-Id`` line and starts an ``X-Injected`` one —
        response splitting, from a header the caller fully controls.

        What is asserted is the absence of a forged header *line*, not the
        absence of the substring: the sanitised value legitimately still
        reads ``trace-9 X-Injected: yes``, which is one value and parses as
        one. The second assertion pins exactly that, so the test cannot
        pass by the whole header having been dropped.
        """
        with _serve(workspace, token=None) as port:
            raw = _raw(
                port,
                (
                    "GET /status HTTP/1.1\r\n"
                    "Host: 127.0.0.1\r\n"
                    "Connection: close\r\n"
                    "X-MindMem-Request-Id: trace-9\r\n"
                    " X-Injected: yes\r\n"  # continuation line: one header value
                    "\r\n"
                ).encode("utf-8"),
            )
        head, _, _body = raw.partition(b"\r\n\r\n")
        assert b"\r\nX-Injected" not in head, head
        assert b"\r\nX-MindMem-Request-Id: trace-9 X-Injected: yes\r\n" in head, head
        assert head.count(b"X-MindMem-Request-Id") == 1, head

    def test_control_bytes_are_stripped_from_the_echo(self, workspace: str) -> None:
        with _serve(workspace, token=None) as port:
            _status, headers, _ = _get(port, audit={ac.HEADER_ACTOR: "ag\tent\x0b7"})
        assert headers["x-mindmem-actor"] == "agent7"

    def test_an_over_long_request_id_is_bounded(self, workspace: str) -> None:
        with _serve(workspace, token=None) as port:
            _status, headers, _ = _get(port, audit={ac.HEADER_REQUEST_ID: "x" * 5000})
        assert headers["x-mindmem-request-id"] == "x" * ac.MAX_REQUEST_ID_LEN

    def test_an_over_long_purpose_is_bounded(self, workspace: str) -> None:
        with _serve(workspace, token=None) as port:
            _status, headers, _ = _get(port, audit={ac.HEADER_PURPOSE: "p" * 5000})
        assert headers["x-mindmem-purpose"] == "p" * ac.MAX_FIELD_LEN

    def test_an_all_control_request_id_is_replaced_not_echoed_empty(self, workspace: str) -> None:
        """A value that sanitises to nothing is an ABSENT value: mint a fresh one."""
        with _serve(workspace, token=None) as port:
            _status, headers, _ = _get(port, audit={ac.HEADER_REQUEST_ID: "\x01\x02\x03"})
        assert uuid.UUID(headers["x-mindmem-request-id"]).version == 4


# ---------------------------------------------------------------------------
# Leg 3 — the structured-log context
# ---------------------------------------------------------------------------


class TestLogContext:
    """Gated behind ``v4.logging_context``, like every other leg of that surface."""

    @contextlib.contextmanager
    def _armed(self, workspace: str, monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> Iterator[None]:
        """Arm the filter for the block and DISARM on the way out.

        The filter is installed on a process-wide handler, so a test that
        only unsets ``MIND_MEM_CONFIG`` leaves it armed for the rest of the
        session. Teardown writes the flag back OFF and asserts the disarm.
        """
        from mind_mem import observability

        cfg = Path(workspace, "mind-mem.json")
        original = cfg.read_text(encoding="utf-8")

        def _write(flag: bool) -> None:
            body = json.loads(original)
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
            cfg.write_text(original, encoding="utf-8")

    def _probe(self, monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any]) -> None:
        """Capture the log context from inside a served handler.

        ``_handle_list_memories`` resolves ``_admitted_blocks`` from module
        globals at call time, so this substitution is genuinely on the
        served path — ``ROUTES`` holds the handler by reference, and a patch
        of the handler itself would not be.
        """
        from mind_mem.v4.logging_context import current_context

        def _capture(workspace: str, *, active_only: bool, surface: str) -> tuple[list[dict[str, Any]], int]:
            seen["ctx"] = current_context()
            return ([], 0)

        monkeypatch.setattr(http_transport, "_admitted_blocks", _capture)

    def test_the_claims_are_bound_for_the_duration_of_the_request(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}
        self._probe(monkeypatch, seen)
        with self._armed(workspace, monkeypatch, enabled=True), _serve(workspace, token=TOKEN) as port:
            status, _headers, _body = _get(
                port,
                PATH_MEMORIES,
                token=TOKEN,
                audit={
                    ac.HEADER_REQUEST_ID: "trace-log-1",
                    ac.HEADER_ACTOR: "agent-7",
                    ac.HEADER_PURPOSE: "audit",
                },
            )
        assert status == 200
        ctx = seen["ctx"]
        assert ctx["correlation_id"] == "trace-log-1"
        assert ctx["request_id"] == "trace-log-1"
        assert ctx["actor_claimed"] == "agent-7"
        assert ctx["purpose"] == "audit"
        assert ctx["transport"] == "http"
        # The identity the DOOR resolved, not the one the caller claimed.
        assert ctx["agent"] == _token_actor(TOKEN)

    def test_flag_off_pushes_nothing(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The off path must cost nothing and change nothing."""
        seen: dict[str, Any] = {}
        self._probe(monkeypatch, seen)
        with self._armed(workspace, monkeypatch, enabled=False), _serve(workspace, token=TOKEN) as port:
            status, _headers, _body = _get(
                port,
                PATH_MEMORIES,
                token=TOKEN,
                audit={ac.HEADER_REQUEST_ID: "trace-log-2"},
            )
        assert status == 200
        assert seen["ctx"] == {}

    def test_a_crlf_actor_cannot_forge_a_log_binding(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}
        self._probe(monkeypatch, seen)
        with self._armed(workspace, monkeypatch, enabled=True), _serve(workspace, token=TOKEN) as port:
            raw = _raw(
                port,
                (
                    f"GET {PATH_MEMORIES} HTTP/1.1\r\n"
                    "Host: 127.0.0.1\r\n"
                    "Connection: close\r\n"
                    f"{AUTH_HEADER}: {TOKEN}\r\n"
                    "X-MindMem-Actor: real\r\n"
                    ' {"level": "info", "event": "forged"}\r\n'
                    "\r\n"
                ).encode("utf-8"),
            )
        assert raw.split(b"\r\n", 1)[0].split(b" ")[1] == b"200", raw[:200]
        bound = seen["ctx"]["actor_claimed"]
        assert "\r" not in bound and "\n" not in bound, repr(bound)


# ---------------------------------------------------------------------------
# Leg 4 — the audit trail
# ---------------------------------------------------------------------------


class TestAuditTrail:
    def test_a_delete_records_the_request_that_asked_for_it(self, workspace: str) -> None:
        with _serve(workspace, token=TOKEN) as port:
            status, _headers, _body = _get(
                port,
                f"{_MEMORY_ID_PREFIX}{SEED_ID}",
                method="DELETE",
                token=TOKEN,
                audit={
                    ac.HEADER_REQUEST_ID: "trace-del-1",
                    ac.HEADER_ACTOR: "agent-7",
                    ac.HEADER_PURPOSE: "right-to-erasure request 4012",
                },
            )
        assert status == 200
        rows = _delete_rows(workspace)
        assert rows, "positive control: the delete must have reached the chain at all"
        meta = rows[0]["metadata"]
        assert meta["request_id"] == "trace-del-1"
        assert meta["actor_claimed"] == "agent-7"
        assert meta["purpose"] == "right-to-erasure request 4012"

    def test_the_recorded_actor_is_the_door_not_the_claim(self, workspace: str) -> None:
        """A provenance field filled from an unauthenticated claim reads like a fact."""
        with _serve(workspace, token=TOKEN) as port:
            status, _headers, _body = _get(
                port,
                f"{_MEMORY_ID_PREFIX}{SEED_ID}",
                method="DELETE",
                token=TOKEN,
                audit={ac.HEADER_ACTOR: "root"},
            )
        assert status == 200
        rows = _delete_rows(workspace)
        assert rows, "positive control: the delete must have reached the chain at all"
        actor = rows[0]["actor"]
        assert actor == _token_actor(TOKEN)
        assert actor.startswith(HTTP_TOKEN_ACTOR_PREFIX)
        assert actor != "root"

    def test_no_audit_headers_records_exactly_what_it_recorded_before(self, workspace: str) -> None:
        """The OPTIONAL half: absence adds no key to the chain entry."""
        with _serve(workspace, token=TOKEN) as port:
            status, _headers, _body = _get(
                port,
                f"{_MEMORY_ID_PREFIX}{SEED_ID}",
                method="DELETE",
                token=TOKEN,
            )
        assert status == 200
        rows = _delete_rows(workspace)
        assert rows, "positive control: the delete must have reached the chain at all"
        meta = rows[0]["metadata"]
        assert "request_id" not in meta
        assert "actor_claimed" not in meta
        assert "purpose" not in meta
        # POSITIVE CONTROL for the three assertions above: the keys the door
        # has always written ARE there, so the probe can see this metadata.
        assert meta["rationale"] == http_transport.DEFAULT_DELETE_RATIONALE
        assert meta["delete_phase"] == "admitted"

    def test_a_mutating_route_records_the_door_identity_on_the_context(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """``agent_authenticated`` is what the door resolved, never the claim.

        Probed through ``_request_audit_metadata``, which the delete door
        resolves from module globals at call time — so this observes the
        real served path rather than a re-derivation of it.
        """
        seen: dict[str, Any] = {}
        original = http_transport._request_audit_metadata

        def _capture() -> dict[str, str] | None:
            ctx = ac.current_audit_context()
            seen["bound"] = ctx is not None
            seen["authenticated"] = ctx.agent_authenticated if ctx is not None else None
            seen["claimed"] = ctx.actor_claimed if ctx is not None else None
            return original()

        monkeypatch.setattr(http_transport, "_request_audit_metadata", _capture)
        with _serve(workspace, token=TOKEN) as port:
            status, _headers, _body = _get(
                port,
                f"{_MEMORY_ID_PREFIX}{SEED_ID}",
                method="DELETE",
                token=TOKEN,
                audit={ac.HEADER_ACTOR: "root"},
            )
        assert status == 200
        assert seen["bound"] is True
        assert seen["authenticated"] == _token_actor(TOKEN)
        assert seen["claimed"] == "root"

    def test_a_read_route_records_no_authenticated_identity(self, workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The route table is the surface: no declared mutation, no attribution.

        The same rule that decides whether a handler is handed an
        ``actor``. Keeping the two under one declaration is what lets
        ``test_governed_delete_http``'s dispatcher twin still reproduce
        the anonymous row it exists to reproduce.
        """
        seen: dict[str, Any] = {}

        def _capture(workspace: str, *, active_only: bool, surface: str) -> tuple[list[dict[str, Any]], int]:
            ctx = ac.current_audit_context()
            seen["bound"] = ctx is not None
            seen["authenticated"] = ctx.agent_authenticated if ctx is not None else None
            return ([], 0)

        monkeypatch.setattr(http_transport, "_admitted_blocks", _capture)
        with _serve(workspace, token=TOKEN) as port:
            status, _headers, _body = _get(port, PATH_MEMORIES, token=TOKEN)
        assert status == 200
        # POSITIVE CONTROL: a context IS bound here, so ``None`` below is a
        # measured absence rather than the probe having seen nothing at all.
        assert seen["bound"] is True
        assert seen["authenticated"] is None

    def test_a_crlf_purpose_cannot_forge_a_chain_field(self, workspace: str) -> None:
        with _serve(workspace, token=TOKEN) as port:
            raw = _raw(
                port,
                (
                    f"DELETE {_MEMORY_ID_PREFIX}{SEED_ID} HTTP/1.1\r\n"
                    "Host: 127.0.0.1\r\n"
                    "Connection: close\r\n"
                    f"{AUTH_HEADER}: {TOKEN}\r\n"
                    "X-MindMem-Purpose: real\r\n"
                    ' {"actor": "root"}\r\n'
                    "\r\n"
                ).encode("utf-8"),
            )
        assert raw.split(b"\r\n", 1)[0].split(b" ")[1] == b"200", raw[:200]
        rows = _delete_rows(workspace)
        assert rows, "positive control: the delete must have reached the chain at all"
        purpose = rows[0]["metadata"]["purpose"]
        assert "\r" not in purpose and "\n" not in purpose, repr(purpose)
        assert rows[0]["actor"] == _token_actor(TOKEN)


# ---------------------------------------------------------------------------
# The primitive, called the way this transport calls it
# ---------------------------------------------------------------------------


class TestRequestAuditMetadata:
    def test_outside_a_request_there_is_no_metadata(self) -> None:
        """A library / in-process caller writes the chain entry it always wrote."""
        assert http_transport._request_audit_metadata() is None

    def test_a_context_with_no_caller_headers_yields_no_metadata(self) -> None:
        ctx = ac.context_from_headers(lambda _name: None, transport="http")
        with ac.bind_audit_context(ctx):
            assert http_transport._request_audit_metadata() is None

    def test_a_context_with_one_caller_header_yields_the_correlation_id(self) -> None:
        ctx = ac.context_from_headers(
            lambda name: "audit" if name == "x-mindmem-purpose" else None,
            transport="http",
        )
        with ac.bind_audit_context(ctx):
            meta = http_transport._request_audit_metadata()
        assert meta is not None
        assert meta["purpose"] == "audit"
        assert meta["request_id"] == ctx.request_id
        assert "actor_claimed" not in meta


class TestSuppliedTracking:
    """``AuditContext.supplied`` — what the CALLER sent, not what we minted."""

    def test_nothing_sent_is_an_empty_set(self) -> None:
        ctx = ac.context_from_headers(lambda _name: None, transport="http")
        assert ctx.supplied == frozenset()
        assert ctx.request_id, "a minted id is still an id"

    def test_each_header_is_tracked_by_its_canonical_spelling(self) -> None:
        sent = {
            "x-mindmem-request-id": "r1",
            "x-mindmem-actor": "a1",
            "x-mindmem-purpose": "p1",
        }
        ctx = ac.context_from_headers(sent.get, transport="http")
        assert ctx.supplied == frozenset({ac.HEADER_REQUEST_ID, ac.HEADER_ACTOR, ac.HEADER_PURPOSE})

    def test_a_value_that_sanitises_to_nothing_was_not_supplied(self) -> None:
        ctx = ac.context_from_headers(
            {"x-mindmem-request-id": "\r\n\x00", "x-mindmem-actor": ""}.get,
            transport="http",
        )
        assert ctx.supplied == frozenset()
        assert uuid.UUID(ctx.request_id).version == 4
