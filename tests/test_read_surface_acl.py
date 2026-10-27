# Copyright 2026 STARGA, Inc.
"""No read surface returns memory the MCP tool path denies to the same caller.

The tool path (``get_block``, ``recall``) enforces two things on every read:
the caller's NAMESPACE ACL and the admission filter (quarantined / pending
content is unresolvable). Two other doors used to enforce only the second:

* the REST ``/v1`` endpoints authenticated in one threadpool frame and ran the
  tool in another, so the tool saw an UNBOUND caller and answered workspace-wide;
* the ``mind-mem://`` resources read fixed root-level corpus files and never
  asked the ACL at all.

Every test pairs the negative (denied / quarantined content absent, in the body,
the counts, the error text, the headers) with a POSITIVE CONTROL (the block the
caller IS allowed is present), because "X not in response" also passes when the
write silently failed. The mutation check is
``test_removing_the_shared_check_turns_the_resource_test_red``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mind_mem import read_authz
from mind_mem.audit_context import bind_current_agent
from mind_mem.init_workspace import init
from mind_mem.mcp import resources as R
from mind_mem.mcp.infra.workspace import use_workspace

TAGS = ("ROOT", "ALICE", "BOB", "SHARED")
ROOT_FILES = ("decisions/DECISIONS.md", "tasks/TASKS.md")


def _blk(block_id: str, stmt: str, status: str = "active", typ: str = "Decision") -> str:
    return f"[{block_id}]\nType: {typ}\nDate: 2026-01-01\nStatement: {stmt} frost telemetry rollout\nStatus: {status}\n\n"


def _w(ws: Path, rel: str, text: str) -> None:
    path = ws / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    init(str(root))
    for ns, tag in (("", "ROOT"), ("agents/alice/", "ALICE"), ("agents/bob/", "BOB"), ("shared/", "SHARED")):
        _w(
            root,
            ns + "decisions/DECISIONS.md",
            _blk(f"D-{tag}-ACT", f"zq{tag}ACTIVEcanary") + _blk(f"D-{tag}-QUA", f"zq{tag}QUARANTINEDcanary", "quarantined"),
        )
        _w(
            root,
            ns + "tasks/TASKS.md",
            _blk(f"T-{tag}-ACT", f"zq{tag}TASKACTIVEcanary", typ="Task")
            + _blk(f"T-{tag}-QUA", f"zq{tag}TASKQUARANTINEDcanary", "quarantined", typ="Task"),
        )
    _w(root, "intelligence/CONTRADICTIONS.md", _blk("C-ROOT-ACT", "zqROOTCONTRACTIVEcanary", typ="Contradiction"))
    _w(
        root,
        "intelligence/SIGNALS.md",
        _blk("SIG-ROOT-ACT", "zqROOTSIGACTIVEcanary", typ="Signal")
        + _blk("SIG-ROOT-PEN", "zqROOTSIGPENDINGcanary", "pending", typ="Signal"),
    )
    _w(root, "entities/projects.md", _blk("PRJ-ROOT-ACT", "zqROOTPROJACTIVEcanary", typ="Project"))
    _w(
        root,
        "shared/intelligence/LEDGER.md",
        _blk("FACT-SH-ACT", "zqSHAREDLEDGERACTIVEcanary", typ="Fact")
        + _blk("FACT-SH-QUA", "zqSHAREDLEDGERQUARCANARY", "quarantined", typ="Fact"),
    )
    # eve: shared only. alice: shared + her namespace + the root corpus dirs.
    # bob exists only so there is a second private namespace to leak from.
    (root / "mind-mem-acl.json").write_text(
        json.dumps(
            {
                "default_policy": "deny",
                "agents": {
                    "eve": {"namespaces": ["shared"], "read": ["shared"], "write": []},
                    "alice": {
                        "namespaces": ["shared", "agents/alice"],
                        "read": ["shared", "agents/alice", "decisions", "tasks", "entities", "intelligence", "memory"],
                        "write": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(root))
    return root


# canaries that no caller below may ever see
ALWAYS_DENIED = tuple(f"zq{tag}{kind}QUARANTINEDcanary" for tag in TAGS for kind in ("", "TASK")) + (
    "zqBOBACTIVEcanary",
    "zqBOBTASKACTIVEcanary",
    "zqSHAREDLEDGERQUARCANARY",
    "zqALICEACTIVEcanary",
    "zqALICETASKACTIVEcanary",
)
ROOT_ACTIVE = ("zqROOTACTIVEcanary", "zqROOTTASKACTIVEcanary", "zqROOTCONTRACTIVEcanary", "zqROOTPROJACTIVEcanary", "zqROOTSIGACTIVEcanary")
ALWAYS_DENIED = ALWAYS_DENIED + ("zqROOTSIGPENDINGcanary",)


def _resource_bodies(ws: Path, principal: str | None) -> dict[str, str]:
    """Read every registered resource through its ``_gated`` wrapper, as *principal*."""
    calls: dict[str, Callable[[], str]] = {
        "decisions": lambda: R._gated(R.get_decisions)(),
        "tasks": lambda: R._gated(R.get_tasks)(),
        "entities": lambda: R._gated(R.get_entities)("projects"),
        "signals": lambda: R._gated(R.get_signals)(),
        "contradictions": lambda: R._gated(R.get_contradictions)(),
        "health": lambda: R._gated(R.get_health)(),
        "recall": lambda: R._gated(R.get_recall)("frost telemetry rollout"),
        "ledger": lambda: R._gated(R.get_ledger)(),
    }
    out: dict[str, str] = {}
    for name, call in calls.items():
        ctx = bind_current_agent(principal) if principal else _nullcontext()
        with ctx, use_workspace(str(ws)):
            out[name] = call()
    return out


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


# --------------------------------------------------------------------------
# MCP resources
# --------------------------------------------------------------------------


def test_every_registered_resource_is_covered_by_this_file() -> None:
    uris: list[str] = []

    class _Capture:
        def resource(self, uri: str):  # type: ignore[no-untyped-def]
            uris.append(uri)
            return lambda fn: fn

    R.register(_Capture())
    assert sorted(uris) == sorted(
        [
            "mind-mem://decisions",
            "mind-mem://tasks",
            "mind-mem://entities/{entity_type}",
            "mind-mem://signals",
            "mind-mem://contradictions",
            "mind-mem://health",
            "mind-mem://recall/{query}",
            "mind-mem://ledger",
        ]
    ), "a new resource must be added to _resource_bodies() and swept for ACL here"


def test_restricted_caller_gets_no_root_or_foreign_content_from_any_resource(ws: Path) -> None:
    bodies = _resource_bodies(ws, "eve")
    for name, body in bodies.items():
        for canary in ALWAYS_DENIED + ROOT_ACTIVE:
            assert canary not in body, f"resource {name} leaked {canary} to a shared-only caller"
    # positive control: the one thing eve may read (shared ledger, active) IS served
    assert "zqSHAREDLEDGERACTIVEcanary" in bodies["ledger"]
    # no leak through counts: the root corpus is not summarised to a caller who cannot read it
    health = json.loads(bodies["health"])
    assert health["files"]["decisions"] == {"total": 0, "active": 0}
    assert health["files"]["tasks"] == {"total": 0, "active": 0}
    assert health["files"]["contradictions"] == {"total": 0, "active": 0}
    # a denied file answers exactly like an absent one
    for name in ("decisions", "tasks", "entities", "contradictions"):
        assert json.loads(bodies[name]) == [], name


def test_granted_caller_gets_exactly_what_it_is_granted_from_every_resource(ws: Path) -> None:
    bodies = _resource_bodies(ws, "alice")
    # positive controls: the root corpus alice was granted IS served
    assert "zqROOTACTIVEcanary" in bodies["decisions"]
    assert "zqROOTTASKACTIVEcanary" in bodies["tasks"]
    assert "zqROOTCONTRACTIVEcanary" in bodies["contradictions"]
    assert "zqROOTSIGACTIVEcanary" in bodies["signals"]
    assert "zqROOTPROJACTIVEcanary" in bodies["entities"]
    assert "zqSHAREDLEDGERACTIVEcanary" in bodies["ledger"]
    assert json.loads(bodies["health"])["files"]["decisions"]["total"] == 2
    # negatives: other namespaces and every quarantined block stay out
    for name, body in bodies.items():
        for canary in (
            "zqBOBACTIVEcanary",
            "zqBOBTASKACTIVEcanary",
            "zqROOTQUARANTINEDcanary",
            "zqROOTTASKQUARANTINEDcanary",
            "zqSHAREDLEDGERQUARCANARY",
        ):
            assert canary not in body, f"resource {name} leaked {canary}"


def test_resource_recall_is_scoped_to_the_caller(ws: Path) -> None:
    with bind_current_agent("eve"), use_workspace(str(ws)):
        body = R.get_recall("frost telemetry rollout")
    assert "zqBOBACTIVEcanary" not in body and "zqROOTACTIVEcanary" not in body
    with bind_current_agent("alice"), use_workspace(str(ws)):
        body = R.get_recall("frost telemetry rollout alpha")
    assert "zqROOTACTIVEcanary" in body  # positive control: the root corpus she was granted


def test_unresolvable_caller_is_refused_not_promoted_to_unbound(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> str | None:
        raise ValueError("authentication context unavailable")

    monkeypatch.setattr("mind_mem.mcp.infra.acl.authenticated_agent_id", broken)
    with use_workspace(str(ws)):
        with pytest.raises(read_authz.CallerUnresolved):
            R.get_decisions()
        body = R._gated(R.get_decisions)()
    assert "zqROOTACTIVEcanary" not in body
    assert "error" in json.loads(body)


def test_unbound_operator_keeps_workspace_level_reads(ws: Path) -> None:
    """Behaviour preserved: stdio / local operator has no principal to restrict."""
    with use_workspace(str(ws)):
        assert "zqROOTACTIVEcanary" in R.get_decisions()
        assert "zqROOTQUARANTINEDcanary" not in R.get_decisions()  # admission still applies


def test_removing_the_shared_check_turns_the_resource_test_red(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: with the ACL decision disabled the sweep above MUST fail."""
    monkeypatch.setattr(R, "source_readable", lambda *a, **k: True)
    bodies = _resource_bodies(ws, "eve")
    assert "zqROOTACTIVEcanary" in bodies["decisions"], "mutation had no effect; the ACL test above cannot fail"


# --------------------------------------------------------------------------
# REST /v1
# --------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

from mind_mem.api import rest  # noqa: E402
from mind_mem.api.api_keys import APIKeyStore  # noqa: E402


@pytest.fixture
def client_keys(ws: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    for name in ("MIND_MEM_TOKEN", "MIND_MEM_ADMIN_TOKEN", "OIDC_ISSUER", "OIDC_AUDIENCE", "MIND_MEM_ALLOW_UNAUTHENTICATED_LOCALHOST"):
        monkeypatch.delenv(name, raising=False)
    db = str(ws.parent / "keys.db")
    monkeypatch.setenv("MIND_MEM_API_KEY_DB", db)
    store = APIKeyStore(db)
    keys = {who: store.create(who, ["user"]) for who in ("eve", "alice", "mallory")}
    with TestClient(rest.create_app(str(ws)), raise_server_exceptions=False) as client:
        yield client, keys


def _auth(keys: dict[str, str], who: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {keys[who]}"}


def test_rest_block_denies_what_the_tool_denies_and_serves_what_it_serves(client_keys) -> None:  # type: ignore[no-untyped-def]
    client, keys = client_keys
    # control: the MCP tool path itself denies the same principal the same source
    from mind_mem.mcp.tools.memory_ops import get_block

    with bind_current_agent("eve"):
        tool = json.loads(get_block("D-ROOT-ACT"))
    assert tool.get("error") == "namespace access denied" or tool.get("found") is not True
    denied = client.get("/v1/block/D-ROOT-ACT", headers=_auth(keys, "eve"))
    assert denied.status_code == 404
    assert "zqROOTACTIVEcanary" not in denied.text
    missing = client.get("/v1/block/D-NOPE-NOPE", headers=_auth(keys, "eve"))
    # denial is indistinguishable from a miss except for the echoed id
    assert denied.json()["detail"]["error"].replace("D-ROOT-ACT", "X") == missing.json()["detail"]["error"].replace("D-NOPE-NOPE", "X")
    assert set(denied.headers) - {"content-length"} == set(missing.headers) - {"content-length"}
    assert "etag" not in {h.lower() for h in denied.headers}
    # positive control: alice, granted the root decisions, gets it
    ok = client.get("/v1/block/D-ROOT-ACT", headers=_auth(keys, "alice"))
    assert ok.status_code == 200 and "zqROOTACTIVEcanary" in ok.text
    # quarantined stays withheld even for the granted caller, with no content
    held = client.get("/v1/block/D-ROOT-QUA", headers=_auth(keys, "alice"))
    assert "zqROOTQUARANTINEDcanary" not in held.text
    assert held.json()["detail"].get("withheld") is True  # positive control that it exists and was withheld, not missing


def test_rest_block_without_a_resolvable_identity_fails_closed(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.requests import Request

    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""}
    request = Request(scope)
    with pytest.raises(fastapi.HTTPException) as err:
        with rest._reading_as(request):
            pytest.fail("a read ran with no authentication decision")
    assert err.value.status_code == 403
    request.state.mindmem_authenticated_agent = "not a valid id!"
    request.state.mindmem_authenticated_scope = "user"
    with pytest.raises(fastapi.HTTPException) as err:
        with rest._reading_as(request):
            pytest.fail("a read ran as an invalid principal")
    assert err.value.status_code == 403


def test_rest_recall_is_scoped_to_the_authenticated_principal(client_keys) -> None:  # type: ignore[no-untyped-def]
    client, keys = client_keys
    eve = client.post("/v1/recall", headers=_auth(keys, "eve"), json={"query": "frost telemetry rollout", "limit": 20})
    assert eve.status_code == 200
    for canary in ALWAYS_DENIED + ROOT_ACTIVE:
        assert canary not in eve.text, f"/v1/recall leaked {canary} to a shared-only caller"
    assert "zqSHAREDACTIVEcanary" in eve.text  # positive control: shared IS readable
    alice = client.post("/v1/recall", headers=_auth(keys, "alice"), json={"query": "frost telemetry rollout beta", "limit": 20})
    assert "zqROOTACTIVEcanary" in alice.text  # positive control: the root corpus she was granted
    for canary in ("zqBOBACTIVEcanary", "zqBOBQUARANTINEDcanary", "zqROOTQUARANTINEDcanary"):
        assert canary not in alice.text


def test_rest_read_endpoint_binds_the_principal_around_the_tool(client_keys, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """Mutation guard on the cause: the tool must SEE the caller, not an unbound context."""
    client, keys = client_keys
    seen: list[str | None] = []
    import mind_mem.mcp.infra.acl as acl

    real = acl.authenticated_agent_id

    def spy() -> str | None:
        who = real()
        seen.append(who)
        return who

    monkeypatch.setattr(acl, "authenticated_agent_id", spy)
    client.post("/v1/recall", headers=_auth(keys, "eve"), json={"query": "frost telemetry rollout gamma", "limit": 5})
    assert seen and set(seen) == {"eve"}


# --- REST scan / contradictions: denied and allowed, parity with the tool ------


def test_rest_scan_and_contradictions_deny_restricted_and_serve_granted(client_keys) -> None:  # type: ignore[no-untyped-def]
    client, keys = client_keys
    for path in ("/v1/scan", "/v1/contradictions"):
        denied = client.get(path, headers=_auth(keys, "eve"))
        assert denied.status_code == 403, path
        assert "total" not in denied.text and "decisions" not in denied.text  # no counts leak through the refusal
        allowed = client.get(path, headers=_auth(keys, "alice"))
        assert allowed.status_code == 200, path  # positive control
    scan = client.get("/v1/scan", headers=_auth(keys, "alice")).json()
    assert scan["checks"]["decisions"]["total"] == 2


def test_scan_and_contradictions_tools_follow_the_same_acl(ws: Path) -> None:
    from mind_mem.mcp.tools.governance import list_contradictions, scan

    with bind_current_agent("eve"), use_workspace(str(ws)):
        assert json.loads(scan()).get("error") == "namespace access denied"
        assert json.loads(list_contradictions()).get("error") == "namespace access denied"
    with bind_current_agent("alice"), use_workspace(str(ws)):
        assert "error" not in json.loads(scan())
        assert "error" not in json.loads(list_contradictions())
    with use_workspace(str(ws)):  # unbound operator keeps workspace-level reads
        assert "error" not in json.loads(scan())


# --- workspace-level transports: refuse while a namespace ACL is configured ----


def test_namespace_acl_configured_predicate(tmp_path: Path) -> None:
    assert read_authz.namespace_acl_configured(str(tmp_path)) is False  # no file
    acl = tmp_path / "mind-mem-acl.json"
    acl.write_text(json.dumps({"default_policy": "read", "agents": {}}), encoding="utf-8")
    assert read_authz.namespace_acl_configured(str(tmp_path)) is False
    acl.write_text(json.dumps({"default_policy": "deny"}), encoding="utf-8")
    assert read_authz.namespace_acl_configured(str(tmp_path)) is True
    acl.write_text(json.dumps({"agents": {"a": {"read": ["shared"]}}}), encoding="utf-8")
    assert read_authz.namespace_acl_configured(str(tmp_path)) is True
    acl.write_text("{ not json", encoding="utf-8")
    assert read_authz.namespace_acl_configured(str(tmp_path)) is True  # cannot tell -> restricted
    acl.write_text("[]", encoding="utf-8")
    assert read_authz.namespace_acl_configured(str(tmp_path)) is True


def _http(port: int, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, str]:
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = json.dumps(body).encode() if body is not None else b""
    try:
        conn.request(method, path, body=payload, headers={"Content-Length": str(len(payload)), "Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read().decode("utf-8", "replace")
    finally:
        conn.close()


def _serve_http(workspace: Path):  # type: ignore[no-untyped-def]
    import socket

    from mind_mem.http_transport import serve_http

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    _t, stop = serve_http(workspace=str(workspace), port=port, host="127.0.0.1", token=None, allow_unauthenticated_localhost=True)
    return port, stop


HTTP_READS = (
    ("GET", "/memories?limit=50&active_only=false", None),
    ("POST", "/query", {"query": "frost telemetry rollout", "limit": 20}),
    ("POST", "/walkthrough", {"topic": "frost telemetry rollout", "limit": 20}),
    ("GET", "/federation/conflicts", None),
    ("GET", "/federation/vclock/D-ROOT-ACT", None),
)


def test_every_non_mutating_http_route_but_status_is_guarded() -> None:
    from mind_mem.http_transport import PATH_STATUS, ROUTES

    unguarded = [
        r.name for r in ROUTES if not r.mutates and r.path != PATH_STATUS and not getattr(r.handler, "_namespace_acl_guarded", False)
    ]
    assert unguarded == [], f"read routes that would bypass a configured namespace ACL: {unguarded}"


def test_http_handlers_refuse_when_called_directly_too(ws: Path) -> None:
    from mind_mem import http_transport as ht

    for status, payload in (
        ht._handle_query(str(ws), {"query": "frost telemetry rollout", "limit": 10}),
        ht._handle_list_memories(str(ws), {}),
        ht._handle_walkthrough(str(ws), {"topic": "frost telemetry rollout"}),
    ):
        assert status == 403 and "zqROOT" not in json.dumps(payload)


def test_http_transport_refuses_every_read_while_a_namespace_acl_is_configured(ws: Path) -> None:
    port, stop = _serve_http(ws)
    try:
        for method, path, body in HTTP_READS:
            status, text = _http(port, method, path, body)
            assert status == 403, (path, status, text[:200])
            for canary in ALWAYS_DENIED + ROOT_ACTIVE:
                assert canary not in text, (path, canary)
            assert "D-ROOT" not in text and "D-BOB" not in text  # not even ids or counts
        assert _http(port, "GET", "/status")[0] == 200  # liveness is not a read
    finally:
        stop()


def test_http_transport_still_serves_when_no_namespace_acl_exists(ws: Path) -> None:
    """Positive control: the refusal is caused by the ACL, not by the route being dead."""
    (ws / "mind-mem-acl.json").unlink()
    port, stop = _serve_http(ws)
    try:
        status, text = _http(port, "GET", "/memories?limit=50&active_only=false")
        assert status == 200 and "D-ROOT-ACT" in text
        assert "zqROOTQUARANTINEDcanary" not in text and "D-ROOT-QUA" not in text  # admission still applies
    finally:
        stop()


def test_grpc_recall_and_scan_refuse_while_a_namespace_acl_is_configured(ws: Path) -> None:
    from mind_mem.api import grpc_server as g
    from mind_mem.mcp.infra.workspace import use_workspace

    with use_workspace(str(ws)):
        with pytest.raises(PermissionError):
            g.handle_recall(g.RecallRequest(query="frost telemetry rollout"))
        scan = g.handle_governance(g.GovernanceRequest(operation="scan"))
        assert scan.ok is False and scan.payload == ""
        (ws / "mind-mem-acl.json").unlink()
        ok = g.handle_recall(g.RecallRequest(query="frost telemetry rollout"))  # positive control
        assert "zqROOTACTIVEcanary" in ok.payload
        assert "zqROOTQUARANTINEDcanary" not in ok.payload
