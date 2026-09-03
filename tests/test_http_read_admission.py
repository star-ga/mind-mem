# Copyright 2026 STARGA, Inc.
"""Every HTTP route, swept with a three-status canary.

The registry-wide sweep that closed the MCP read surface
(``tests/test_read_surface_admission.py``) enumerates the **tool
registry**: 102 tools, every one classified and exercised against a
corpus holding one ``active``, one ``pending`` and one ``quarantined``
block. It does not enumerate this transport. So for as long as it has
been green, ``GET /memories`` has been calling
``store.get_all(active_only=False)`` and putting the result on the wire
with no admission on it at all — the same defect the MCP sweep exists to
catch, one surface over, invisible because the sweep's unit of
enumeration stopped at the registry.

Two things were wrong with that endpoint and they had to be fixed in one
change, because fixing either alone is worse than fixing neither:

1. **Every summary was empty.** The projection read ``id`` / ``type`` /
   ``subject`` / ``timestamp``; blocks carry ``_id`` and the canonical
   capitalised fields on all five backends. Measured before the fix: a
   three-block corpus answered ``{"count": 3, "memories": [{"id": null,
   "type": null, "category": null, "subject": null, "timestamp": null},
   ...]}``. The endpoint was useless to a client — and, by accident, the
   only reason the missing admission had not yet leaked block text.
2. **No admission.** ``total`` counted the withheld blocks, and the
   moment the projection was repaired the withheld *statements* would
   have gone out with it.

What makes the file below a proof rather than a page of green rows:

* **The route table is the unit of enumeration.** ``ROUTES`` is what the
  dispatcher consults, every :class:`~mind_mem.http_transport.Route`
  carries a ``verdict`` with no default, and the tests check the table
  against the module's ``_handle_*`` functions in BOTH directions — a new
  endpoint fails the build until someone classifies it.
* **Positive control on the seed.** The withheld blocks are shown to be
  on disk and in the index before anything asserts their absence.
* **Positive control on the canary.** The same block, with its status
  flipped to ``active``, comes back through ``GET /memories`` — so "the
  canary did not appear" means the gate held, not that the canary was
  broken.
* **Positive control on the sweep.** Every route classified ``content``
  must actually reach the corpus, and the measured reach set is asserted
  EQUAL to the declared one. A route that starts serving block content
  fails until it is reclassified.
* **No invocation may be refused before its handler ran.** A 404 from the
  router, a 401, a 429 or a "feature disabled" 503 carries no canary and
  would read as a clean pass — the vacuous-pass shape. Detected, with its
  own positive control.
* **The sweep can see a leak.** A deliberately leaking route is added to
  the table, served by a real server, and must be caught by the same
  assertion the sweep uses.
"""

from __future__ import annotations

import ast
import http.client
import inspect
import json
import os
import shutil
import socket
import sqlite3
import sys
import tempfile
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

import pytest
from test_read_surface_classification import ACTIVE_ID, PENDING_ID, QUARANTINED_ID, SWEEP_FLAGS

from mind_mem import http_transport
from mind_mem.http_transport import CONTENT, NO_CONTENT, ROUTES, Route, content_routes, serve_http

#: Improbable tokens: any hit is the seeded block and nothing else. Distinct
#: from the MCP sweep's, so neither file can be made vacuous by an edit to
#: the other's seed.
CANARIES: dict[str, str] = {
    "active": "zqxHTTPACTIVEcanary",
    "pending": "zqxHTTPPENDINGcanary",
    "quarantined": "zqxHTTPQUARANTINEcanary",
}
WITHHELD_CANARIES = (CANARIES["pending"], CANARIES["quarantined"])
SEEDED = ((ACTIVE_ID, "active"), (PENDING_ID, "pending"), (QUARANTINED_ID, "quarantined"))

#: Refusals produced by the dispatcher's guards or by a disabled feature —
#: everything the caller can be told *before* a handler body runs. A row
#: carrying one of these has not exercised the surface it claims to have
#: swept, so the sweep treats it as a hole rather than a pass.
_PRE_HANDLER_REFUSALS = frozenset(
    {
        "not found",
        "method not allowed",
        "missing or invalid token",
        "cross-origin request rejected",
        "rate limit exceeded",
        "bad request body",
        "empty body",
        "source IP not on MIND_MEM_FED_PEERS allowlist",
        "federation module unavailable",
        "federation feature disabled",
        "federation unavailable",
    }
)


def _refused_before_the_handler(status: int, body: dict[str, Any]) -> bool:
    """True when the response was produced without reaching a handler."""
    error = body.get("error")
    return isinstance(error, str) and error in _PRE_HANDLER_REFUSALS


# ---------------------------------------------------------------------------
# The seed
# ---------------------------------------------------------------------------


def _render(block_id: str, status: str, canary: str) -> str:
    return (
        f"[{block_id}]\n"
        f"Date: 2026-01-01\n"
        f"Status: {status}\n"
        f"Scope: global\n"
        f"Statement: The {canary} architecture decision governs frost telemetry.\n"
        f"Rationale: {canary} rationale for the frost telemetry architecture rollout.\n"
        f"Tags: architecture, frost\n"
        f"Sources: -\n"
        f"Supersedes: -\n\n"
    )


def _seed(workspace: str) -> None:
    """One active, one pending, one quarantined canary block, plus the index.

    Written to the corpus directly: what this file needs is a corpus in a
    known state with the withheld blocks unambiguously present, not a test
    of some ingest door.

    Every registered v4 flag is turned on, reusing the MCP sweep's own
    ``SWEEP_FLAGS`` list so the two surfaces cannot drift apart on
    coverage. Four routes here (the federation family) answer ``503
    feature disabled`` otherwise, and a disabled route is not a route the
    sweep has checked.
    """
    from mind_mem import sqlite_index
    from mind_mem.init_workspace import init

    init(workspace)
    body = "".join(_render(bid, status, CANARIES[status]) for bid, status in SEEDED)
    with open(os.path.join(workspace, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write("\n" + body)

    config_path = os.path.join(workspace, "mind-mem.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("v4", {})
    for flag in SWEEP_FLAGS:
        config["v4"][flag] = {"enabled": True}
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)

    sqlite_index.build_index(workspace)


@pytest.fixture(scope="module")
def seed_template(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Seeded once; every invocation gets its own copy.

    A shared workspace would let ``POST /clear`` and ``DELETE
    /memories/{id}`` change what the next route sees, and a sweep whose
    corpus depends on execution order proves whatever it happens to.
    """
    template = str(tmp_path_factory.mktemp("http_read_seed") / "ws")
    _seed(template)
    return template


def _fresh(template: str) -> str:
    target = tempfile.mkdtemp(prefix="mm_http_sweep_")
    shutil.rmtree(target)
    shutil.copytree(template, target)
    return target


# ---------------------------------------------------------------------------
# The transport, driven for real
# ---------------------------------------------------------------------------


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _request(port: int, method: str, path: str, body: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = json.dumps(body).encode("utf-8") if body is not None else b""
    headers = {"Content-Length": str(len(payload))}
    if payload:
        headers["Content-Type"] = "application/json"
    try:
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            parsed = {"_raw": raw.decode("utf-8", "replace")}
        return (response.status, parsed if isinstance(parsed, dict) else {"_raw": parsed})
    finally:
        conn.close()


@contextmanager
def _serve(workspace: str) -> Iterator[int]:
    """A real loopback server over *workspace*, torn down on exit."""
    port = _free_port()
    _thread, stop = serve_http(
        workspace=workspace,
        port=port,
        host="127.0.0.1",
        token=None,
        allow_unauthenticated_localhost=True,
    )
    try:
        yield port
    finally:
        stop()


def _call(workspace: str, route: Route, invocation: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """One request, through the real dispatcher, against *workspace*.

    Driven over a socket rather than by calling the handler function: a
    handler that is not in :data:`~mind_mem.http_transport.ROUTES` is
    unreachable however well it behaves, and a sweep that called the
    functions directly would report a surface nobody can reach as swept.
    """
    from mind_mem.governance_gate import evict_gate

    path = route.path + str(invocation.get("tail", ""))
    params = invocation.get("params")
    if params:
        path = f"{path}?{urlencode(params)}"
    previous = {key: os.environ.get(key) for key in ("MIND_MEM_WORKSPACE", "MIND_MEM_CONFIG")}
    # The v4 flag resolver reads ``MIND_MEM_CONFIG`` / ``MIND_MEM_WORKSPACE``
    # from the process environment, NOT the workspace the server was handed
    # (``v4.feature_flags._config_path``). Without this the four federation
    # routes answer 503 "feature disabled" whatever the seeded config says,
    # and four rows of the sweep would be green over a surface it never
    # reached -- which is exactly what the refusal check caught when this
    # file was first run.
    os.environ["MIND_MEM_WORKSPACE"] = workspace
    os.environ["MIND_MEM_CONFIG"] = os.path.join(workspace, "mind-mem.json")
    try:
        with _serve(workspace) as port:
            return _request(port, route.method, path, invocation.get("body"))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # The gate is cached per workspace path and these are throwaway
        # directories; a cached gate pointing at a deleted tree would
        # follow the next invocation into a workspace it never opened.
        evict_gate(workspace)


# ---------------------------------------------------------------------------
# The invocation table — how the sweep calls each route
# ---------------------------------------------------------------------------

#: ``route.name`` -> the request(s) the sweep makes. A route with no entry
#: is not swept, and ``test_the_sweep_called_every_route`` fails the build
#: for it. Nothing passes :data:`ACTIVE_ID` in, so a route echoing its own
#: argument can never be read as having reached the corpus.
INVOCATIONS: dict[str, tuple[dict[str, Any], ...]] = {
    "GET /status": ({},),
    "GET /memories": (
        {},
        {"params": {"active_only": "true", "limit": "1000"}},
        {"params": {"active_only": "false", "limit": "1000"}},
    ),
    "GET /federation/conflicts": ({"params": {"limit": "100"}},),
    "GET /federation/vclock/": ({"tail": QUARANTINED_ID},),
    "POST /query": (
        {"body": {"query": "frost telemetry architecture", "limit": 10}},
        {"body": {"query": "frost telemetry architecture", "limit": 10, "active_only": False, "persona": "detailed"}},
    ),
    "POST /consolidate": ({"body": {"dry_run": True}},),
    "POST /walkthrough": ({"body": {"topic": "frost telemetry architecture", "limit": 10}},),
    "POST /clear": (
        {"body": {"rationale": "sweeping the http read surface for withheld content", "confirm": "yes-i-really-want-to-clear"}},
    ),
    "POST /federation/write": ({"body": {"block_id": QUARANTINED_ID, "agent_id": "sweep"}},),
    "POST /federation/resolve": ({"body": {"block_id": QUARANTINED_ID, "strategy": "last_writer_wins"}},),
    "DELETE /memories/": ({"tail": PENDING_ID},),
}


@pytest.fixture(scope="module")
def sweep(seed_template: str) -> dict[str, dict[str, Any]]:
    """Call every route, each invocation in its own workspace and server."""
    results: dict[str, dict[str, Any]] = {}
    by_name = {route.name: route for route in ROUTES}
    for name in sorted(INVOCATIONS):
        route = by_name.get(name)
        if route is None:  # a stale row; the coverage test reports it
            continue
        outputs: list[tuple[dict, int, str]] = []
        for invocation in INVOCATIONS[name]:
            workspace = _fresh(seed_template)
            try:
                status, body = _call(workspace, route, invocation)
            finally:
                shutil.rmtree(workspace, ignore_errors=True)
            outputs.append((invocation, status, json.dumps(body, default=str)))
        blob = "\n".join(text for _, _, text in outputs)
        # Naming a withheld block's id is a weaker disclosure than serving its
        # text, and a different one: it says the block exists. Measured only
        # over invocations that did NOT pass that id in, so an echo of the
        # caller's own argument never counts -- four routes here are given a
        # withheld id on purpose.
        unprompted = "\n".join(
            text for invocation, _s, text in outputs if not any(wid in str(invocation) for wid in (PENDING_ID, QUARANTINED_ID))
        )
        results[name] = {
            "outputs": outputs,
            "leaked": sorted(status for status in ("pending", "quarantined") if CANARIES[status] in blob),
            "reached": CANARIES["active"] in blob,
            "names_withheld_ids": any(wid in unprompted for wid in (PENDING_ID, QUARANTINED_ID)),
            "refused": [
                (invocation, status, text) for invocation, status, text in outputs if _refused_before_the_handler(status, json.loads(text))
            ],
        }
    return results


# ---------------------------------------------------------------------------
# Positive controls on the seed
# ---------------------------------------------------------------------------


def test_the_withheld_canaries_really_are_in_the_corpus(seed_template: str) -> None:
    """Without this, every "canary absent" assertion below is vacuous."""
    from mind_mem.block_parser import parse_file

    blocks = {b["_id"]: b for b in parse_file(os.path.join(seed_template, "decisions", "DECISIONS.md")) if b.get("_id")}
    for block_id, status in SEEDED:
        assert block_id in blocks, f"seed failed: {block_id} is not in the corpus"
        assert blocks[block_id]["Status"] == status
        assert CANARIES[status] in json.dumps(blocks[block_id]), f"seed failed: {block_id} carries no canary"

    from mind_mem.sqlite_index import DB_REL_PATH

    conn = sqlite3.connect(os.path.join(seed_template, DB_REL_PATH))
    try:
        rows = dict(conn.execute("SELECT id, status FROM blocks WHERE id IN (?,?,?)", (ACTIVE_ID, PENDING_ID, QUARANTINED_ID)).fetchall())
    finally:
        conn.close()
    assert rows == {ACTIVE_ID: "active", PENDING_ID: "pending", QUARANTINED_ID: "quarantined"}, f"index does not hold the seed: {rows}"


def test_the_seeded_statuses_are_the_ones_admission_withholds() -> None:
    """The seed must exercise the gate, not sit on the servable side of it."""
    from mind_mem.admissibility import is_admissible_status

    assert is_admissible_status("active")
    assert not is_admissible_status("pending")
    assert not is_admissible_status("quarantined")


def test_the_canary_is_retrievable_when_it_is_not_withheld(seed_template: str) -> None:
    """The positive control the whole sweep rests on.

    Same block, same canary token, same route — only the status differs.
    Flip the quarantined block to ``active`` and its text comes back
    through ``GET /memories``. Without this, "the quarantined canary did
    not appear" would be equally consistent with a canary the endpoint
    could never have served in the first place.
    """
    workspace = _fresh(seed_template)
    try:
        path = os.path.join(workspace, "decisions", "DECISIONS.md")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        flipped = text.replace(
            f"[{QUARANTINED_ID}]\nDate: 2026-01-01\nStatus: quarantined",
            f"[{QUARANTINED_ID}]\nDate: 2026-01-01\nStatus: active",
        )
        assert flipped != text, "the corpus rewrite did not match; the control would prove nothing"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(flipped)
        status, body = _call(workspace, _route("GET /memories"), {})
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    blob = json.dumps(body)
    assert status == 200
    assert CANARIES["quarantined"] in blob, f"the canary is not retrievable even when admissible: {blob[:400]}"
    assert QUARANTINED_ID in blob


# ---------------------------------------------------------------------------
# The table and the module must agree, both ways
# ---------------------------------------------------------------------------


_MODULE_SOURCE = inspect.getsource(http_transport)
_MODULE_TREE = ast.parse(_MODULE_SOURCE)


def _route(name: str) -> Route:
    for route in ROUTES:
        if route.name == name:
            return route
    raise KeyError(name)


def _module_handlers() -> set[str]:
    """Every module-level ``_handle_*`` function, by AST, not by import.

    The same discovery shape the MCP tool-count gate uses: a static walk,
    so a handler that exists but is never imported anywhere is still seen.
    """
    return {node.name for node in _MODULE_TREE.body if isinstance(node, ast.FunctionDef) and node.name.startswith("_handle_")}


def test_every_handler_in_the_module_is_routed() -> None:
    """Tripwire A. A new endpoint fails the build until it is classified.

    This is the property the MCP registry sweep has and this transport did
    not: there, a tool that is registered but unclassified fails
    ``test_every_registered_tool_is_classified``. Here, a ``_handle_*``
    function that no ``Route`` names is either unreachable (dead weight,
    say so) or reachable through a dispatch path outside the table (the
    hole this table closes).
    """
    routed = {route.handler.__name__ for route in ROUTES}
    unrouted = sorted(_module_handlers() - routed)
    assert not unrouted, (
        f"handlers with no Route: {unrouted}. Add each to http_transport.ROUTES with a "
        f"verdict ({CONTENT!r} if its response can carry workspace block content, else "
        f"{NO_CONTENT!r}) and give it an entry in INVOCATIONS so the canary sweep exercises it."
    )


def test_the_route_table_names_no_handler_the_module_lost() -> None:
    """Tripwire B. A renamed or deleted handler cannot linger as a stale row."""
    routed = {route.handler.__name__ for route in ROUTES}
    ghosts = sorted(routed - _module_handlers())
    assert not ghosts, f"ROUTES names handlers this module does not define: {ghosts}"


def test_every_route_carries_one_of_the_two_verdicts() -> None:
    bad = {route.name: route.verdict for route in ROUTES if route.verdict not in (CONTENT, NO_CONTENT)}
    assert not bad, f"unknown verdicts: {bad}"


def test_a_route_cannot_be_constructed_without_a_valid_verdict() -> None:
    """The classification is enforced at import, not at review time.

    ``verdict`` has no default, so it cannot be omitted; and a value that
    is not one of the two states raises before the module finishes
    loading. Positive control for the two tripwires above: they check a
    table that cannot hold an unclassified row in the first place.

    ``mutates`` is the same shape for the same reason — see
    ``tests/test_governed_delete_http.py`` for the attribution half.
    """
    with pytest.raises(TypeError):
        Route("GET", "/x", http_transport._handle_status, "workspace")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Route("GET", "/x", http_transport._handle_status, "workspace", NO_CONTENT)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="verdict"):
        Route("GET", "/x", http_transport._handle_status, "workspace", "probably-fine", mutates=False)
    with pytest.raises(ValueError, match="takes"):
        Route("GET", "/x", http_transport._handle_status, "whatever", NO_CONTENT, mutates=False)


def test_no_two_routes_claim_the_same_method_and_path() -> None:
    names = [route.name for route in ROUTES]
    assert len(names) == len(set(names)), f"duplicate routes: {sorted({n for n in names if names.count(n) > 1})}"


def test_tripwire_a_fails_on_an_unrouted_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test-of-the-test: the tripwire must go red on exactly this."""
    real = _module_handlers()
    monkeypatch.setattr(sys.modules[__name__], "_module_handlers", lambda: real | {"_handle_leak_everything"})
    with pytest.raises(AssertionError, match="_handle_leak_everything"):
        test_every_handler_in_the_module_is_routed()


def test_tripwire_b_fails_on_a_ghost_row(monkeypatch: pytest.MonkeyPatch) -> None:
    real = _module_handlers()
    monkeypatch.setattr(sys.modules[__name__], "_module_handlers", lambda: real - {"_handle_status"})
    with pytest.raises(AssertionError, match="_handle_status"):
        test_the_route_table_names_no_handler_the_module_lost()


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(INVOCATIONS))
def test_no_route_serves_withheld_content(name: str, sweep: dict[str, dict[str, Any]]) -> None:
    """The sweep. Every route, every invocation, no withheld canary."""
    result = sweep[name]
    if not result["leaked"]:
        return
    offending = [
        (invocation, status, text[:400]) for invocation, status, text in result["outputs"] if any(c in text for c in WITHHELD_CANARIES)
    ]
    raise AssertionError(f"{name} served withheld block content ({result['leaked']}): {offending}")


def test_the_sweep_reached_exactly_the_content_routes(sweep: dict[str, dict[str, Any]]) -> None:
    """Positive control on the sweep, and the check that keeps the table true.

    ``reached`` means the response carried the ACTIVE canary, and no
    invocation passes that block's id in, so reaching is not an echo.
    SET EQUALITY, not containment: a route that starts returning block
    content joins this set and fails until it is reclassified, and a
    ``content`` route that quietly stops reaching (a projection changed, a
    flag default flipped) fails rather than degrading into a canary check
    over an error string.
    """
    reached = {name for name, result in sweep.items() if result["reached"]}
    declared = content_routes()
    assert reached == declared, (
        f"HTTP read-surface classification disagrees with measured behaviour. "
        f"reached the corpus but classified {NO_CONTENT!r}: {sorted(reached - declared)}; "
        f"classified {CONTENT!r} but never reached the corpus: {sorted(declared - reached)}"
    )


def test_no_route_names_a_withheld_block_id_it_was_not_given(sweep: dict[str, dict[str, Any]]) -> None:
    """The weaker disclosure, pinned rather than ignored.

    None of these serves withheld TEXT — the sweep above covers that.
    Naming an id the caller did not supply is a smaller leak of the same
    kind: it tells a caller that a block it may not read exists. The set
    is committed as EMPTY and equality is asserted, so a route that starts
    naming withheld blocks fails here and somebody decides, rather than it
    arriving as a silent default. The MCP surface pins the same property
    with a non-empty set (``ID_DISCLOSING``); this transport has no route
    that needs it.
    """
    naming = sorted(name for name, result in sweep.items() if result["names_withheld_ids"])
    assert naming == [], f"routes naming a withheld block id they were not given: {naming}"


def test_the_sweep_called_every_route(sweep: dict[str, dict[str, Any]]) -> None:
    """A route nobody swept is a route nobody checked."""
    swept = set(sweep)
    declared = {route.name for route in ROUTES}
    assert swept == declared, (
        f"routes in ROUTES with no INVOCATIONS entry: {sorted(declared - swept)}; "
        f"INVOCATIONS rows naming no route: {sorted(swept - declared)}"
    )


def test_no_invocation_was_refused_before_its_handler_ran(sweep: dict[str, dict[str, Any]]) -> None:
    """A request the router or a guard refused is not a route the sweep checked.

    This is the HTTP form of the binding-error check on the MCP sweep. A
    404 from the route matcher, a 401, a 429 or a "federation feature
    disabled" 503 carries no canary, so it reads as a clean pass while
    measuring the guard rather than the surface. Absence of a finding is
    only evidence when the search actually ran.
    """
    broken = {name: result["refused"] for name, result in sweep.items() if result["refused"]}
    assert not broken, f"invocations refused before the handler ran: {broken}"


def test_the_refusal_detector_sees_a_request_that_never_reached_a_handler(seed_template: str) -> None:
    """Positive control for the check above, against real responses."""
    workspace = _fresh(seed_template)
    try:
        with _serve(workspace) as port:
            unrouted = _request(port, "GET", "/no-such-endpoint", None)
            wrong_method = _request(port, "OPTIONS", "/status", None)
            well_formed = _request(port, "GET", "/status", None)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    assert _refused_before_the_handler(*unrouted), f"an unrouted path was not detected: {unrouted}"
    assert _refused_before_the_handler(*wrong_method), f"a rejected method was not detected: {wrong_method}"
    assert not _refused_before_the_handler(*well_formed), f"a served request was flagged as refused: {well_formed}"


def test_the_sweep_catches_a_route_that_leaks(seed_template: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test-of-the-test: a deliberately leaking route, served for real.

    The route is added to the table the dispatcher reads, its handler
    returns the corpus raw — which is what ``GET /memories`` would have
    done the moment its projection was repaired — and the identical
    assertion the sweep uses must catch it. If it does not, every green
    row above means nothing.
    """

    def _handle_leak(workspace: str, _params: dict[str, str]) -> tuple[int, dict[str, Any]]:
        from mind_mem.block_parser import parse_file

        blocks = parse_file(os.path.join(workspace, "decisions", "DECISIONS.md"))
        return (200, {"blocks": blocks})

    leaky = Route("GET", "/leak", _handle_leak, "params", NO_CONTENT, mutates=False)
    monkeypatch.setattr(http_transport, "ROUTES", ROUTES + (leaky,))

    workspace = _fresh(seed_template)
    try:
        status, body = _call(workspace, leaky, {})
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    text = json.dumps(body, default=str)
    assert status == 200, f"the leak fixture never ran: {status} {text[:200]}"
    leaked = sorted(name for name in ("pending", "quarantined") if CANARIES[name] in text)
    assert leaked == ["pending", "quarantined"], f"the leak fixture did not leak; the sweep's detector is untested: {text[:400]}"
    # Positive control for the id-disclosure detector too: this route was
    # given no block id at all and names both withheld ones, so the empty
    # set asserted above is a measurement rather than a detector that
    # cannot fire.
    assert all(withheld_id in text for withheld_id in (PENDING_ID, QUARANTINED_ID)), "the id-disclosure detector is untested"


# ---------------------------------------------------------------------------
# GET /memories — the two defects, pinned
# ---------------------------------------------------------------------------


def _list_memories(workspace: str, **params: str) -> tuple[int, dict[str, Any]]:
    return _call(workspace, _route("GET /memories"), {"params": params} if params else {})


def test_the_listing_serves_the_admitted_block_and_withholds_the_other_two(seed_template: str) -> None:
    workspace = _fresh(seed_template)
    try:
        status, body = _list_memories(workspace)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    assert status == 200
    blob = json.dumps(body)
    for canary in WITHHELD_CANARIES:
        assert canary not in blob, f"GET /memories served withheld content: {blob[:400]}"
    assert CANARIES["active"] in blob, "positive control: the admitted block was not served either"
    assert [m["id"] for m in body["memories"]] == [ACTIVE_ID]
    assert body["withheld"] == 2, f"withheld should be 2, got {body.get('withheld')}"
    assert body["count"] == 1 and body["total"] == 1


def test_the_listing_does_not_report_the_corpus_size_as_the_total(seed_template: str) -> None:
    """The disclosure that was live before the fix, pinned.

    Measured across every route on the HEAD blob and on this tree with the
    same three-status seed: no route put a withheld block's *text* on the
    wire, and the only reason ``GET /memories`` did not is that its
    projection was emitting nulls. What it did put on the wire was the
    count — ``{"count": 3, "total": 3}`` over a corpus with one servable
    block — so any caller could read off how many blocks were being
    withheld from them and watch that number move.

    ``total`` is now the size of the admitted set. The withheld count is
    still reported, deliberately and under its own name: that is the
    shipped seam's decision (``ReadAdmission.withheld``, and
    ``export_memory``'s ``withheld_count``) — a count, never the ids,
    because a surface that named what it withheld would tell every caller
    which blocks exist, which is most of what withholding them was for.
    The difference is that it is now labelled rather than presented as the
    number of memories the caller may see.
    """
    workspace = _fresh(seed_template)
    try:
        _status, body = _list_memories(workspace, limit="1000")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    assert body["total"] == 1, f"total is not the admitted count: {body}"
    assert body["count"] == len(body["memories"]) == 1
    assert body["withheld"] == 2
    assert not any(withheld_id in json.dumps(body) for withheld_id in (PENDING_ID, QUARANTINED_ID)), "the listing names a withheld block id"


def test_the_listing_reports_the_withheld_count_even_when_it_is_zero(tmp_path: Path) -> None:
    """Silently short is the failure; a key that appears only sometimes is
    a key readers learn to ignore. Positive control for the count itself:
    a corpus with nothing withheld must say ``0``, not omit the field."""
    from mind_mem import sqlite_index
    from mind_mem.init_workspace import init

    workspace = str(tmp_path / "ws")
    init(workspace)
    with open(os.path.join(workspace, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write("\n" + _render(ACTIVE_ID, "active", CANARIES["active"]))
    sqlite_index.build_index(workspace)

    status, body = _list_memories(workspace)
    assert status == 200
    assert body["withheld"] == 0
    assert body["count"] == 1


def test_every_summary_carries_the_block_id(seed_template: str) -> None:
    """The plain bug: ``b.get("id")`` on rows keyed ``_id``.

    Measured before the fix: every summary was ``{"id": null, "type":
    null, "category": null, "subject": null, "timestamp": null}`` — 200 OK
    over a list of empty shapes, on every backend, for any corpus.
    """
    workspace = _fresh(seed_template)
    try:
        _status, body = _list_memories(workspace, limit="1000", active_only="false")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    assert body["memories"], "positive control: the listing returned no rows at all"
    for summary in body["memories"]:
        assert summary["id"], f"summary carries no id: {summary}"
        assert summary["subject"], f"summary carries no subject: {summary}"
        assert summary["timestamp"], f"summary carries no timestamp: {summary}"
    assert set(body["memories"][0]) == {"id", "type", "category", "subject", "timestamp"}, "the wire shape changed"


def test_active_only_false_does_not_widen_the_listing(seed_template: str) -> None:
    """``active_only`` is a caller's filter; admission is not negotiable.

    It defaults to ``false`` and the endpoint used to hand
    ``get_all(active_only=False)`` straight to the wire, so the default
    request was the widest possible read of the corpus.
    """
    workspace = _fresh(seed_template)
    try:
        _s1, permissive = _list_memories(workspace, active_only="false", limit="1000")
        _s2, strict = _list_memories(workspace, active_only="true", limit="1000")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    assert [m["id"] for m in permissive["memories"]] == [ACTIVE_ID]
    assert [m["id"] for m in strict["memories"]] == [ACTIVE_ID]
    for canary in WITHHELD_CANARIES:
        assert canary not in json.dumps(permissive)


def test_the_listing_follows_the_corpus_when_the_index_is_stale(seed_template: str) -> None:
    """The stale-cache direction: a block quarantined AFTER it was indexed.

    The index caches ``status`` and goes stale fail-OPEN. ``admit_read``
    refreshes from the corpus, so the endpoint must follow the corpus and
    not the cache — otherwise quarantining a block would not take effect
    on this surface until someone remembered to reindex.
    """
    workspace = _fresh(seed_template)
    try:
        path = os.path.join(workspace, "decisions", "DECISIONS.md")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        flipped = text.replace(
            f"[{ACTIVE_ID}]\nDate: 2026-01-01\nStatus: active",
            f"[{ACTIVE_ID}]\nDate: 2026-01-01\nStatus: quarantined",
        )
        assert flipped != text, "the corpus rewrite did not match; the test would prove nothing"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(flipped)
        _status, body = _list_memories(workspace, limit="1000")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    assert body["memories"] == [], f"the listing served a block the corpus has quarantined: {body}"
    assert body["withheld"] == 3
    assert CANARIES["active"] not in json.dumps(body)


def _code_without_docstring(func: Any) -> str:
    """*func*'s source with its docstring removed.

    A source-text assertion that a word is absent trips over the
    docstring that explains why it is absent — a check that can only pass
    while nobody documents the rule is not a check worth having.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        node.body = node.body[1:]
    return ast.unparse(node)


def test_the_listing_has_no_bypass_parameter() -> None:
    """The decision, enforced rather than documented.

    A full-fidelity read of the corpus is ``snapshot()``. Re-adding an
    ``include_withheld``-shaped argument here would put the leak back
    behind a query parameter, so the signature is pinned — the same pin
    ``export_memory`` carries on the MCP side.
    """
    params = set(inspect.signature(http_transport._handle_list_memories).parameters)
    assert params == {"workspace", "params"}, f"_handle_list_memories grew a parameter: {sorted(params)}"
    # The query string is the other way in, so the code -- docstring
    # stripped, since it says the word on purpose -- must not read one.
    body = _code_without_docstring(http_transport._handle_list_memories)
    assert "include_withheld" not in body, "the handler reads a widening query parameter"


# ---------------------------------------------------------------------------
# One reader — the structural half
# ---------------------------------------------------------------------------

#: The functions allowed to read the store without admission, and why.
#: Both are on the DELETE path, where reaching a withheld block is the
#: whole point: a quarantined block an operator cannot delete is a block the
#: product cannot govern.
#:
#: ``_handle_delete_memory`` was a third entry, for an existence pre-check
#: it ran before opening the scope. The pre-check is gone — it answered
#: "does this id exist?" ahead of the gate and left no record of the
#: question — so the exemption went with it rather than being left here
#: naming code that no longer exists.
RAW_READ_ALLOWLIST: dict[str, str] = {
    "_admitted_blocks": "the seam itself — it is what applies admit_read",
    "_corpus_block_ids": "POST /clear must enumerate withheld blocks to delete them",
}

_RAW_READS = ("get_all", "get_by_id")


def _raw_store_readers(tree: ast.Module) -> dict[str, list[str]]:
    """``function name -> the raw store reads it makes``.

    Attribute calls, so ``store.get_all(...)`` is found wherever the store
    came from, and a helper that renames the local does not hide.
    """
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr in _RAW_READS:
                found.setdefault(node.name, []).append(inner.func.attr)
    return found


def test_only_the_delete_path_reads_the_store_without_admission() -> None:
    """Rule 1 of wiring discipline, made structural for this module.

    Any leg that reads blocks calls the shared admission filter. The three
    exceptions are enumerated with their reasons; a fourth fails the build
    rather than being noticed in review.
    """
    readers = _raw_store_readers(_MODULE_TREE)
    unexpected = sorted(set(readers) - set(RAW_READ_ALLOWLIST))
    assert not unexpected, (
        f"functions in http_transport reading the store without admission: "
        f"{ {name: readers[name] for name in unexpected} }. Read through _admitted_blocks, "
        f"or add the function to RAW_READ_ALLOWLIST with the reason it must see withheld blocks."
    )


def test_the_raw_read_scanner_finds_the_allowlisted_calls() -> None:
    """Positive control: an allowlist of ghosts would prove nothing.

    If the scanner could not see these three, the assertion above would be
    green over a module it never actually read.
    """
    readers = _raw_store_readers(_MODULE_TREE)
    missing = sorted(set(RAW_READ_ALLOWLIST) - set(readers))
    assert not missing, f"the scanner does not see the raw reads it allows: {missing}"


def test_the_raw_read_scanner_catches_a_new_raw_reader() -> None:
    """Test-of-the-test, on source the scanner has never seen."""
    fixture = ast.parse(
        "def _handle_new_listing(workspace, params):\n"
        "    store = get_block_store(workspace)\n"
        "    return (200, {'memories': store.get_all(active_only=False)})\n"
    )
    assert _raw_store_readers(fixture) == {"_handle_new_listing": ["get_all"]}


def test_the_listing_admits_through_the_shared_seam() -> None:
    """Not a hand-rolled status check — the same function recall uses."""
    source = inspect.getsource(http_transport._admitted_blocks)
    assert "admit_read(" in source, "_admitted_blocks does not call the shared read-admission seam"
    assert http_transport.admit_read.__module__ == "mind_mem.admission"
    assert "_admitted_blocks(" in inspect.getsource(http_transport._handle_list_memories)


def test_the_dispatcher_reads_the_route_table_and_nothing_else() -> None:
    """Reachability comes from the table, not from an ``if`` chain.

    The dispatcher rewrite is the by-construction half of this file: a
    handler that is not in ``ROUTES`` cannot be served, so "classified"
    and "reachable" are the same list by construction rather than by two
    lists agreeing.
    """
    source = inspect.getsource(http_transport.build_handler)
    assert "_match_route(" in source, "build_handler does not consult the route table"
    called_directly = sorted(name for name in _module_handlers() if name in source)
    assert not called_directly, f"build_handler names handlers directly: {called_directly}; the table is not the only dispatch path"
