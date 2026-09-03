# Copyright 2026 STARGA, Inc.
"""Both in-tree clients must only call operations the server actually serves.

This is the gate that gives ``sdk/spec/openapi.json`` a job. The artifact says
what the server serves; ``sdk/go/routes.go`` and ``sdk/js/src/routes.ts``
declare what each client calls; this module joins the two and fails when they
part company.

It exists because they had already parted company. Before the route tables
landed, both clients issued:

* ``GET /v1/recall`` with query parameters — the server serves ``POST`` with a
  JSON body, so every recall through either SDK was a 405;
* ``GET /v1/blocks/{id}`` — the server serves the singular
  ``/v1/block/{block_id}``, so every block fetch was a 404.

Nothing caught it. The Go suite asserted the path it sent matched the path it
meant to send, the JS suite did the same, and neither had ever been compared
with the server. Two clients agreeing with themselves is not a contract.

Why parse the route tables instead of the clients
-------------------------------------------------
The tables are small declarative files whose format this repository controls,
and each language's own suite proves its client actually issues the declared
route (``TestRoutes_MethodsUseTheDeclaredRoutes`` in Go, ``describe("route
table")`` in the JS suite). So the chain is: client behaviour → route table →
OpenAPI artifact → live app. Every link has a test, and this module is the
middle one. Parsing arbitrary client source with a regex would be the fragile
version of the same idea.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "sdk" / "spec" / "openapi.json"
GO_ROUTES = REPO_ROOT / "sdk" / "go" / "routes.go"
TS_ROUTES = REPO_ROOT / "sdk" / "js" / "src" / "routes.ts"
GO_SOURCES = sorted((REPO_ROOT / "sdk" / "go").glob("*.go"))
TS_SOURCES = sorted((REPO_ROOT / "sdk" / "js" / "src").glob("*.ts"))

_GO_ROUTE_RE = re.compile(r'Route\{Method:\s*"([A-Z]+)",\s*Path:\s*"([^"]+)"\}')
_TS_ROUTE_RE = re.compile(r'\{\s*method:\s*"([A-Z]+)",\s*path:\s*"([^"]+)"\s*\}')

#: What both clients are expected to call. Written out rather than derived so
#: a route silently disappearing from a table is a failure, not a smaller set
#: that still passes every "subset of the spec" assertion.
EXPECTED_ROUTES = {
    ("POST", "/v1/recall"),
    ("GET", "/v1/block/{block_id}"),
    ("GET", "/v1/contradictions"),
    ("GET", "/v1/health"),
    ("GET", "/v1/scan"),
}

#: The two forms the clients used to send. Neither has ever been served.
RETIRED_ROUTES = {
    ("GET", "/v1/recall"),
    ("GET", "/v1/blocks/{block_id}"),
}


def _parse(pattern: re.Pattern[str], path: Path) -> set[tuple[str, str]]:
    return {(m.group(1), m.group(2)) for m in pattern.finditer(path.read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def spec_operations() -> set[tuple[str, str]]:
    spec = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return {(method.upper(), path) for path, operations in spec["paths"].items() for method in operations}


@pytest.fixture(scope="module")
def go_routes() -> set[tuple[str, str]]:
    return _parse(_GO_ROUTE_RE, GO_ROUTES)


@pytest.fixture(scope="module")
def ts_routes() -> set[tuple[str, str]]:
    return _parse(_TS_ROUTE_RE, TS_ROUTES)


class TestParsersSeeSomething:
    """Positive controls. Every assertion below is worthless if these fail."""

    def test_go_table_parses_to_the_expected_size(self, go_routes: set[tuple[str, str]]) -> None:
        assert len(go_routes) == len(EXPECTED_ROUTES), f"parsed {sorted(go_routes)} from {GO_ROUTES}"

    def test_ts_table_parses_to_the_expected_size(self, ts_routes: set[tuple[str, str]]) -> None:
        assert len(ts_routes) == len(EXPECTED_ROUTES), f"parsed {sorted(ts_routes)} from {TS_ROUTES}"

    def test_spec_operations_are_populated(self, spec_operations: set[tuple[str, str]]) -> None:
        assert len(spec_operations) >= 10, sorted(spec_operations)

    def test_go_parser_would_see_a_retired_route(self) -> None:
        # Proof the pattern is not simply blind: hand it the exact literal the
        # old client carried and require a hit. Without this, the "no retired
        # route is declared" assertions below could pass on a broken regex.
        sample = 'x = Route{Method: "GET", Path: "/v1/blocks/{block_id}"}'
        assert _GO_ROUTE_RE.findall(sample) == [("GET", "/v1/blocks/{block_id}")]

    def test_ts_parser_would_see_a_retired_route(self) -> None:
        sample = 'export const X: Route = { method: "GET", path: "/v1/blocks/{block_id}" };'
        assert _TS_ROUTE_RE.findall(sample) == [("GET", "/v1/blocks/{block_id}")]


class TestClientsAgreeWithTheSpec:
    def test_go_routes_are_all_served(self, go_routes: set[tuple[str, str]], spec_operations: set[tuple[str, str]]) -> None:
        unserved = go_routes - spec_operations
        assert not unserved, f"sdk/go/routes.go declares operations the server does not serve: {sorted(unserved)}"

    def test_ts_routes_are_all_served(self, ts_routes: set[tuple[str, str]], spec_operations: set[tuple[str, str]]) -> None:
        unserved = ts_routes - spec_operations
        assert not unserved, f"sdk/js/src/routes.ts declares operations the server does not serve: {sorted(unserved)}"

    def test_both_clients_declare_the_same_surface(self, go_routes: set[tuple[str, str]], ts_routes: set[tuple[str, str]]) -> None:
        assert go_routes == ts_routes, (
            "the two clients no longer cover the same operations; "
            f"Go only: {sorted(go_routes - ts_routes)}, JS only: {sorted(ts_routes - go_routes)}"
        )

    def test_the_expected_surface_is_the_declared_surface(self, go_routes: set[tuple[str, str]]) -> None:
        assert go_routes == EXPECTED_ROUTES


class TestRetiredRoutesAreGone:
    def test_no_client_declares_a_retired_route(self, go_routes: set[tuple[str, str]], ts_routes: set[tuple[str, str]]) -> None:
        assert not (go_routes | ts_routes) & RETIRED_ROUTES

    def test_no_retired_route_is_served(self, spec_operations: set[tuple[str, str]]) -> None:
        # The other half of the same claim: these are absent from the clients
        # because the server never had them, not because someone tidied up.
        assert not spec_operations & RETIRED_ROUTES

    @pytest.mark.parametrize("source", GO_SOURCES + TS_SOURCES, ids=lambda p: p.name)
    def test_no_client_source_still_builds_the_plural_blocks_path(self, source: Path) -> None:
        # A route table is only authoritative if nothing bypasses it. The
        # plural path is a string literal, so a leftover hand-built URL would
        # still be visible here. Comments are stripped first so the retired
        # form can be *described* in prose without tripping the gate.
        text = source.read_text(encoding="utf-8")
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("//", "*", "/*")))
        assert "/v1/blocks" not in code, f"{source} still builds the retired plural path"
