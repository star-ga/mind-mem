"""``list_cores`` — the read side of the ``.mmcore`` lifecycle, previously unpinned.

``CoreRegistry`` itself is well covered (tests/test_context_core.py), and the
folded ``core("list")`` facade has one assertion — ``"cores" in env or "error"
in env`` — which passes on *either* branch and therefore proves nothing about
which one you got. Between those two, the MCP tool that actually answers "what
is mounted right now?" had no test at all: nothing checked its envelope, its
workspace gate, or that the numbers it reports match the bundle that was
loaded. A drifted key name or a swallowed gate would have shipped green.

What these tests pin, and why each one is worth a test:

* **The envelope is a listing, never a bare list.** ``{"cores": [...],
  "_schema_version": "1.0"}`` with those keys and nothing else. Callers branch
  on the shape.
* **An empty registry answers ``[]``, not an error.** "Nothing is mounted" is a
  fact, not a failure, and a caller that treats it as one will retry forever.
* **A refusal is distinguishable from an empty listing.** This is the sharp
  edge: both are "no cores came back", but only one means the store was
  consulted. So the refusal envelope must NOT carry a ``cores`` key, and the
  workspace gate must run *before* the registry — a mounted core must not leak
  through a refused call.
* **The reported counts are the loaded bundle's, not a restatement of the
  request.** ``blocks``/``edges``/``content_hash`` are cross-checked against
  what ``build_core`` wrote and what ``load_core`` returned.
* **Order is deterministic** (sorted by namespace) and **the registry is keyed
  by namespace**, so re-loading a namespace replaces the row rather than
  duplicating it.
* **ACL classification.** ``list_cores`` is read-only and user-scoped, so it
  must work at the default scope — but it must still fail closed on the
  ``"deny"`` sentinel that means authentication context was unavailable.

One deliberate non-fix is recorded in ``TestRegistryIsProcessWideNotPerWorkspace``:
the tool is workspace-*gated* but the registry behind it is process-global, so
a core mounted while workspace A was active is listed when workspace B is
active. ``CoreRegistry`` documents itself as "process-local", so this is
existing intended behaviour and the test pins it as-is rather than asserting a
per-workspace answer nothing in ``src/`` promises.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.core import build_core, list_cores, load_core, unload_core


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Give back the shared MCP call budget this module spends.

    ``mcp_tool_observe`` enforces a per-client sliding window (120 calls / 60s)
    through a module-global ``rate_limit._rate_limiters``, and every test in the
    session shares one client id. These tests make several calls apiece, so
    without this they quietly eat the budget and some LATER, unrelated test
    fails with "Rate limit exceeded" under a random test order.
    """
    from mind_mem.mcp.infra import rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    yield
    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Hand each test a private registry and put the process one back after.

    ``_helpers._CORE_REGISTRY`` is a lazily-built module singleton shared by
    every core tool in the process. Mounting into the real one would make these
    tests order-dependent among themselves and would leak mounted namespaces
    into the rest of the session.
    """
    from mind_mem.context_core import CoreRegistry
    from mind_mem.mcp.tools import _helpers

    saved = _helpers._CORE_REGISTRY
    _helpers._CORE_REGISTRY = CoreRegistry()
    try:
        yield _helpers._CORE_REGISTRY
    finally:
        _helpers._CORE_REGISTRY = saved


@pytest.fixture
def ws(tmp_path):
    """A workspace the ws-gate accepts.

    On the default (Markdown/SQLite) backend ``_check_workspace`` defines a
    valid workspace as one with a ``decisions/`` directory, so without it every
    ws-gated tool fails closed before reaching its own logic.
    """
    w = tmp_path / "ws"
    (w / "decisions").mkdir(parents=True)
    (w / ".mind-mem-index").mkdir(parents=True)
    return w


# --------------------------------------------------------------------------
# Call helpers — every tool call goes through the real decorated entry point
# (rate limit + ACL + observability), not the undecorated __wrapped__, because
# the gates are part of the contract being pinned.
# --------------------------------------------------------------------------


def _list_raw(w) -> str:
    with use_workspace(str(w)):
        return list_cores()


def _list(w) -> dict:
    return json.loads(_list_raw(w))


def _build(w, namespace: str, version: str, **kw) -> dict:
    with use_workspace(str(w)):
        return json.loads(build_core(namespace, version, **kw))


def _load(w, filename: str, **kw) -> dict:
    with use_workspace(str(w)):
        return json.loads(load_core(filename, **kw))


def _unload(w, namespace: str) -> dict:
    with use_workspace(str(w)):
        return json.loads(unload_core(namespace))


def _mount(w, namespace: str, version: str = "1.0.0") -> dict:
    """Build a bundle in *w* and mount it; returns the ``load_core`` envelope."""
    built = _build(w, namespace, version)
    assert "error" not in built, built
    return _load(w, f"{namespace}-{version}.mmcore")


class TestEnvelope:
    def test_an_empty_registry_lists_nothing_rather_than_failing(self, ws) -> None:
        """Nothing mounted is an answer, not an error — callers must not retry."""
        out = _list(ws)
        assert out["cores"] == []
        assert "error" not in out

    def test_top_level_keys_are_exactly_cores_and_schema_version(self, ws) -> None:
        assert set(_list(ws)) == {"cores", "_schema_version"}

    def test_schema_version_is_pinned_at_1_0(self, ws) -> None:
        assert _list(ws)["_schema_version"] == "1.0"

    def test_output_is_pretty_printed(self, ws) -> None:
        """The tool serialises with ``indent=2``; the reader is an agent transcript."""
        raw = _list_raw(ws)
        assert raw.startswith("{\n")
        assert '\n  "cores"' in raw


class TestListsWhatIsActuallyMounted:
    def test_a_mounted_core_appears_with_its_manifest_identity(self, ws) -> None:
        loaded = _mount(ws, "acme", "2.3.1")
        assert loaded["loaded"] is True

        cores = _list(ws)["cores"]
        assert len(cores) == 1
        entry = cores[0]
        assert entry["namespace"] == "acme"
        assert entry["version"] == "2.3.1"
        assert entry["content_hash"] == loaded["content_hash"]

    def test_entry_keys_are_exactly_the_documented_five(self, ws) -> None:
        _mount(ws, "acme")
        assert set(_list(ws)["cores"][0]) == {
            "namespace",
            "version",
            "blocks",
            "edges",
            "content_hash",
        }

    def test_counts_come_from_the_bundle_not_from_the_request(self, ws) -> None:
        """``blocks``/``edges`` must be what was loaded, not what was asked for.

        The fixture workspace has an empty index and no knowledge graph, so the
        honest answer here is zero on both — and it has to agree with the count
        ``load_core`` reported for the same bundle.
        """
        loaded = _mount(ws, "acme")
        entry = _list(ws)["cores"][0]
        assert entry["blocks"] == loaded["blocks"] == 0
        assert entry["edges"] == loaded["edges"] == 0

    def test_content_hash_matches_the_manifest_written_at_build_time(self, ws) -> None:
        built = _build(ws, "acme", "1.0.0")
        _load(ws, "acme-1.0.0.mmcore")
        assert _list(ws)["cores"][0]["content_hash"] == built["manifest"]["content_hash"]

    def test_unloading_removes_the_row(self, ws) -> None:
        _mount(ws, "acme")
        assert [c["namespace"] for c in _list(ws)["cores"]] == ["acme"]

        assert _unload(ws, "acme")["unloaded"] is True
        assert _list(ws)["cores"] == []

    def test_listing_is_sorted_by_namespace_not_by_load_order(self, ws) -> None:
        """Two agents comparing listings must see the same order."""
        _mount(ws, "zeta")
        _mount(ws, "alpha")
        _mount(ws, "mid")
        assert [c["namespace"] for c in _list(ws)["cores"]] == ["alpha", "mid", "zeta"]

    def test_reloading_a_namespace_replaces_the_row_rather_than_duplicating_it(self, ws) -> None:
        """The registry is keyed by namespace, so the newest mount wins."""
        _mount(ws, "acme", "1.0.0")
        _mount(ws, "acme", "2.0.0")

        cores = _list(ws)["cores"]
        assert len(cores) == 1
        assert cores[0]["version"] == "2.0.0"

    def test_a_bundle_that_failed_to_load_is_not_listed(self, ws) -> None:
        """A refused load must leave no half-mounted trace behind."""
        failed = _load(ws, "does-not-exist.mmcore")
        assert "error" in failed
        assert _list(ws)["cores"] == []


class TestWorkspaceGate:
    def test_a_missing_workspace_is_refused_with_the_init_hint(self, tmp_path) -> None:
        out = _list(tmp_path / "nope")
        assert out == {"error": "Workspace not found. Run: mind-mem-init <path>"}

    def test_a_workspace_without_decisions_is_refused(self, tmp_path) -> None:
        bare = tmp_path / "bare"
        (bare / ".mind-mem-index").mkdir(parents=True)
        out = _list(bare)
        assert "error" in out
        assert "decisions/" in out["error"]

    def test_a_refusal_carries_no_cores_key(self, tmp_path) -> None:
        """The whole point: "refused" must not be readable as "nothing mounted".

        Both outcomes are "no cores came back", but only one of them means the
        registry was consulted. A caller doing ``env.get("cores", [])`` would
        silently turn a broken workspace into a confident empty answer.
        """
        assert "cores" not in _list(tmp_path / "nope")

    def test_the_gate_runs_before_the_registry_is_consulted(self, ws, tmp_path) -> None:
        """A mounted core must not leak out through a refused workspace."""
        _mount(ws, "secret-ns")
        assert _list(ws)["cores"][0]["namespace"] == "secret-ns"

        refused = _list_raw(tmp_path / "nope")
        assert "secret-ns" not in refused
        assert json.loads(refused)["error"].startswith("Workspace not found")


class TestAccessControl:
    def test_the_tool_is_registered_on_the_core_family(self) -> None:
        """A tool nothing registers is unreachable however well it behaves."""
        registered: list[str] = []

        class _Mcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        from mind_mem.mcp.tools.core import register

        register(_Mcp())
        assert "list_cores" in registered

    def test_it_is_classified_user_scope_not_admin(self) -> None:
        """Read-only listing behind an admin gate would be a usability bug;
        unclassified entirely would make it uncallable at any scope."""
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "list_cores" in USER_TOOLS
        assert "list_cores" not in ADMIN_TOOLS

    def test_it_answers_at_the_default_user_scope(self, ws, monkeypatch) -> None:
        monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
        _mount(ws, "acme")
        assert _list(ws)["cores"][0]["namespace"] == "acme"

    def test_it_still_answers_for_an_admin_caller(self, ws, monkeypatch) -> None:
        monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
        _mount(ws, "acme")
        assert _list(ws)["cores"][0]["namespace"] == "acme"

    def test_an_unavailable_auth_context_fails_closed_even_for_a_read(self, ws, monkeypatch) -> None:
        """``"deny"`` is the sentinel for "token introspection raised".

        It has to block user-scoped reads too, otherwise a transient
        introspection failure degrades to plain "user" and the mounted-core
        inventory is served to an unauthenticated caller.
        """
        _mount(ws, "acme")
        monkeypatch.setattr(
            "mind_mem.mcp.infra.observability._get_request_scope",
            lambda: "deny",
        )
        out = _list(ws)
        assert out["scope"] == "deny"
        assert out["error"] == "Permission denied: authentication context unavailable"
        assert "cores" not in out


class TestRegistryIsProcessWideNotPerWorkspace:
    def test_a_core_mounted_under_one_workspace_is_listed_under_another(self, tmp_path) -> None:
        """Pinned as-is: the gate is per-workspace, the registry is not.

        ``CoreRegistry`` documents itself as a *process-local* registry of
        mounted cores, and ``_core_registry()`` is a module singleton shared by
        every workspace this process serves. So switching workspaces does not
        change the answer. That is worth a test either way — if someone later
        makes the registry workspace-scoped, this test should be updated
        deliberately rather than discovered by an agent reading another
        workspace's inventory.
        """
        a = tmp_path / "a"
        b = tmp_path / "b"
        for w in (a, b):
            (w / "decisions").mkdir(parents=True)
            (w / ".mind-mem-index").mkdir(parents=True)

        _mount(a, "from-a")
        assert [c["namespace"] for c in _list(b)["cores"]] == ["from-a"]
