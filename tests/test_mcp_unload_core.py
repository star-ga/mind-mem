"""``unload_core`` — the MCP tool that had no test anywhere.

``CoreRegistry.unload`` is well covered in ``tests/test_context_core.py``, and
that coverage is exactly why nobody noticed: the *library* method was tested,
the *tool* that agents actually call was not. The tool is not a thin alias —
between an MCP caller and ``CoreRegistry.unload`` sit a workspace gate, a
type/emptiness check, a ``strip()``, a boolean-coercing JSON envelope, an ACL
classification, and a registration call. Every one of those can rot without a
single ``test_context_core`` assertion changing colour.

What these tests pin, in the order the tool applies them:

* the workspace gate fires **first** — before argument validation — and is a
  real gate: a refused call must not mutate the registry;
* a bad namespace is refused with a stable message and no ``unloaded`` key, so
  a caller can never read a refusal as a successful no-op;
* unloading returns ``unloaded: true`` only when something was actually
  removed, ``false`` when the namespace was not mounted — a "not found" is a
  fact to report, not an error to raise, and the second unload of the same
  namespace must not keep saying ``true``;
* the envelope is one boolean key, and *not* the ``_schema_version`` shape its
  three sibling core tools return (see ``TestEnvelope`` — recorded as actual
  behaviour, not endorsed);
* the tool is reachable at all: registered on the core family and classified in
  the ACL as user scope, which is what makes it callable without admin.

One property is documented rather than asserted-as-desirable: the registry
behind the tool is process-global, not workspace-keyed, so the workspace gate
checks that *a* valid workspace is active and nothing more. That is real,
observable, and worth a test that will notice if it ever changes.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.core import build_core, list_cores, load_core, unload_core


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Return the shared MCP call budget this module spends.

    ``mcp_tool_observe`` enforces 120 calls / 60s per client through a
    module-global registry, and the whole test session shares one client id.
    This file makes several calls per test, so without giving the budget back
    the failure lands in some later, unrelated module under random ordering.
    """
    from mind_mem.mcp.infra import rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    yield
    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()


@pytest.fixture(autouse=True)
def _fresh_core_registry():
    """Hand every test an empty core registry, and leave one behind.

    ``_helpers._core_registry`` is a lazily-built process singleton shared by
    build/load/unload/list. A core mounted by one test would otherwise still be
    mounted in the next one — and, worse, in unrelated modules that call
    ``list_cores``. This is the same class of shared-state leak as the rate
    limiter above, so it gets the same treatment.
    """
    from mind_mem.mcp.tools import _helpers

    _helpers._CORE_REGISTRY = None
    yield
    _helpers._CORE_REGISTRY = None


@pytest.fixture
def ws(tmp_path):
    """A workspace that passes the gate on the default (Markdown) backend."""
    return _make_ws(tmp_path / "ws")


def _make_ws(path):
    (path / ".mind-mem-index").mkdir(parents=True)
    # _check_workspace requires the Markdown corpus layout on the default
    # backend; without decisions/ every ws-gated tool fails closed.
    (path / "decisions").mkdir(parents=True)
    return path


def _mount(w, namespace: str, version: str = "1.0") -> None:
    """Build and load a real .mmcore so the registry holds a genuine core."""
    with use_workspace(str(w)):
        built = json.loads(build_core(namespace, version))
        assert "error" not in built, built
        loaded = json.loads(load_core(f"{namespace}-{version}.mmcore"))
        assert loaded["loaded"] is True, loaded


def _unload(w, namespace) -> dict:
    with use_workspace(str(w)):
        return json.loads(unload_core(namespace))


def _mounted(w) -> list[str]:
    with use_workspace(str(w)):
        return [c["namespace"] for c in json.loads(list_cores())["cores"]]


class TestUnloadContract:
    def test_a_mounted_core_is_reported_unloaded_and_actually_leaves(self, ws) -> None:
        """The return value has to agree with the registry, not just look right."""
        _mount(ws, "alpha")
        assert _mounted(ws) == ["alpha"]

        assert _unload(ws, "alpha") == {"unloaded": True}
        assert _mounted(ws) == []

    def test_an_unmounted_namespace_is_false_rather_than_an_error(self, ws) -> None:
        """Nothing-to-do is a fact the caller can act on, not a failure."""
        out = _unload(ws, "never-loaded")
        assert out == {"unloaded": False}
        assert "error" not in out

    def test_the_second_unload_of_the_same_namespace_reports_false(self, ws) -> None:
        """A registry that keeps answering 'true' would hide a double-unload bug."""
        _mount(ws, "alpha")
        assert _unload(ws, "alpha")["unloaded"] is True
        assert _unload(ws, "alpha")["unloaded"] is False

    def test_surrounding_whitespace_in_the_namespace_is_stripped(self, ws) -> None:
        """The tool strips before lookup; an agent-supplied ' alpha ' must hit."""
        _mount(ws, "alpha")
        assert _unload(ws, "  alpha\t")["unloaded"] is True
        assert _mounted(ws) == []

    def test_only_the_named_core_is_unloaded(self, ws) -> None:
        """Namespace-targeted, not 'clear everything mounted'."""
        _mount(ws, "alpha")
        _mount(ws, "beta")
        assert sorted(_mounted(ws)) == ["alpha", "beta"]

        assert _unload(ws, "alpha")["unloaded"] is True
        assert _mounted(ws) == ["beta"]

    def test_namespace_matching_is_exact_not_prefix(self, ws) -> None:
        _mount(ws, "alpha")
        assert _unload(ws, "alph")["unloaded"] is False
        assert _unload(ws, "alphabet")["unloaded"] is False
        assert _mounted(ws) == ["alpha"]


class TestEnvelope:
    def test_the_success_envelope_is_exactly_one_boolean_key(self, ws) -> None:
        """Pins the shape a caller parses — including where it differs.

        ``build_core``/``load_core``/``list_cores`` all stamp
        ``_schema_version: "1.0"``; ``unload_core`` does not. That asymmetry is
        recorded here as observed behaviour, not endorsed: if someone adds the
        stamp for consistency this test fails loudly and they can update the
        pin deliberately, which is the point of pinning an envelope at all.
        """
        _mount(ws, "alpha")
        out = _unload(ws, "alpha")
        assert set(out) == {"unloaded"}
        assert isinstance(out["unloaded"], bool)

    def test_the_miss_envelope_has_the_same_shape_as_the_hit(self, ws) -> None:
        """Hit and miss must be distinguishable by value, never by key set."""
        _mount(ws, "alpha")
        hit = _unload(ws, "alpha")
        miss = _unload(ws, "alpha")
        assert set(hit) == set(miss) == {"unloaded"}
        assert hit["unloaded"] is True
        assert miss["unloaded"] is False

    def test_the_result_is_always_a_json_string(self, ws) -> None:
        with use_workspace(str(ws)):
            raw = unload_core("alpha")
        assert isinstance(raw, str)
        assert json.loads(raw) == {"unloaded": False}


class TestRefusals:
    @pytest.mark.parametrize("bad", ["", "   ", "\t\n", None, 7, 0, True, [], {}, b"alpha"])
    def test_a_non_string_or_blank_namespace_is_refused(self, ws, bad) -> None:
        out = _unload(ws, bad)
        assert out == {"error": "namespace must be a non-empty string"}

    def test_a_refusal_carries_no_unloaded_key(self, ws) -> None:
        """A caller doing ``result.get('unloaded')`` must not read a refusal as False.

        Both are falsy in Python; only the absent key distinguishes 'I refused'
        from 'nothing was mounted'.
        """
        assert "unloaded" not in _unload(ws, "")

    def test_a_refused_call_leaves_the_registry_untouched(self, ws) -> None:
        """Validation must fail before it can do damage."""
        _mount(ws, "alpha")
        assert "error" in _unload(ws, "")
        assert _mounted(ws) == ["alpha"]


class TestWorkspaceGate:
    def test_a_missing_workspace_is_refused(self, tmp_path) -> None:
        out = _unload(tmp_path / "does-not-exist", "alpha")
        assert out == {"error": "Workspace not found. Run: mind-mem-init <path>"}

    def test_a_workspace_without_decisions_is_refused(self, tmp_path) -> None:
        half = tmp_path / "half"
        (half / ".mind-mem-index").mkdir(parents=True)
        out = _unload(half, "alpha")
        assert out["error"] == ("Workspace is missing the 'decisions/' directory. Run: mind-mem-init <path>")

    def test_the_workspace_gate_runs_before_argument_validation(self, tmp_path) -> None:
        """Order is observable, so pin it: an uninitialised workspace is the
        first thing the caller is told about, even when the arguments are also
        bad. Reversing the two checks would send an operator chasing the
        namespace instead of running mind-mem-init."""
        half = tmp_path / "half"
        (half / ".mind-mem-index").mkdir(parents=True)
        out = _unload(half, "")
        assert "decisions" in out["error"]
        assert "namespace" not in out["error"]

    def test_the_gate_is_real_it_blocks_the_unload(self, tmp_path) -> None:
        """A gate that returns an error but still unloads would be decorative."""
        good = _make_ws(tmp_path / "good")
        _mount(good, "alpha")

        bad = tmp_path / "bad"
        bad.mkdir()
        assert "error" in _unload(bad, "alpha")
        assert _mounted(good) == ["alpha"]


class TestRegistryScope:
    def test_a_core_mounted_under_one_workspace_unloads_from_another(self, tmp_path) -> None:
        """Observed behaviour, recorded deliberately.

        The registry behind these tools is a process-wide singleton keyed by
        namespace alone — it is not partitioned per workspace. So the workspace
        gate proves only that *some* initialised workspace is active; it does
        not scope which cores a call can reach. Whether that is the intended
        blast radius for a multi-workspace server is a design question this
        test does not answer. It exists so the answer cannot change silently.
        """
        a = _make_ws(tmp_path / "a")
        b = _make_ws(tmp_path / "b")
        _mount(a, "alpha")

        assert _mounted(b) == ["alpha"]
        assert _unload(b, "alpha")["unloaded"] is True
        assert _mounted(a) == []


class TestReachability:
    def test_the_tool_is_registered_on_the_core_family(self) -> None:
        """Unregistered is unreachable — the defect class this file exists for."""
        registered: list[str] = []

        class _Mcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        from mind_mem.mcp.tools.core import register

        register(_Mcp())
        assert "unload_core" in registered

    def test_the_tool_is_classified_as_user_scope(self) -> None:
        """Registered but unclassified is refused as an unknown tool by the ACL."""
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "unload_core" in USER_TOOLS
        assert "unload_core" not in ADMIN_TOOLS

    def test_it_is_callable_without_admin_scope(self, ws, monkeypatch) -> None:
        """The ACL classification has to hold through the live decorator, not
        just in the frozenset: user scope must reach the tool body."""
        monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
        monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
        _mount(ws, "alpha")
        assert _unload(ws, "alpha") == {"unloaded": True}


class TestPublicFacadeParity:
    """``core(action='unload')`` folds this tool into the compact surface.

    Two entry points to one behaviour is two things to keep in step, so the
    parity is asserted rather than assumed.
    """

    def test_the_facade_unload_matches_the_dedicated_tool(self, ws) -> None:
        from mind_mem.mcp.tools.public import core

        _mount(ws, "alpha")
        with use_workspace(str(ws)):
            out = json.loads(core("unload", namespace="alpha"))
        assert out == {"unloaded": True}
        assert _mounted(ws) == []

    def test_the_facade_forwards_the_namespace_refusal(self, ws) -> None:
        from mind_mem.mcp.tools.public import core

        with use_workspace(str(ws)):
            out = json.loads(core("unload"))
        assert out == {"error": "namespace must be a non-empty string"}
