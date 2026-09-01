"""``ontology_load`` -- the MCP door onto the OWL-lite schema layer.

``tests/test_ontology.py`` is thorough about the *model*: ``EntityType``,
``Ontology.validate``, the registry. It never imports the MCP surface. So the
tool that actually exposes any of that to an agent -- the one that parses
attacker-shaped JSON, gates on the workspace, and decides what becomes the
schema every subsequent ``ontology_validate`` call is judged against -- had no
test anywhere. Everything it does beyond ``Ontology.from_dict`` was unpinned:
the whole input-validation ladder, the workspace refusal, the envelope, and
``make_active``.

``make_active`` is the part that matters. It is not a status flag; it rebinds
the ontology the *next* tool validates against, process-wide, for every client
of that server. So the test below does not assert ``"active": true`` and stop
there -- it loads a schema, promotes it, and then proves through
``ontology_validate`` that the promotion took, and that the previously active
in-box ontology is genuinely gone. An assertion on the boolean alone would pass
just as happily if ``load()`` dropped the ontology on the floor.

The refusal tests pin messages, not just "some error". The ladder is ordered
(size before parse, parse before shape, shape before ``from_dict``) and the
order is load-bearing: the 1 MiB cap exists to keep a hostile spec away from
``json.loads``, which a "some error key is present" assertion would not notice
had stopped happening.

One genuine hole is pinned as-is rather than as-it-should-be, in
``TestNonMappingTypesEscapesTheEnvelope`` -- see that class.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.ontology import ontology_load, ontology_validate


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Give back the shared MCP call budget this module spends.

    ``mcp_tool_observe`` enforces a per-client sliding window (120 calls / 60s)
    through a module-global ``rate_limit._rate_limiters``, and every test in the
    session shares one client id. Spending the budget here is how an unrelated
    test three files away fails with "Rate limit exceeded" under CI's random
    ordering while this file stays green locally.
    """
    from mind_mem.mcp.infra import rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    yield
    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()


@pytest.fixture(autouse=True)
def _pristine_registry():
    """Isolate the process-global ontology registry around every test.

    ``_helpers._ONTOLOGY_REGISTRY`` is a lazy module singleton, so a test that
    calls ``ontology_load(..., make_active=True)`` permanently rebinds the
    active schema for the rest of the session. Dropping the singleton to None
    on both sides forces the next access to rebuild it with only the in-box
    ``se-1.0`` preloaded -- which is both what these tests assume as their
    starting state and what later test modules are entitled to find.
    """
    from mind_mem.mcp.tools import _helpers

    _helpers._ONTOLOGY_REGISTRY = None
    yield
    _helpers._ONTOLOGY_REGISTRY = None


@pytest.fixture(autouse=True)
def _user_scope(monkeypatch):
    """Run at plain user scope -- loading an ontology is not an admin act.

    Explicit rather than inherited: another module leaving MIND_MEM_SCOPE=admin
    set would otherwise make these pass for the wrong reason, and the point is
    that an ordinary caller can reach this tool.
    """
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)


@pytest.fixture
def ws(tmp_path):
    """A minimal valid workspace on the default Markdown/SQLite backend."""
    w = tmp_path / "ws"
    (w / "decisions").mkdir(parents=True)
    (w / ".mind-mem-index").mkdir(parents=True)
    with use_workspace(str(w)) as resolved:
        yield resolved


def _spec(version: str = "test-1.0", **types) -> str:
    """Serialise an ontology spec in the shape ``Ontology.from_dict`` wants."""
    return json.dumps({"version": version, "types": types})


WIDGET = {
    "name": "WIDGET",
    "required": ["sku"],
    "optional": ["color"],
    "property_types": {"sku": "str", "color": "str"},
}
ANVIL = {"name": "ANVIL", "required": ["mass"], "property_types": {"mass": "int"}}


class TestWorkspaceGate:
    """The tool refuses before parsing anything when the workspace is wrong.

    Both refusals name the remedy (``mind-mem-init``), and they are distinct
    messages because the two failures need different operator actions: one
    directory does not exist, the other exists but was never initialised.
    """

    def test_missing_workspace_is_refused(self, tmp_path):
        with use_workspace(str(tmp_path / "does-not-exist")):
            out = json.loads(ontology_load(_spec(WIDGET=WIDGET)))
        assert out == {"error": "Workspace not found. Run: mind-mem-init <path>"}

    def test_workspace_without_decisions_dir_is_refused(self, tmp_path):
        bare = tmp_path / "bare"
        bare.mkdir()
        with use_workspace(str(bare)):
            out = json.loads(ontology_load(_spec(WIDGET=WIDGET)))
        assert out == {"error": "Workspace is missing the 'decisions/' directory. Run: mind-mem-init <path>"}

    def test_workspace_gate_precedes_spec_validation(self, tmp_path):
        """A bad spec against a bad workspace reports the workspace.

        Ordering matters for the error to be actionable: telling a caller their
        JSON is malformed when the real problem is an uninitialised workspace
        sends them to debug the wrong thing.
        """
        with use_workspace(str(tmp_path / "does-not-exist")):
            out = json.loads(ontology_load("{ not json"))
        assert out["error"] == "Workspace not found. Run: mind-mem-init <path>"


class TestSpecRefusals:
    """Every malformed spec becomes a JSON envelope with an ``error`` key.

    The tool's return type is ``str``; it never raises for input it anticipates.
    These pin the exact messages so a refactor cannot quietly collapse nine
    distinguishable failures into one generic "bad spec".
    """

    def test_empty_spec_is_refused(self, ws):
        assert json.loads(ontology_load(""))["error"] == "spec must be a non-empty JSON string"

    def test_whitespace_only_spec_is_refused(self, ws):
        assert json.loads(ontology_load("   \n\t "))["error"] == "spec must be a non-empty JSON string"

    def test_non_string_spec_is_refused(self, ws):
        """MCP clients are not obliged to honour the type hint."""
        assert json.loads(ontology_load(123))["error"] == "spec must be a non-empty JSON string"

    def test_oversize_spec_is_refused_before_it_is_parsed(self, ws):
        """The 1 MiB cap is a guard, so it must fire ahead of ``json.loads``.

        The payload here is also invalid JSON. Getting the size message rather
        than the parse message is the evidence that a hostile multi-megabyte
        spec never reaches the parser.
        """
        out = json.loads(ontology_load("x" * 1_048_577))
        assert out["error"] == "spec must be ≤1 MiB"

    def test_spec_at_the_size_limit_is_not_refused_for_size(self, ws):
        """The cap is inclusive -- exactly 1 MiB is allowed through to parsing."""
        padding = "y" * (1_048_576 - len('{"version":"big-1.0","types":{},"pad":""}'))
        spec = '{"version":"big-1.0","types":{},"pad":"' + padding + '"}'
        assert len(spec) == 1_048_576
        out = json.loads(ontology_load(spec))
        assert out.get("loaded") is True
        assert out["version"] == "big-1.0"

    def test_malformed_json_is_refused_with_the_parser_reason(self, ws):
        out = json.loads(ontology_load("{nope"))
        assert out["error"].startswith("spec is not valid JSON: ")

    def test_json_array_spec_is_refused(self, ws):
        """Valid JSON, wrong top-level shape."""
        assert json.loads(ontology_load("[1, 2]"))["error"] == "spec must decode to a JSON object"

    def test_json_scalar_spec_is_refused(self, ws):
        assert json.loads(ontology_load('"just a string"'))["error"] == "spec must decode to a JSON object"


class TestInvalidOntologyRefusals:
    """Structurally valid JSON that is not a valid ontology.

    These exercise the ``except (ValueError, KeyError, TypeError)`` arm, whose
    job is to turn every ``Ontology.from_dict`` / ``__post_init__`` rejection
    into an envelope. The underlying reason is preserved in the message, which
    is the only thing that makes the refusal debuggable by the caller.
    """

    def test_missing_version_is_refused(self, ws):
        out = json.loads(ontology_load('{"types": {}}'))
        assert out["error"] == "invalid ontology: 'version'"

    def test_lowercase_type_name_is_refused(self, ws):
        """Type names are UPPER_SNAKE_CASE by construction."""
        out = json.loads(ontology_load('{"version": "v", "types": {"widget": {"name": "widget"}}}'))
        assert out["error"] == "invalid ontology: EntityType.name must be UPPER_SNAKE_CASE, got 'widget'"

    def test_key_not_matching_type_name_is_refused(self, ws):
        """The dict key and the declared name are one identity, not two."""
        out = json.loads(ontology_load('{"version": "v", "types": {"FOO": {"name": "BAR"}}}'))
        assert out["error"] == "invalid ontology: Ontology key 'FOO' does not match EntityType.name='BAR'"

    def test_dangling_parent_is_refused(self, ws):
        """A parent outside the ontology would make inheritance unresolvable."""
        out = json.loads(ontology_load('{"version": "v", "types": {"FOO": {"name": "FOO", "parent": "GHOST"}}}'))
        assert out["error"] == "invalid ontology: EntityType 'FOO' references unknown parent 'GHOST'"

    def test_property_required_and_optional_is_refused(self, ws):
        out = json.loads(ontology_load(_spec(FOO={"name": "FOO", "required": ["x"], "optional": ["x"]})))
        assert out["error"] == "invalid ontology: property cannot be both required and optional: ['x']"

    def test_scalar_in_place_of_a_type_body_is_refused(self, ws):
        """A string where a type declaration belongs raises TypeError, caught."""
        out = json.loads(ontology_load('{"version": "v", "types": {"FOO": "not-a-type-body"}}'))
        assert out["error"] == "invalid ontology: string indices must be integers, not 'str'"

    def test_a_refused_spec_does_not_reach_the_registry(self, ws):
        """Refusal means refusal -- nothing half-loaded is left behind."""
        from mind_mem.mcp.tools import _helpers

        ontology_load('{"version": "poison-1.0", "types": {"widget": {"name": "widget"}}}')
        assert "poison-1.0" not in _helpers._ontology_registry().versions()


class TestNonMappingTypesEscapesTheEnvelope:
    """A JSON array under ``types`` is the one bad spec that is NOT an envelope.

    ``from_dict`` calls ``data.get("types", {}).items()``, and the ``except``
    arm lists ValueError/KeyError/TypeError -- not AttributeError. So
    ``{"version": "v", "types": []}`` propagates out of the tool instead of
    returning the documented ``{"error": ...}`` string, and (not being a
    database error) the decorator's backstop re-raises it too.

    Pinned as observed, not as preferred. Widening the ``except`` is a one-word
    src change and this test is what would then have to be updated deliberately
    -- which is the point of writing it down rather than leaving the tool's only
    uncaught input path undiscovered. It is a crash on hostile input, so it is
    worth fixing; the shape of the fix is a maintainer's call, not a test's.
    """

    def test_types_as_array_raises_instead_of_returning_an_error_envelope(self, ws):
        with pytest.raises(AttributeError, match="'list' object has no attribute 'items'"):
            ontology_load('{"version": "v", "types": []}')

    def test_types_as_string_raises_the_same_way(self, ws):
        with pytest.raises(AttributeError):
            ontology_load('{"version": "v", "types": "WIDGET"}')


class TestSuccessEnvelope:
    """What a successful load promises the caller."""

    def test_envelope_has_exactly_the_documented_keys(self, ws):
        out = json.loads(ontology_load(_spec("test-1.0", WIDGET=WIDGET, ANVIL=ANVIL)))
        assert set(out) == {"loaded", "version", "types", "active", "_schema_version"}

    def test_envelope_values(self, ws):
        out = json.loads(ontology_load(_spec("test-1.0", WIDGET=WIDGET, ANVIL=ANVIL)))
        assert out["loaded"] is True
        assert out["version"] == "test-1.0"
        assert out["_schema_version"] == "1.0"

    def test_types_are_reported_sorted(self, ws):
        """``type_names()`` sorts, so the listing is stable across calls.

        Declaration order here is WIDGET-then-ANVIL; the response is not.
        """
        out = json.loads(ontology_load(_spec("test-1.0", WIDGET=WIDGET, ANVIL=ANVIL)))
        assert out["types"] == ["ANVIL", "WIDGET"]

    def test_an_ontology_with_no_types_still_loads(self, ws):
        """Empty is legal -- ``Ontology`` only requires a version."""
        out = json.loads(ontology_load('{"version": "empty-1.0", "types": {}}'))
        assert out["loaded"] is True
        assert out["types"] == []

    def test_loaded_ontology_is_retrievable_by_version(self, ws):
        """``loaded: true`` is a claim about the registry; check the registry."""
        from mind_mem.mcp.tools import _helpers

        ontology_load(_spec("test-1.0", WIDGET=WIDGET))
        stored = _helpers._ontology_registry().get("test-1.0")
        assert stored is not None
        assert stored.type_names() == ["WIDGET"]


class TestMakeActive:
    """``make_active`` rebinds what ``ontology_validate`` judges against.

    This is the tool's only side effect on other tools, so it is checked
    through ``ontology_validate`` rather than through the ``active`` boolean.
    """

    def test_default_load_does_not_displace_the_in_box_ontology(self, ws):
        out = json.loads(ontology_load(_spec("test-1.0", WIDGET=WIDGET)))
        assert out["active"] is False

    def test_default_load_leaves_ontology_validate_on_the_in_box_schema(self, ws):
        """The registry preloads ``se-1.0``; a passive load must not steal it."""
        ontology_load(_spec("test-1.0", WIDGET=WIDGET))
        checked = json.loads(ontology_validate(json.dumps({"name": "Ada", "role": "eng"}), "PERSON"))
        assert checked["ontology_version"] == "se-1.0"
        assert checked["valid"] is True

    def test_make_active_reports_active(self, ws):
        out = json.loads(ontology_load(_spec("test-1.0", WIDGET=WIDGET), make_active=True))
        assert out["active"] is True

    def test_make_active_rebinds_ontology_validate_to_the_new_schema(self, ws):
        """The promise, end to end: a type only the new ontology knows."""
        ontology_load(_spec("test-1.0", WIDGET=WIDGET, ANVIL=ANVIL), make_active=True)
        checked = json.loads(ontology_validate(json.dumps({"sku": "A-1", "color": "red"}), "WIDGET"))
        assert checked["ontology_version"] == "test-1.0"
        assert checked["valid"] is True
        assert checked["errors"] == []

    def test_promoted_schema_actually_enforces_its_own_property_types(self, ws):
        """Not just reachable -- in force. A wrong-typed property is caught."""
        ontology_load(_spec("test-1.0", WIDGET=WIDGET), make_active=True)
        checked = json.loads(ontology_validate(json.dumps({"sku": 7}), "WIDGET"))
        assert checked["valid"] is False
        assert checked["errors"] == ["type mismatch for 'sku': expected str, got int"]

    def test_promotion_displaces_the_previous_active_ontology(self, ws):
        """``se-1.0``'s types stop resolving once something else is active.

        The registry has exactly one active version, so promotion is a
        replacement, not an addition -- a caller relying on PERSON must load an
        ontology that declares it.
        """
        ontology_load(_spec("test-1.0", WIDGET=WIDGET), make_active=True)
        checked = json.loads(ontology_validate(json.dumps({"name": "Ada"}), "PERSON"))
        assert checked["valid"] is False
        assert checked["errors"] == ["unknown type: 'PERSON'"]

    def test_truthy_non_bool_make_active_is_coerced(self, ws):
        """``bool(make_active)`` normalises, and the envelope reports a real bool."""
        out = json.loads(ontology_load(_spec("test-1.0", WIDGET=WIDGET), make_active="yes"))
        assert out["active"] is True

    def test_reloading_the_active_version_replaces_it_in_place(self, ws):
        """Version string is the registry's identity key.

        Loading ``se-1.0`` overwrites the in-box ``se-1.0`` even without
        ``make_active``, because ``load()`` stores by version and the active
        pointer is that same version. ``active: true`` is therefore honest --
        the loaded ontology *is* the active one now -- but the caller has
        silently replaced the in-box schema without asking to. Worth knowing
        before naming an ontology after one that already exists.
        """
        out = json.loads(ontology_load(_spec("se-1.0", WIDGET=WIDGET)))
        assert out["active"] is True
        checked = json.loads(ontology_validate(json.dumps({"name": "Ada"}), "PERSON"))
        assert checked["errors"] == ["unknown type: 'PERSON'"]

    def test_loading_a_second_version_does_not_evict_the_first(self, ws):
        """The registry accumulates versions; only the active pointer moves."""
        from mind_mem.mcp.tools import _helpers

        ontology_load(_spec("test-1.0", WIDGET=WIDGET))
        ontology_load(_spec("test-2.0", ANVIL=ANVIL), make_active=True)
        assert _helpers._ontology_registry().versions() == ["se-1.0", "test-1.0", "test-2.0"]
        assert _helpers._ontology_registry().active().version == "test-2.0"


class TestReachability:
    """Registered and classified -- the two ways a tool becomes unreachable."""

    def test_tool_is_registered_on_the_ontology_family(self):
        registered = []

        class _Mcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        from mind_mem.mcp.tools.ontology import register

        register(_Mcp())
        assert "ontology_load" in registered

    def test_tool_is_user_scoped_not_admin(self):
        """Unclassified is unreachable at every scope, not merely unprivileged.

        User scope is the deliberate classification: the tool writes only to a
        process-local registry, never to the store or the corpus. Pinning it
        also catches an accidental promotion to ADMIN_TOOLS, which would break
        every ordinary caller.
        """
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "ontology_load" in USER_TOOLS
        assert "ontology_load" not in ADMIN_TOOLS

    def test_tool_is_callable_at_user_scope(self, ws):
        """The ACL classification, exercised rather than asserted."""
        out = json.loads(ontology_load(_spec("test-1.0", WIDGET=WIDGET)))
        assert out.get("loaded") is True
