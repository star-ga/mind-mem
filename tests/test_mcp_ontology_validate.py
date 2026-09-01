"""``ontology_validate`` — the MCP tool, as distinct from the ontology library.

``tests/test_ontology.py`` covers ``Ontology.validate`` thoroughly, and that
coverage is exactly why this tool slipped through: a reachability pass over the
98 registered MCP tools found ``ontology_validate`` with no test anywhere, while
the library underneath it looked well tested. The two are not the same contract.
The library returns a ``list[str]``; the tool has to decide what a *caller*
gets — which refusals are the caller's fault, what the JSON envelope looks like,
and whether the workspace and the active ontology are in a fit state to answer
at all. None of that was pinned.

So these tests exercise the tool's decisions rather than the validator's:

* The **verdict/refusal split**. An unknown ``type_name`` is a validation
  *verdict* (``valid: false`` with an ``errors`` entry), not a refusal — a
  caller that treats every non-empty ``errors`` list as "my block is wrong" and
  every ``error`` key as "my request is wrong" must not have those two swapped.
* The **envelope**, key by key. Callers branch on ``valid`` and read
  ``ontology_version`` to know *which* schema judged them; a silent rename is a
  silent break.
* The **refusal paths** — malformed JSON, a non-object, an oversized payload, a
  blank ``type_name`` — each of which returns a bare ``{"error": ...}`` and must
  never be mistaken for a passing verdict.
* The **workspace gate**, including its ordering: the gate runs before argument
  validation, so a bad workspace is reported even when the arguments are also
  bad. That ordering is a contract, not an accident — it stops a caller
  "fixing" its JSON in a workspace that was never going to answer.
* The **preload claim** in the module docstring: that validation works on a
  fresh workspace with no ``ontology_load`` step. That claim is load-bearing
  (it is why the tool is usable at all out of the box) and was untested.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools import _helpers
from mind_mem.mcp.tools.ontology import ontology_load, ontology_validate

# Every property below is inherited from ENTITY (name) or declared on PERSON
# (role) in the bundled se-1.0 ontology.
VALID_PERSON = {"name": "Ada", "role": "engineer"}


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Give back the shared MCP call budget this module spends.

    ``mcp_tool_observe`` enforces a per-client sliding window (120 calls / 60s)
    through a module-global ``rate_limit._rate_limiters``, and the whole test
    session shares one client id. Without this, a file that makes a few dozen
    calls quietly eats the budget and some LATER, unrelated test fails with
    "Rate limit exceeded" under CI's random ordering.
    """
    from mind_mem.mcp.infra import rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    yield
    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()


@pytest.fixture(autouse=True)
def _isolate_ontology_registry():
    """Restore the process-global ontology registry after each test.

    ``_ontology_registry`` is a lazy-init singleton shared by every tool in the
    process, and ``ontology_load(make_active=True)`` mutates it. Leaking a
    test-local ontology out of this file would leave a later test validating
    against the wrong schema — a failure that would look like a validator bug.
    """
    original = _helpers._ONTOLOGY_REGISTRY
    yield
    _helpers._ONTOLOGY_REGISTRY = original


@pytest.fixture
def ws(tmp_path):
    """A workspace shaped the way ``_check_workspace`` demands.

    On the default (Markdown/SQLite) backend the gate requires the local
    ``decisions/`` directory; without it every ws-gated tool fails closed
    before its body runs.
    """
    w = tmp_path / "ws"
    (w / "decisions").mkdir(parents=True)
    (w / ".mind-mem-index").mkdir(parents=True)
    return w


def _validate(w, **kwargs):
    with use_workspace(str(w)):
        return json.loads(ontology_validate(**kwargs))


class TestVerdictEnvelope:
    """What a caller gets back when the tool actually answers."""

    def test_a_conforming_block_is_valid_with_no_errors(self, ws) -> None:
        out = _validate(ws, block=json.dumps(VALID_PERSON), type_name="PERSON")
        assert out["valid"] is True
        assert out["errors"] == []

    def test_the_envelope_carries_exactly_the_documented_keys(self, ws) -> None:
        """Pinned as an exact set, not a subset.

        A caller cannot branch on a key that silently disappeared, and a new key
        appearing unannounced is a schema change that should bump
        ``_schema_version`` rather than arrive quietly.
        """
        out = _validate(ws, block=json.dumps(VALID_PERSON), type_name="PERSON")
        assert set(out) == {"valid", "errors", "type", "ontology_version", "_schema_version"}
        assert out["_schema_version"] == "1.0"
        assert out["type"] == "PERSON"

    def test_the_verdict_names_the_ontology_that_judged_it(self, ws) -> None:
        """``valid: true`` is meaningless without knowing which schema said so."""
        out = _validate(ws, block=json.dumps(VALID_PERSON), type_name="PERSON")
        assert out["ontology_version"] == "se-1.0"

    def test_type_is_echoed_even_when_the_type_does_not_exist(self, ws) -> None:
        out = _validate(ws, block=json.dumps(VALID_PERSON), type_name="NOSUCHTYPE")
        assert out["type"] == "NOSUCHTYPE"
        assert out["ontology_version"] == "se-1.0"


class TestValidationVerdicts:
    """Cases where the block is judged — the tool answers, the answer is 'no'."""

    def test_inherited_required_properties_are_enforced(self, ws) -> None:
        """PERSON declares only ``role``; ``name`` comes from its ENTITY parent.

        This is the whole point of the parent chain, and the one thing a caller
        would notice immediately if inheritance regressed to a flat lookup.
        """
        out = _validate(ws, block=json.dumps({"role": "engineer"}), type_name="PERSON")
        assert out["valid"] is False
        assert out["errors"] == ["missing required property: 'name'"]

    def test_a_present_but_empty_required_property_still_counts_as_missing(self, ws) -> None:
        """``{"name": ""}`` is not a filled-in name.

        Treating presence as satisfaction would let empty strings through the
        governance surface, so the emptiness check is deliberate — pin it.
        """
        out = _validate(ws, block=json.dumps({"name": "", "role": "engineer"}), type_name="PERSON")
        assert out["valid"] is False
        assert out["errors"] == ["missing required property: 'name'"]

    def test_declared_property_types_are_checked(self, ws) -> None:
        block = dict(VALID_PERSON, github=7)
        out = _validate(ws, block=json.dumps(block), type_name="PERSON")
        assert out["valid"] is False
        assert out["errors"] == ["type mismatch for 'github': expected str, got int"]

    def test_an_unknown_type_is_a_verdict_not_a_refusal(self, ws) -> None:
        """The distinction this tool's callers depend on most.

        A missing type is reported through ``errors`` with ``valid: false`` --
        NOT through the ``error`` key. A client that keys on ``error`` to mean
        "malformed request" must not start seeing it for "your ontology has no
        such type", and vice versa.
        """
        out = _validate(ws, block=json.dumps(VALID_PERSON), type_name="NOSUCHTYPE")
        assert out["valid"] is False
        assert out["errors"] == ["unknown type: 'NOSUCHTYPE'"]
        assert "error" not in out


class TestStrictness:
    """``strict`` is the only knob, so what it switches has to be exact."""

    def test_strict_is_the_default_and_rejects_unexpected_properties(self, ws) -> None:
        """Called with no ``strict`` argument at all — the default is the contract."""
        block = dict(VALID_PERSON, nickname="A")
        out = _validate(ws, block=json.dumps(block), type_name="PERSON")
        assert out["valid"] is False
        assert out["errors"] == ["unexpected property: 'nickname'"]

    def test_non_strict_permits_extra_properties(self, ws) -> None:
        block = dict(VALID_PERSON, nickname="A")
        out = _validate(ws, block=json.dumps(block), type_name="PERSON", strict=False)
        assert out["valid"] is True
        assert out["errors"] == []

    def test_non_strict_still_enforces_required_and_types(self, ws) -> None:
        """Relaxing 'extra' must not relax everything.

        ``strict=False`` is the mode an agent reaches for when its blocks carry
        incidental metadata; if it also stopped catching missing required
        properties it would be indistinguishable from not validating at all.
        """
        out = _validate(
            ws,
            block=json.dumps({"role": "engineer", "tags": "not-a-list", "extra": 1}),
            type_name="PERSON",
            strict=False,
        )
        assert out["valid"] is False
        assert "missing required property: 'name'" in out["errors"]
        assert "type mismatch for 'tags': expected list, got str" in out["errors"]
        assert not any("extra" in e for e in out["errors"])

    def test_framework_private_underscore_fields_survive_strict_mode(self, ws) -> None:
        """Blocks come back from recall carrying ``_id``/``_score``.

        If strict mode rejected those, the obvious workflow -- recall a block,
        validate it -- would fail on every block the product itself produced.
        """
        block = dict(VALID_PERSON, _id="DEC-20200101-001", _score=0.5)
        out = _validate(ws, block=json.dumps(block), type_name="PERSON")
        assert out["valid"] is True
        assert out["errors"] == []


class TestRefusals:
    """Bad arguments get an ``error`` envelope — never a passing verdict."""

    @pytest.mark.parametrize(
        ("block", "expected"),
        [
            ("", "block must be a non-empty JSON string"),
            ("   ", "block must be a non-empty JSON string"),
            (None, "block must be a non-empty JSON string"),
            (123, "block must be a non-empty JSON string"),
            ("[1, 2]", "block must decode to a JSON object"),
            ('"a string"', "block must decode to a JSON object"),
            ("null", "block must decode to a JSON object"),
        ],
    )
    def test_block_must_be_a_non_empty_json_object_string(self, ws, block, expected) -> None:
        out = _validate(ws, block=block, type_name="PERSON")
        assert out["error"] == expected
        assert "valid" not in out, "a refusal must not be readable as a verdict"

    def test_malformed_json_is_refused_with_the_parser_reason(self, ws) -> None:
        """The decoder's message is forwarded so the caller can fix the payload."""
        out = _validate(ws, block="{not json", type_name="PERSON")
        assert out["error"].startswith("block is not valid JSON: ")
        assert "valid" not in out

    def test_an_oversized_block_is_refused_before_it_is_parsed(self, ws) -> None:
        """A 1 MiB ceiling on the raw string, not on the decoded object.

        Refusing on the string length is what keeps a hostile payload from being
        parsed at all; a size check after ``json.loads`` would have already paid
        the cost it exists to avoid.
        """
        oversized = json.dumps({"name": "a" * 1_100_000})
        assert len(oversized) > 1_048_576
        out = _validate(ws, block=oversized, type_name="PERSON")
        assert out["error"] == "block must be ≤1 MiB"

    @pytest.mark.parametrize("type_name", ["", "   ", None, 42])
    def test_type_name_must_be_a_non_empty_string(self, ws, type_name) -> None:
        out = _validate(ws, block=json.dumps(VALID_PERSON), type_name=type_name)
        assert out["error"] == "type_name must be a non-empty string"

    def test_a_blank_type_name_is_refused_rather_than_reported_as_unknown(self, ws) -> None:
        """Contrast with ``NOSUCHTYPE``, which is a verdict.

        An absent argument is the caller's bug; a name the ontology happens not
        to declare is a data answer. Collapsing the two would make a typo'd
        argument look like a legitimate 'no such type' result.
        """
        blank = _validate(ws, block=json.dumps(VALID_PERSON), type_name="  ")
        unknown = _validate(ws, block=json.dumps(VALID_PERSON), type_name="NOSUCHTYPE")
        assert "error" in blank and "valid" not in blank
        assert "valid" in unknown and "error" not in unknown

    def test_refusal_envelopes_are_bare_error_objects(self, ws) -> None:
        """Pins observed behaviour, and it is asymmetric with the success path.

        Verdicts carry ``_schema_version``; refusals do not. That is the shape
        callers must code against today, so it is what this asserts -- but it
        means an error envelope cannot be version-negotiated the way a verdict
        can. Worth revisiting in a schema bump; not something to change under a
        test that merely describes it.
        """
        out = _validate(ws, block="[1, 2]", type_name="PERSON")
        assert set(out) == {"error"}


class TestWorkspaceGate:
    def test_a_missing_workspace_is_refused(self, tmp_path) -> None:
        missing = tmp_path / "nope"
        out = _validate(missing, block=json.dumps(VALID_PERSON), type_name="PERSON")
        assert out["error"] == "Workspace not found. Run: mind-mem-init <path>"

    def test_an_uninitialised_workspace_is_refused(self, tmp_path) -> None:
        """A directory is not a workspace until ``mind-mem-init`` has shaped it."""
        bare = tmp_path / "bare"
        bare.mkdir()
        out = _validate(bare, block=json.dumps(VALID_PERSON), type_name="PERSON")
        assert out["error"] == ("Workspace is missing the 'decisions/' directory. Run: mind-mem-init <path>")

    def test_the_workspace_gate_runs_before_argument_validation(self, tmp_path) -> None:
        """Ordering is part of the contract, not an implementation detail.

        With both a bad workspace and a bad block, the caller hears about the
        workspace. The reverse order would send someone off fixing their JSON
        for a workspace that was never going to answer.
        """
        bare = tmp_path / "bare"
        bare.mkdir()
        out = _validate(bare, block="{not json", type_name="")
        assert out["error"].startswith("Workspace is missing")


class TestActiveOntology:
    def test_validation_works_on_a_fresh_process_with_no_load_step(self, ws) -> None:
        """The module docstring's standing claim, pinned.

        ``_ontology_registry`` preloads the in-box software-engineering ontology
        on first touch precisely so a caller need not run ``ontology_load``
        first. Dropping ``_ONTOLOGY_REGISTRY`` to ``None`` reproduces a
        never-touched process.
        """
        _helpers._ONTOLOGY_REGISTRY = None
        out = _validate(ws, block=json.dumps(VALID_PERSON), type_name="PERSON")
        assert out["ontology_version"] == "se-1.0"
        assert out["valid"] is True

    def test_with_no_active_ontology_the_tool_says_so_instead_of_passing(self, ws, monkeypatch) -> None:
        """The one state the preload is meant to make unreachable.

        It is still reachable through the registry API (``OntologyRegistry()``
        starts empty), and the failure mode if it were unhandled would be an
        ``AttributeError`` on ``None`` -- a crash mid-session rather than a
        structured answer. Pin the refusal.
        """
        import mind_mem.mcp.tools.ontology as ontology_tools
        from mind_mem.ontology import OntologyRegistry

        monkeypatch.setattr(ontology_tools, "_ontology_registry", OntologyRegistry)
        out = _validate(ws, block=json.dumps(VALID_PERSON), type_name="PERSON")
        assert out["error"] == "no active ontology; call ontology_load first"

    def test_a_loaded_ontology_redirects_validation(self, ws) -> None:
        """The documented ``ontology_load(make_active=True)`` handoff.

        Both directions matter: the new types become validatable AND the old
        ones stop being, because 'active' means exactly one ontology answers.
        """
        spec = json.dumps(
            {
                "version": "widgets-1.0",
                "types": {"WIDGET": {"name": "WIDGET", "required": ["sku"], "property_types": {"sku": "str"}}},
            }
        )
        with use_workspace(str(ws)):
            loaded = json.loads(ontology_load(spec=spec, make_active=True))
        assert loaded["loaded"] is True and loaded["active"] is True

        widget = _validate(ws, block=json.dumps({"sku": "W-1"}), type_name="WIDGET")
        assert widget["valid"] is True
        assert widget["ontology_version"] == "widgets-1.0"

        person = _validate(ws, block=json.dumps(VALID_PERSON), type_name="PERSON")
        assert person["valid"] is False
        assert person["errors"] == ["unknown type: 'PERSON'"]
        assert person["ontology_version"] == "widgets-1.0"

    def test_loading_without_make_active_leaves_the_verdict_authority_alone(self, ws) -> None:
        """Loading is not activating; a stray load must not silently retarget."""
        spec = json.dumps({"version": "widgets-2.0", "types": {"WIDGET": {"name": "WIDGET"}}})
        with use_workspace(str(ws)):
            ontology_load(spec=spec)
        out = _validate(ws, block=json.dumps(VALID_PERSON), type_name="PERSON")
        assert out["ontology_version"] == "se-1.0"
        assert out["valid"] is True


class TestReachability:
    """Registered-but-unclassified is unreachable, which is how tools go dark."""

    def test_the_tool_is_registered_on_the_ontology_family(self) -> None:
        registered: list[str] = []

        class _Mcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        from mind_mem.mcp.tools.ontology import register

        register(_Mcp())
        assert "ontology_validate" in registered

    def test_the_tool_is_user_scoped_not_admin(self) -> None:
        """Validation reads a process-local schema and touches no store.

        Admin-scoping it would break the ordinary agent workflow for no gain;
        leaving it out of BOTH sets would make it unreachable at every scope.
        """
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "ontology_validate" in USER_TOOLS
        assert "ontology_validate" not in ADMIN_TOOLS

    def test_it_answers_under_the_default_user_scope(self, ws, monkeypatch) -> None:
        """The classification, checked behaviourally rather than by set membership."""
        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        out = _validate(ws, block=json.dumps(VALID_PERSON), type_name="PERSON")
        assert out["valid"] is True
