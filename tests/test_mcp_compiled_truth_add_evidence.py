"""The one compiled-truth tool that WRITES, and the only one nothing tested.

``compiled_truth_add_evidence`` has been registered on the MCP surface since
v1.x and folded into the v3.2.0 ``compiled_truth`` dispatcher, but the
reachability pass found no test anywhere that calls it. Its two siblings are
read-only; this is the tool that appends to the append-only evidence trail and
recompiles the page, so a silent break here corrupts the record rather than
just returning a bad answer. That asymmetry is why it is worth real tests and
not a "returns a string" smoke test.

What these pin, in order of what actually matters:

* **The append-only + recompile contract.** Each call must add exactly one
  entry, leave every earlier entry byte-identical and un-superseded, bump the
  version by one, and regenerate ``compiled_section`` from the live evidence.
  A page that quietly drops or rewrites history is worse than no page.
* **The refusal envelope.** Bad input comes back as
  ``{"_schema_version", "error"}`` with no result keys, and — the part that
  actually protects the corpus — a refused call must not half-write: an
  existing page keeps its version and its evidence count.
* **The success envelope**, key for key, including ``path`` pointing at the
  file that was really written.
* **Its ACL classification.** The tool is USER-scoped by design (agents append
  observations); the test asserts both the classification and that a live call
  under an enforced ACL at default user scope is not refused. Promote it to
  ADMIN and both fail loudly instead of every agent silently losing the ability
  to record what it learned.

``TestUnvalidatedInput`` is deliberately different in kind: those are
CHARACTERIZATION tests of behaviour this tool has today and probably should not
(``entity_id`` reaches ``os.path.join`` unvalidated, and an observation
containing an evidence-trail header line reappears as a separate, forged
evidence entry on reload). Nothing in ``src/`` was changed to accommodate them.
They assert the behaviour that exists so that hardening the tool turns into a
red test with a note attached, rather than a silent change nobody reviews.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.kernels import compiled_truth_add_evidence, compiled_truth_load


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Give back the shared MCP call budget this module spends.

    ``mcp_tool_observe`` enforces a per-client sliding window (120 calls / 60s)
    through a module-global ``rate_limit._rate_limiters``, and every test in the
    session shares one client id. Spending it here makes a LATER, unrelated test
    fail with "Rate limit exceeded" under CI's random ordering.
    """
    from mind_mem.mcp.infra import rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    yield
    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()


@pytest.fixture(autouse=True)
def _enforced_user_scope(monkeypatch):
    """Run every call the way a plain agent runs it: ACL on, user scope.

    ``MIND_MEM_ACL_DISABLED`` leaking in from the environment would turn the
    ACL assertions below into tautologies, so it is removed rather than trusted.
    """
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """A workspace in the shape ``mind-mem-init`` leaves behind."""
    w = tmp_path / "ws"
    (w / "decisions").mkdir(parents=True)
    (w / ".mind-mem-index").mkdir(parents=True)
    return w


def _add(workspace: Path, **kwargs) -> dict:
    with use_workspace(str(workspace)):
        return json.loads(compiled_truth_add_evidence(**kwargs))


def _load(workspace: Path, entity_id: str) -> dict:
    with use_workspace(str(workspace)):
        return json.loads(compiled_truth_load(entity_id))


class TestSuccessEnvelope:
    """The documented shape of a successful add, key for key."""

    def test_envelope_carries_exactly_the_documented_keys(self, ws: Path) -> None:
        out = _add(ws, entity_id="PRJ-mind-mem", observation="5.0.0 ships the reachability gate.")

        assert set(out) == {"_schema_version", "entity_id", "version", "evidence_count", "path", "message"}
        assert out["_schema_version"] == "1.0"
        assert out["entity_id"] == "PRJ-mind-mem"
        assert out["version"] == 1
        assert out["evidence_count"] == 1
        assert out["message"] == "Evidence added and page recompiled (v1)."

    def test_reported_path_is_the_file_that_was_written(self, ws: Path) -> None:
        """``path`` is a claim about the disk; check the disk, not the claim."""
        out = _add(ws, entity_id="PRJ-mind-mem", observation="first observation")

        path = Path(out["path"])
        assert path == (ws / "entities" / "compiled" / "PRJ-mind-mem.md")
        assert path.is_file()

        text = path.read_text(encoding="utf-8")
        assert "entity_id: PRJ-mind-mem" in text
        assert "version: 1" in text
        assert "first observation" in text

    def test_first_call_creates_the_page_rather_than_refusing(self, ws: Path) -> None:
        """No page exists yet, so the tool is also the page constructor."""
        assert not (ws / "entities" / "compiled" / "NEW.md").exists()

        out = _add(ws, entity_id="NEW", observation="bootstrap")

        assert "error" not in out
        assert out["version"] == 1
        assert (ws / "entities" / "compiled" / "NEW.md").is_file()


class TestAppendOnlyAndRecompile:
    """The contract in the docstring: 'add evidence ... and auto-recompile'."""

    def test_each_call_adds_exactly_one_entry_and_bumps_the_version_by_one(self, ws: Path) -> None:
        first = _add(ws, entity_id="E", observation="obs-1")
        second = _add(ws, entity_id="E", observation="obs-2")
        third = _add(ws, entity_id="E", observation="obs-3")

        assert [first["version"], second["version"], third["version"]] == [1, 2, 3]
        assert [first["evidence_count"], second["evidence_count"], third["evidence_count"]] == [1, 2, 3]

    def test_earlier_evidence_survives_verbatim_and_unsuperseded(self, ws: Path) -> None:
        """Append-only means append-only: the trail is the audit surface."""
        _add(ws, entity_id="E", observation="obs-1", source="session-a", confidence="low")
        before = _load(ws, "E")["evidence"][0]

        _add(ws, entity_id="E", observation="obs-2", source="session-b", confidence="high")
        page = _load(ws, "E")

        assert page["evidence"][0] == before
        assert page["evidence"][0]["observation"] == "obs-1"
        assert page["evidence"][0]["source"] == "session-a"
        assert page["evidence"][0]["confidence"] == "low"
        assert page["evidence"][0]["superseded"] is False
        assert page["evidence"][1]["observation"] == "obs-2"
        assert page["evidence"][1]["source"] == "session-b"

    def test_recompiled_section_is_newest_first_bullets_over_live_evidence(self, ws: Path) -> None:
        _add(ws, entity_id="E", observation="older claim", confidence="low")
        _add(ws, entity_id="E", observation="newer claim", confidence="high")

        compiled = _load(ws, "E")["compiled_section"]

        assert compiled == "- **[HIGH]** newer claim\n- **[LOW]** older claim"

    def test_default_source_and_confidence_are_recorded_not_dropped(self, ws: Path) -> None:
        _add(ws, entity_id="E", observation="obs")

        entry = _load(ws, "E")["evidence"][0]
        assert entry["source"] == "mcp_tool"
        assert entry["confidence"] == "medium"

    def test_the_written_page_round_trips_through_the_sibling_load_tool(self, ws: Path) -> None:
        """A write nobody can read back is not a write."""
        out = _add(ws, entity_id="RT", observation="round trip", entity_type="project")

        page = _load(ws, "RT")
        assert "error" not in page
        assert page["entity_id"] == "RT"
        assert page["entity_type"] == "project"
        assert page["version"] == out["version"]
        assert page["evidence_count"] == out["evidence_count"]


class TestRefusals:
    """Bad input comes back as a structured error, and changes nothing."""

    def test_invalid_confidence_returns_the_error_envelope(self, ws: Path) -> None:
        out = _add(ws, entity_id="E", observation="obs", confidence="certain")

        assert set(out) == {"_schema_version", "error"}
        assert out["_schema_version"] == "1.0"
        assert out["error"].startswith("Failed to add evidence:")
        assert "Invalid confidence 'certain'" in out["error"]

    def test_confidence_is_case_sensitive(self, ws: Path) -> None:
        """``VALID_CONFIDENCE_LEVELS`` is lowercase; "HIGH" is not "high".

        Pinned because the compiled section upper-cases confidence for display,
        which makes "HIGH" look like a legal value to a caller reading a page.
        """
        out = _add(ws, entity_id="E", observation="obs", confidence="HIGH")

        assert "error" in out
        assert "Invalid confidence 'HIGH'" in out["error"]

    def test_a_refused_call_writes_nothing_at_all(self, ws: Path) -> None:
        _add(ws, entity_id="E", observation="obs", confidence="bogus")

        assert not (ws / "entities" / "compiled" / "E.md").exists()

    def test_a_refused_call_leaves_an_existing_page_untouched(self, ws: Path) -> None:
        """The dangerous failure is a half-write: version bumped, evidence lost."""
        _add(ws, entity_id="E", observation="the good observation")
        before = (ws / "entities" / "compiled" / "E.md").read_text(encoding="utf-8")

        refused = _add(ws, entity_id="E", observation="the bad one", confidence="bogus")

        assert "error" in refused
        assert (ws / "entities" / "compiled" / "E.md").read_text(encoding="utf-8") == before
        page = _load(ws, "E")
        assert page["version"] == 1
        assert page["evidence_count"] == 1
        assert page["evidence"][0]["observation"] == "the good observation"

    def test_a_non_string_observation_is_reported_not_raised(self, ws: Path) -> None:
        """The MCP surface must never let an exception escape as a transport error."""
        out = _add(ws, entity_id="E", observation=123)

        assert set(out) == {"_schema_version", "error"}
        assert "Failed to add evidence:" in out["error"]


class TestAclAndReachability:
    """Registered, classified, and callable by the scope that needs it."""

    def test_the_tool_is_registered_on_the_kernels_family(self) -> None:
        registered: list[str] = []

        class _Mcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        from mind_mem.mcp.tools.kernels import register

        register(_Mcp())
        assert "compiled_truth_add_evidence" in registered

    def test_it_is_user_scoped_not_admin_scoped(self) -> None:
        """Registered-but-unclassified is unreachable at EVERY scope.

        And admin-classified would silently strip every agent of the ability to
        record an observation, which is the tool's entire reason to exist.
        """
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "compiled_truth_add_evidence" in USER_TOOLS
        assert "compiled_truth_add_evidence" not in ADMIN_TOOLS

    def test_a_user_scope_call_passes_the_enforced_acl(self, ws: Path) -> None:
        """Behavioural counterpart to the classification assert above."""
        out = _add(ws, entity_id="E", observation="obs")

        assert "error" not in out
        assert out["version"] == 1


class TestUnvalidatedInput:
    """SECURITY REGRESSION — these pinned a defect and now pin its fix.

    ``entity_id`` arrives from an MCP tool argument and used to be joined into
    a filesystem path unchecked, so ``../`` escaped the workspace and this tool
    would WRITE an arbitrary ``.md`` file anywhere on the host. Closed in 5.0.0
    by ``compiled_truth._compiled_page_path``, which refuses anything that is
    not a bare name and then re-checks containment with realpath so a symlink
    cannot slip past the string test.
    """

    def test_an_uninitialised_workspace_is_created_rather_than_refused(self, tmp_path: Path) -> None:
        """GAP: this tool never calls ``_check_workspace``.

        Every ws-gated MCP tool fails closed on a workspace with no
        ``decisions/`` ("Run: mind-mem-init <path>"). This one does not consult
        the gate at all, so a typo'd workspace path silently materialises a new
        tree and the evidence lands somewhere nobody will look. A fix would
        return the ``_check_workspace`` error envelope here.
        """
        missing = tmp_path / "never-initialised"
        assert not missing.exists()

        out = _add(missing, entity_id="E", observation="obs")

        assert "error" not in out
        assert out["version"] == 1
        assert (missing / "entities" / "compiled" / "E.md").is_file()

    def test_a_path_shaped_entity_id_is_refused(self, ws: Path) -> None:
        """A traversing entity_id must not WRITE outside entities/compiled/.

        ``get_mind_kernel``, in this very module, has always guarded its ``name``
        with ``^[a-zA-Z0-9_-]{1,64}$`` before touching disk. This tool did not,
        so a traversing ``entity_id`` wrote an arbitrary ``.md`` file anywhere
        the process could reach. Closed in 5.0.0.
        """
        out = _add(ws, entity_id="../escaped", observation="obs")

        assert "error" in out and "bare name" in out["error"]
        assert "path" not in out, "a refused write must not report a path"
        assert not (ws / "entities" / "escaped.md").exists(), "the file was written anyway"
        assert not (ws / "entities" / "compiled" / "escaped.md").exists()

    def test_an_empty_entity_id_is_refused_not_written_as_a_dotfile(self, ws: Path) -> None:
        """An empty id used to produce a hidden ``.md`` dotfile page."""
        out = _add(ws, entity_id="", observation="obs")

        assert "error" in out and "non-empty string" in out["error"]
        assert "path" not in out
        assert not (ws / "entities" / "compiled" / ".md").exists()

    def test_a_legitimate_entity_id_still_writes(self, ws: Path) -> None:
        """The guard must not break the tool it protects."""
        out = _add(ws, entity_id="PRJ-ok", observation="obs")
        assert "error" not in out, out
        assert (ws / "entities" / "compiled" / "PRJ-ok.md").is_file()
