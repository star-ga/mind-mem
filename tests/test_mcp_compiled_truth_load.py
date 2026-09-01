"""The compiled-truth read tool -- registered since v3.2.0, never tested.

``compiled_truth_load`` is one of the 98 tools the MCP server advertises and
one of the 10 a reachability pass found with no test anywhere. The library
underneath it (``mind_mem.compiled_truth``) has its own tests; the *tool* had
none, which is a different thing. The library is exercised through Python
objects, while the tool is exercised through a JSON envelope over a workspace
resolved from a ContextVar -- and every one of the failure modes below lives in
that gap, not in the library.

What these tests pin:

* **The success envelope.** Consumers read ``entity_id``, ``entity_type``,
  ``version``, ``last_compiled``, ``compiled_section``, ``evidence_count`` and
  ``evidence`` off the top level, and each evidence entry carries exactly five
  fields. The key set is asserted whole, not sampled, so a silently dropped
  field fails here rather than in a caller.
* **The two refusals, kept distinct.** An entity with no page returns a
  ``No compiled truth page found`` error plus a hint pointing at
  ``compiled_truth_add_evidence``; a page that exists but will not parse
  returns ``Failed to load truth page: ...``. Collapsing those two into one
  message would turn "you have not written this yet" into "your evidence trail
  is corrupt" -- indistinguishable to an agent, and the corrupt case is the one
  that needs a human.
* **Append-only semantics survive the read.** Superseded entries are returned,
  not filtered, and observations are returned in full. That is what separates
  this tool from its two siblings: ``recompile`` drops superseded evidence from
  the compiled section, and ``compiled_truth_contradictions`` truncates
  observations to 100 characters. ``load`` is the one surface that shows the
  trail as recorded, and a caller auditing an entity depends on it.
* **It is deliberately NOT workspace-gated.** Unlike most ws-scoped tools it
  never calls ``_check_workspace``, so a workspace with no ``decisions/``
  directory -- or no directory at all -- reports a missing *page*, not a
  missing *workspace*. That is worth pinning precisely because it is the
  exception: someone tidying the tools into a uniform gate would change an
  error string that callers branch on.
* **Its ACL classification.** It is in ``USER_TOOLS``. A registered tool in
  neither ACL set is unreachable at every scope, and promoting this one to
  admin would lock out the default stdio agent, so the classification is
  asserted both structurally and functionally (a real call at user scope).

``TestEntityIdIsUnsanitised`` is a characterization test, not an endorsement --
see its docstring.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.compiled_truth import CompiledTruthPage, EvidenceEntry, save_truth_page
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.kernels import compiled_truth_load

# The exact top-level keys the success envelope promises. Asserted as a set
# equality so an added key is as loud as a removed one.
_SUCCESS_KEYS = {
    "_schema_version",
    "entity_id",
    "entity_type",
    "version",
    "last_compiled",
    "compiled_section",
    "evidence_count",
    "evidence",
}

_EVIDENCE_KEYS = {"timestamp", "source", "observation", "confidence", "superseded"}


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Give back the shared MCP call budget this module spends.

    ``mcp_tool_observe`` enforces a per-client sliding window (120 calls / 60s)
    through a module-global ``rate_limit._rate_limiters``, and every test in the
    session shares one client id. Spending it here makes an unrelated test fail
    later with "Rate limit exceeded", which under random ordering looks like a
    bug in whatever ran next.
    """
    from mind_mem.mcp.infra import rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    yield
    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()


@pytest.fixture(autouse=True)
def _user_scope(monkeypatch):
    """Run every call at the default, unprivileged scope.

    Pinned rather than inherited from the ambient environment: the point of
    ``test_user_scope_can_actually_read_a_page`` is that this tool needs no
    admin grant, and an inherited ``MIND_MEM_SCOPE=admin`` or a stray
    ``MIND_MEM_ACL_DISABLED`` would make that assertion vacuous.
    """
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)


def _load(workspace, entity_id: str) -> dict:
    """Call the tool against *workspace* and parse its JSON reply.

    ``json.loads`` here is itself an assertion: every response this tool can
    produce, refusals included, must be a JSON object.
    """
    with use_workspace(str(workspace)):
        raw = compiled_truth_load(entity_id)
    assert isinstance(raw, str), "the MCP contract is a JSON string, not an object"
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return parsed


def _page(**overrides) -> CompiledTruthPage:
    """A two-entry page whose evidence is stored OLDEST-first on purpose.

    Storing it out of timestamp order is what lets the ordering test below
    distinguish "returned the trail as recorded" from "happened to be sorted".
    """
    fields = {
        "entity_id": "PRJ-mind-mem",
        "entity_type": "project",
        "compiled_section": "- **[HIGH]** mind-mem ships a governed memory store.",
        "evidence_entries": [
            EvidenceEntry(
                timestamp="2026-01-01T00:00:00+00:00",
                source="decisions/DECISIONS.md",
                observation="Initial claim, later replaced.",
                confidence="low",
                superseded=True,
            ),
            EvidenceEntry(
                timestamp="2026-02-02T00:00:00+00:00",
                source="maintenance/2026-02-02.md",
                observation="mind-mem ships a governed memory store.",
                confidence="high",
                superseded=False,
            ),
        ],
        "last_compiled": "2026-02-02T12:00:00+00:00",
        "version": 7,
    }
    fields.update(overrides)
    return CompiledTruthPage(**fields)


@pytest.fixture
def ws(tmp_path):
    """A workspace holding one saved truth page.

    Deliberately minimal: no ``decisions/``, no ``.mind-mem-index/``. This tool
    reads a single markdown file and must not require the corpus layout, and
    ``test_a_workspace_without_decisions_still_serves_pages`` pins that.
    """
    w = tmp_path / "ws"
    w.mkdir()
    save_truth_page(str(w), _page())
    return w


class TestSuccessEnvelope:
    def test_envelope_carries_exactly_the_documented_keys(self, ws):
        out = _load(ws, "PRJ-mind-mem")
        assert set(out) == _SUCCESS_KEYS
        assert out["_schema_version"] == "1.0"

    def test_page_metadata_is_echoed_verbatim_from_disk(self, ws):
        out = _load(ws, "PRJ-mind-mem")
        assert out["entity_id"] == "PRJ-mind-mem"
        assert out["entity_type"] == "project"
        assert out["version"] == 7
        assert out["last_compiled"] == "2026-02-02T12:00:00+00:00"
        assert out["compiled_section"] == "- **[HIGH]** mind-mem ships a governed memory store."

    def test_version_is_an_int_not_the_frontmatter_string(self, ws):
        """Callers compare versions; a string "7" would sort against "10" wrong."""
        out = _load(ws, "PRJ-mind-mem")
        assert isinstance(out["version"], int)

    def test_each_evidence_entry_carries_exactly_five_fields(self, ws):
        out = _load(ws, "PRJ-mind-mem")
        assert [set(e) for e in out["evidence"]] == [_EVIDENCE_KEYS, _EVIDENCE_KEYS]

    def test_evidence_count_agrees_with_the_evidence_list(self, ws):
        out = _load(ws, "PRJ-mind-mem")
        assert out["evidence_count"] == 2
        assert out["evidence_count"] == len(out["evidence"])

    def test_evidence_fields_round_trip_through_markdown(self, ws):
        out = _load(ws, "PRJ-mind-mem")
        assert out["evidence"][1] == {
            "timestamp": "2026-02-02T00:00:00+00:00",
            "source": "maintenance/2026-02-02.md",
            "observation": "mind-mem ships a governed memory store.",
            "confidence": "high",
            "superseded": False,
        }

    def test_superseded_evidence_is_returned_not_filtered(self, ws):
        """The trail is append-only; only the *compiled section* forgets.

        ``recompile_truth`` drops superseded entries from the summary. If this
        read did the same, an auditor asking "what did we once believe?" would
        get silence instead of a retracted claim.
        """
        out = _load(ws, "PRJ-mind-mem")
        superseded = [e for e in out["evidence"] if e["superseded"]]
        assert len(superseded) == 1
        assert superseded[0]["observation"] == "Initial claim, later replaced."
        assert superseded[0]["confidence"] == "low"

    def test_trail_is_returned_in_recorded_order_not_re_sorted(self, ws):
        """Fixture stores oldest-first, so a sort would be visible here."""
        assert [e["timestamp"] for e in _load(ws, "PRJ-mind-mem")["evidence"]] == [
            "2026-01-01T00:00:00+00:00",
            "2026-02-02T00:00:00+00:00",
        ]

    def test_observations_are_returned_in_full_not_truncated(self, tmp_path):
        """The sibling ``compiled_truth_contradictions`` clips to 100 chars.

        This tool must not: it is the only surface that shows an evidence entry
        as written, and a caller quoting it would silently quote a fragment.
        """
        long_observation = "B" * 250
        w = tmp_path / "ws"
        w.mkdir()
        save_truth_page(
            str(w),
            _page(
                evidence_entries=[
                    EvidenceEntry(
                        timestamp="2026-03-03T00:00:00+00:00",
                        source="notes.md",
                        observation=long_observation,
                        confidence="medium",
                        superseded=False,
                    )
                ]
            ),
        )
        out = _load(w, "PRJ-mind-mem")
        assert out["evidence"][0]["observation"] == long_observation

    def test_page_with_no_evidence_reports_an_empty_trail(self, tmp_path):
        w = tmp_path / "ws"
        w.mkdir()
        save_truth_page(str(w), _page(evidence_entries=[], compiled_section="(No active evidence.)"))
        out = _load(w, "PRJ-mind-mem")
        assert out["evidence_count"] == 0
        assert out["evidence"] == []
        assert "error" not in out, "an empty trail is a page, not a failure"


class TestRefusals:
    def test_unknown_entity_refuses_with_an_error_and_a_hint(self, ws):
        out = _load(ws, "PRJ-does-not-exist")
        assert out["error"] == "No compiled truth page found for 'PRJ-does-not-exist'."
        assert out["hint"] == "Create one with compiled_truth_add_evidence."
        assert out["_schema_version"] == "1.0"

    def test_the_refusal_leaks_no_page_shaped_keys(self, ws):
        """A caller must not be able to mistake a refusal for an empty page."""
        out = _load(ws, "PRJ-does-not-exist")
        assert set(out) == {"_schema_version", "error", "hint"}

    def test_a_page_that_will_not_parse_is_a_distinct_failure(self, ws):
        """Corrupt trail vs. absent trail are different problems.

        Absence is normal and self-service (write evidence). A page on disk
        that will not parse means the recorded trail is unreadable, which needs
        a human -- so the two must not share a message.
        """
        (ws / "entities" / "compiled" / "PRJ-corrupt.md").write_text("this file has no frontmatter\n", encoding="utf-8")
        out = _load(ws, "PRJ-corrupt")
        assert out["error"].startswith("Failed to load truth page:")
        assert "No compiled truth page found" not in out["error"]
        assert "hint" not in out, "a parse failure is not fixed by adding evidence"

    def test_an_unparsable_version_field_is_also_a_load_failure(self, ws):
        """Frontmatter present but ``version`` is not an int."""
        (ws / "entities" / "compiled" / "PRJ-badver.md").write_text(
            "---\nentity_id: PRJ-badver\nentity_type: project\nlast_compiled: 2026-01-01T00:00:00+00:00\nversion: seven\n---\n\n",
            encoding="utf-8",
        )
        out = _load(ws, "PRJ-badver")
        assert out["error"].startswith("Failed to load truth page:")

    def test_a_missing_workspace_reports_a_missing_page_not_a_missing_workspace(self, tmp_path):
        """This tool skips ``_check_workspace`` -- pinned because it is the odd one.

        Most ws-scoped tools answer "Workspace not found. Run: mind-mem-init".
        This one has no such gate, so an agent pointed at a bad workspace is
        told the *entity* is unknown. Folding it into the common gate would
        change an error string callers branch on, and that should be a
        deliberate decision, not a tidy-up.
        """
        out = _load(tmp_path / "no-such-workspace", "PRJ-mind-mem")
        assert out["error"] == "No compiled truth page found for 'PRJ-mind-mem'."
        assert "mind-mem-init" not in out["error"]

    def test_a_workspace_without_decisions_still_serves_pages(self, ws):
        """The corpus layout is not required to read one markdown file."""
        assert not (ws / "decisions").exists()
        out = _load(ws, "PRJ-mind-mem")
        assert out["entity_id"] == "PRJ-mind-mem"
        assert "error" not in out

    def test_an_empty_entity_id_is_refused_rather_than_serving_a_directory(self, ws):
        out = _load(ws, "")
        # 5.0.0: an empty id is now refused at the SHAPE check, before the
        # path is built at all -- a stricter and earlier refusal than the old
        # "no such page" answer, which had already touched the filesystem.
        assert "non-empty string" in out["error"]


class TestAclAndRegistration:
    def test_the_tool_is_registered_on_the_kernels_family(self):
        """Registered-but-unimported is exactly how the anchor tools went dark."""
        registered: list[str] = []

        class _Mcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        from mind_mem.mcp.tools.kernels import register

        register(_Mcp())
        assert "compiled_truth_load" in registered

    def test_the_tool_is_user_scope_not_admin(self):
        """A read of a local markdown page needs no admin grant.

        Classification is not cosmetic: a tool in NEITHER set is rejected as
        "not in ACL policy" at every scope, and one moved into ADMIN_TOOLS
        stops answering the default stdio agent.
        """
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "compiled_truth_load" in USER_TOOLS
        assert "compiled_truth_load" not in ADMIN_TOOLS

    def test_user_scope_can_actually_read_a_page(self, ws):
        """The functional half of the classification claim, at MIND_MEM_SCOPE=user."""
        out = _load(ws, "PRJ-mind-mem")
        assert out["entity_id"] == "PRJ-mind-mem"
        assert "not in ACL policy" not in json.dumps(out)


class TestEntityIdCannotEscapeTheStore:
    """SECURITY REGRESSION — this class used to pin the defect; now it pins the fix.

    Until 5.0.0 ``load_truth_page`` built its path as
    ``os.path.join(workspace, "entities/compiled", f"{entity_id}.md")`` with no
    validation, and ``entity_id`` arrives from an MCP tool argument. A
    caller-supplied ``../`` escaped the workspace, so at USER scope this tool
    read and returned ANY ``.md`` file on the host, and the two distinct error
    strings made it a file-existence oracle for arbitrary paths.

    Verified live before the fix: ``load_truth_page(ws, "../../../SECRET")``
    returned the contents of a file outside the workspace.

    ``_compiled_page_path`` now applies two checks — a shape check (an entity id
    is a NAME, never a path) and a realpath containment check that also catches
    a symlink, which no string inspection can.
    """

    @pytest.mark.parametrize(
        "payload",
        ["../../../elsewhere", "..", "a/b", "./x", "/etc/passwd", ""],
    )
    def test_a_path_shaped_entity_id_is_refused(self, tmp_path, payload):
        ws = tmp_path / "ws"
        (ws / "entities" / "compiled").mkdir(parents=True)
        out = _load(ws, payload)
        assert "error" in out, f"{payload!r} was not refused"
        assert "entity_id" not in out or "compiled_section" not in out

    def test_an_off_workspace_page_is_no_longer_served(self, tmp_path):
        """The original exploit, inverted."""
        ws = tmp_path / "ws"
        (ws / "entities" / "compiled").mkdir(parents=True)
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        save_truth_page(str(outside), _page(entity_id="PRJ-private", compiled_section="off-workspace"))

        out = _load(ws, "../../../elsewhere/entities/compiled/PRJ-private")
        assert "error" in out
        assert out.get("compiled_section") != "off-workspace"

    def test_the_existence_oracle_is_closed(self, tmp_path):
        """Present and absent off-workspace paths must be indistinguishable."""
        ws = tmp_path / "ws"
        (ws / "entities" / "compiled").mkdir(parents=True)
        (tmp_path / "present.md").write_text("not a truth page\n", encoding="utf-8")

        present = _load(ws, "../../../present")
        absent = _load(ws, "../../../absent")

        # The messages echo the caller's own entity_id, which reveals nothing
        # the caller did not already supply. What must NOT differ is the KIND of
        # refusal: both are rejected on shape, before the filesystem is touched,
        # so nothing in the answer depends on whether the target exists.
        def _kind(msg: str) -> str:
            return msg.split(":")[0] + ("|bare name" if "bare name" in msg else "")

        assert _kind(present["error"]) == _kind(absent["error"]), (
            f"refusal KIND differs -- still an existence oracle: {present['error']!r} vs {absent['error']!r}"
        )
        assert "bare name" in present["error"], "expected the shape refusal, not a disk answer"

    def test_a_legitimate_entity_id_still_works(self, tmp_path):
        """The guard must not break the tool it protects."""
        ws = tmp_path / "ws"
        (ws / "entities" / "compiled").mkdir(parents=True)
        save_truth_page(str(ws), _page(entity_id="PRJ-normal", compiled_section="fine"))
        out = _load(ws, "PRJ-normal")
        assert out["entity_id"] == "PRJ-normal"
        assert out["compiled_section"] == "fine"
