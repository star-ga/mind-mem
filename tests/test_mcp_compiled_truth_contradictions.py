"""``compiled_truth_contradictions`` — the detective half of the truth pages.

A compiled truth page is only worth keeping because its evidence trail is
append-only: nothing is ever edited away, so two observations recorded months
apart can disagree and both survive on disk. ``compiled_truth_contradictions``
is the tool that says so. It shipped in v1.9.0 alongside the rest of the
compiled-truth surface and has been registered, ACL-classified and documented
in ``docs/mcp-integration.md`` ever since — with no test anywhere. The library
function underneath it (``compiled_truth.detect_contradictions``) is covered by
``tests/test_compiled_truth.py``; the MCP tool that wraps it, and therefore
every promise the *tool* makes to a caller, was not.

That gap matters more than the usual "wrapper is thin" argument, because the
wrapper is where the two refusal paths live and where a caller's mistake gets
expensive. The tool answers a safety question — "is this entity's record
self-consistent?" — and its refusal envelope carries no ``contradiction_count``
at all. A caller that reads the answer as ``resp.get("contradiction_count", 0)``
turns "I could not find that page" and "I could not parse that page" into a
clean bill of health. So the sharpest thing these tests pin is not the happy
path: it is that a refusal is structurally distinguishable from a zero.

What is pinned here:

* the success envelope's exact key set, and that ``contradictions`` carries the
  two entries and the human reason;
* both refusal envelopes (no page, unparseable page) and the fact that neither
  can be mistaken for "zero contradictions";
* the detector semantics a caller depends on — superseded evidence is out of
  scope, observations are truncated to 100 characters in the response, and a
  pair is reported at most once;
* one honest limitation (a pair where *both* sides negate is not flagged) and
  one unfixed input gap (``entity_id`` is not path-validated), recorded as
  characterisation tests rather than left to be re-discovered;
* reachability: registered on the kernels family, classified USER (not admin),
  and reachable through the consolidated ``compiled_truth`` dispatcher.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.kernels import compiled_truth_contradictions

_FRONTMATTER = """---
entity_id: {entity_id}
entity_type: topic
last_compiled: 2026-01-01T00:00:00+00:00
version: 1
---

# {entity_id} — Compiled Truth

## Current Understanding

(irrelevant to contradiction detection — the detector reads the trail.)

## Evidence Trail

"""


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Give back the shared MCP call budget this module spends.

    ``mcp_tool_observe`` enforces a per-client sliding window (120 calls / 60s)
    through a module-global ``rate_limit._rate_limiters``, and every test in the
    session shares one client id. Spending the budget here makes some *later*,
    unrelated test fail with "Rate limit exceeded" under CI's random ordering.
    """
    from mind_mem.mcp.infra import rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    yield
    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()


@pytest.fixture(autouse=True)
def _user_scope(monkeypatch):
    """Run at the default (unprivileged) scope.

    This tool is in ``USER_TOOLS``: reading a page's contradictions is a
    read-only diagnosis, so it must work with no admin grant at all. Clearing
    the env var means a regression that reclassified it as admin would surface
    as an ACL refusal in every test below, not just in ``TestReachability``.
    """
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)


@pytest.fixture
def ws(tmp_path):
    """A workspace laid out the way ``mind-mem-init`` leaves one."""
    w = tmp_path / "ws"
    (w / "decisions").mkdir(parents=True)
    (w / ".mind-mem-index").mkdir(parents=True)
    (w / "entities" / "compiled").mkdir(parents=True)
    return w


def _write_page(w, entity_id: str, entries, *, frontmatter_id: str | None = None):
    """Write a truth page whose trail is *entries* — ``(observation, superseded)``.

    Built as markdown rather than through ``compiled_truth_add_evidence`` on
    purpose: the timestamps have to be fixed for the assertions below, and the
    page under test should not depend on the write tool being correct.
    """
    body = _FRONTMATTER.format(entity_id=frontmatter_id or entity_id)
    for i, (observation, superseded) in enumerate(entries):
        marker = " ~~SUPERSEDED~~" if superseded else ""
        body += f"### 2026-01-{i + 1:02d}T00:00:00+00:00 [HIGH] (source: s{i}.md){marker}\n\n{observation}\n\n"
    path = w / "entities" / "compiled" / f"{entity_id}.md"
    path.write_text(body, encoding="utf-8")
    return path


def _contradictions(w, entity_id: str) -> dict:
    with use_workspace(str(w)):
        return json.loads(compiled_truth_contradictions(entity_id))


# The canonical flagged pair used across several tests: same topic, one side
# carries a negation word ("disabled") the other does not.
_ENABLED = "the recall cache is enabled for every workspace"
_DISABLED = "the recall cache is disabled for every workspace"


class TestRefusals:
    """The two ways the tool declines, and why they must not read as a zero."""

    def test_unknown_entity_is_refused_and_names_the_entity(self, ws) -> None:
        resp = _contradictions(ws, "PRJ-does-not-exist")

        assert resp["_schema_version"] == "1.0"
        assert resp["error"] == "No compiled truth page found for 'PRJ-does-not-exist'."

    def test_refusal_carries_no_count_so_it_cannot_read_as_clean(self, ws) -> None:
        """A missing page must not answer the question it was asked.

        ``contradiction_count`` absent is the only thing standing between
        "never compiled" and "audited, found nothing".
        """
        resp = _contradictions(ws, "PRJ-does-not-exist")

        assert set(resp) == {"_schema_version", "error"}
        assert "contradiction_count" not in resp
        assert "contradictions" not in resp

    def test_unparseable_page_is_reported_not_raised(self, ws) -> None:
        """A corrupt page on disk is a tool-level error, never a server crash.

        ``parse_truth_page`` raises ``ValueError`` on bad frontmatter; the tool
        owns that and converts it, so one damaged file cannot drop the stdio
        session for every other tool.
        """
        (ws / "entities" / "compiled" / "CORRUPT.md").write_text("not a truth page at all\n", encoding="utf-8")

        resp = _contradictions(ws, "CORRUPT")

        assert set(resp) == {"_schema_version", "error"}
        assert resp["error"] == "Failed to detect contradictions: Missing or malformed frontmatter"

    def test_missing_workspace_refuses_at_the_page_level(self, tmp_path) -> None:
        """This tool is deliberately NOT ``_check_workspace``-gated.

        Sibling ws-gated tools answer a workspace that was never initialised
        with "Workspace not found. Run: mind-mem-init <path>". This one never
        calls that gate — it only ever opens one file — so it degrades to the
        page-level refusal instead. Pinned because the difference is visible to
        a caller trying to distinguish "wrong workspace" from "wrong entity",
        and because a future ws-gate would change this string.
        """
        resp = _contradictions(tmp_path / "never-initialised", "PRJ-x")

        assert resp["error"] == "No compiled truth page found for 'PRJ-x'."
        assert "mind-mem-init" not in resp["error"]


class TestResultEnvelope:
    """What a successful answer promises."""

    def test_clean_page_reports_zero_with_the_full_shape(self, ws) -> None:
        _write_page(ws, "CLEAN", [(_ENABLED, False), ("the index was rebuilt on tuesday", False)])

        resp = _contradictions(ws, "CLEAN")

        assert set(resp) == {"_schema_version", "entity_id", "contradiction_count", "contradictions"}
        assert resp["_schema_version"] == "1.0"
        assert resp["entity_id"] == "CLEAN"
        assert resp["contradiction_count"] == 0
        assert resp["contradictions"] == []

    def test_negation_asymmetry_pair_names_both_entries_and_the_reason(self, ws) -> None:
        _write_page(ws, "NEG", [(_ENABLED, False), (_DISABLED, False)])

        resp = _contradictions(ws, "NEG")

        assert resp["contradiction_count"] == 1
        (conflict,) = resp["contradictions"]
        assert set(conflict) == {"entry_a", "entry_b", "reason"}
        assert conflict["entry_a"] == {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "observation": _ENABLED,
        }
        assert conflict["entry_b"] == {
            "timestamp": "2026-01-02T00:00:00+00:00",
            "observation": _DISABLED,
        }
        assert conflict["reason"].startswith("Negation asymmetry:")
        assert "'disabled'" in conflict["reason"]

    def test_antonym_pair_is_flagged_when_neither_side_negates(self, ws) -> None:
        """The second heuristic — no negation word anywhere, but pass vs fail."""
        _write_page(
            ws,
            "ANT",
            [("the keystone gate is a pass on arm", False), ("the keystone gate is a fail on arm", False)],
        )

        resp = _contradictions(ws, "ANT")

        assert resp["contradiction_count"] == 1
        reason = resp["contradictions"][0]["reason"]
        assert reason.startswith("Antonym pair:")
        assert "'pass'" in reason and "'fail'" in reason

    def test_entries_are_ordered_a_before_b_in_trail_order(self, ws) -> None:
        """``entry_a`` is always the older half of the pair.

        The detector walks ``i < j`` over the trail, so a caller can read
        ``entry_b`` as "the later claim" — worth pinning, because that is the
        one an operator has to decide whether to supersede.
        """
        _write_page(ws, "ORDER", [(_ENABLED, False), (_DISABLED, False)])

        (conflict,) = _contradictions(ws, "ORDER")["contradictions"]

        assert conflict["entry_a"]["timestamp"] < conflict["entry_b"]["timestamp"]

    def test_observations_are_truncated_to_100_characters(self, ws) -> None:
        """The response is a summary, not a copy of the trail.

        Both halves are cut at 100 chars so a page of long observations cannot
        blow up the tool result; the full text stays available via
        ``compiled_truth_load``.
        """
        long_enabled = "the deterministic build pipeline is enabled for every supported substrate and stays stable across releases"
        long_disabled = "the deterministic build pipeline is disabled for every supported substrate and stays stable across releases"
        assert len(long_enabled) > 100 and len(long_disabled) > 100
        _write_page(ws, "LONG", [(long_enabled, False), (long_disabled, False)])

        (conflict,) = _contradictions(ws, "LONG")["contradictions"]

        assert conflict["entry_a"]["observation"] == long_enabled[:100]
        assert conflict["entry_b"]["observation"] == long_disabled[:100]
        assert len(conflict["entry_a"]["observation"]) == 100


class TestDetectorScope:
    """Which evidence counts, and how often a pair is reported."""

    def test_superseded_evidence_is_out_of_scope(self, ws) -> None:
        """Superseding is how an operator RESOLVES a contradiction.

        If the retired half kept being flagged, the resolution would never
        stick and the tool would nag forever about a settled question.
        """
        _write_page(ws, "SUP", [(_ENABLED, False), (_DISABLED, True)])

        resp = _contradictions(ws, "SUP")

        assert resp["contradiction_count"] == 0

    def test_a_pair_is_reported_at_most_once(self, ws) -> None:
        """Three entries, two flagged pairs — not four, and never the same pair twice.

        Both heuristics short-circuit per pair, so the count is bounded by
        ``n*(n-1)/2`` and every reported pair is distinct.
        """
        _write_page(
            ws,
            "MULTI",
            [
                ("the gate is enabled for release builds", False),
                ("the gate is disabled for release builds", False),
                ("the gate cannot run for release builds", False),
            ],
        )

        resp = _contradictions(ws, "MULTI")

        pairs = [(c["entry_a"]["timestamp"], c["entry_b"]["timestamp"]) for c in resp["contradictions"]]
        assert resp["contradiction_count"] == len(pairs) == len(set(pairs))
        assert pairs == [
            ("2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"),
            ("2026-01-01T00:00:00+00:00", "2026-01-03T00:00:00+00:00"),
        ]

    def test_two_mutually_negating_claims_are_not_flagged(self, ws) -> None:
        """A known blind spot, recorded rather than left to be re-discovered.

        Heuristic 1 fires on *asymmetry* — one side negates, the other does
        not. "disabled" versus "cannot run" both carry negation words, so the
        asymmetry test is false and the pair falls through to the antonym
        table, which has no entry for that wording. The two statements plainly
        disagree and the tool stays silent.

        This asserts the tool's real behaviour, not its desirable behaviour. If
        the detector is ever taught symmetric negation, flip this to assert the
        pair IS reported — do not delete it.
        """
        _write_page(
            ws,
            "BOTHNEG",
            [
                ("the gate is disabled for release builds", False),
                ("the gate cannot run for release builds", False),
            ],
        )

        resp = _contradictions(ws, "BOTHNEG")

        assert resp["contradiction_count"] == 0


class TestEntityIdHandling:
    """What the tool does with the one argument it takes."""

    def test_entity_id_is_echoed_from_the_request_not_the_page(self, ws) -> None:
        """The envelope's ``entity_id`` is the caller's string, verbatim.

        The page's own frontmatter id is never consulted for the response, so a
        page whose filename and frontmatter disagree reports the filename. Pinned
        because a caller correlating responses to requests can rely on the echo,
        and must NOT read it as "the page says it is about this entity".
        """
        _write_page(ws, "FILENAME-ID", [(_ENABLED, False)], frontmatter_id="FRONTMATTER-ID")

        resp = _contradictions(ws, "FILENAME-ID")

        assert resp["entity_id"] == "FILENAME-ID"

    def test_a_path_shaped_entity_id_is_refused(self, ws) -> None:
        """Characterisation test for an unfixed input gap — do not read as approval.

        ``get_mind_kernel``, the sibling tool in this same module, validates its
        one string argument against ``^[a-zA-Z0-9_-]{1,64}$`` before touching the
        filesystem. This tool does not: ``entity_id`` is joined straight into
        ``{ws}/entities/compiled/{entity_id}.md``, so ``..`` segments escape the
        compiled directory and any readable ``.md`` file that happens to parse as
        a truth page is summarised back to the caller.

        The reach is bounded — the workspace root is still the base, only ``.md``
        files are opened, and a non-page returns the parse error — but it is wider
        SECURITY REGRESSION: this pinned the gap and now pins the fix. Closed in
        5.0.0 by ``compiled_truth._compiled_page_path``, which refuses anything
        that is not a bare name and then re-checks containment with realpath so
        a symlink cannot slip past the string test.
        """
        _write_page(ws, "escaped", [(_ENABLED, False), (_DISABLED, False)])
        (ws / "entities" / "compiled" / "escaped.md").rename(ws / "escaped.md")

        resp = _contradictions(ws, "../../escaped")

        assert "error" in resp and "bare name" in resp["error"]
        assert "contradiction_count" not in resp, "a refused id must not report contradictions from an off-store page"


class TestReachability:
    """Registered, classified, and reachable from the consolidated surface."""

    def test_tool_is_registered_on_the_kernels_family(self) -> None:
        registered = []

        class _Mcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        from mind_mem.mcp.tools.kernels import register

        register(_Mcp())

        assert "compiled_truth_contradictions" in registered

    def test_tool_is_classified_user_scope(self) -> None:
        """Registered but unclassified is unreachable, not merely unprivileged.

        And it belongs in USER: it reads one file and writes nothing, so
        gating it behind admin would make contradiction detection unavailable
        to exactly the unprivileged agent that most needs the warning.
        """
        assert "compiled_truth_contradictions" in USER_TOOLS
        assert "compiled_truth_contradictions" not in ADMIN_TOOLS

    def test_answers_at_default_scope_with_no_admin_grant(self, ws) -> None:
        """The ACL gate runs for real here — the fixture cleared MIND_MEM_SCOPE."""
        _write_page(ws, "SCOPE", [(_ENABLED, False), (_DISABLED, False)])

        resp = _contradictions(ws, "SCOPE")

        assert "error" not in resp
        assert resp["contradiction_count"] == 1

    def test_dispatcher_action_returns_the_same_envelope(self, ws) -> None:
        """``compiled_truth(action="contradictions")`` must not diverge.

        The v3.2.0 dispatcher reaches the tool through ``__wrapped__``, which
        strips the per-tool ACL gate — so the two entry points can drift. Pin
        that they agree byte for byte on a real page.
        """
        from mind_mem.mcp.tools import public

        _write_page(ws, "DISPATCH", [(_ENABLED, False), (_DISABLED, False)])

        with use_workspace(str(ws)):
            direct = compiled_truth_contradictions("DISPATCH")
            via_dispatcher = public.compiled_truth("contradictions", entity_id="DISPATCH")

        assert json.loads(via_dispatcher)["contradiction_count"] == 1
        assert via_dispatcher == direct

    def test_dispatcher_requires_an_entity_id(self, ws) -> None:
        """The dispatcher refuses before it can ask about the empty entity."""
        from mind_mem.mcp.tools import public

        with use_workspace(str(ws)):
            resp = json.loads(public.compiled_truth("contradictions"))

        assert resp["error"] == "action='contradictions' requires 'entity_id'"
        assert "contradiction_count" not in resp
