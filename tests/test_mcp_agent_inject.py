# Copyright 2026 STARGA, Inc.
"""``agent_inject`` — the MCP tool, not the formatter underneath it.

``tests/test_agent_bridge.py`` has covered :class:`AgentFormatter` since the
bridge landed, and that coverage is what made this tool look tested: the
rendering is exercised, so nobody noticed that *nothing anywhere* called the
registered MCP tool. The formatter is the easy half. The tool is the half that
resolves a workspace, refuses bad input, is gated by the ACL, runs a recall,
and reduces whatever recall hands back to a list of blocks — five places to
break silently while ``test_agent_bridge.py`` stays green.

So these tests deliberately do not re-test rendering. They pin the tool's own
contract:

* the workspace gate fires **before** any argument is validated and before any
  recall is spent — a tool that reads a corpus must fail closed on "no corpus";
* every refusal is a JSON object with an ``error`` key, and the unknown-agent
  refusal carries the full ``valid`` list (it is the only discovery surface a
  caller has for which agents exist);
* the success envelope is exactly
  ``{agent, query, snippet, attestation, _schema_version}`` with
  ``_schema_version == "1.0"`` — callers parse this, so an added or renamed
  key is a breaking change and should fail here. ``attestation`` joined the
  envelope in 5.0.2: this tool renders block content straight into a system
  prompt and was the one content door whose reply carried no receipt, so the
  recall's own ``RECALL_ATTEST_v2`` record is forwarded onto it (forwarded,
  not re-derived — two derivations of one run can disagree, and then neither
  is evidence);
* ``limit`` is enforced twice — threaded into recall *and* used as the
  formatter's ``max_blocks`` — so a recall that over-returns still cannot blow
  past the caller's budget into an agent's context window;
* ``scoring_instant`` is threaded through as the docstring promises (empty
  string means "today", i.e. ``None``, not the invalid literal ``""``), and a
  pinned instant makes the snippet reproducible, which is the whole reason the
  parameter exists;
* the three-branch shape reduction over recall's return (dict / list /
  anything else) is real code with real callers and is pinned per branch.

One documented sharp edge, tested as **actual** behaviour rather than desired:
when the recall underneath returns an error envelope (an invalid
``scoring_instant`` is the easy way to trigger it), ``agent_inject`` reads
``raw.get("results", [])``, gets nothing, and returns a *success-shaped*
envelope carrying an empty context snippet. The error is swallowed. That is
arguably wrong — an agent asking for memory and receiving "no memory matched"
cannot tell that from "your request was rejected" — but it is deliberate-looking
(the same branch absorbs list and scalar returns) and out of scope to change
here, so ``test_a_recall_error_is_swallowed_into_an_empty_snippet`` pins what
the code does today and will fail loudly if someone changes it, in either
direction.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.data_marking import DATA_PREAMBLE
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.agent import agent_inject


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Return the shared MCP call budget this module spends.

    ``mcp_tool_observe`` enforces a per-client sliding window (120 calls / 60s)
    through a module-global ``rate_limit._rate_limiters``, and every test in the
    session shares one client id. Without this, a file that makes a few dozen
    tool calls quietly eats a chunk of the budget and some *later, unrelated*
    test fails with "Rate limit exceeded" under random ordering.
    """
    from mind_mem.mcp.infra import rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    yield
    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()


@pytest.fixture
def ws(tmp_path):
    """A minimal Markdown-backend workspace with two recallable decisions.

    ``_check_workspace`` treats the presence of ``decisions/`` as the definition
    of a valid workspace on the default SQLite/Markdown backend, and the BM25
    leg scans that corpus directly — so this is a real corpus, not a stub, and
    the end-to-end tests below retrieve out of it for real.
    """
    w = tmp_path / "ws"
    for sub in ("decisions", "tasks", "entities", "intelligence"):
        (w / sub).mkdir(parents=True)
    (w / "mind-mem.json").write_text('{"retrieval": {"backend": "bm25"}}', encoding="utf-8")
    (w / "decisions" / "DECISIONS.md").write_text(
        "[DEC-20200101-001]\n"
        "Statement: The kernel scheduler uses a deterministic quantum.\n"
        "Status: active\nDate: 2020-01-01\n\n"
        "[DEC-20200102-002]\n"
        "Statement: Vector search degrades to a lexical scan when embeddings are absent.\n"
        "Status: active\nDate: 2020-01-02\n\n",
        encoding="utf-8",
    )
    return w


@pytest.fixture
def empty_dir(tmp_path):
    """An existing directory that is not a workspace (no ``decisions/``)."""
    d = tmp_path / "not-a-workspace"
    d.mkdir()
    return d


# A frozen instant so the recency layer scores the same on every run; without
# it the snippet depends on the wall clock and these assertions would rot.
INSTANT = "2020-06-01"

# The whole generic rendering when nothing was recalled. Built from the shared
# preamble rather than pasted, so a reworded preamble does not silently turn
# these into assertions about a string the product no longer emits.
_EMPTY_GENERIC = "Query: {query}\n" + DATA_PREAMBLE + "\nContext:\n"


def call(workspace, **kwargs) -> dict:
    """Invoke the tool inside *workspace* and parse its JSON string return."""
    with use_workspace(str(workspace)):
        return json.loads(agent_inject(**kwargs))


def snippet_of(workspace, **kwargs) -> str:
    out = call(workspace, **kwargs)
    assert "snippet" in out, out
    return out["snippet"]


def _exploding_recall(*_args, **_kwargs):
    raise AssertionError("recall must not run: the call should have been refused first")


class TestWorkspaceGate:
    """A memory tool with no corpus must refuse, not answer emptily."""

    def test_missing_workspace_directory_is_refused_by_name(self, tmp_path) -> None:
        out = call(tmp_path / "does-not-exist", query="scheduler")
        assert out == {"error": "Workspace not found. Run: mind-mem-init <path>"}

    def test_directory_without_decisions_is_refused_as_uninitialised(self, empty_dir) -> None:
        out = call(empty_dir, query="scheduler")
        assert "error" in out
        assert "decisions/" in out["error"]
        assert "mind-mem-init" in out["error"]

    def test_the_workspace_gate_precedes_argument_validation(self, empty_dir) -> None:
        """An uninitialised workspace is reported even when the args are junk.

        Ordering is the point: if arguments were checked first, an agent
        pointed at the wrong directory would be told "bad query" and keep
        retrying against a workspace that can never answer.
        """
        out = call(empty_dir, query="", agent="not-an-agent", limit=999)
        assert "decisions/" in out["error"]

    def test_the_workspace_gate_precedes_the_recall(self, empty_dir, monkeypatch) -> None:
        monkeypatch.setattr("mind_mem.mcp_server._recall_impl", _exploding_recall)
        assert "error" in call(empty_dir, query="scheduler")


class TestArgumentRefusals:
    """Every refusal is a JSON object with an ``error`` key. Pin the messages."""

    @pytest.mark.parametrize("query", ["", "   ", "\n\t "])
    def test_blank_query_is_refused(self, ws, query) -> None:
        assert call(ws, query=query) == {"error": "query must be a non-empty string"}

    @pytest.mark.parametrize("query", [None, 42, ["scheduler"]])
    def test_non_string_query_is_refused_not_coerced(self, ws, query) -> None:
        assert call(ws, query=query) == {"error": "query must be a non-empty string"}

    def test_unknown_agent_refusal_lists_every_valid_agent(self, ws) -> None:
        """The ``valid`` list is the caller's only discovery surface."""
        from mind_mem.agent_bridge import KNOWN_AGENTS

        out = call(ws, query="scheduler", agent="chatgpt")
        assert out["error"] == "unknown agent: 'chatgpt'"
        assert out["valid"] == list(KNOWN_AGENTS)
        assert "generic" in out["valid"]

    def test_agent_none_is_refused_rather_than_defaulted(self, ws) -> None:
        assert "unknown agent" in call(ws, query="scheduler", agent=None)["error"]

    @pytest.mark.parametrize("limit", [0, -1, 101, 1000])
    def test_limit_outside_one_to_hundred_is_refused(self, ws, limit) -> None:
        assert call(ws, query="scheduler", limit=limit) == {"error": "limit must be in [1, 100]"}

    @pytest.mark.parametrize("limit", [1, 100])
    def test_the_boundary_limits_are_accepted(self, ws, limit) -> None:
        """1 and 100 are inside the documented range; an off-by-one would break real callers."""
        assert "snippet" in call(ws, query="scheduler", limit=limit, scoring_instant=INSTANT)

    def test_a_bad_agent_does_not_spend_a_recall(self, ws, monkeypatch) -> None:
        """Validation is cheap; recall is not. Refuse before paying for it."""
        monkeypatch.setattr("mind_mem.mcp_server._recall_impl", _exploding_recall)
        assert "error" in call(ws, query="scheduler", agent="chatgpt")
        assert "error" in call(ws, query="scheduler", limit=0)


class TestSuccessEnvelope:
    def test_envelope_has_exactly_the_documented_keys(self, ws) -> None:
        out = call(ws, query="scheduler", agent="codex", limit=3, scoring_instant=INSTANT)
        assert set(out) == {"agent", "query", "snippet", "attestation", "_schema_version"}
        assert out["_schema_version"] == "1.0"

    def test_the_envelope_carries_the_recall_attestation(self, ws) -> None:
        """A door that serves block content has to say which run served it.

        ``agent_inject`` renders straight into a system prompt, which is the
        surface where "where did this come from" is hardest to answer after
        the fact -- and until 5.0.2 it was the one content tool whose reply
        carried no receipt at all. The record is the recall's own, forwarded
        rather than re-derived, so the two surfaces cannot disagree about
        what was served.
        """
        from mind_mem.recall_digests import served_set_digest

        out = call(ws, query="scheduler quantum", agent="generic", scoring_instant=INSTANT)
        attestation = out["attestation"]
        assert isinstance(attestation, dict), attestation
        assert attestation["schema"] == "RECALL_ATTEST_v2"
        # Positive control on the forwarding: a receipt that is merely
        # PRESENT proves nothing -- it has to commit to the ranking this
        # snippet was rendered from. The snippet names DEC-20200101-001 and
        # only that, so the canonical digest of exactly that served list is
        # what the record must carry. Forward the wrong run's attestation
        # and this goes red.
        assert "DEC-20200101-001" in out["snippet"]
        assert attestation["results_digest"] == served_set_digest(["DEC-20200101-001"])

    def test_the_attestation_is_null_rather_than_invented_when_recall_has_none(self, ws, monkeypatch) -> None:
        """Absent evidence is reported absent. A fabricated receipt is worse
        than no receipt, because it reads as proof."""
        monkeypatch.setattr(
            "mind_mem.mcp_server._recall_impl",
            lambda *a, **k: json.dumps([{"id": "B9", "excerpt": "no envelope, no attestation"}]),
        )
        out = call(ws, query="q")
        assert out["attestation"] is None
        assert "B9" in out["snippet"]

    def test_envelope_echoes_the_agent_and_query_it_was_given(self, ws) -> None:
        out = call(ws, query="deterministic quantum", agent="aider", scoring_instant=INSTANT)
        assert out["agent"] == "aider"
        assert out["query"] == "deterministic quantum"
        assert isinstance(out["snippet"], str)

    def test_the_snippet_carries_matched_corpus_content_not_just_the_query(self, ws) -> None:
        """End-to-end through the real recall: the block must actually come back."""
        text = snippet_of(ws, query="scheduler quantum", agent="generic", scoring_instant=INSTANT)
        assert "DEC-20200101-001" in text
        assert "deterministic quantum" in text

    def test_a_query_matching_nothing_yields_an_empty_context_not_an_error(self, ws) -> None:
        out = call(ws, query="zzzquux nonexistent term", agent="generic", scoring_instant=INSTANT)
        assert "error" not in out
        assert "DEC-2020" not in out["snippet"]


class TestAgentFormatIsDispatched:
    """The tool's job is choosing the target agent's shape — pin that it does."""

    def test_each_agent_gets_its_own_rendering_of_the_same_recall(self, ws) -> None:
        rendered = {
            agent: snippet_of(ws, query="scheduler quantum", agent=agent, scoring_instant=INSTANT)
            for agent in ("claude-code", "codex", "aider", "generic")
        }
        assert rendered["claude-code"].startswith("# mind-mem context")
        assert rendered["codex"].startswith("# Context for: scheduler quantum")
        assert rendered["aider"].startswith("repo_map:")
        assert rendered["generic"].startswith("Query: scheduler quantum")
        assert len(set(rendered.values())) == 4, "two agents collapsed to the same format"

    def test_every_known_agent_is_dispatchable_through_the_tool(self, ws) -> None:
        """``KNOWN_AGENTS`` is the accept-list; a name on it must not 500 or refuse."""
        from mind_mem.agent_bridge import KNOWN_AGENTS

        for agent in KNOWN_AGENTS:
            out = call(ws, query="scheduler", agent=agent, limit=2, scoring_instant=INSTANT)
            assert out.get("agent") == agent, out


class TestLimitIsEnforcedTwice:
    def test_limit_is_threaded_into_the_recall(self, ws, monkeypatch) -> None:
        seen = {}

        def fake(query, **kwargs):
            seen.update(query=query, **kwargs)
            return json.dumps({"results": []})

        monkeypatch.setattr("mind_mem.mcp_server._recall_impl", fake)
        call(ws, query="scheduler", limit=7)
        assert seen["limit"] == 7

    def test_an_over_returning_recall_is_still_capped_at_limit(self, ws, monkeypatch) -> None:
        """Defence in depth: the formatter's ``max_blocks`` is the caller's budget.

        A recall that ignores ``limit`` (a cache bug, a backend that pads)
        would otherwise dump unbounded text straight into an agent's context.
        """
        results = [{"id": f"B{i}", "excerpt": f"block {i}"} for i in range(9)]
        monkeypatch.setattr(
            "mind_mem.mcp_server._recall_impl",
            lambda *a, **k: json.dumps({"results": results}),
        )
        text = snippet_of(ws, query="scheduler", agent="generic", limit=3)
        assert text.count("- [B") == 3
        assert "B3" not in text


class TestScoringInstant:
    def test_empty_instant_becomes_none_not_the_empty_string(self, ws, monkeypatch) -> None:
        """ "" means "today in UTC". Passing it through literally is an invalid date."""
        seen = {}
        monkeypatch.setattr(
            "mind_mem.mcp_server._recall_impl",
            lambda query, **kw: (seen.update(kw), json.dumps({"results": []}))[1],
        )
        call(ws, query="scheduler")
        assert seen["scoring_instant"] is None

    def test_a_given_instant_reaches_recall_verbatim(self, ws, monkeypatch) -> None:
        seen = {}
        monkeypatch.setattr(
            "mind_mem.mcp_server._recall_impl",
            lambda query, **kw: (seen.update(kw), json.dumps({"results": []}))[1],
        )
        call(ws, query="scheduler", scoring_instant=INSTANT)
        assert seen["scoring_instant"] == INSTANT

    def test_the_same_instant_reproduces_the_same_snippet(self, ws) -> None:
        """The reason the parameter exists: a pinned instant is a replayable answer."""
        first = snippet_of(ws, query="scheduler quantum", agent="codex", scoring_instant=INSTANT)
        second = snippet_of(ws, query="scheduler quantum", agent="codex", scoring_instant=INSTANT)
        assert first == second

    def test_two_agents_on_one_instant_see_the_same_blocks(self, ws) -> None:
        """Different renderings, identical evidence — the docstring's promise."""
        claude = snippet_of(ws, query="scheduler quantum", agent="claude-code", scoring_instant=INSTANT)
        gemini = snippet_of(ws, query="scheduler quantum", agent="gemini", scoring_instant=INSTANT)
        assert claude != gemini
        assert ("DEC-20200101-001" in claude) == ("DEC-20200101-001" in gemini)
        assert "DEC-20200101-001" in claude


class TestRecallShapeReduction:
    """Three branches of real code over recall's return. One test each."""

    def _pin(self, monkeypatch, payload: str) -> None:
        monkeypatch.setattr("mind_mem.mcp_server._recall_impl", lambda *a, **k: payload)

    def test_a_dict_envelope_is_read_from_its_results_key(self, ws, monkeypatch) -> None:
        self._pin(monkeypatch, json.dumps({"results": [{"id": "B1", "excerpt": "hello"}]}))
        assert "- [B1] (provenance: unknown) <evidence>hello</evidence>" in snippet_of(ws, query="q")

    def test_a_bare_list_is_treated_as_the_results(self, ws, monkeypatch) -> None:
        self._pin(monkeypatch, json.dumps([{"id": "B2", "excerpt": "listed"}]))
        assert "- [B2] (provenance: unknown) <evidence>listed</evidence>" in snippet_of(ws, query="q")

    @pytest.mark.parametrize("payload", ["5", '"a string"', "null", "true"])
    def test_a_scalar_return_degrades_to_an_empty_context(self, ws, monkeypatch, payload) -> None:
        self._pin(monkeypatch, payload)
        out = call(ws, query="q")
        assert out["snippet"] == _EMPTY_GENERIC.format(query="q")

    def test_a_null_results_key_is_treated_as_empty_not_iterated(self, ws, monkeypatch) -> None:
        """``raw.get("results", []) or []`` — the ``or`` exists for this case."""
        self._pin(monkeypatch, json.dumps({"results": None}))
        assert call(ws, query="q")["snippet"] == _EMPTY_GENERIC.format(query="q")

    def test_a_recall_error_is_swallowed_into_an_empty_snippet(self, ws) -> None:
        """DOCUMENTED SHARP EDGE, pinned as actual behaviour, not endorsed.

        An invalid ``scoring_instant`` makes ``_recall_impl`` return
        ``{"error": ...}``. ``agent_inject`` has no error branch for that: it
        reads the absent ``results`` key and returns a success-shaped envelope
        with an empty context. The caller cannot distinguish "nothing matched"
        from "your request was rejected". Changing that is a behaviour change
        for every existing caller, so this test pins today's contract and will
        fail if it moves — deliberately or not.
        """
        out = call(ws, query="scheduler", agent="generic", scoring_instant="not-a-date")
        assert "error" not in out
        assert out["snippet"] == _EMPTY_GENERIC.format(query="scheduler")


class TestAclAndReachability:
    def test_the_tool_is_registered_on_the_agent_family(self) -> None:
        """A tool nothing registers is unreachable from the product."""
        registered: list[str] = []

        class _Mcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        from mind_mem.mcp.tools.agent import register

        register(_Mcp())
        assert "agent_inject" in registered

    def test_the_tool_is_exported_from_the_mcp_server_module(self) -> None:
        from mind_mem import mcp_server

        assert getattr(mcp_server, "agent_inject", None) is not None

    def test_the_tool_is_classified_user_scope_not_admin(self) -> None:
        """Registered-but-unclassified returns "not in ACL policy" — unreachable.

        ``agent_inject`` only reads, so admin scope would wrongly lock the
        documented agent-bridge surface out of every default (user-scope) client.
        """
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "agent_inject" in USER_TOOLS
        assert "agent_inject" not in ADMIN_TOOLS

    def test_default_user_scope_can_call_it(self, ws, monkeypatch) -> None:
        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        assert "snippet" in call(ws, query="scheduler", scoring_instant=INSTANT)

    def test_admin_scope_gets_the_same_answer_as_user_scope(self, ws, monkeypatch) -> None:
        monkeypatch.setenv("MIND_MEM_SCOPE", "user")
        as_user = snippet_of(ws, query="scheduler quantum", scoring_instant=INSTANT)
        monkeypatch.setenv("MIND_MEM_SCOPE", "admin")
        assert snippet_of(ws, query="scheduler quantum", scoring_instant=INSTANT) == as_user

    def test_a_broken_auth_context_fails_closed_on_this_user_tool_too(self, ws, monkeypatch) -> None:
        """Issue #526: introspection failure must deny USER tools, not degrade to user scope.

        The regression this guards is silent — a tool that kept answering
        while token introspection was throwing would look perfectly healthy.
        """

        def _boom():
            raise RuntimeError("token introspection unavailable")

        monkeypatch.setattr("mind_mem.mcp.infra.acl.get_access_token", _boom)
        out = call(ws, query="scheduler")
        assert out["scope"] == "deny"
        assert out["error"] == "Permission denied: authentication context unavailable"
