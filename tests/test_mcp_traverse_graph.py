"""``traverse_graph`` — the causal-graph tool that nothing tested.

A 5.0.0 reachability pass over the 98 registered MCP tools found ten with no
test anywhere. ``traverse_graph`` was one of them: registered, ACL-classified,
documented in the ``graph`` dispatcher table, and never once called by the
suite. It is the impact-analysis surface — "if I change this block, what breaks"
— so a silent regression here does not crash anything, it just quietly answers
"nothing depends on this" and the operator believes it.

What these tests pin, in the order they matter:

* **The two refusals happen before any I/O.** ``block_id`` must match
  ``^[A-Z]+-...$`` and ``direction`` must be one of three words. Both are
  rejected with a bare ``{_schema_version, error}`` envelope carrying no partial
  results, and — the assertion that actually proves ordering — a malformed id
  against a bare directory leaves no ``.mind-mem-index/`` behind.
* **``depth`` is clamped, not trusted, and the clamp is reported.** The response
  echoes the *effective* depth after ``max(1, min(depth, 5))``, so a caller who
  asked for 99 is told it got 5 rather than silently believing it saw the whole
  graph. The clamp really bounds the walk: the downstream BFS emits exactly
  ``depth`` layers of a longer chain.
* **``direction`` selects which half of the envelope exists at all.** The
  ``upstream``/``downstream`` keys are absent, not empty, when not asked for —
  a consumer that does ``result["downstream"]`` on an upstream call gets a
  KeyError, and that is the contract.
* **An unknown block is an answer, not an error.** Impact analysis on a block
  with no edges legitimately returns "nothing", and the tool distinguishes that
  from a malformed request.

Two divergences are pinned as *actual* behaviour with the reasoning written
down rather than quietly fixed (this file does not touch ``src/``):

1. ``traverse_graph`` is the one tool in ``graph.py`` that never calls
   ``_check_workspace``. Its siblings refuse an un-initialised workspace with
   "Run: mind-mem-init"; this one constructs a ``CausalGraph``, which creates
   ``.mind-mem-index/`` and answers from an empty database.
2. A non-string ``block_id`` escapes as a ``TypeError`` out of ``re.match``
   instead of becoming an error envelope like every other bad input.

Both are marked in-place so that if either is ever tightened, the failing test
names the intended change instead of looking like a regression.
"""

from __future__ import annotations

import json

import pytest

from mind_mem.causal_graph import CausalGraph
from mind_mem.mcp.infra.workspace import use_workspace
from mind_mem.mcp.tools.graph import traverse_graph

# A four-node diamond plus a straight chain, laid out so every traversal
# assertion below has a unique correct answer:
#
#   ROOT ← MID_A ← TIP          (TIP depends_on MID_A depends_on ROOT)
#   ROOT ← MID_B ← TIP          (the diamond: TIP reaches ROOT two ways)
#   ROOT ← C1 ← C2 ← C3 ← C4 ← C5 ← C6   (long chain, for the depth clamp)
#
# Edge direction is `source depends_on target`, so "upstream" of a block is its
# outgoing edges (what it needs) and "downstream" is its incoming ones (what
# needs it).
ROOT = "DEC-20200101-001"
MID_A = "DEC-20200102-001"
MID_B = "DEC-20200103-001"
TIP = "DEC-20200104-001"
CHAIN = [f"DEC-2021010{i}-001" for i in range(1, 8)]  # CHAIN[0] is nearest ROOT


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Return the shared MCP call budget this module spends.

    ``mcp_tool_observe`` enforces a 120-call/60s sliding window in a
    module-global keyed by client id, and the whole session is one client. This
    file makes well over thirty calls; without this fixture it eats a third of
    the budget and some *later, unrelated* test fails with "Rate limit
    exceeded" under CI's random ordering.
    """
    from mind_mem.mcp.infra import rate_limit

    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()
    yield
    with rate_limit._rate_limiters_lock:
        rate_limit._rate_limiters.clear()


@pytest.fixture(autouse=True)
def _user_scope(monkeypatch):
    """Run at the default scope on purpose.

    ``traverse_graph`` is a USER tool (read-only impact analysis). Leaking an
    ``MIND_MEM_SCOPE=admin`` from another module's env would hide an accidental
    reclassification into ``ADMIN_TOOLS``, so the variable is cleared for every
    test here and ``TestAclAndReachability`` asserts the classification.
    """
    monkeypatch.delenv("MIND_MEM_SCOPE", raising=False)
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)


@pytest.fixture
def ws(tmp_path):
    """An initialised workspace carrying the diamond and the chain."""
    w = tmp_path / "ws"
    (w / ".mind-mem-index").mkdir(parents=True)
    # Sibling ws-gated tools fail closed without decisions/ on the default
    # SQLite/Markdown backend. traverse_graph does not check (see module
    # docstring), but the workspace is built correctly so that the tests which
    # *do* care about the ungated behaviour use a deliberately bare directory.
    (w / "decisions").mkdir(parents=True)

    cg = CausalGraph(str(w))
    cg.add_edge(MID_A, ROOT, "depends_on", weight=0.25)
    cg.add_edge(MID_B, ROOT, "supersedes", weight=2.0)
    cg.add_edge(TIP, MID_A, "depends_on")
    cg.add_edge(TIP, MID_B, "informs")
    cg.add_edge(CHAIN[0], ROOT, "depends_on")
    for near, far in zip(CHAIN[1:], CHAIN):
        cg.add_edge(near, far, "depends_on")
    return w


def _traverse(workspace, **kwargs) -> dict:
    with use_workspace(str(workspace)):
        return json.loads(traverse_graph(**kwargs))


class TestInputRefusal:
    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "dec-20200101-001",  # lowercase prefix
            "20200101-001",  # no alpha prefix
            "DEC20200101001",  # no separator
            "DEC-",  # empty suffix
            "DEC 20200101 001",  # spaces
            "DEC-001; DROP TABLE causal_edges",
            "../../etc/passwd",
            "DEC-001' OR '1'='1",
        ],
    )
    def test_a_malformed_block_id_is_refused_by_name(self, ws, bad) -> None:
        out = _traverse(ws, block_id=bad)
        assert out["error"] == f"Invalid block_id format: {bad}"
        assert out["_schema_version"] == "1.0"

    @pytest.mark.parametrize("bad", ["outgoing", "incoming", "BOTH", "", "up", "both "])
    def test_an_unrecognised_direction_is_refused_and_the_options_are_listed(self, ws, bad) -> None:
        """The error names the three legal values.

        A traversal tool that says only "invalid direction" leaves the caller
        guessing between at least four plausible vocabularies
        (upstream/downstream, in/out, forward/backward, parents/children).
        """
        out = _traverse(ws, block_id=ROOT, direction=bad)
        assert out["error"] == (f"Invalid direction: {bad}. Use 'upstream', 'downstream', or 'both'.")
        assert out["_schema_version"] == "1.0"

    def test_a_refusal_carries_no_partial_results(self, ws) -> None:
        """Exactly two keys: the version and the error.

        A refusal that also shipped an empty ``upstream``/``downstream`` would
        read to a consumer as "traversed successfully, found nothing" — the
        single worst failure mode for an impact-analysis tool.
        """
        assert set(_traverse(ws, block_id="nope").keys()) == {"_schema_version", "error"}
        assert set(_traverse(ws, block_id=ROOT, direction="sideways").keys()) == {
            "_schema_version",
            "error",
        }

    def test_validation_runs_before_the_workspace_is_touched(self, tmp_path) -> None:
        """A rejected id must not create a database.

        This is the ordering assertion: ``CausalGraph.__init__`` does
        ``makedirs`` + ``executescript``, so if validation moved below the
        workspace resolution, a malformed id from a poisoned agent would
        provision an index directory anywhere the process can write.
        """
        bare = tmp_path / "never-initialised"
        bare.mkdir()
        out = _traverse(bare, block_id="not a block id")
        assert out["error"].startswith("Invalid block_id format")
        assert list(bare.iterdir()) == []


class TestDepthClamp:
    @pytest.mark.parametrize(
        ("asked", "effective"),
        [(-99, 1), (0, 1), (1, 1), (2, 2), (5, 5), (6, 5), (99, 5)],
    )
    def test_the_response_reports_the_effective_depth_not_the_request(self, ws, asked, effective) -> None:
        """``max(1, min(depth, 5))``, echoed back.

        Reporting the requested depth would let a caller who asked for 99
        conclude the graph really is only five hops deep.
        """
        assert _traverse(ws, block_id=ROOT, depth=asked)["depth"] == effective

    @pytest.mark.parametrize(("asked", "layers"), [(1, 1), (2, 2), (3, 3), (5, 5), (99, 5)])
    def test_the_downstream_walk_stops_at_the_effective_depth(self, ws, asked, layers) -> None:
        """The chain has six descendants, so every clamp value is observable.

        Walking downstream from the chain's near end reaches exactly one new
        node per layer, so the emitted depth labels *are* the realised walk.
        """
        out = _traverse(ws, block_id=CHAIN[0], depth=asked, direction="downstream")
        nodes = out["downstream"]["reachable_nodes"]
        assert [n["depth"] for n in nodes] == list(range(1, layers + 1))

    def test_the_upstream_chain_grows_with_depth_and_then_stops(self, ws) -> None:
        """``causal_chain`` is bounded by the same clamped depth.

        Pinned as the *observed* bound rather than a formula: the DFS appends
        the path when ``depth > max_depth``, so a max_depth of N yields a path
        of N+2 ids until the real chain runs out (here at 8 ids: CHAIN[6..0]
        plus ROOT).
        """

        def _chain_len(d: int) -> int:
            out = _traverse(ws, block_id=CHAIN[-1], depth=d, direction="upstream")
            return len(out["upstream"]["causal_chains"][0])

        lengths = [_chain_len(d) for d in (1, 2, 3, 5, 99)]
        assert lengths == [3, 4, 5, 7, 7]

    def test_a_walk_shorter_than_the_depth_terminates_instead_of_padding(self, ws) -> None:
        """MID_A's downstream is one hop deep; asking for five must not invent four."""
        out = _traverse(ws, block_id=MID_A, depth=5, direction="downstream")
        assert [n["block_id"] for n in out["downstream"]["reachable_nodes"]] == [TIP]


class TestDirectionSelectsHalves:
    def test_upstream_omits_the_downstream_half_entirely(self, ws) -> None:
        out = _traverse(ws, block_id=MID_A, direction="upstream")
        assert set(out) == {"_schema_version", "block_id", "direction", "depth", "upstream", "graph_summary"}
        assert "downstream" not in out

    def test_downstream_omits_the_upstream_half_entirely(self, ws) -> None:
        out = _traverse(ws, block_id=MID_A, direction="downstream")
        assert set(out) == {"_schema_version", "block_id", "direction", "depth", "downstream", "graph_summary"}
        assert "upstream" not in out

    def test_both_is_the_default_and_carries_both_halves(self, ws) -> None:
        out = _traverse(ws, block_id=MID_A)
        assert out["direction"] == "both"
        assert set(out["upstream"]) == {"direct_dependencies", "causal_chains"}
        assert set(out["downstream"]) == {"direct_dependents", "reachable_nodes"}

    def test_the_graph_summary_is_present_in_every_direction(self, ws) -> None:
        """The summary describes the whole graph, so it does not vary by direction."""
        summaries = [_traverse(ws, block_id=MID_A, direction=d)["graph_summary"] for d in ("upstream", "downstream", "both")]
        assert summaries[0] == summaries[1] == summaries[2]
        assert summaries[0] == {
            "total_edges": 11,
            "unique_nodes": 11,
            "edges_by_type": {"depends_on": 9, "informs": 1, "supersedes": 1},
            "stale_blocks": 0,
        }


class TestUpstream:
    def test_direct_dependencies_name_the_target_type_and_weight(self, ws) -> None:
        """These three fields are the whole payload — pin them by value.

        ``weight`` in particular must survive as a float: it is the only
        numeric an impact ranking can sort on, and a stringified 0.25 would
        sort after 2.0.
        """
        deps = _traverse(ws, block_id=TIP, direction="upstream")["upstream"]["direct_dependencies"]
        assert sorted(deps, key=lambda e: e["target"]) == [
            {"target": MID_A, "edge_type": "depends_on", "weight": 1.0},
            {"target": MID_B, "edge_type": "informs", "weight": 1.0},
        ]

    def test_a_non_default_weight_round_trips_as_a_number(self, ws) -> None:
        deps = _traverse(ws, block_id=MID_A, direction="upstream")["upstream"]["direct_dependencies"]
        assert deps == [{"target": ROOT, "edge_type": "depends_on", "weight": 0.25}]
        assert isinstance(deps[0]["weight"], float)

    def test_causal_chains_start_at_the_queried_block_and_end_at_a_root(self, ws) -> None:
        chains = _traverse(ws, block_id=TIP, direction="upstream", depth=3)["upstream"]["causal_chains"]
        assert sorted(chains) == [[TIP, MID_A, ROOT], [TIP, MID_B, ROOT]]

    def test_a_root_block_has_no_dependencies_and_a_single_trivial_chain(self, ws) -> None:
        """ROOT is depended on by four blocks but depends on nothing itself."""
        up = _traverse(ws, block_id=ROOT, direction="upstream")["upstream"]
        assert up["direct_dependencies"] == []
        assert up["causal_chains"] == [[ROOT]]


class TestDownstream:
    def test_direct_dependents_name_the_source_type_and_weight(self, ws) -> None:
        dependents = _traverse(ws, block_id=ROOT, direction="downstream")["downstream"]["direct_dependents"]
        assert sorted(dependents, key=lambda e: e["source"]) == [
            {"source": MID_A, "edge_type": "depends_on", "weight": 0.25},
            {"source": MID_B, "edge_type": "supersedes", "weight": 2.0},
            {"source": CHAIN[0], "edge_type": "depends_on", "weight": 1.0},
        ]

    def test_each_reachable_node_records_the_parent_it_was_found_through(self, ws) -> None:
        """``depends_on`` here is the BFS parent, i.e. the edge that pulled the
        node into the blast radius — that provenance is what makes the result
        auditable rather than an unordered set of ids."""
        nodes = _traverse(ws, block_id=MID_A, depth=2, direction="downstream")["downstream"]["reachable_nodes"]
        assert nodes == [{"block_id": TIP, "depends_on": MID_A, "edge_type": "depends_on", "depth": 1}]

    def test_a_diamond_yields_each_node_once_at_its_shallowest_depth(self, ws) -> None:
        """TIP reaches ROOT through both MID_A and MID_B.

        Without the ``visited`` set the walk would report TIP twice, and an
        operator counting the blast radius would double-count it. Both mids sit
        at depth 1, TIP at depth 2 via whichever mid the BFS saw first.
        """
        nodes = _traverse(ws, block_id=ROOT, depth=2, direction="downstream")["downstream"]["reachable_nodes"]
        found = {n["block_id"]: n["depth"] for n in nodes}
        assert len(nodes) == len(found), "a node was emitted more than once"
        assert found[MID_A] == 1 and found[MID_B] == 1
        assert found[TIP] == 2
        assert nodes[[n["block_id"] for n in nodes].index(TIP)]["depends_on"] in (MID_A, MID_B)

    def test_a_leaf_block_reports_an_empty_blast_radius(self, ws) -> None:
        down = _traverse(ws, block_id=TIP, direction="downstream")["downstream"]
        assert down["direct_dependents"] == []
        assert down["reachable_nodes"] == []


class TestUnknownBlock:
    def test_an_unedged_block_is_answered_not_refused(self, ws) -> None:
        """ "Nothing depends on this" is a legitimate impact-analysis answer.

        It must be distinguishable from "your request was malformed", which is
        why this asserts the success envelope and the absence of ``error``.
        """
        out = _traverse(ws, block_id="DEC-19990101-999")
        assert "error" not in out
        assert out["block_id"] == "DEC-19990101-999"
        assert out["upstream"] == {"direct_dependencies": [], "causal_chains": [["DEC-19990101-999"]]}
        assert out["downstream"] == {"direct_dependents": [], "reachable_nodes": []}

    def test_the_summary_still_describes_the_real_graph(self, ws) -> None:
        """An unknown block does not make the graph look empty."""
        assert _traverse(ws, block_id="DEC-19990101-999")["graph_summary"]["total_edges"] == 11


class TestUngatedWorkspace:
    """Divergence 1, pinned as observed — see the module docstring.

    Every other tool in ``graph.py`` opens with ``_check_workspace(ws)``.
    ``traverse_graph`` does not, so an un-initialised path is answered from a
    freshly created empty database instead of refused.
    """

    def test_a_bare_directory_answers_empty_instead_of_refusing(self, tmp_path) -> None:
        bare = tmp_path / "bare"
        bare.mkdir()
        out = _traverse(bare, block_id=ROOT)
        # If this ever fails with "Run: mind-mem-init", the ws gate was added
        # on purpose — adopt the refusal here, do not revert the source.
        assert "error" not in out
        assert out["graph_summary"] == {
            "total_edges": 0,
            "unique_nodes": 0,
            "edges_by_type": {},
            "stale_blocks": 0,
        }

    def test_answering_provisions_the_index_directory(self, tmp_path) -> None:
        """The read-only-looking call is a filesystem write. Worth knowing."""
        bare = tmp_path / "bare2"
        bare.mkdir()
        _traverse(bare, block_id=ROOT)
        assert (bare / ".mind-mem-index" / "causal.db").is_file()

    def test_a_missing_workspace_directory_is_created_rather_than_reported(self, tmp_path) -> None:
        absent = tmp_path / "does-not-exist"
        out = _traverse(absent, block_id=ROOT)
        assert "error" not in out
        assert absent.is_dir()


class TestValidatorGaps:
    """Divergence 2 and the regex anchor, pinned as observed."""

    def test_a_trailing_newline_slips_past_the_block_id_anchor(self, ws) -> None:
        """``re.match(..."$")`` matches before a final newline; fullmatch would not.

        Harmless downstream — the id only ever reaches SQLite as a bound
        parameter, asserted here by the graph surviving intact — but it means
        the format claim in the error message is slightly wider than the regex
        reads. If this test starts failing because the id is now REFUSED, that
        is the ``re.fullmatch`` fix landing: update this test, do not revert it.
        """
        out = _traverse(ws, block_id=f"{ROOT}\n")
        assert "error" not in out
        assert out["block_id"] == f"{ROOT}\n"
        assert out["upstream"]["direct_dependencies"] == []
        assert out["graph_summary"]["total_edges"] == 11

    @pytest.mark.parametrize("bad", [None, 123, ["DEC-1"]])
    def test_a_non_string_block_id_escapes_as_a_typeerror(self, ws, bad) -> None:
        """The one bad input that is not converted to an error envelope.

        ``re.match`` raises before the ``isinstance`` check other tools in this
        module perform on their string arguments, and ``mcp_tool_observe`` only
        converts *database* errors, so this propagates. FastMCP's own signature
        validation keeps it off the wire, which is why it has survived — but it
        is reachable from every in-process caller, including the ``graph``
        dispatcher's ``__wrapped__`` path.
        """
        with pytest.raises(TypeError, match="expected string or bytes-like object"):
            _traverse(ws, block_id=bad)


class TestPublicDispatcher:
    """The ``graph`` consolidated tool routes ``action="traverse"`` here."""

    def test_the_traverse_action_reaches_the_tool(self, ws) -> None:
        from mind_mem.mcp.tools.public import graph as graph_tool

        with use_workspace(str(ws)):
            out = json.loads(graph_tool("traverse", block_id=ROOT, direction="downstream", depth=2))
        assert out["block_id"] == ROOT
        assert {n["block_id"] for n in out["downstream"]["reachable_nodes"]} == {MID_A, MID_B, TIP, CHAIN[0], CHAIN[1]}

    def test_the_dispatchers_default_direction_is_rejected_by_the_tool(self, ws) -> None:
        """Observed defect, pinned rather than fixed.

        ``graph``'s shared ``direction`` parameter defaults to ``"outgoing"``
        for the knowledge-graph ``query`` branch, but the causal tool's
        vocabulary is upstream/downstream/both — so ``graph("traverse",
        block_id=...)`` with defaults can never succeed, and the failure is a
        direction error rather than anything naming the dispatcher. Callers
        must pass ``direction=`` explicitly today.
        """
        from mind_mem.mcp.tools.public import graph as graph_tool

        with use_workspace(str(ws)):
            out = json.loads(graph_tool("traverse", block_id=ROOT))
        assert out["error"].startswith("Invalid direction: outgoing.")


class TestAclAndReachability:
    def test_the_tool_is_registered_on_the_graph_family(self) -> None:
        """Unregistered is the defect this whole reachability pass exists to catch."""
        registered: list[str] = []

        class _Mcp:
            def tool(self, fn):
                registered.append(fn.__name__)
                return fn

        from mind_mem.mcp.tools.graph import register

        register(_Mcp())
        assert "traverse_graph" in registered

    def test_it_is_classified_as_a_user_tool_not_an_admin_one(self) -> None:
        """Registered but unclassified is unreachable at every scope."""
        from mind_mem.mcp.infra.acl import ADMIN_TOOLS, USER_TOOLS

        assert "traverse_graph" in USER_TOOLS
        assert "traverse_graph" not in ADMIN_TOOLS

    def test_it_answers_at_the_default_user_scope(self, ws) -> None:
        """Read-only impact analysis must not need MIND_MEM_SCOPE=admin."""
        out = _traverse(ws, block_id=ROOT, direction="downstream")
        assert "error" not in out

    def test_a_failed_scope_introspection_denies_the_call(self, ws, monkeypatch) -> None:
        """The ``deny`` sentinel is honoured against user tools too, not just admin ones.

        Degrading a failed token introspection to "user" would make every
        USER_TOOLS entry — this one included — silently reachable during an
        auth outage.
        """
        monkeypatch.setattr(
            "mind_mem.mcp.infra.observability._get_request_scope",
            lambda: "deny",
        )
        out = _traverse(ws, block_id=ROOT)
        assert out["scope"] == "deny"
        assert out["error"] == "Permission denied: authentication context unavailable"
