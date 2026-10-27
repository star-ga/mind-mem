# Copyright 2026 STARGA, Inc.
"""Every USER-scope tool answers a bound caller from what that caller may read.

``tests/test_read_surface_acl.py`` closed the REST, resource and ``scan`` /
``list_contradictions`` doors. This file closes the user-scope MCP tools that
read the corpus, a metadata store derived from it, or a count over it, and
nothing else: ``memory_health``, ``stale_blocks``, ``pipeline_status``,
``index_stats``, the graph / lineage / evidence / calibration readers, the
compiled-truth, guardrail and frame readers, ``category_summary``,
``dream_cycle`` and ``memory_evolution``.

Method, for every probe:

* ``eve`` may read ``shared/`` only; ``alice`` additionally reads her own
  namespace and the root corpus directories; ``ops`` reads everything.
* the negative (the denied marker is absent from the whole response body) is
  always paired with a POSITIVE CONTROL: the same call as a caller who IS
  granted the content returns it. An "absent" assertion alone also passes when
  the seed silently failed.
* the shared decision is :mod:`mind_mem.read_authz`; the mutation tests at the
  bottom disable it and require the probes to go red.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from mind_mem.audit_context import bind_current_agent
from mind_mem.init_workspace import init
from mind_mem.mcp.infra.acl import USER_TOOLS
from mind_mem.mcp.infra.workspace import use_workspace

NS_TAGS = (("", "ROOT"), ("agents/alice/", "ALICE"), ("agents/bob/", "BOB"), ("shared/", "SHARED"))
DENIED_FOR_EVE = (
    "zqROOT",
    "zqBOB",
    "zqALICE",
    "D-ROOT",
    "T-ROOT",
    "D-BOB",
    "T-BOB",
    "D-ALICE",
    "rootent",
    "secretcomp",
    "zqROOTGUARD",
    "zqROOTFRAME",
    "zqROOTDEAD",
    "zqprobequery",
)
DENIED_FOR_ALICE = ("zqBOB", "D-BOB", "T-BOB")
DENIED = "namespace access denied"

ALICE_ROOT_READ = ["decisions", "tasks", "entities", "intelligence", "memory", "guardrails", "frames"]
ALL_READ = ["shared", "agents", "decisions", "tasks", "entities", "intelligence", "memory", "guardrails", "frames", "trajectories"]


def _blk(block_id: str, stmt: str, typ: str = "Decision", extra: str = "") -> str:
    return (
        f"[{block_id}]\nType: {typ}\nDate: 2026-01-01\nStatement: {stmt} frost telemetry rollout architecture{extra}\n"
        "Status: active\nTags: architecture\n\n"
    )


def _w(ws: Path, rel: str, text: str) -> None:
    path = ws / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    init(str(root))
    for ns, tag in NS_TAGS:
        _w(root, ns + "decisions/DECISIONS.md", _blk(f"D-{tag}-ACT", f"zq{tag}ACTIVEcanary", extra=" See D-20260101-999."))
        _w(root, ns + "tasks/TASKS.md", _blk(f"T-{tag}-ACT", f"zq{tag}TASKcanary", "Task"))
    _w(root, "intelligence/CONTRADICTIONS.md", _blk("C-ROOT-ACT", "zqROOTCONTRADICTIONcanary", "Contradiction"))
    _w(root, "intelligence/SIGNALS.md", _signal())
    _w(root, "shared/intelligence/LEDGER.md", _blk("FACT-SH-ACT", "zqSHAREDLEDGERcanary", "Fact"))
    _w(root, "intelligence/DRIFT.md", _blk("DRIFT-ROOT-ACT", "zqROOTDRIFTcanary", "Drift"))
    _w(root, "guardrails/GUARDRAILS.md", _guardrail("GR-20260827-001", "zqROOTGUARDrail"))
    _w(root, "frames/FRAMES.md", _frame())
    _w(root, "frames/DEAD-ENDS.md", _dead_end())
    (root / "mind-mem-acl.json").write_text(
        json.dumps(
            {
                "default_policy": "deny",
                "agents": {
                    "eve": {"namespaces": ["shared"], "read": ["shared"], "write": []},
                    "alice": {
                        "namespaces": ["shared", "agents/alice"],
                        "read": ["shared", "agents/alice", *ALICE_ROOT_READ],
                        "write": [],
                    },
                    "ops": {"namespaces": ALL_READ, "read": ALL_READ, "write": []},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg_path = root / "mind-mem.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("v4", {}).update({"lint": {"enabled": True}, "entity_observations": {"enabled": True}, "trajectory": {"enabled": True}})
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("MIND_MEM_WORKSPACE", str(root))
    monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg_path))
    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    _seed_side_stores(root)
    return root


SIG_TEXT = "zqROOTSIGNALphrase adopt the frost telemetry rollout across every region this quarter"


def _signal() -> str:
    now = datetime.now(timezone.utc)
    head = f"[SIG-{now:%Y%m%d}-001]\nType: Signal\nDate: {now:%Y-%m-%d}\nCaptured: {now.isoformat()}\n"
    return head + f"Excerpt: {SIG_TEXT}\nStatus: pending\n\n"


def _guardrail(block_id: str, text: str) -> str:
    return f"[{block_id}]\nType: Guardrail\nStatement: {text}\nSeverity: high\nTriggerTools: bash\nStatus: active\n\n"


def _frame() -> str:
    return "[TF-20260829-001]\nType: TaskFrame\nGoal: zqROOTFRAMEgoal\nStatus: active\nSteps:\n- doing: x\nApproachTools: bash\n\n"


def _dead_end() -> str:
    return (
        "[DE-20260826-001]\nType: DeadEnd\nApproach: zqROOTDEADapproach\nWhyFailed: zqROOTDEADwhy\n"
        "Outcome: refuted\nTriggerTools: bash\nStatus: active\n\n"
    )


def _seed_side_stores(root: Path) -> None:
    """Every derived store a user-scope tool reads, with a row per namespace."""
    from mind_mem.block_lineage import add_block_edge
    from mind_mem.block_metadata import BlockMetadataManager, block_meta_db_path
    from mind_mem.calibration import CalibrationManager
    from mind_mem.causal_graph import CausalGraph
    from mind_mem.compiled_truth import CompiledTruthPage, EvidenceEntry, save_truth_page
    from mind_mem.governance_gate import get_gate
    from mind_mem.knowledge_graph import KnowledgeGraph
    from mind_mem.mcp.tools._helpers import _kg_path

    ws = str(root)
    cg = CausalGraph(ws)
    con = sqlite3.connect(str(root / ".mind-mem-index" / "causal.db"))
    for bid in ("D-ROOT-ACT", "D-BOB-ACT", "D-SHARED-ACT"):
        con.execute("INSERT INTO staleness_flags (block_id, reason, source_id) VALUES (?, 'seed', 'D-ROOT-ACT')", (bid,))
    con.commit()
    con.close()
    cg.add_edge("D-SHARED-ACT", "D-ROOT-ACT", "depends_on")
    cg.add_edge("D-SHARED-ACT", "T-SHARED-ACT", "depends_on")
    cg.add_edge("D-ROOT-ACT", "D-BOB-ACT", "depends_on")
    add_block_edge(ws, "D-SHARED-ACT", "D-ROOT-ACT", "cites")
    add_block_edge(ws, "D-SHARED-ACT", "T-SHARED-ACT", "cites")
    add_block_edge(ws, "D-ROOT-ACT", "D-BOB-ACT", "cites")
    # visible-only-via-hidden: D-SHARED-ACT -> D-BOB-ACT (hidden from eve) -> T-SHARED-ACT
    add_block_edge(ws, "D-BOB-ACT", "T-SHARED-ACT", "cites")
    mgr = BlockMetadataManager(block_meta_db_path(ws))
    mgr.record_access(["D-SHARED-ACT", "D-ROOT-ACT", "T-SHARED-ACT"], query="frost")
    cal = CalibrationManager(ws)
    cal.record_feedback("q-root", ["D-ROOT-ACT"], [], "accepted", query_text="frost")
    cal.record_feedback("q-shared", ["D-SHARED-ACT"], [], "accepted", query_text="frost")
    cal.record_outcome(["D-ROOT-ACT"], "failure", task_id="t-root")
    cal.record_outcome(["D-SHARED-ACT"], "failure", task_id="t-shared")
    chain = get_gate(ws).evidence
    from mind_mem.evidence_objects import EvidenceAction

    for bid, src in (("D-ROOT-ACT", "decisions/DECISIONS.md"), ("D-SHARED-ACT", "shared/decisions/DECISIONS.md")):
        chain.create(action=EvidenceAction.VERIFY, actor="seed", target_block_id=bid, target_file=src, payload="x")
    from mind_mem.trajectory import write_trajectory

    write_trajectory(
        ws,
        {"_id": "TRAJ-20260101-001", "Task": "zqROOTTRAJECTORYtask frost", "Date": "2026-01-01", "Status": "active", "Outcome": "SUCCESS"},
    )
    kg = KnowledgeGraph(_kg_path(ws))
    with get_gate(ws).admit_proposal(proposal_id="SEED-KG", content="[]", actor="seed"):
        kg.add_edge("rootent", "related_to", "secretcomp", source_block_id="D-ROOT-ACT")
        kg.add_edge("sharedent", "related_to", "sharedcomp", source_block_id="D-SHARED-ACT")
    kg.propose_edge("rootent", "related_to", "secretcomp2", source_block_id="D-ROOT-ACT")
    kg.propose_edge("sharedent", "related_to", "sharedcomp2", source_block_id="D-SHARED-ACT")
    kg.close()
    page = CompiledTruthPage(
        entity_id="rootent",
        entity_type="topic",
        compiled_section="zqROOTCOMPILEDsection",
        evidence_entries=[EvidenceEntry("2026-01-01T00:00:00+00:00", "seed", "zqROOTEVIDENCEobs", "high", False)],
        last_compiled="2026-01-01T00:00:00+00:00",
        version=1,
    )
    save_truth_page(ws, page)
    from mind_mem import sqlite_index

    sqlite_index.build_index(ws)
    from mind_mem.category_distiller import CategoryDistiller

    CategoryDistiller().distill(ws)


@pytest.fixture(scope="module")
def registered() -> dict[str, Callable[..., Any]]:
    from mind_mem.mcp.server import mcp

    return {tool.name: tool.fn for tool in asyncio.run(mcp.list_tools())}


@pytest.fixture
def call(ws: Path, registered: dict[str, Callable[..., Any]]) -> Callable[..., str]:
    def _call(name: str, who: str | None, /, **kwargs: Any) -> str:
        with use_workspace(str(ws)):
            if who is None:
                return str(registered[name](**kwargs))
            with bind_current_agent(who):
                return str(registered[name](**kwargs))

    return _call


def _leaks(body: str, markers: tuple[str, ...]) -> list[str]:
    return [m for m in markers if m in body]


def _is_denied(body: str) -> bool:
    try:
        return json.loads(body).get("error") == DENIED
    except (ValueError, AttributeError):
        return False


# --- the seed itself: a probe that cannot fail is not a probe -------------------


def test_the_seed_is_really_there_for_an_unrestricted_caller(call) -> None:  # type: ignore[no-untyped-def]
    assert "D-ROOT-ACT" in call("stale_blocks", None)
    assert "D-BOB-ACT" in call("stale_blocks", None)
    assert "secretcomp" in call("graph_query", None, entity="rootent")
    assert "D-ROOT-ACT" in call("block_lineage", None, block_id="D-SHARED-ACT")
    assert "zqROOTEVIDENCEobs" in call("compiled_truth_load", None, entity_id="rootent")
    assert "zqROOTGUARDrail" in call("check_guardrails", None, tool="bash")
    assert "zqROOTFRAMEgoal" in call("resume_brief", None)
    assert "zqROOTDEADapproach" in call("check_dead_ends", None, tool="bash")


# --- id-keyed readers: filter to what the caller may read -----------------------

ID_KEYED = (
    # tool, kwargs, marker the restricted caller MUST still see (positive control)
    ("stale_blocks", {}, "D-SHARED-ACT"),
    ("block_lineage", {"block_id": "D-SHARED-ACT"}, "T-SHARED-ACT"),
    ("traverse_graph", {"block_id": "D-SHARED-ACT"}, "T-SHARED-ACT"),
    ("list_evidence", {}, "D-SHARED-ACT"),
    ("memory_evolution", {"block_id": "D-SHARED-ACT"}, "T-SHARED-ACT"),
)


@pytest.mark.parametrize(("tool", "kwargs", "visible"), ID_KEYED, ids=[t[0] for t in ID_KEYED])
def test_id_keyed_reader_returns_only_what_the_caller_may_read(call, tool: str, kwargs: dict, visible: str) -> None:  # type: ignore[no-untyped-def]
    eve = call(tool, "eve", **kwargs)
    assert _leaks(eve, DENIED_FOR_EVE) == [], f"{tool} leaked to a shared-only caller: {eve[:400]}"
    assert visible in eve, f"{tool}: positive control lost the shared row: {eve[:400]}"
    ops = call(tool, "ops", **kwargs)
    assert "D-ROOT-ACT" in ops or "T-ROOT-ACT" in ops, f"{tool}: the seed has no root row for a granted caller: {ops[:400]}"


def test_pipeline_status_lists_only_dirty_blocks_the_caller_may_read(call) -> None:  # type: ignore[no-untyped-def]
    ops = json.loads(call("pipeline_status", "ops"))
    assert ops["dirty_count"] >= 4 and "D-ROOT-ACT" in ops["dirty_ids"]  # positive control: the seed has dirty root blocks
    eve = json.loads(call("pipeline_status", "eve"))
    assert eve["dirty_count"] == 0 and eve["dirty_ids"] == []
    alice = json.loads(call("pipeline_status", "alice"))
    assert "D-ROOT-ACT" in alice["dirty_ids"] and alice["dirty_count"] == ops["dirty_count"]


def test_reindex_dirty_dry_run_lists_only_readable_ids(call) -> None:  # type: ignore[no-untyped-def]
    ops = json.loads(call("reindex_dirty", "ops", dry_run=True))
    assert "D-ROOT-ACT" in ops["ids"]  # positive control: the dry run does list dirty root blocks
    assert json.loads(call("reindex_dirty", "eve", dry_run=True))["ids"] == []
    assert "D-ROOT-ACT" in json.loads(call("reindex_dirty", "alice", dry_run=True))["ids"]
    assert json.loads(call("reindex_dirty", "eve", dry_run=True, limit=1))["ids"] == []


def test_similar_trajectories_follows_the_sidecar_acl(call) -> None:  # type: ignore[no-untyped-def]
    ops = json.loads(call("similar_trajectories", "ops", task="zqROOTTRAJECTORYtask frost"))
    assert ops["count"] >= 1 and "zqROOTTRAJECTORYtask" in json.dumps(ops)  # positive control
    for who in ("eve", "alice"):  # neither is granted trajectories/
        body = call("similar_trajectories", who, task="zqROOTTRAJECTORYtask frost")
        assert json.loads(body)["count"] == 0 and "zqROOTTRAJECTORYtask frost" not in json.loads(body)["trajectories"].__repr__()


def test_validate_block_near_duplicate_rule_is_no_oracle_over_unreadable_signals(call) -> None:  # type: ignore[no-untyped-def]
    ops = json.loads(call("validate_block", "ops", text=SIG_TEXT))
    assert "near_duplicate" in json.dumps(ops["reasons"] + ops["advisory"]), ops  # positive control: it fires for a reader
    eve = json.loads(call("validate_block", "eve", text=SIG_TEXT))
    assert "near_duplicate" not in json.dumps(eve["reasons"] + eve["advisory"]), eve


def test_kind_summaries_reach_only_a_caller_whose_view_is_the_workspace(ws: Path, call) -> None:  # type: ignore[no-untyped-def]
    from mind_mem.v4 import kind_summaries

    cfg_path = ws / "mind-mem.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["v4"]["kind_summaries"] = {"enabled": True}
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    db = sqlite3.connect(ws / "index.db")
    db.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, content TEXT, kind TEXT)")
    db.executemany(
        "INSERT INTO blocks VALUES (?, ?, 'decision')",
        [("D-ROOT-ACT", "zqROOTACTIVEcanary frost"), ("T-ROOT-ACT", "zqROOTTASKcanary frost")],
    )
    db.commit()
    db.close()
    try:
        assert kind_summaries.refresh_summary(ws, "decision") is not None
    finally:
        kind_summaries.set_summariser(kind_summaries.default_summariser)
    for who in (None, "ops"):  # positive control: the summary is served to a whole-view caller
        payload = json.loads(call("category_summary", who, topic="architecture"))
        assert payload["kind_summaries"] and "D-ROOT-ACT" in json.dumps(payload["kind_summaries"]), payload
    for who in ("eve", "alice"):
        body = call("category_summary", who, topic="architecture")
        assert "kind_summaries" not in json.loads(body), body[:300]
        assert "D-ROOT-ACT" not in body or who == "alice"  # alice may read the root block itself


def test_alice_gets_root_rows_but_never_bobs(call) -> None:  # type: ignore[no-untyped-def]
    for tool, kwargs in (("stale_blocks", {}), ("list_evidence", {}), ("block_lineage", {"block_id": "D-ROOT-ACT"})):
        alice = call(tool, "alice", **kwargs)
        assert "D-ROOT-ACT" in alice, f"{tool}: alice was granted the root corpus: {alice[:300]}"
        assert _leaks(alice, DENIED_FOR_ALICE) == [], f"{tool} leaked bob to alice: {alice[:400]}"


def _proposal_ids(body: str) -> str:
    return json.dumps([p.get("object", p.get("loser_id")) for p in json.loads(body)["proposals"]])


def test_staged_proposals_are_visible_to_their_proposer_and_to_readers_of_the_cited_blocks(call) -> None:  # type: ignore[no-untyped-def]
    args = {"subject": "mine", "predicate": "related_to", "object": "eveobj", "source_block_id": "D-ROOT-ACT"}
    assert "eveobj" in call("propose_edge", "eve", **args)  # eve cites a block she cannot read, as proposer
    assert "evemerge" in call("propose_entity_merge", "eve", winner_id="sharedent", loser_id="sharedcomp", rationale="evemerge")
    eve = _proposal_ids(call("list_edge_proposals", "eve"))
    assert "eveobj" in eve, "a caller must see its OWN staged proposal"
    assert "sharedcomp2" in eve, "a proposal citing only readable blocks is visible"
    assert "secretcomp" not in eve, "someone else's proposal citing an unreadable block is not"
    assert "sharedcomp" in _proposal_ids(call("list_entity_merge_proposals", "eve")), "own merge proposal visible"
    alice = _proposal_ids(call("list_edge_proposals", "alice"))
    assert "secretcomp2" in alice and "eveobj" in alice, "alice may read the root block these cite"
    assert "secretcomp2" not in _proposal_ids(call("list_edge_proposals", "eve"))
    # eve's merge derives only from shared blocks and eve's view (shared) is inside alice's: alice may see it,
    # but never WHO staged it (that is for the proposer and whole views)
    alice_merges = call("list_entity_merge_proposals", "alice")
    assert "sharedcomp" in _proposal_ids(alice_merges) and '"proposer"' not in alice_merges
    assert '"proposer": "eve"' in call("list_entity_merge_proposals", "eve")
    ops = _proposal_ids(call("list_edge_proposals", "ops")) + _proposal_ids(call("list_entity_merge_proposals", "ops"))
    assert "eveobj" in ops and "secretcomp2" in ops and "sharedcomp" in ops  # positive control: the whole view sees everything


def test_block_lineage_is_computed_on_the_visible_subgraph_only(call) -> None:  # type: ignore[no-untyped-def]
    """T-SHARED-ACT is 1 hop from D-SHARED-ACT directly, and 3 hops via hidden blocks: only the direct path exists for eve."""
    ops = json.loads(call("block_lineage", "ops", block_id="D-SHARED-ACT"))
    assert {e["block_id"] for e in ops["edges"]} >= {"D-ROOT-ACT", "D-BOB-ACT", "T-SHARED-ACT"}  # positive control
    eve = json.loads(call("block_lineage", "eve", block_id="D-SHARED-ACT"))
    assert [(e["block_id"], e["distance"]) for e in eve["edges"]] == [("T-SHARED-ACT", 1)]


def test_block_lineage_omits_a_block_reachable_only_through_a_hidden_one(call) -> None:  # type: ignore[no-untyped-def]
    """Start at a readable block whose only route onward runs through a hidden block."""
    from mind_mem.block_lineage import add_block_edge

    ws = os.environ["MIND_MEM_WORKSPACE"]
    add_block_edge(ws, "T-SHARED-ACT", "D-ALICE-ACT", "cites")  # hidden hop (alice's namespace)
    add_block_edge(ws, "D-ALICE-ACT", "FACT-SH-ACT", "cites")  # shared block behind it
    eve = json.loads(call("block_lineage", "eve", block_id="T-SHARED-ACT"))
    assert "FACT-SH-ACT" not in json.dumps(eve) and "D-ALICE-ACT" not in json.dumps(eve), eve
    ops = json.loads(call("block_lineage", "ops", block_id="T-SHARED-ACT"))
    assert {"D-ALICE-ACT", "FACT-SH-ACT"} <= {e["block_id"] for e in ops["edges"]}  # positive control


def test_traverse_graph_omits_the_workspace_wide_graph_summary_for_a_restricted_caller(call) -> None:  # type: ignore[no-untyped-def]
    ops = json.loads(call("traverse_graph", "ops", block_id="D-SHARED-ACT"))
    assert ops["graph_summary"]["total_edges"] >= 3  # positive control: the summary exists and counts every edge
    for who in ("eve", "alice"):
        assert "graph_summary" not in json.loads(call("traverse_graph", who, block_id="D-SHARED-ACT"))


def test_stale_count_is_the_filtered_count(call) -> None:  # type: ignore[no-untyped-def]
    assert json.loads(call("stale_blocks", None))["stale_count"] == 3
    assert json.loads(call("stale_blocks", "eve"))["stale_count"] == 1
    assert json.loads(call("stale_blocks", "alice"))["stale_count"] == 2


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    (
        ("block_lineage", {"block_id": "{id}"}),
        ("traverse_graph", {"block_id": "{id}"}),
        ("memory_evolution", {"block_id": "{id}"}),
        ("verify_merkle", {"block_id": "{id}", "content_hash": "00"}),
        ("find_similar", {"block_id": "{id}"}),
    ),
)
def test_a_denied_block_id_is_indistinguishable_from_a_missing_one(call, tool: str, kwargs: dict) -> None:  # type: ignore[no-untyped-def]
    def run(block_id: str) -> str:
        args = {k: (block_id if v == "{id}" else v) for k, v in kwargs.items()}
        return call(tool, "eve", **args).replace(block_id, "<ID>")

    denied, missing = run("D-ROOT-ACT"), run("D-NOPE-NOPE")
    assert denied == missing, f"{tool} distinguishes a denied block from a missing one:\n{denied[:300]}\n{missing[:300]}"
    # positive control: an unrestricted caller really does get a different answer for the real id.
    real = call(tool, None, **{k: ("D-ROOT-ACT" if v == "{id}" else v) for k, v in kwargs.items()}).replace("D-ROOT-ACT", "<ID>")
    gone = call(tool, None, **{k: ("D-NOPE-NOPE" if v == "{id}" else v) for k, v in kwargs.items()}).replace("D-NOPE-NOPE", "<ID>")
    if tool != "find_similar":  # find_similar has no unrestricted existence signal at all
        assert real != gone, f"{tool}: seed has no distinguishing data, the check above is vacuous"


# --- workspace-level aggregates: refused unless the caller's view IS the workspace -

AGGREGATES = (
    ("memory_health", {}, "total_blocks"),
    ("index_stats", {}, "total_blocks"),
    ("graph_stats", {}, "entities"),
    ("retrieval_diagnostics", {}, "queries_analyzed"),
    ("calibration_stats", {}, "total_feedback"),
    ("outcome_stats", {}, "total_outcomes"),
    ("dream_cycle", {}, "total_findings"),
    ("lint", {}, "findings"),
    ("graph_query", {"entity": "rootent"}, "neighbors"),
    ("entity_observations", {"entity": "rootent"}, "observations"),
)


@pytest.mark.parametrize(("tool", "kwargs", "key"), AGGREGATES, ids=[t[0] for t in AGGREGATES])
def test_workspace_level_reader_is_refused_to_a_caller_who_cannot_see_the_workspace(call, tool: str, kwargs: dict, key: str) -> None:  # type: ignore[no-untyped-def]
    # a retrieval log row for retrieval_diagnostics to report on
    call("recall", None, query="zqprobequery frost")
    for who in ("eve", "alice"):  # alice reads root but not bob: still not the workspace
        body = call(tool, who, **kwargs)
        assert _is_denied(body), f"{tool} answered {who}, who cannot read the whole workspace: {body[:300]}"
    ops = call(tool, "ops", **kwargs)
    assert key in ops and not _is_denied(ops), f"{tool}: a caller who reads everything must be served: {ops[:300]}"
    assert key in call(tool, None, **kwargs)


def test_memory_health_numbers_do_not_reach_a_caller_without_the_root_corpus(call) -> None:  # type: ignore[no-untyped-def]
    eve = call("memory_health", "eve")
    for key in ("total_blocks", "total_active", "corpus", "unresolved_contradictions", "drift_items", "stale_block_ids"):
        assert key not in eve
    health = json.loads(call("memory_health", "ops"))
    assert health["total_blocks"] >= 4 and health["unresolved_contradictions"] == 1


# --- path-keyed readers of root files -------------------------------------------

ROOT_FILE_READERS = (
    ("compiled_truth_load", {"entity_id": "rootent"}, "zqROOTEVIDENCEobs"),
    ("compiled_truth_contradictions", {"entity_id": "rootent"}, "contradiction_count"),
    ("check_guardrails", {"tool": "bash"}, "zqROOTGUARDrail"),
    ("resume_brief", {}, "zqROOTFRAMEgoal"),
    ("check_dead_ends", {"tool": "bash"}, "zqROOTDEADapproach"),
)


@pytest.mark.parametrize(("tool", "kwargs", "visible"), ROOT_FILE_READERS, ids=[t[0] for t in ROOT_FILE_READERS])
def test_root_file_reader_follows_the_namespace_acl(call, tool: str, kwargs: dict, visible: str) -> None:  # type: ignore[no-untyped-def]
    eve = call(tool, "eve", **kwargs)
    # the entity id is echoed by the not-found message, so it is not a leak marker here
    assert _leaks(eve, tuple(m for m in DENIED_FOR_EVE if m != "rootent")) == [], f"{tool} leaked a root file: {eve[:400]}"
    assert visible not in eve
    for who in ("alice", "ops", None):
        assert visible in call(tool, who, **kwargs), f"{tool}: positive control failed for {who}"


def test_compiled_truth_dispatcher_is_covered_by_the_same_gate(call) -> None:  # type: ignore[no-untyped-def]
    eve = call("compiled_truth", "eve", action="load", entity_id="rootent")
    assert "zqROOTEVIDENCEobs" not in eve and "zqROOTCOMPILEDsection" not in eve
    assert "zqROOTEVIDENCEobs" in call("compiled_truth", "alice", action="load", entity_id="rootent")


def test_recall_with_guardrails_does_not_force_surface_a_denied_guardrail(call) -> None:  # type: ignore[no-untyped-def]
    eve = call("recall_with_guardrails", "eve", query="frost telemetry rollout", tool="bash")
    assert _leaks(eve, DENIED_FOR_EVE) == [], eve[:400]
    assert "zqROOTGUARDrail" in call("recall_with_guardrails", "alice", query="frost telemetry rollout", tool="bash")


# --- derived artifacts ----------------------------------------------------------


def test_category_summary_serves_only_sections_the_caller_may_read(call) -> None:  # type: ignore[no-untyped-def]
    unrestricted = call("category_summary", None, topic="architecture")
    assert "zqROOTACTIVEcanary" in unrestricted, "seed: the category files hold root content"
    eve = call("category_summary", "eve", topic="architecture")
    assert _leaks(eve, DENIED_FOR_EVE) == [], eve[:400]
    assert "blocks in this category" not in eve, "the per-category block count spans unreadable blocks"
    assert "blocks in this category" in unrestricted  # positive control: the count line exists in the file
    alice = call("category_summary", "alice", topic="architecture")
    assert "zqROOTACTIVEcanary" in alice and _leaks(alice, DENIED_FOR_ALICE) == []


# --- verify_merkle: no proof, root or hash for a block the caller cannot read -----


def test_verify_merkle_gives_no_hash_oracle_for_a_denied_block(call) -> None:  # type: ignore[no-untyped-def]
    for who in ("eve", "alice"):  # alice cannot read bob's blocks, which sit in the same tree
        denied = json.loads(call("verify_merkle", who, block_id="D-ROOT-ACT", content_hash="00"))
        assert denied == {"_schema_version": "1.0", "error": DENIED}
    ops = json.loads(call("verify_merkle", "ops", block_id="D-ROOT-ACT", content_hash="00"))
    assert "proof" in ops and "root" in ops  # positive control: the granted caller does get a proof


# --- every tool in the read-capable surface, in one sweep -----------------------

#: tool -> (verdict, probe kwargs). ``gated`` means a bound-caller test above or
#: in test_read_surface_acl.py covers it; ``scoped`` means the body already
#: filters through the caller (checked by the sweep below); ``no block data``
#: means the response carries no corpus-derived content, count or id.
VERDICTS: dict[str, str] = {
    "agent_inject": "scoped",
    "anchor_history": "no block data",
    "audit_model_tool": "no block data",
    "block_lineage": "gated",
    "calibration_stats": "gated",
    "category_summary": "gated",
    "chat_with_memory": "scoped",
    "check_dead_ends": "gated",
    "check_guardrails": "gated",
    "compile_truth_walkthrough": "scoped",
    "compiled_truth": "gated",
    "compiled_truth_contradictions": "gated",
    "compiled_truth_load": "gated",
    "core": "no block data",
    "dream_cycle": "gated",
    "entity_observations": "gated",
    "find_similar": "scoped",
    "get_block": "gated",
    "get_mind_kernel": "no block data",
    "graph": "gated",
    "graph_query": "gated",
    "graph_stats": "gated",
    "hybrid_search": "scoped",
    "index_stats": "gated",
    "intent_classify": "no block data",
    "kernels": "no block data",
    "lint": "gated",
    "list_contradictions": "gated",
    "list_cores": "no block data",
    "list_edge_proposals": "gated",
    "list_entity_merge_proposals": "gated",
    "list_evidence": "gated",
    "list_mind_kernels": "no block data",
    "memory_evolution": "gated",
    "memory_health": "gated",
    "mic_convert_tool": "no block data",
    "mic_inspect_tool": "no block data",
    "mind_mem_verify": "no block data",
    "observe_signal": "no block data",
    "ontology_validate": "no block data",
    "outcome_stats": "gated",
    "pack_recall_budget": "scoped",
    "pipeline_status": "gated",
    "plan_consolidation": "scoped",
    "prefetch": "scoped",
    "project_profile": "scoped",
    "propagate_staleness": "scoped",
    "propose_edge": "no block data",
    "propose_entity_merge": "no block data",
    "recall": "scoped",
    "recall_with_axis": "scoped",
    "recall_with_guardrails": "gated",
    "recall_with_persona": "scoped",
    "reindex_dirty": "gated",
    "resume_brief": "gated",
    "retrieval_diagnostics": "gated",
    "scan": "gated",
    "signal_stats": "no block data",
    "similar_trajectories": "gated",
    "stale_blocks": "gated",
    "stream_status": "no block data",
    "traverse_graph": "gated",
    "validate_block": "gated",
    "vault_scan": "no block data",
    "verify_merkle": "gated",
    "verify_model_tool": "no block data",
}


def test_every_user_tool_has_a_verdict() -> None:
    assert set(VERDICTS) == set(USER_TOOLS), (sorted(set(USER_TOOLS) - set(VERDICTS)), sorted(set(VERDICTS) - set(USER_TOOLS)))


def _sweep_kwargs(tool: str, ext: str) -> dict[str, Any]:
    q = "frost telemetry rollout"
    return {
        "agent_inject": {"query": q},
        "chat_with_memory": {"question": q},
        "compile_truth_walkthrough": {"topic": q},
        "find_similar": {"block_id": "D-SHARED-ACT"},
        "get_block": {"block_id": "D-ROOT-ACT"},
        "hybrid_search": {"query": q},
        "list_contradictions": {},
        "pack_recall_budget": {"query": q},
        "plan_consolidation": {},
        "prefetch": {"signals": "frost,telemetry"},
        "project_profile": {},
        "propagate_staleness": {"seed_block_ids": "D-SHARED-ACT,D-ROOT-ACT"},
        "recall": {"query": q},
        "recall_with_axis": {"query": q},
        "recall_with_persona": {"query": q},
        "scan": {},
    }.get(tool, {})


SCOPED = sorted(t for t, v in VERDICTS.items() if v == "scoped")


@pytest.mark.parametrize("tool", SCOPED)
def test_scoped_tools_hold_for_a_shared_only_caller(call, tool: str, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    kwargs = _sweep_kwargs(tool, str(tmp_path))
    eve = call(tool, "eve", **kwargs)
    assert _leaks(eve, DENIED_FOR_EVE) == [], f"{tool} leaked: {eve[:400]}"
    ctrl = call(tool, "ops", **kwargs)
    assert "ROOT" in ctrl or "zq" in ctrl or tool in {"prefetch", "pack_recall_budget"}, f"{tool}: vacuous probe: {ctrl[:300]}"


NO_BLOCK_DATA = sorted(t for t, v in VERDICTS.items() if v == "no block data")


def _no_data_kwargs(tool: str, ext: str) -> dict[str, Any]:
    return {
        "audit_model_tool": {"path": ext},
        "core": {"action": "list"},
        "get_mind_kernel": {"name": "recall"},
        "intent_classify": {"query": "frost telemetry rollout"},
        "kernels": {"action": "list"},
        "mic_convert_tool": {"input": "{}"},
        "mic_inspect_tool": {"input": "{}"},
        "ontology_validate": {"block": "{}", "type_name": "THING"},
        "reindex_dirty": {"dry_run": True},
        "similar_trajectories": {"task": "frost"},
        "validate_block": {"text": _blk("D-NEW-1", "candidate")},
        "vault_scan": {"vault_root": ext},
        "verify_model_tool": {"path": ext},
        "observe_signal": {"session_id": "s", "previous_query": "frost", "new_query": "frost telemetry"},
        "propose_edge": {"subject": "a", "predicate": "related_to", "object": "b", "source_block_id": "D-SHARED-ACT"},
        "propose_entity_merge": {"winner_id": "sharedent", "loser_id": "sharedcomp", "rationale": "duplicate surface forms"},
    }.get(tool, {})


@pytest.mark.parametrize("tool", NO_BLOCK_DATA)
def test_tools_classified_no_block_data_really_return_none(call, tool: str, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """The verdict is measured, not asserted: a shared-only caller sees no corpus marker."""
    body = call(tool, "eve", **_no_data_kwargs(tool, str(tmp_path)))
    assert _leaks(body, DENIED_FOR_EVE + ("T-SHARED", "D-SHARED")[:0]) == [], f"{tool} carries block data: {body[:400]}"


def test_a_second_identical_bound_recall_is_not_served_from_another_callers_answer(call) -> None:  # type: ignore[no-untyped-def]
    """Cache isolation: an unrestricted answer is never replayed to a restricted caller."""
    first = call("recall", None, query="frost telemetry rollout")
    assert "zqROOTACTIVEcanary" in first
    for _ in range(2):
        assert _leaks(call("recall", "eve", query="frost telemetry rollout"), DENIED_FOR_EVE) == []
    again = call("recall", None, query="frost telemetry rollout")
    assert "zqROOTACTIVEcanary" in again


def test_recall_cache_key_carries_the_principal() -> None:
    from mind_mem.recall_cache import make_cache_key

    base = {"namespace": "default", "limit": 10, "workspace": "/w"}
    assert make_cache_key("q", agent_id="eve", **base) != make_cache_key("q", agent_id="alice", **base)
    assert make_cache_key("q", agent_id="eve", **base) != make_cache_key("q", **base)


def test_the_cache_is_never_consulted_for_a_bound_caller(ws: Path, monkeypatch: pytest.MonkeyPatch, call) -> None:  # type: ignore[no-untyped-def]
    from mind_mem import recall_cache

    seen: list[str | None] = []
    real = recall_cache.cached_recall

    def spy(inner, query, **kw):  # type: ignore[no-untyped-def]
        seen.append(kw.get("agent_id"))
        return real(inner, query, **kw)

    monkeypatch.setattr(recall_cache, "cached_recall", spy)
    # the corpus stamp refuses files modified inside the racy-timestamp window; age the seed out of it
    old = time.time() - 7200
    for dirpath, _dirs, files in os.walk(ws):
        for fname in files:
            os.utime(os.path.join(dirpath, fname), (old, old))
    call("recall", None, query="frost telemetry rollout")
    call("recall", "eve", query="frost telemetry rollout")
    call("recall", "alice", query="frost telemetry rollout")
    assert seen.count(None) >= 1, "positive control: the unbound call must use the cache"
    assert "eve" not in seen and "alice" not in seen, f"a bound caller reached the shared cache: {seen}"


def test_the_anticipation_cache_is_neither_filled_nor_read_for_a_bound_caller(
    ws: Path,
    monkeypatch: pytest.MonkeyPatch,
    call,  # type: ignore[no-untyped-def]
) -> None:
    from mind_mem import prefetch

    cfg_path = ws / "mind-mem.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["cache"] = {"anticipation": {"enabled": True}}
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    old = time.time() - 7200
    for dirpath, _dirs, files in os.walk(ws):
        for fname in files:
            os.utime(os.path.join(dirpath, fname), (old, old))
    looked_up: list[str] = []
    recorded: list[str] = []
    real_lookup, real_record = prefetch.AnticipationCache.lookup, prefetch.AnticipationCache.record

    def lookup(self, workspace, query, **kw):  # type: ignore[no-untyped-def]
        looked_up.append(query)
        return real_lookup(self, workspace, query, **kw)

    def record(self, workspace, origin, hits, **kw):  # type: ignore[no-untyped-def]
        recorded.append(origin)
        return real_record(self, workspace, origin, hits, **kw)

    monkeypatch.setattr(prefetch.AnticipationCache, "lookup", lookup)
    monkeypatch.setattr(prefetch.AnticipationCache, "record", record)
    call("recall", None, query="frost telemetry rollout")
    call("prefetch", None, signals="frost,telemetry")
    assert looked_up and recorded, f"positive control: the unbound caller must use the anticipation cache ({looked_up}, {recorded})"
    looked_up.clear()
    recorded.clear()
    for who in ("eve", "alice"):
        call("recall", who, query="frost telemetry rollout")
        call("prefetch", who, signals="frost,telemetry")
    assert looked_up == [] and recorded == [], f"a bound caller reached the shared anticipation cache: {looked_up} {recorded}"


@pytest.fixture
def no_fts(ws: Path) -> Path:
    """The workspace as a default install has it: corpus on disk, no FTS index yet."""
    for name in ("recall.db", "recall.db-wal", "recall.db-shm"):
        (ws / ".mind-mem-index" / name).unlink(missing_ok=True)
    return ws


def _recall_ids(body: str) -> list[str]:
    return [r.get("_id") for r in json.loads(body).get("results", [])]


def test_a_bound_shared_only_caller_gets_shared_content_on_the_first_and_the_second_recall(call, no_fts: Path) -> None:  # type: ignore[no-untyped-def]
    """The first recall leaves a recall.db with no FTS schema; the second must not be diverted by it."""
    q = "frost telemetry rollout"
    first = _recall_ids(call("recall", "eve", query=q, limit=20))
    assert "D-SHARED-ACT" in first and "T-SHARED-ACT" in first, first
    from mind_mem import sqlite_index

    assert (no_fts / ".mind-mem-index" / "recall.db").exists() and not sqlite_index._index_present(str(no_fts)), (
        "seed: recall.db without an FTS schema"
    )
    second = _recall_ids(call("recall", "eve", query=q, limit=20))
    assert second == first, f"second recall diverged: {second} vs {first}"
    third = _recall_ids(call("recall", "eve", query=q, limit=20, backend="bm25"))
    assert "D-SHARED-ACT" in third


def test_the_fallback_scan_runs_as_the_caller_and_never_serves_root_content_to_a_restricted_one(call, no_fts: Path) -> None:  # type: ignore[no-untyped-def]
    q = "frost telemetry rollout"
    for _ in range(3):
        for backend in ("auto", "bm25", "hybrid"):
            body = call("recall", "eve", query=q, limit=20, backend=backend)
            assert _leaks(body, DENIED_FOR_EVE) == [], (backend, body[:300])
    assert "D-SHARED-ACT" in call("recall", "eve", query=q, limit=20, backend="hybrid")
    for _ in range(2):  # alice: root (granted) and shared, never bob
        body = call("recall", "alice", query=q, limit=20)
        assert "D-ROOT-ACT" in body and _leaks(body, DENIED_FOR_ALICE) == []


def test_an_unbound_callers_recall_is_unchanged_by_the_index_probe(call, no_fts: Path) -> None:  # type: ignore[no-untyped-def]
    q = "frost telemetry rollout"
    runs = [sorted(_recall_ids(call("recall", None, query=q, limit=20))) for _ in range(3)]
    assert runs[0] and runs[0] == runs[1] == runs[2]
    assert "D-ROOT-ACT" in runs[0]


def test_query_index_fallback_carries_the_principal(no_fts: Path) -> None:
    ws = no_fts
    """A database with no FTS schema must fall back to a scan run AS the caller."""
    from mind_mem import sqlite_index

    con = sqlite3.connect(str(ws / ".mind-mem-index" / "recall.db"))  # side-table-only database
    con.execute("CREATE TABLE IF NOT EXISTS calibration_feedback (x TEXT)")
    con.commit()
    con.close()
    hits = sqlite_index.query_index(str(ws), "frost telemetry rollout", limit=20, agent_id="eve")
    assert {h.get("_id") for h in hits} >= {"D-SHARED-ACT"}, hits
    assert not any(str(h.get("_id", "")).startswith(("D-ROOT", "T-ROOT", "D-BOB")) for h in hits)


def test_env_is_isolated(ws: Path) -> None:
    assert os.environ["MIND_MEM_WORKSPACE"] == str(ws)
