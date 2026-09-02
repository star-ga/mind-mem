"""The registry-wide read-surface classification — the committed table.

The tripwire this replaces enumerated ONE module. ``tests/test_quarantine_redteam.py``
asked ``recall.register(_Probe())`` for its tool names and asserted that set had
not grown, which is a real tripwire over a sixth of the surface. Two leaks lived
in the other five sixths for as long as they existed: ``get_block`` (USER scope)
served quarantined block content verbatim, and ``export_memory`` exported it. The
recall family was clean the whole time, so the tripwire was green the whole time.

So the unit of enumeration here is the **tool registry**, discovered exactly the
way ``version-check`` discovers it (``scripts/count_mcp_tools._tool_names``, a
static AST walk over every ``register(mcp)`` module). Every registered tool must
appear in :data:`CLASSIFICATION` with one of two verdicts:

``content``
    The response can carry workspace block content. It must be exercised against
    the three-status canary seed in ``test_read_surface_admission.py`` AND it
    must be shown to actually reach the corpus, or its "no canary" result would
    only mean the invocation never got that far.

``no-content``
    The response never carries workspace block content. It is swept anyway — the
    sweep is cheap and a misclassification is exactly what it exists to catch —
    but it carries no reach obligation.

Two properties keep the table honest, and neither rests on anyone's judgement:

1. The table is checked against the registry in BOTH directions, so a new tool
   fails the build until someone classifies it, and a deleted tool cannot linger.
2. The ``content`` set is checked against **measured behaviour**: the sweep
   records which tools actually reached the seeded corpus, and
   ``test_read_surface_admission.py`` asserts that measured set equals the
   ``content`` set. A tool that starts returning block content joins the reach
   set and breaks the build until it is reclassified; a ``content`` tool that
   stops reaching (a flag default flipped, a dependency vanished) breaks it too,
   rather than degrading into a canary check over an error string.

The invocation table lives here as well, because "classified content" and "swept"
have to be the same list or the classification is decoration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

CONTENT = "content"
NO_CONTENT = "no-content"
VERDICTS = frozenset({CONTENT, NO_CONTENT})


def registered_tools() -> set[str]:
    """Every registered MCP tool name, by the discovery ``version-check`` trusts.

    Static AST over the ``register(mcp)`` modules — not a runtime import — so
    the set is identical in a CI job that does not install the ``mcp`` extra.
    """
    from count_mcp_tools import _tool_names, _tool_source_files

    names: set[str] = set()
    for path in _tool_source_files():
        names.update(_tool_names(path))
    return names


def tool_module(name: str) -> str:
    """Import path of the module defining registered tool *name*."""
    from count_mcp_tools import _tool_names, _tool_source_files

    for path in _tool_source_files():
        if name in _tool_names(path):
            if path.name == "mcp_server.py":
                return "mind_mem.mcp_server"
            return f"mind_mem.mcp.tools.{path.stem}"
    raise KeyError(name)


# ---------------------------------------------------------------------------
# The table. Every registered tool, one of two verdicts.
# ---------------------------------------------------------------------------

CLASSIFICATION: dict[str, str] = {
    # --- content: the response can carry workspace block content -----------
    "agent_inject": CONTENT,
    "category_summary": CONTENT,
    "chat_with_memory": CONTENT,
    "export_memory": CONTENT,
    "get_block": CONTENT,
    "hybrid_search": CONTENT,
    "pack_recall_budget": CONTENT,
    "prefetch": CONTENT,
    "recall": CONTENT,
    "recall_with_axis": CONTENT,
    "recall_with_guardrails": CONTENT,
    "recall_with_persona": CONTENT,
    # --- no-content --------------------------------------------------------
    "add_block_edge": NO_CONTENT,
    "anchor_history": NO_CONTENT,
    "anchor_root": NO_CONTENT,
    "approve_apply": NO_CONTENT,
    "approve_edge": NO_CONTENT,
    "arch_baseline": NO_CONTENT,
    "arch_check_rules": NO_CONTENT,
    "arch_delta": NO_CONTENT,
    "arch_history": NO_CONTENT,
    "arch_metric_explain": NO_CONTENT,
    "arch_session_end": NO_CONTENT,
    "arch_session_start": NO_CONTENT,
    "audit_model_tool": NO_CONTENT,
    "block_lineage": NO_CONTENT,
    "build_core": NO_CONTENT,
    "calibration_feedback": NO_CONTENT,
    "calibration_stats": NO_CONTENT,
    "check_dead_ends": NO_CONTENT,
    "check_guardrails": NO_CONTENT,
    "compact": NO_CONTENT,
    "compiled_truth": NO_CONTENT,
    # MEASURED, not assumed: its ``subject`` field reads Statement/Subject/
    # content and is a content channel by construction, but the recall rows it
    # projects carry ``excerpt`` rather than ``Statement``, so it emits block
    # ids and an empty subject. The reach check in
    # ``test_read_surface_admission`` is what keeps this honest -- the day the
    # projection starts carrying text, the reach set grows and the build fails
    # until this row says ``CONTENT``.
    "compile_truth_walkthrough": NO_CONTENT,
    "compiled_truth_add_evidence": NO_CONTENT,
    "compiled_truth_contradictions": NO_CONTENT,
    "compiled_truth_load": NO_CONTENT,
    "core": NO_CONTENT,
    "decrypt_file": NO_CONTENT,
    "delete_memory_item": NO_CONTENT,
    "dream_cycle": NO_CONTENT,
    "encrypt_file": NO_CONTENT,
    "entity_add_observation": NO_CONTENT,
    "entity_observations": NO_CONTENT,
    "export_core": NO_CONTENT,
    "find_similar": NO_CONTENT,
    "get_mind_kernel": NO_CONTENT,
    "governance_health_bench": NO_CONTENT,
    "graph": NO_CONTENT,
    "graph_add_edge": NO_CONTENT,
    "graph_query": NO_CONTENT,
    "graph_stats": NO_CONTENT,
    "index_stats": NO_CONTENT,
    "intent_classify": NO_CONTENT,
    "kernels": NO_CONTENT,
    "lint": NO_CONTENT,
    "lint_autofix": NO_CONTENT,
    "list_contradictions": NO_CONTENT,
    "list_cores": NO_CONTENT,
    "list_edge_proposals": NO_CONTENT,
    "list_evidence": NO_CONTENT,
    "list_mind_kernels": NO_CONTENT,
    "load_core": NO_CONTENT,
    "memory_evolution": NO_CONTENT,
    "memory_health": NO_CONTENT,
    "memory_verify": NO_CONTENT,
    "mic_convert_tool": NO_CONTENT,
    "mic_inspect_tool": NO_CONTENT,
    "mind_mem_verify": NO_CONTENT,
    "observe_signal": NO_CONTENT,
    "ontology_load": NO_CONTENT,
    "ontology_validate": NO_CONTENT,
    "outcome_stats": NO_CONTENT,
    "pipeline_status": NO_CONTENT,
    "plan_consolidation": NO_CONTENT,
    "project_profile": NO_CONTENT,
    "propagate_staleness": NO_CONTENT,
    "propose_edge": NO_CONTENT,
    "propose_update": NO_CONTENT,
    "reindex": NO_CONTENT,
    "reindex_dirty": NO_CONTENT,
    "reject_edge": NO_CONTENT,
    "reject_proposal": NO_CONTENT,
    "report_outcome": NO_CONTENT,
    "resume_brief": NO_CONTENT,
    "retrieval_diagnostics": NO_CONTENT,
    "rollback_proposal": NO_CONTENT,
    "scan": NO_CONTENT,
    "sign_model_tool": NO_CONTENT,
    "signal_stats": NO_CONTENT,
    "similar_trajectories": NO_CONTENT,
    "staged_change": NO_CONTENT,
    "stale_blocks": NO_CONTENT,
    "stream_status": NO_CONTENT,
    "traverse_graph": NO_CONTENT,
    "unload_core": NO_CONTENT,
    "validate_block": NO_CONTENT,
    "vault_scan": NO_CONTENT,
    "vault_sync": NO_CONTENT,
    "verify_chain": NO_CONTENT,
    "verify_merkle": NO_CONTENT,
    "verify_model_tool": NO_CONTENT,
}


def content_tools() -> set[str]:
    return {name for name, verdict in CLASSIFICATION.items() if verdict == CONTENT}


# ---------------------------------------------------------------------------
# The invocation table — how the sweep calls each tool.
# ---------------------------------------------------------------------------

#: Placeholder substituted with the seeded workspace path at call time.
WS = "$WS"

#: v4 flags the sweep turns ON. A surface answering "disabled" is not a surface
#: the sweep has checked, so every gated tool that can be reached offline is
#: reached. Pinned here rather than left to a default so the sweep's coverage
#: is a committed fact.
SWEEP_FLAGS: tuple[str, ...] = (
    "block_metadata",
    "chat",
    "compliance_export",
    "core_export",
    "entity_observations",
    "evidence",
    "granularity_align",
    "kind_summaries",
    "lint",
    "provenance",
    "redaction",
    "trajectory",
    "typed_edges",
)

#: Block ids seeded by the sweep. ``ACTIVE_ID`` is never passed as an argument
#: (asserted below), so its appearance in a response is proof the tool reached
#: the corpus rather than an echo of its own input.
#: Personas ``recall_with_persona`` is swept through. ``detailed`` is the
#: identity projection and the widest surface; ``brief`` truncates. Sweeping
#: only the narrow one would check the projection, not the tool.
PERSONAS_SWEPT: tuple[str, ...] = ("brief", "detailed", "technical")

ACTIVE_ID = "D-20260101-001"
PENDING_ID = "D-20260102-001"
QUARANTINED_ID = "D-20260103-001"

#: tool -> the argument sets the sweep calls it with. A tool with no entry is
#: not swept, and ``test_read_surface_admission`` fails the build for any
#: ``content`` tool in that position.
TOOL_INVOCATIONS: dict[str, tuple[dict, ...]] = {
    "add_block_edge": ({"src": QUARANTINED_ID, "dst": PENDING_ID, "kind": "derives"},),
    "approve_apply": ({"proposal_id": "no-such-proposal", "dry_run": True},),
    "approve_edge": ({"proposal_id": "no-such-proposal"},),
    "arch_baseline": ({"repo": WS},),
    "arch_check_rules": ({"repo": WS},),
    "arch_delta": ({"repo": WS, "before": "", "after": ""},),
    "arch_metric_explain": ({"metric": "coupling", "fixture": ""},),
    "arch_session_end": ({"repo": WS},),
    "arch_session_start": ({"repo": WS},),
    "compiled_truth_add_evidence": ({"entity_id": "frost", "observation": "a swept observation", "source": QUARANTINED_ID},),
    "decrypt_file": ({"file_path": "decisions/DECISIONS.md"},),
    "graph_add_edge": ({"subject": "a", "predicate": "relates_to", "object": "b", "source_block_id": QUARANTINED_ID},),
    "mic_convert_tool": ({"input": "{}", "input_format": "mic", "output_format": "mic-b"},),
    "mind_mem_verify": ({},),
    "ontology_load": ({"spec": "{}"},),
    "propose_edge": ({"subject": "a", "predicate": "relates_to", "object": "b", "source_block_id": QUARANTINED_ID},),
    "reject_edge": ({"proposal_id": "no-such-proposal"},),
    "reject_proposal": ({"proposal_id": "no-such-proposal", "reason": "swept"},),
    "rollback_proposal": ({"receipt_ts": "1970-01-01T00:00:00Z", "reason": "swept"},),
    "sign_model_tool": ({"path": WS},),
    "vault_sync": ({"vault_root": WS, "block_id": QUARANTINED_ID, "relative_path": "swept.md", "body": "swept"},),
    "agent_inject": ({"query": "architecture decision", "agent": "claude-code", "limit": 10},),
    "anchor_history": ({"limit": 10},),
    "anchor_root": ({"chain": "test", "tx_hash": "0x1", "block_height": 1},),
    "arch_history": ({"repo": WS},),
    "audit_model_tool": ({"path": WS},),
    "block_lineage": ({"block_id": QUARANTINED_ID},),
    "build_core": ({"namespace": "sweep", "version": "1.0"},),
    "calibration_feedback": ({"query_id": "q1", "block_ids_useful": QUARANTINED_ID},),
    "calibration_stats": ({},),
    "category_summary": ({"topic": "architecture", "limit": 10}, {"topic": "frost telemetry", "limit": 10}),
    "chat_with_memory": ({"question": "architecture decision", "limit": 5},),
    "check_dead_ends": ({"tool": "bash", "command": "ls", "intent": "architecture decision"},),
    "check_guardrails": ({"tool": "bash", "command": "ls", "intent": "architecture decision"},),
    "compact": ({"dry_run": True},),
    "compile_truth_walkthrough": ({"topic": "architecture decision", "limit": 10},),
    "compiled_truth": ({"action": "load"},),
    "compiled_truth_contradictions": ({"entity_id": "frost"},),
    "compiled_truth_load": ({"entity_id": "frost"},),
    "core": ({"action": "list"},),
    "delete_memory_item": ({"block_id": PENDING_ID},),
    "dream_cycle": ({},),
    "encrypt_file": ({"file_path": "decisions/DECISIONS.md"},),
    "entity_add_observation": ({"entity": "frost", "fact": "a swept fact"},),
    "entity_observations": ({"entity": "frost"},),
    "export_core": ({"name": "sweep-1.0.mmcore", "format": "markdown"},),
    "export_memory": ({},),
    "find_similar": ({"block_id": QUARANTINED_ID, "limit": 10},),
    "get_block": ({"block_id": ACTIVE_ID}, {"block_id": PENDING_ID}, {"block_id": QUARANTINED_ID}),
    "get_mind_kernel": ({"name": "recall"},),
    "governance_health_bench": ({},),
    "graph": ({"action": "stats"},),
    "graph_query": ({"entity": "frost", "depth": 2},),
    "graph_stats": ({},),
    "hybrid_search": ({"query": "architecture decision", "limit": 10},),
    "index_stats": ({},),
    "intent_classify": ({"query": "architecture decision"},),
    "kernels": ({"action": "list"},),
    "lint": ({},),
    "lint_autofix": ({"finding_id": "unknown-finding"},),
    "list_contradictions": ({},),
    "list_cores": ({},),
    "list_edge_proposals": ({},),
    "list_evidence": ({"limit": 50},),
    "list_mind_kernels": ({},),
    "load_core": ({"filename": "sweep-1.0.mmcore"},),
    "memory_evolution": ({"block_id": QUARANTINED_ID},),
    "memory_health": ({},),
    "memory_verify": ({},),
    "mic_inspect_tool": ({"input": "{}", "input_format": "mic"},),
    "observe_signal": ({"session_id": "s", "previous_query": "frost", "new_query": "frost telemetry"},),
    "ontology_validate": ({"block": '{"_id": "' + QUARANTINED_ID + '"}', "type_name": "decision"},),
    "outcome_stats": ({"top_n": 10},),
    "pack_recall_budget": ({"query": "architecture decision", "max_tokens": 2000},),
    "pipeline_status": ({},),
    "plan_consolidation": ({},),
    "prefetch": ({"signals": "architecture decision"},),
    "project_profile": ({"name": "frost", "top_k": 5},),
    "propagate_staleness": ({"seed_block_ids": QUARANTINED_ID},),
    "propose_update": ({"block_type": "decision", "statement": "a swept statement", "rationale": "swept"},),
    "recall": ({"query": "architecture decision", "limit": 10},),
    "recall_with_axis": ({"query": "architecture decision", "limit": 10},),
    "recall_with_guardrails": ({"query": "architecture decision", "limit": 10},),
    "recall_with_persona": tuple({"query": "architecture decision", "persona": p, "limit": 10} for p in PERSONAS_SWEPT),
    "reindex": ({},),
    "reindex_dirty": ({"dry_run": True},),
    "report_outcome": ({"block_ids": QUARANTINED_ID, "outcome": "success"},),
    "resume_brief": ({},),
    "retrieval_diagnostics": ({},),
    "scan": ({},),
    "signal_stats": ({},),
    "similar_trajectories": ({"task": "architecture decision", "limit": 10},),
    "staged_change": ({"phase": "propose"},),
    "stale_blocks": ({"limit": 20},),
    "stream_status": ({},),
    "traverse_graph": ({"block_id": QUARANTINED_ID, "depth": 2},),
    "unload_core": ({"namespace": "sweep"},),
    "validate_block": ({"text": "Statement: architecture decision"},),
    "vault_scan": ({"vault_root": WS},),
    "verify_chain": ({},),
    "verify_merkle": ({"block_id": QUARANTINED_ID, "content_hash": ""},),
    "verify_model_tool": ({"path": WS},),
}


# ---------------------------------------------------------------------------
# Tripwires A and B — the table and the registry must agree, both ways
# ---------------------------------------------------------------------------


def test_every_registered_tool_is_classified() -> None:
    """Tripwire A. A new tool fails the build until someone classifies it."""
    unclassified = sorted(registered_tools() - set(CLASSIFICATION))
    assert not unclassified, (
        f"registered MCP tools with no read-surface classification: {unclassified}. "
        f"Add each to CLASSIFICATION as '{CONTENT}' (its response can carry workspace "
        f"block content -- then give it an entry in TOOL_INVOCATIONS so the canary "
        f"sweep exercises it) or '{NO_CONTENT}' (it cannot)."
    )


def test_the_classification_names_no_unregistered_tool() -> None:
    """Tripwire B. A renamed or removed tool cannot linger as a stale row."""
    ghosts = sorted(set(CLASSIFICATION) - registered_tools())
    assert not ghosts, f"CLASSIFICATION names tools that are not registered: {ghosts}"


def test_every_verdict_is_one_of_the_two_states() -> None:
    bad = {name: v for name, v in CLASSIFICATION.items() if v not in VERDICTS}
    assert not bad, f"unknown verdicts: {bad}"


def test_the_active_block_id_is_never_an_argument() -> None:
    """Reach is measured by the ACTIVE id appearing in a response.

    If any invocation passed that id in, a tool echoing its own arguments back
    would read as "reached the corpus" and the positive control would be a
    tautology. Nothing may pass it.
    """
    for tool, invocations in TOOL_INVOCATIONS.items():
        for kwargs in invocations:
            if tool == "get_block":
                continue  # the one surface whose whole job is to be given an id
            for key, value in kwargs.items():
                assert ACTIVE_ID not in str(value), f"{tool}({key}={value!r}) passes the active block id; reach would be an echo"


# ---------------------------------------------------------------------------
# Test-of-the-test — the tripwire has to be able to fail
# ---------------------------------------------------------------------------


def test_tripwire_a_fails_on_an_unclassified_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fixture tool that returns raw block text and is NOT classified.

    The tripwire is only worth its line count if it goes red on exactly this.
    Registration is simulated at the discovery seam, because that is the seam
    the tripwire reads -- patching the assertion instead would prove nothing.
    """
    real = registered_tools()
    monkeypatch.setattr(sys.modules[__name__], "registered_tools", lambda: real | {"leak_raw_block_text"})
    with pytest.raises(AssertionError, match="leak_raw_block_text"):
        test_every_registered_tool_is_classified()


def test_tripwire_b_fails_on_a_ghost_row(monkeypatch: pytest.MonkeyPatch) -> None:
    real = registered_tools()
    monkeypatch.setattr(sys.modules[__name__], "registered_tools", lambda: real - {"get_block"})
    with pytest.raises(AssertionError, match="get_block"):
        test_the_classification_names_no_unregistered_tool()


def test_discovery_sees_the_whole_registry_not_one_module() -> None:
    """The defect that let two leaks live: enumeration scoped to one module.

    ``recall.py`` holds 8 of the tools; the registry holds 102. A discovery
    that returns only the recall family is the bug, not a smaller version of
    the fix.
    """
    from count_mcp_tools import _tool_names

    found = registered_tools()
    recall_only = set(_tool_names(_ROOT / "src" / "mind_mem" / "mcp" / "tools" / "recall.py"))
    assert len(found) > 5 * len(recall_only), f"discovery looks module-scoped: {len(found)} tools vs {len(recall_only)} in recall.py"
    modules = {tool_module(name) for name in found}
    assert len(modules) > 15, f"discovery reached only {len(modules)} modules: {sorted(modules)}"


def test_the_ast_discovery_matches_a_live_registration_probe() -> None:
    """Static discovery must see what a live ``register(mcp)`` would register.

    ``count_mcp_tools`` parses; the server imports. If the two ever disagree the
    classification is over a registry nobody actually serves. Any module whose
    import fails here (an uninstalled extra) is reported, not skipped.
    """
    import importlib

    from count_mcp_tools import _tool_names, _tool_source_files

    live: set[str] = set()
    unimportable: list[str] = []
    for path in _tool_source_files():
        if not _tool_names(path):
            continue
        modname = "mind_mem.mcp_server" if path.name == "mcp_server.py" else f"mind_mem.mcp.tools.{path.stem}"
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            unimportable.append(f"{modname}: {type(exc).__name__}")
            continue

        class _Probe:
            def tool(self, fn):  # noqa: ANN001, ANN202 - FastMCP's shape
                live.add(getattr(fn, "__name__", str(fn)))
                return fn

        mod.register(_Probe())

    assert not unimportable, f"tool modules that would not import: {unimportable}"
    assert live == registered_tools(), (
        f"static discovery and live registration disagree; "
        f"only-static={sorted(registered_tools() - live)} only-live={sorted(live - registered_tools())}"
    )


def test_every_registered_tool_has_an_invocation() -> None:
    """Coverage. A classification nobody exercises is a label, not a gate.

    ``no-content`` is a claim about behaviour, and the sweep is what checks it.
    A tool with no invocation is a tool the sweep never called, so its verdict
    rests on somebody's reading of the code -- which is exactly how the two
    leaks survived. Every registered tool gets called; there is no exempt list
    for this, deliberately, because an exempt list is where the next one hides.
    """
    missing = sorted(registered_tools() - set(TOOL_INVOCATIONS))
    assert not missing, f"registered tools with no sweep invocation: {missing}"
    ghosts = sorted(set(TOOL_INVOCATIONS) - registered_tools())
    assert not ghosts, f"TOOL_INVOCATIONS names tools that are not registered: {ghosts}"


#: Tools that name corpus block IDS they were not given -- withheld blocks
#: included -- without serving any of their content. Naming an id is a weaker
#: disclosure than serving text and a legitimate one for these three (``lint``
#: reports defects an operator must repair, and the two pipeline surfaces
#: report which blocks need re-stamping; a maintenance surface that hid the
#: blocks needing maintenance would be useless). It is still a channel, so the
#: set is pinned: a fourth tool joining it is a decision somebody makes on
#: purpose, not a diff nobody reads.
ID_DISCLOSING: frozenset[str] = frozenset({"lint", "pipeline_status", "reindex_dirty"})
