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

#: v4 flags the sweep turns ON: **every registered flag**, not a chosen few.
#: A surface answering "disabled" is not a surface the sweep has checked, so a
#: content path behind an off-by-default flag is dark to the canary in exactly
#: the way the whole read surface was dark to the one-module tripwire this file
#: replaced. That was not hypothetical here -- the first version of this sweep
#: turned on 14 of the 52 registered flags, leaving 38 flag-gated paths
#: (``long_context_recall``, ``streaming_recall``, ``surprise_retrieval``,
#: ``fusion``, ``context_budget``, ``ingest_serve`` among them) never exercised.
#:
#: Pinned as a literal rather than computed at import, so the sweep's coverage
#: is a committed fact a reader can diff. ``test_no_registered_flag_is_left_dark``
#: checks it against :data:`mind_mem.v4.feature_flags.ALL_V4_FLAGS`, one way
#: round: every registered flag must appear here, and a name here that is no
#: longer registered is harmless (it sets a config key nobody reads) rather than
#: a failure, so a flag deleted elsewhere cannot red-light this file.
SWEEP_FLAGS: tuple[str, ...] = (
    "backpressure",
    "block_kinds",
    "block_metadata",
    "bootstrap_corpus",
    "chat",
    "circuit_breaker",
    "cognitive_kernel",
    "compliance_export",
    "context_budget",
    "contradiction_states",
    "contradiction_stream",
    "contraindicates_edges",
    "core_export",
    "embedding_fallback",
    "embedding_pipeline",
    "entity_observations",
    "evidence",
    "federation",
    "fusion",
    "granularity_align",
    "health",
    "hnsw_kind_index",
    "idle_ingest",
    "ingest_serve",
    "kind_summaries",
    "lint",
    "llm_noise_profile",
    "logging_context",
    "long_context_recall",
    "maintenance_layout",
    "mind_kernels",
    "multi_modal",
    "observability",
    "online_training",
    "pq",
    "prompt_schema",
    "provenance",
    "redaction",
    "retrieval_metrics",
    "rust_hot_path",
    "self_editing",
    "self_heal",
    "streaming_recall",
    "surprise_retrieval",
    "tenant_chains",
    "tenant_kms",
    "time_bounded_recall",
    "trajectory",
    "typed_edges",
    "viewer",
    "vocabulary",
    "world_staleness",
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
    # The fixture path is required, not optional: without it these four bound
    # nothing and every call died in ``TypeError: ... missing 1 required
    # positional argument`` before the tool body ran. Four "no canary" rows
    # measuring the signature. ``test_no_invocation_is_rejected_by_the_signature``
    # is the tripwire that found it and keeps it found.
    "arch_baseline": ({"repo": WS, "fixture": f"{WS}/.arch-mind/fixture.json"},),
    "arch_check_rules": ({"repo": WS, "fixture": f"{WS}/.arch-mind/fixture.json", "rules": None, "mode": "report"},),
    "arch_delta": ({"repo": WS, "before": "", "after": ""},),
    "arch_metric_explain": ({"metric": "coupling", "fixture": f"{WS}/.arch-mind/fixture.json"},),
    "arch_session_end": ({"repo": WS, "fixture": f"{WS}/.arch-mind/fixture.json"},),
    "arch_session_start": ({"repo": WS, "fixture": f"{WS}/.arch-mind/fixture.json", "agent_id": "sweep", "commit_sha": "0" * 40},),
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
    "agent_inject": (
        {"query": "architecture decision", "agent": "claude-code", "limit": 10},
        {"query": "frost telemetry", "agent": "generic", "limit": 10, "scoring_instant": "2026-06-01"},
    ),
    "anchor_history": ({"limit": 10},),
    "anchor_root": ({"chain": "test", "tx_hash": "0x1", "block_height": 1},),
    "arch_history": ({"repo": WS},),
    "audit_model_tool": ({"path": WS},),
    "block_lineage": ({"block_id": QUARANTINED_ID},),
    "build_core": ({"namespace": "sweep", "version": "1.0"},),
    "calibration_feedback": ({"query_id": "q1", "block_ids_useful": QUARANTINED_ID},),
    "calibration_stats": ({},),
    "category_summary": ({"topic": "architecture", "limit": 10}, {"topic": "frost telemetry", "limit": 10}),
    "chat_with_memory": (
        {"question": "architecture decision", "limit": 5},
        {"question": "frost telemetry", "limit": 5, "generator": "extractive", "on_invalid": "raise", "require_in_evidence": True},
    ),
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
    "export_memory": (
        {},
        {"format": "jsonl", "include_metadata": True, "max_blocks": 10000},
    ),
    "find_similar": ({"block_id": QUARANTINED_ID, "limit": 10},),
    "get_block": ({"block_id": ACTIVE_ID}, {"block_id": PENDING_ID}, {"block_id": QUARANTINED_ID}),
    "get_mind_kernel": ({"name": "recall"},),
    "governance_health_bench": ({},),
    "graph": ({"action": "stats"},),
    "graph_query": ({"entity": "frost", "depth": 2},),
    "graph_stats": ({},),
    "hybrid_search": (
        {"query": "architecture decision", "limit": 10},
        {"query": "frost telemetry", "limit": 10, "active_only": True, "explain": True, "scoring_instant": "2026-06-01"},
    ),
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
    "pack_recall_budget": (
        {"query": "architecture decision", "max_tokens": 2000},
        {"query": "frost telemetry", "max_tokens": 2000, "limit": 20, "model": "generic", "scoring_instant": "2026-06-01"},
    ),
    "pipeline_status": ({},),
    "plan_consolidation": ({},),
    "prefetch": ({"signals": "architecture decision"}, {"signals": "frost telemetry", "limit": 5}),
    "project_profile": ({"name": "frost", "top_k": 5},),
    "propagate_staleness": ({"seed_block_ids": QUARANTINED_ID},),
    "propose_update": ({"block_type": "decision", "statement": "a swept statement", "rationale": "swept"},),
    # Every dispatch mode, not just the default one. ``recall`` in ``public.py``
    # is a nine-mode dispatcher, and a sweep that calls ``mode="auto"`` only is
    # the same defect one level down from the one-module tripwire: it enumerates
    # one leg of the surface it claims to cover. ``test_every_recall_mode_is_swept``
    # reads the mode list out of the dispatcher's own source, so a tenth mode
    # fails the build until it is swept.
    "recall": (
        {"query": "architecture decision", "limit": 10},
        {"query": "frost telemetry", "mode": "bm25", "active_only": False, "scoring_instant": "2026-06-01"},
        {"query": "frost telemetry", "mode": "hybrid", "active_only": True},
        {"query": "frost telemetry", "backend": "hybrid"},
        {"query": "frost telemetry", "mode": "similar", "block_id": QUARANTINED_ID},
        {"query": "frost telemetry", "mode": "axis", "axes": "lexical,semantic", "weights": "0.5,0.5"},
        {"query": "frost telemetry", "mode": "pack", "max_tokens": 2000},
        {"query": "frost telemetry", "mode": "prefetch", "signals": "architecture decision"},
        {"query": "frost telemetry", "mode": "classify"},
        {"query": "frost telemetry", "mode": "diagnostics"},
    ),
    "recall_with_axis": (
        {"query": "architecture decision", "limit": 10},
        {
            "query": "frost telemetry",
            "axes": "lexical,semantic",
            "weights": "0.5,0.5",
            "limit": 10,
            "active_only": True,
            "adversarial": True,
            "allow_rotation": False,
            "scoring_instant": "2026-06-01",
        },
    ),
    "recall_with_guardrails": (
        {"query": "architecture decision", "limit": 10},
        {
            "query": "frost telemetry",
            "tool": "bash",
            "command": "ls",
            "intent": "architecture decision",
            "paths": ["src/"],
            "limit": 10,
            "active_only": True,
        },
    ),
    "recall_with_persona": tuple({"query": "architecture decision", "persona": p, "limit": 10} for p in PERSONAS_SWEPT)
    + ({"query": "frost telemetry", "persona": "detailed", "limit": 10, "active_only": True, "scoring_instant": "2026-06-01"},),
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


# ---------------------------------------------------------------------------
# Tripwire C — no registered flag leaves a content path dark
# ---------------------------------------------------------------------------


def test_no_registered_flag_is_left_dark() -> None:
    """Every v4 flag is ON during the sweep.

    A content path behind an off-by-default flag is invisible to a canary that
    never turns it on -- the same blindness, one layer in, as a tripwire that
    enumerated one module. Checked one way round on purpose: a flag registered
    and not swept is a hole, a name swept and no longer registered is inert.
    """
    from mind_mem.v4.feature_flags import ALL_V4_FLAGS

    dark = sorted(set(ALL_V4_FLAGS) - set(SWEEP_FLAGS))
    assert not dark, f"registered v4 flags the sweep never enables: {dark}. Add each to SWEEP_FLAGS so the paths they gate are exercised."


def test_tripwire_c_fails_on_a_flag_the_sweep_leaves_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "SWEEP_FLAGS", tuple(f for f in SWEEP_FLAGS if f != "streaming_recall"))
    with pytest.raises(AssertionError, match="streaming_recall"):
        test_no_registered_flag_is_left_dark()


# ---------------------------------------------------------------------------
# Tripwire D — a content tool is swept across its whole argument space
# ---------------------------------------------------------------------------


def swept_parameters(tool: str) -> set[str]:
    """Every keyword the sweep passes to *tool*, across all its invocations."""
    invocations = TOOL_INVOCATIONS.get(tool, ())
    keys: set[str] = set()
    for kwargs in invocations:
        keys |= set(kwargs)
    return keys


def tool_parameters(tool: str) -> set[str]:
    """The live signature's named parameters."""
    import importlib
    import inspect

    fn = getattr(importlib.import_module(tool_module(tool)), tool)
    return {
        p.name
        for p in inspect.signature(fn).parameters.values()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def test_every_content_tool_sweeps_its_whole_parameter_space() -> None:
    """One argument set is one leg of a surface, not the surface.

    A parameter that changes WHAT a tool serves -- ``active_only``, ``mode``,
    ``format``, ``include_metadata``, ``block_id`` -- is a separate read path,
    and a canary that never passes it has not checked it. Requiring every
    named parameter of every ``content`` tool to appear in at least one
    invocation makes a NEW parameter on a content tool fail the build until
    somebody sweeps it, which is the same contract the classification table
    imposes on a new tool.
    """
    gaps = {}
    for tool in sorted(content_tools()):
        missing = sorted(tool_parameters(tool) - swept_parameters(tool))
        if missing:
            gaps[tool] = missing
    assert not gaps, f"content tools with parameters the canary sweep never passes: {gaps}"


def test_tripwire_d_fails_on_an_unswept_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test-of-the-test: drop a swept keyword, watch it go red."""
    trimmed = dict(TOOL_INVOCATIONS)
    trimmed["export_memory"] = ({},)
    monkeypatch.setattr(sys.modules[__name__], "TOOL_INVOCATIONS", trimmed)
    with pytest.raises(AssertionError, match="include_metadata"):
        test_every_content_tool_sweeps_its_whole_parameter_space()


# ---------------------------------------------------------------------------
# Tripwire E — every dispatch mode of the recall dispatcher is swept
# ---------------------------------------------------------------------------


def declared_recall_modes() -> set[str]:
    """The mode names ``public.recall`` itself declares valid.

    Read out of the dispatcher's own ``valid_modes=[...]`` error payload by AST
    rather than copied into this file, so a tenth mode is swept the day it is
    added instead of the day somebody remembers this list exists.
    """
    import ast

    source = (_ROOT / "src" / "mind_mem" / "mcp" / "tools" / "public.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "recall")
    modes: set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.keyword) and node.arg == "valid_modes" and isinstance(node.value, ast.List):
            modes |= {elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)}
    assert modes, "could not read valid_modes out of public.recall; the mode tripwire would be vacuous"
    return modes


def swept_recall_modes() -> set[str]:
    """The modes the sweep actually dispatches, default included.

    ``backend=`` is the v3.1.x alias the dispatcher folds into ``mode`` when
    ``mode`` is left at ``auto``, so it counts as the mode it becomes.
    """
    modes: set[str] = set()
    for kwargs in TOOL_INVOCATIONS["recall"]:
        mode = kwargs.get("mode", "auto")
        if mode == "auto" and kwargs.get("backend"):
            mode = kwargs["backend"]
        modes.add(mode)
    return modes


def test_every_recall_mode_is_swept() -> None:
    unswept = sorted(declared_recall_modes() - swept_recall_modes())
    assert not unswept, f"recall dispatch modes the canary sweep never calls: {unswept}"


def test_tripwire_e_fails_on_an_unswept_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    trimmed = dict(TOOL_INVOCATIONS)
    trimmed["recall"] = tuple(kw for kw in TOOL_INVOCATIONS["recall"] if kw.get("mode") != "axis")
    monkeypatch.setattr(sys.modules[__name__], "TOOL_INVOCATIONS", trimmed)
    with pytest.raises(AssertionError, match="axis"):
        test_every_recall_mode_is_swept()


# ---------------------------------------------------------------------------
# Tripwire F — a shadowed tool name is swept as the SERVER serves it
# ---------------------------------------------------------------------------


def _registration_order() -> list[str]:
    """Tool module stems in the order ``mcp/server.py`` registers them.

    Pure AST over the server source: the ``from mind_mem.mcp.tools import (x as
    _alias)`` bindings, then the ``_alias.register(mcp)`` call sequence.
    """
    import ast

    source = (_ROOT / "src" / "mind_mem" / "mcp" / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    alias_to_stem: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "mind_mem.mcp.tools":
            for alias in node.names:
                alias_to_stem[alias.asname or alias.name] = alias.name
    order: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in alias_to_stem
        ):
            order.append(alias_to_stem[node.func.value.id])
    assert order, "could not read the registration order out of mcp/server.py"
    return order


def _modules_defining(name: str) -> list[str]:
    from count_mcp_tools import _tool_names, _tool_source_files

    return [path.stem for path in _tool_source_files() if name in _tool_names(path)]


def test_a_shadowed_tool_name_is_swept_as_the_server_serves_it() -> None:
    """``recall`` is registered twice. The sweep must exercise the winner.

    ``public.recall`` (the nine-mode dispatcher) and ``recall.recall`` (the
    ranked-pipeline leg) share a name. FastMCP keeps the LAST registration --
    measured, not assumed: importing ``mind_mem.mcp.server`` logs "Component
    already exists: tool:recall" and ``get_tool("recall").fn`` resolves to
    ``mind_mem.mcp.tools.public``, which is why ``server.py`` registers
    ``public`` last on purpose.

    ``tool_module`` resolves by sorted filename, and today that agrees by
    coincidence (``public.py`` sorts before ``recall.py``). Rename either file
    and the sweep would start exercising the shadowed function while the server
    served the other one -- 121 green rows over a tool nobody calls. This pins
    the agreement instead of relying on the alphabet.
    """
    order = _registration_order()
    mismatched = {}
    for name in sorted(registered_tools()):
        defining = _modules_defining(name)
        if len(defining) < 2:
            continue
        winner = max(defining, key=lambda stem: order.index(stem) if stem in order else -1)
        swept = tool_module(name).rsplit(".", 1)[-1]
        if swept != winner:
            mismatched[name] = {"server serves": winner, "sweep exercises": swept}
    assert not mismatched, f"shadowed tool names the sweep resolves differently from the server: {mismatched}"


def test_tripwire_f_would_catch_a_reordered_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flip the registration order; the sweep's resolution must go stale."""
    reordered = [stem for stem in _registration_order() if stem != "public"]
    reordered.insert(0, "public")
    monkeypatch.setattr(sys.modules[__name__], "_registration_order", lambda: reordered)
    with pytest.raises(AssertionError, match="recall"):
        test_a_shadowed_tool_name_is_swept_as_the_server_serves_it()
