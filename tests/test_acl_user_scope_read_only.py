"""No USER-scope MCP tool may write, in any argument combination.

Architecture review F14: seven tools classified ``USER_TOOLS`` wrote state in
at least one mode -- ``add_block_edge``, ``stale_blocks(clear_block_id)``,
``memory_evolution(action="update")``, ``compiled_truth_add_evidence``,
``dream_cycle`` (which always ran with ``dry_run=False``), ``vault_sync`` and
``ontology_load`` (a process-global registry, so on HTTP one client changed
what every other client's ``ontology_validate`` saw). Measuring every user
tool rather than only the named seven found more of the same class:
``sign_model_tool`` (writes key material and sidecars at any host path),
``reindex_dirty(dry_run=False)`` (rewrites corpus files), ``build_core`` /
``export_core`` (artifact writes), ``load_core`` / ``unload_core``
(process-global core registry), the ``core`` / ``compiled_truth`` dispatcher
branches that reach those, ``governance_health_bench`` (its drift probe
writes signals into the workspace drift store) and ``graph_query`` (looking
up an unknown entity minted it into the knowledge graph).

The rule the ACL now carries: a tool that writes in any mode is ADMIN, or its
write mode is rejected at user scope by ``WRITE_MODE_GATES`` before the body
runs. This module proves it by EXECUTION, not by set membership:

* every tool in ``USER_TOOLS`` is called through the registered FastMCP
  callable, at user scope, with the most write-shaped arguments it accepts,
  on a seeded tmp workspace; the workspace (logically -- SQLite compared by
  table rows, not bytes), an out-of-workspace directory and the process-global
  registries are snapshotted around the call and must not change, except for
  the declared read-accounting stores and the pinned HITL staging set;
* positive control: the same write-shaped call at ADMIN scope must change the
  state the user-scope run did not -- otherwise the snapshot is blind and the
  user-scope pass is vacuous.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable

import pytest

from mind_mem.mcp.infra import rate_limit as _rate_limit
from mind_mem.mcp.infra.acl import (
    ADMIN_TOOLS,
    STAGING_STORE_READERS,
    USER_STAGING_WRITES,
    USER_TOOLS,
    WRITE_MODE_GATES,
    cross_client_reader_config_keys,
    staging_store_token,
)
from mind_mem.mcp.infra.workspace import use_workspace

D1 = "D-20260101-001"
D2 = "D-20260101-002"

_DECISIONS = "".join(
    f"[{bid}]\nDate: 2026-01-01\nStatus: active\nScope: global\n"
    f"Statement: Use frost telemetry for {name}.\nRationale: measured rollout.\nTags: frost\n"
    "Sources: -\nSupersedes: -\n\n"
    for bid, name in ((D1, "alpha"), (D2, "beta"))
)

_ONTOLOGY = json.dumps({"version": "acl-probe-1", "types": {"THING": {"name": "THING", "required": ["a"]}}})

#: Stores a READ is allowed to touch: the served-read ledger (the governance
#: receipt of what was served), the retrieval log + co-retrieval counters and
#: the per-block access counters A-MEM ranking is built on. These record that
#: a read happened; none of them is reachable as a client-chosen write, and
#: ``block_meta`` importance -- the column ``memory_evolution(update)`` writes --
#: is snapshotted separately below so this allowance cannot mask it.
READ_ACCOUNTING = frozenset(
    {
        ".mind-mem-ledger/served.head",
        ".mind-mem-ledger/served.jsonl",
        ".mind-mem-index/recall.db::retrieval_log",
        ".mind-mem-index/recall.db::co_retrieval",
        # The computed-abstention prediction a recall WITH a stated intended use
        # records so report_outcome can score it later. A record that a read was
        # served, keyed by the run's content-derived identity; the table has no
        # reader on the ranking path (tests/test_abstention_store.py poisons it
        # and shows recall unchanged). Created by the first prediction only.
        ".mind-mem-index/recall.db::recall_abstention",
        ".mind-mem-index/recall.db::sqlite_sequence",
        ".mind-mem/block_meta.db::block_meta",
        # The 9-type intent router's adaptive confidence weights, updated by
        # the recall path it classifies for.
        "memory/intent_router_stats.json",
    }
)


def _probe_args(ext: str) -> dict[str, dict[str, Any]]:
    """The most write-shaped arguments each tool accepts."""
    model = os.path.join(ext, "model")
    vault = os.path.join(ext, "vault")
    return {
        # --- moved to ADMIN_TOOLS (write in every mode) -----------------
        "add_block_edge": dict(src=D1, dst=D2, kind="cites"),
        "build_core": dict(namespace="aclprobe", version="2.0"),
        # Recall reads calibration_feedback on every query (CalibrationManager
        # in _recall_core) and reweights served scores: a cross-client write.
        "calibration_feedback": dict(query_id="q1", block_ids_useful=[D1]),
        # recall_outcome has a serving reader (the validity gate's outcome
        # attribution): a row is served once that opt-in turns on, whenever
        # it was written, so every mode of report_outcome is admin.
        "report_outcome": dict(block_ids=[D1], outcome="failure", task_id="t1"),
        "compiled_truth_add_evidence": dict(entity_id="ent1", observation="observed"),
        "export_core": dict(name="aclprobe-1.0.mmcore"),
        "governance_health_bench": {},
        "load_core": dict(filename="aclprobe-1.0.mmcore"),
        "ontology_load": dict(spec=_ONTOLOGY, make_active=True),
        "sign_model_tool": dict(path=model, generate_key_prefix=os.path.join(ext, "key")),
        "unload_core": dict(namespace="aclprobe"),
        "vault_sync": dict(vault_root=vault, block_id=D1, relative_path="n.md", body="hello"),
        # --- user tools with a gated write mode -------------------------
        "compiled_truth": dict(action="add_evidence", entity_id="ent1", observation="observed"),
        "core": dict(action="build", namespace="aclprobe", version="2.0"),
        "dream_cycle": dict(auto_repair=True),
        "graph": dict(action="add_edge", subject="a", predicate="related_to", object="b", source_block_id=D1),
        "memory_evolution": dict(block_id=D1, action="update"),
        "reindex_dirty": dict(dry_run=False),
        "stale_blocks": dict(clear_block_id=D1),
        # --- HITL staging (pinned USER_STAGING_WRITES) -------------------
        "observe_signal": dict(
            session_id="s",
            previous_query="frost telemetry alpha",
            new_query="frost telemetry alpha beta",
        ),
        "propose_edge": dict(subject="a", predicate="related_to", object="b", source_block_id=D1),
        "propose_entity_merge": dict(winner_id="alpha", loser_id="beta", rationale="duplicate surface forms"),
        # --- read tools --------------------------------------------------
        "agent_inject": dict(query="frost"),
        "anchor_history": {},
        "audit_model_tool": dict(path=model, include_manifest=True),
        "block_lineage": dict(block_id=D1),
        "calibration_stats": {},
        "category_summary": dict(topic="frost"),
        "chat_with_memory": dict(question="frost?"),
        "check_dead_ends": dict(tool="bash", command="ls"),
        "check_guardrails": dict(tool="bash", command="ls"),
        "compile_truth_walkthrough": dict(topic="frost"),
        "compiled_truth_contradictions": dict(entity_id="ent1"),
        "compiled_truth_load": dict(entity_id="ent1"),
        "entity_observations": dict(entity="ent1"),
        "find_similar": dict(block_id=D1),
        "get_block": dict(block_id=D1),
        "get_mind_kernel": dict(name="recall"),
        "graph_query": dict(entity="never-seen-entity"),
        "graph_stats": {},
        "hybrid_search": dict(query="frost"),
        "index_stats": {},
        "intent_classify": dict(query="frost"),
        "kernels": dict(action="list"),
        "lint": {},
        "list_contradictions": {},
        "list_cores": {},
        "list_edge_proposals": {},
        "list_entity_merge_proposals": {},
        "list_evidence": {},
        "list_mind_kernels": {},
        "memory_health": {},
        "mic_convert_tool": dict(input="{}"),
        "mic_inspect_tool": dict(input="{}"),
        "mind_mem_verify": {},
        "ontology_validate": dict(block="{}", type_name="THING"),
        "outcome_stats": {},
        "pack_recall_budget": dict(query="frost"),
        "pipeline_status": {},
        "plan_consolidation": {},
        "prefetch": dict(signals="frost"),
        "project_profile": {},
        "propagate_staleness": dict(seed_block_ids=D1),
        # The most write-shaped recall: a stated intended use is what makes it record
        # a prediction (the only write a recall can add to its read accounting).
        "recall": dict(query="frost", intended_use="governed_write"),
        "recall_with_axis": dict(query="frost"),
        "recall_with_guardrails": dict(query="frost"),
        "recall_with_persona": dict(query="frost"),
        "resume_brief": {},
        "retrieval_diagnostics": {},
        "scan": {},
        "signal_stats": {},
        "similar_trajectories": dict(task="frost"),
        "stream_status": {},
        "traverse_graph": dict(block_id=D1),
        "validate_block": dict(text=_DECISIONS.split("\n\n")[0]),
        "vault_scan": dict(vault_root=vault),
        "verify_merkle": dict(block_id=D1, content_hash="00"),
        "verify_model_tool": dict(path=model),
    }


#: Tools whose positive control (admin scope) must show a change. Every tool
#: moved out of USER_TOOLS and every gated write mode is here.
WRITE_CONTROLS = (
    "add_block_edge",
    "build_core",
    "calibration_feedback",
    "compiled_truth",
    "compiled_truth_add_evidence",
    "core",
    "dream_cycle",
    "export_core",
    "governance_health_bench",
    "graph",
    "load_core",
    "memory_evolution",
    "ontology_load",
    "reindex_dirty",
    "report_outcome",
    "sign_model_tool",
    "stale_blocks",
    "unload_core",
    "vault_sync",
)


#: Staging tools whose write-shaped probe is a GATED mode: the positive
#: control for the staging allowance runs the default (staging-only) mode.
STAGING_ARGS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def _db_tables(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            rows = sorted(repr(r) for r in con.execute(f'SELECT * FROM "{table}"'))
            if rows:
                out[table] = hashlib.sha256("\n".join(rows).encode()).hexdigest()
        if any(t == "block_meta" for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")):
            imp = sorted(repr(r) for r in con.execute("SELECT id, importance FROM block_meta WHERE importance IS NOT 1.0"))
            if imp:
                out["block_meta.importance"] = hashlib.sha256("\n".join(imp).encode()).hexdigest()
    finally:
        con.close()
    return out


def _snapshot_files(root: str) -> dict[str, str]:
    """Logical snapshot: SQLite by table content, other files by bytes.

    WAL/SHM side files are skipped (a read checkpoints them) and an empty
    table is absent (a read that only creates a schema wrote nothing).
    """
    out: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            if fname.endswith(("-wal", "-shm", "-journal")):
                continue
            path = os.path.join(dirpath, fname)
            rel = Path(os.path.relpath(path, root)).as_posix()  # allowlists use '/' on every OS
            with open(path, "rb") as handle:
                data = handle.read()
            if data.startswith(b"SQLite format 3"):
                for table, digest in _db_tables(path).items():
                    out[f"{rel}::{table}"] = digest
            else:
                out[rel] = hashlib.sha256(data).hexdigest()
    return out


def _snapshot_globals() -> dict[str, Any]:
    from mind_mem.mcp.tools import _helpers

    ontologies = _helpers._ontology_registry()
    active = ontologies.active()
    return {
        "ontology_versions": tuple(ontologies.versions()),
        "ontology_active": active.version if active else None,
        "cores": json.dumps(_helpers._core_registry().stats(), sort_keys=True, default=str),
    }


def _diff(before: dict[str, str], after: dict[str, str]) -> set[str]:
    return {k for k in set(before) | set(after) if before.get(k) != after.get(k)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _enable_dotted(cfg: dict[str, Any], dotted: str) -> None:
    """Turn ``a.b.c`` on as ``{"enabled": true}``, keeping nested sections."""
    node = cfg
    for part in dotted.split("."):
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node["enabled"] = True


def _build_template(v4_flags: dict[str, Any], config_keys: tuple[str, ...] = ()) -> Any:
    from mind_mem import sqlite_index
    from mind_mem.block_metadata import BlockMetadataManager, block_meta_db_path
    from mind_mem.causal_graph import CausalGraph
    from mind_mem.init_workspace import init
    from mind_mem.mcp.tools import core as core_tools

    root = tempfile.mkdtemp(prefix="mm_acl_ro_")
    ws = os.path.join(root, "ws")
    ext = os.path.join(root, "ext")
    init(ws)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write("\n" + _DECISIONS)
    cfg_path = os.path.join(ws, "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg.setdefault("v4", {}).update(v4_flags)
    for dotted in config_keys:
        _enable_dotted(cfg, dotted)
    with open(cfg_path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle)
    sqlite_index.build_index(ws)
    # A staleness flag for stale_blocks(clear) to clear.
    CausalGraph(ws)
    con = sqlite3.connect(os.path.join(ws, ".mind-mem-index", "causal.db"))
    con.execute("INSERT INTO staleness_flags (block_id, reason, source_id) VALUES (?, 'seed', ?)", (D1, D2))
    con.commit()
    con.close()
    # A block_meta row whose importance memory_evolution(update) will move.
    mgr = BlockMetadataManager(block_meta_db_path(ws))
    mgr.record_access([D1, D2], query="frost")
    con = sqlite3.connect(block_meta_db_path(ws))
    con.execute("UPDATE block_meta SET importance = 0.123")
    con.commit()
    con.close()
    # Two knowledge-graph entities, so propose_entity_merge can stage, and a
    # real graph file, so graph_query's no-mint check runs against one.
    from mind_mem.knowledge_graph import KnowledgeGraph
    from mind_mem.mcp.tools._helpers import _kg_path

    kg = KnowledgeGraph(_kg_path(ws))
    kg.entities.resolve("alpha")
    kg.entities.resolve("beta")
    kg.close()
    # A built core for load/unload/export.
    with use_workspace(ws):
        built = json.loads(core_tools.build_core.__wrapped__("aclprobe", "1.0", ""))
    assert "path" in built, built
    os.makedirs(os.path.join(ext, "model"))
    with open(os.path.join(ext, "model", "w.bin"), "wb") as handle:
        handle.write(b"x" * 64)
    os.makedirs(os.path.join(ext, "vault"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


_BASE_V4 = {"core_export": {"enabled": True}, "lint": {"enabled": True}}


@pytest.fixture(scope="module")
def template() -> Any:
    yield from _build_template(_BASE_V4)


@pytest.fixture(scope="module")
def template_v4_on() -> Any:
    """The template with EVERY v4 flag and EVERY cross-client reader on.

    Flag-gated side writes (a sidecar a default-mode call emits only when its
    flag is set) are invisible to the base template, and so is a staged store
    that a reader serves to other clients only under an opt-in. Both lists are
    generated from the code -- ``ALL_V4_FLAGS`` and the enabling keys of the
    serving readers in ``STAGING_STORE_READERS`` (which an AST scan in
    tests/test_acl_staging_readers.py holds complete) -- so a new flag or a
    new reader cannot be left out of this measurement.
    """
    from mind_mem.v4.feature_flags import ALL_V4_FLAGS

    yield from _build_template({flag: {"enabled": True} for flag in ALL_V4_FLAGS}, cross_client_reader_config_keys())


@pytest.fixture(scope="module")
def registered() -> dict[str, Callable[..., Any]]:
    import asyncio

    from mind_mem.mcp.server import mcp

    return {tool.name: tool.fn for tool in asyncio.run(mcp.list_tools())}


@pytest.fixture
def run_tool(template: str, registered: dict[str, Callable[..., Any]], monkeypatch: pytest.MonkeyPatch):
    """Run one registered tool at *scope* on a fresh copy of the template."""
    yield from _runner(template, registered, monkeypatch)


@pytest.fixture
def run_tool_v4_on(template_v4_on: str, registered: dict[str, Callable[..., Any]], monkeypatch: pytest.MonkeyPatch):
    """``run_tool`` on the every-v4-flag-on template."""
    yield from _runner(template_v4_on, registered, monkeypatch)


def _runner(template: str, registered: dict[str, Callable[..., Any]], monkeypatch: pytest.MonkeyPatch):
    from mind_mem.mcp.tools import _helpers

    saved_ontology = _helpers._ONTOLOGY_REGISTRY
    saved_cores = _helpers._CORE_REGISTRY
    _helpers._ONTOLOGY_REGISTRY = None
    _helpers._CORE_REGISTRY = None
    workdirs: list[str] = []

    def _run(name: str, scope: str, args: dict[str, Any] | None = None) -> tuple[str, set[str], dict[str, Any]]:
        work = tempfile.mkdtemp(prefix="mm_acl_run_")
        workdirs.append(work)
        shutil.copytree(template, work, dirs_exist_ok=True)
        ws = os.path.join(work, "ws")
        ext = os.path.join(work, "ext")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
        monkeypatch.setenv("MIND_MEM_CONFIG", os.path.join(ws, "mind-mem.json"))
        monkeypatch.setenv("MIND_MEM_VAULT_ALLOWLIST", ext)
        monkeypatch.setenv("MIND_MEM_SCOPE", scope)
        monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
        with _rate_limit._rate_limiters_lock:
            _rate_limit._rate_limiters.clear()
        _helpers._ONTOLOGY_REGISTRY = None
        _helpers._CORE_REGISTRY = None
        if name in ("unload_core", "export_core"):
            with use_workspace(ws):
                _helpers._core_registry().load(os.path.join(ws, "memory", "cores", "aclprobe-1.0.mmcore"))
        call_args = _probe_args(ext)[name] if args is None else args
        globals_before = _snapshot_globals()
        files_before = _snapshot_files(work)
        with use_workspace(ws):
            result = registered[name](**call_args)
        changed = {k.split("/", 1)[1] if k.startswith("ws/") else k for k in _diff(files_before, _snapshot_files(work))}
        globals_after = _snapshot_globals()
        global_changes = {k: (globals_before[k], globals_after[k]) for k in globals_before if globals_before[k] != globals_after[k]}
        return str(result), changed, global_changes

    try:
        yield _run
    finally:
        _helpers._ONTOLOGY_REGISTRY = saved_ontology
        _helpers._CORE_REGISTRY = saved_cores
        for work in workdirs:
            shutil.rmtree(work, ignore_errors=True)
        with _rate_limit._rate_limiters_lock:
            _rate_limit._rate_limiters.clear()


# ---------------------------------------------------------------------------
# The enumeration
# ---------------------------------------------------------------------------


def test_the_registered_surface_is_the_one_measured(registered: dict[str, Callable[..., Any]]) -> None:
    """All 107 tools are classified, and every user tool has a probe."""
    assert len(registered) == 107
    assert set(registered) == ADMIN_TOOLS | USER_TOOLS
    missing = sorted(USER_TOOLS - set(_probe_args("/x")))
    assert missing == [], f"user-scope tools with no write probe: {missing}"


@pytest.mark.parametrize("name", sorted(USER_TOOLS))
def test_user_scope_tool_writes_nothing(name: str, run_tool) -> None:
    result, changed, global_changes = run_tool(name, "user")
    allowed = READ_ACCOUNTING | set(USER_STAGING_WRITES.get(name, ()))
    unexpected = sorted(changed - allowed)
    assert unexpected == [], f"{name} at USER scope wrote {unexpected}; result={result[:300]}"
    assert global_changes == {}, f"{name} at USER scope changed process-global state {global_changes}"


def test_staging_exceptions_are_exactly_pinned() -> None:
    """The HITL staging set is a ratchet: adding to it is a reviewed decision."""
    assert set(USER_STAGING_WRITES) == {
        "observe_signal",
        "propose_edge",
        "propose_entity_merge",
    }
    assert set(USER_STAGING_WRITES) <= USER_TOOLS


@pytest.mark.parametrize("name", sorted(USER_STAGING_WRITES))
def test_staging_tools_really_write_their_declared_store(name: str, run_tool) -> None:
    """Positive control for the staging allowance: it is not a blind spot."""
    result, changed, _globals = run_tool(name, "user", STAGING_ARGS.get(name))
    assert changed & set(USER_STAGING_WRITES[name]), f"{name} wrote nothing visible: {result[:300]}"


# ---------------------------------------------------------------------------
# Positive controls -- the snapshot can see each write
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", WRITE_CONTROLS)
def test_admin_scope_write_is_visible(name: str, run_tool) -> None:
    result, changed, global_changes = run_tool(name, "admin")
    real = changed - READ_ACCOUNTING
    assert real or global_changes, f"{name} at ADMIN scope changed nothing -- the user-scope pass would be vacuous; result={result[:300]}"


@pytest.mark.parametrize("name", WRITE_CONTROLS)
def test_user_scope_write_is_denied_with_a_scope_error(name: str, run_tool) -> None:
    # dream_cycle's probe is auto_repair=True -- the gated mode; its default
    # mode is a user read, pinned by the next test.
    result, changed, global_changes = run_tool(name, "user")
    assert "requires admin scope" in result, result[:300]
    assert not (changed - READ_ACCOUNTING) and not global_changes


def test_dream_cycle_default_mode_is_a_read_at_user_scope(run_tool) -> None:
    result, changed, _globals = run_tool("dream_cycle", "user", {})
    payload = json.loads(result)
    assert "error" not in payload, payload
    assert "total_findings" in payload
    assert not (changed - READ_ACCOUNTING), changed
    # Same call at admin persists the integrity summary.
    _result, admin_changed, _g = run_tool("dream_cycle", "admin", {})
    assert any(k.startswith("memory/dream-cycle-") for k in admin_changed), admin_changed


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("stale_blocks", {}),
        ("memory_evolution", {"block_id": D1}),
        ("reindex_dirty", {"dry_run": True}),
        ("core", {"action": "list"}),
        ("compiled_truth", {"action": "load", "entity_id": "ent1"}),
        ("graph", {"action": "stats"}),
    ],
)
def test_read_modes_of_gated_tools_still_work_at_user_scope(name: str, args: dict[str, Any], run_tool) -> None:
    result, changed, global_changes = run_tool(name, "user", args)
    assert "requires admin scope" not in result, result[:300]
    allowed = READ_ACCOUNTING | set(USER_STAGING_WRITES.get(name, ()))
    assert not (changed - allowed) and not global_changes


def test_write_mode_gates_name_only_user_tools() -> None:
    """A gate on an admin tool is dead code; a gate on an unknown name is stale."""
    assert set(WRITE_MODE_GATES) <= USER_TOOLS


def test_write_mode_gate_binds_positional_arguments(template: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate binds the signature, so a positional write arg is still seen."""
    from mind_mem.mcp.tools import memory_ops

    monkeypatch.setenv("MIND_MEM_SCOPE", "user")
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters.clear()
    with use_workspace(os.path.join(template, "ws")):
        result = memory_ops.stale_blocks(20, D1)
    assert "requires admin scope" in result, result


# ---------------------------------------------------------------------------
# Flag-gated side writes -- the same measurement with every v4 flag on
# ---------------------------------------------------------------------------

#: The sidecars report_outcome's DEFAULT mode emits when a v4 flag is on. A
#: trajectory file is served verbatim to every client by the user-scope
#: similar_trajectories tool, and the noise profile feeds calibration_stats:
#: both are cross-client publications, not HITL staging.
_OUTCOME_SIDECARS = {
    "trajectory": "trajectories/",
    "llm_noise_profile": "intelligence/llm_profiles.json",
}

_OUTCOME_DEFAULT = dict(block_ids=[D1], outcome="success", task_id="deploy the frost service", tool_id="bash", evidence="ok")


@pytest.mark.parametrize("name", sorted(USER_TOOLS))
def test_user_scope_tool_writes_nothing_with_every_v4_flag_on(name: str, run_tool_v4_on) -> None:
    result, changed, global_changes = run_tool_v4_on(name, "user")
    allowed = READ_ACCOUNTING | set(USER_STAGING_WRITES.get(name, ()))
    unexpected = sorted(changed - allowed)
    assert unexpected == [], f"{name} at USER scope (v4 on) wrote {unexpected}; result={result[:300]}"
    assert global_changes == {}, f"{name} at USER scope (v4 on) changed process-global state {global_changes}"


@pytest.mark.parametrize("name", sorted(USER_STAGING_WRITES))
def test_staging_tools_write_only_their_store_with_every_v4_flag_on(name: str, run_tool_v4_on) -> None:
    """The staging pin is exact under every flag, not only the default config."""
    result, changed, global_changes = run_tool_v4_on(name, "user", STAGING_ARGS.get(name))
    unexpected = sorted(changed - READ_ACCOUNTING - set(USER_STAGING_WRITES[name]))
    assert unexpected == [], f"{name} at USER scope (v4 on) wrote {unexpected}; result={result[:300]}"
    assert global_changes == {}


def test_report_outcome_default_mode_sidecars_are_admin_scope(run_tool_v4_on) -> None:
    """With a sidecar flag on, the default mode is a cross-client write.

    User scope gets the scope error and nothing lands; the identical call at
    admin scope writes both sidecars (positive control: the snapshot sees
    them, so the user-scope pass is not blind).
    """
    result, changed, _g = run_tool_v4_on("report_outcome", "user", _OUTCOME_DEFAULT)
    assert "requires admin scope" in result, result[:300]
    assert not (changed - READ_ACCOUNTING), changed

    result, changed, _g = run_tool_v4_on("report_outcome", "admin", _OUTCOME_DEFAULT)
    assert json.loads(result).get("status") == "recorded", result[:300]
    assert any(k.startswith(_OUTCOME_SIDECARS["trajectory"]) for k in changed), changed
    assert _OUTCOME_SIDECARS["llm_noise_profile"] in changed, changed


@pytest.mark.parametrize("flag", sorted(_OUTCOME_SIDECARS))
def test_each_report_outcome_sidecar_flag_alone_is_admin_only(flag: str, run_tool, template: str) -> None:
    """With one sidecar flag on, the user call is denied and writes nothing;
    the admin call writes that sidecar (positive control per flag)."""
    cfg_path = os.path.join(template, "ws", "mind-mem.json")
    with open(cfg_path, encoding="utf-8") as handle:
        original = handle.read()
    cfg = json.loads(original)
    cfg["v4"][flag] = {"enabled": True}
    try:
        with open(cfg_path, "w", encoding="utf-8") as handle:
            json.dump(cfg, handle)
        result, changed, _g = run_tool("report_outcome", "user", _OUTCOME_DEFAULT)
        assert "requires admin scope" in result, result[:300]
        assert not (changed - READ_ACCOUNTING), changed
        _result, admin_changed, _g = run_tool("report_outcome", "admin", _OUTCOME_DEFAULT)
        assert any(k.startswith(_OUTCOME_SIDECARS[flag]) for k in admin_changed), admin_changed
    finally:
        with open(cfg_path, "w", encoding="utf-8") as handle:
            handle.write(original)


def test_report_outcome_is_admin_with_every_flag_and_reader_off(run_tool) -> None:
    """No config makes report_outcome staging: rows written now are served as
    soon as an operator turns the outcome-attribution reader on."""
    result, changed, _g = run_tool("report_outcome", "user", _OUTCOME_DEFAULT)
    assert "requires admin scope" in result, result[:300]
    assert not (changed - READ_ACCOUNTING), changed
    _result, admin_changed, _g = run_tool("report_outcome", "admin", _OUTCOME_DEFAULT)
    assert ".mind-mem-index/recall.db::recall_outcome" in admin_changed, admin_changed


def _serves_others(name: str) -> bool:
    return any(
        reader.role == "serving"
        for store in USER_STAGING_WRITES[name]
        for reader in STAGING_STORE_READERS.get(staging_store_token(store), {}).values()
    )


def test_no_staged_store_has_a_serving_reader() -> None:
    """A serving reader -- on or off -- takes its store out of user scope."""
    assert [name for name in USER_STAGING_WRITES if _serves_others(name)] == []
    assert any(r.role == "serving" for readers in STAGING_STORE_READERS.values() for r in readers.values()), (
        "positive control: the reader table still knows the served store"
    )
    assert "report_outcome" in ADMIN_TOOLS
