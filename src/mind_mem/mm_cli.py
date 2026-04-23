# Copyright 2026 STARGA, Inc.
"""``mm`` — unified mind-mem CLI for non-MCP agents (v2.7.0).

Usage::

    mm recall "<query>"             # search memory
    mm context "<query>"            # generate token-budgeted snippet
    mm inject --agent <name> "<q>"  # render snippet for a specific agent
    mm vault scan <vault_root>      # list parsed vault blocks (JSON)
    mm vault write <vault_root> <id> --type <t> --body <b>
    mm status                       # workspace summary

Filesystem-watcher integration and the per-agent hook installer
remain deferred (need ``watchdog`` and per-agent setup scripts).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Workspace resolution (mirrors mcp_server._workspace)
# ---------------------------------------------------------------------------


def _workspace() -> str:
    ws = os.environ.get("MIND_MEM_WORKSPACE", "").strip()
    if not ws:
        ws = os.getcwd()
    return os.path.realpath(ws)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_recall(args: argparse.Namespace) -> int:
    from mind_mem.recall import recall

    results = recall(_workspace(), args.query, limit=args.limit, active_only=args.active_only)
    print(json.dumps(results, indent=2, default=str))
    return 0


def _cmd_context(args: argparse.Namespace) -> int:
    from mind_mem.cognitive_forget import pack_to_budget
    from mind_mem.recall import recall

    results = recall(_workspace(), args.query, limit=args.limit, active_only=False)
    if not isinstance(results, list):
        results = [results] if isinstance(results, dict) else []
    packed = pack_to_budget(results, max_tokens=args.max_tokens)
    payload = {
        "query": args.query,
        "included": packed.included,
        "dropped": packed.dropped,
        **packed.as_dict(),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _cmd_inject(args: argparse.Namespace) -> int:
    from mind_mem.agent_bridge import AgentFormatter
    from mind_mem.recall import recall

    fmt = AgentFormatter(max_blocks=args.limit)
    results = recall(_workspace(), args.query, limit=args.limit, active_only=False)
    if not isinstance(results, list):
        results = [results] if isinstance(results, dict) else []
    print(fmt.inject(args.agent, args.query, results), end="")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    ws = _workspace()
    info: dict[str, Any] = {"workspace": ws, "exists": os.path.isdir(ws)}
    cfg_path = os.path.join(ws, "mind-mem.json")
    info["config_exists"] = os.path.isfile(cfg_path)
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        info[f"{sub}_dir_exists"] = os.path.isdir(os.path.join(ws, sub))
    print(json.dumps(info, indent=2))
    return 0 if info["exists"] else 1


def _cmd_vault_scan(args: argparse.Namespace) -> int:
    from mind_mem.agent_bridge import VaultBridge

    bridge = VaultBridge(vault_root=args.vault_root)
    sync_dirs = args.sync_dirs or None
    blocks = bridge.scan(sync_dirs=sync_dirs)
    print(json.dumps([b.as_dict() for b in blocks], indent=2, default=str))
    return 0


def _cmd_install(args: argparse.Namespace) -> int:
    """Install mind-mem config for a single named agent."""
    from mind_mem.hook_installer import install_config

    ws = _workspace()
    try:
        result = install_config(args.agent, ws, dry_run=args.dry_run, force=args.force)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


def _cmd_install_all(args: argparse.Namespace) -> int:
    """Auto-detect installed AI clients and configure all of them."""
    from mind_mem.hook_installer import detect_installed_agents, install_all

    ws = _workspace()
    agents = args.agent or None  # None → auto-detect
    results = install_all(
        ws,
        dry_run=args.dry_run,
        force=args.force,
        agents=agents,
        include_mcp=not getattr(args, "no_mcp", False),
    )
    errored = sum(1 for r in results if "error" in r)
    summary: dict[str, Any] = {
        "workspace": ws,
        "agents": agents if agents is not None else detect_installed_agents(ws),
        "results": results,
        "summary": {
            "written": sum(1 for r in results if r.get("written")),
            "merged": sum(1 for r in results if r.get("merged")),
            "skipped": sum(1 for r in results if r.get("skipped")),
            "errored": errored,
            "total": len(results),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0 if errored == 0 else 1


def _cmd_detect(args: argparse.Namespace) -> int:
    """List AI clients detected on the current machine."""
    from mind_mem.hook_installer import AGENT_REGISTRY, detect_installed_agents

    ws = _workspace()
    detected = detect_installed_agents(ws)
    out = {
        "workspace": ws,
        "detected": detected,
        "details": {
            name: {
                "description": AGENT_REGISTRY[name].description,
                "config_path": AGENT_REGISTRY[name].expand_path(ws),
            }
            for name in detected
        },
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_vault_write(args: argparse.Namespace) -> int:
    from mind_mem.agent_bridge import VaultBlock, VaultBridge

    bridge = VaultBridge(vault_root=args.vault_root)
    block = VaultBlock(
        relative_path=args.relative_path,
        block_id=args.id,
        block_type=args.type,
        title=args.title or args.id,
        body=args.body,
    )
    target = bridge.write(block, overwrite=args.overwrite)
    print(json.dumps({"written": target}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Debug visualization subcommands (v3.2.0)
# ---------------------------------------------------------------------------

# ANSI colour codes — used only when stdout is a TTY
_C_RESET = "\033[0m"
_C_BOLD = "\033[1m"
_C_DIM = "\033[2m"
_C_GREEN = "\033[32m"
_C_YELLOW = "\033[33m"
_C_CYAN = "\033[36m"
_C_RED = "\033[31m"

_INSPECT_FIXED_FIELDS = ("_id", "Statement", "Status", "Date", "Tags")
_INSPECT_SKIP = frozenset({"_source_file", "_line_number", "_raw"})


def _use_color() -> bool:
    return sys.stdout.isatty()


def _c(text: str, *codes: str) -> str:
    if not _use_color():
        return text
    return "".join(codes) + text + _C_RESET


def _build_provenance(workspace: str, block_id: str) -> dict[str, Any]:
    """Gather edges + causal chains for a block from the CausalGraph."""
    from mind_mem.causal_graph import CausalGraph

    graph = CausalGraph(workspace)
    deps = [e.to_dict() for e in graph.dependencies(block_id)]
    chains = graph.causal_chain(block_id, max_depth=3)
    contradiction_types = {"contradicts", "supersedes"}
    all_edges = deps + [e.to_dict() for e in graph.dependents(block_id)]
    contradictions = [e for e in all_edges if e.get("edge_type") in contradiction_types]
    return {
        "block_id": block_id,
        "dependencies": deps,
        "causal_chains": chains,
        "contradictions": contradictions,
    }


def _render_inspect_text(block: dict[str, Any], provenance: dict[str, Any]) -> str:
    lines: list[str] = []
    block_id = block.get("_id", "?")
    lines.append(_c(f"Block: {block_id}", _C_BOLD, _C_CYAN))
    lines.append(_c("─" * 60, _C_DIM))
    for key in _INSPECT_FIXED_FIELDS:
        if key in block and key not in _INSPECT_SKIP:
            lines.append(f"{_c(f'  {key}:', _C_BOLD)} {block[key]}")
    remaining = sorted(k for k in block if k not in _INSPECT_FIXED_FIELDS and k not in _INSPECT_SKIP)
    for key in remaining:
        lines.append(f"{_c(f'  {key}:', _C_DIM)} {block[key]}")

    lines.append("")
    lines.append(_c("Provenance", _C_BOLD, _C_YELLOW))
    lines.append(_c("─" * 60, _C_DIM))
    deps = provenance.get("dependencies", [])
    if deps:
        lines.append(_c("  Direct dependencies:", _C_BOLD))
        for edge in deps:
            tid = edge["target_id"]
            etype = edge["edge_type"]
            w = edge.get("weight", 1.0)
            lines.append(f"  {_c(f'→ {tid}', _C_GREEN)}  [{etype}]  weight={w:.2f}")
    else:
        lines.append(_c("  No direct dependencies.", _C_DIM))

    chains = provenance.get("causal_chains", [])
    if chains:
        lines.append(_c("  Causal chains (depth ≤ 3):", _C_BOLD))
        for chain in chains:
            lines.append("    " + _c(" → ", _C_DIM).join(chain))

    contradictions = provenance.get("contradictions", [])
    if contradictions:
        lines.append(_c("  Contradictions / supersessions:", _C_BOLD))
        for edge in contradictions:
            lines.append(f"  {_c(edge['source_id'], _C_RED)} [{edge['edge_type']}] {_c(edge['target_id'], _C_RED)}")
    return "\n".join(lines)


def _cmd_inspect(args: argparse.Namespace) -> int:
    ws = _workspace()
    from mind_mem.block_store import MarkdownBlockStore

    store = MarkdownBlockStore(ws)
    block = store.get_by_id(args.block_id)
    if block is None:
        print(json.dumps({"error": f"Block not found: {args.block_id}"}, indent=2), file=sys.stderr)
        return 1

    provenance = _build_provenance(ws, args.block_id)
    fmt = getattr(args, "format", "text") or "text"
    if fmt == "json":
        print(json.dumps({"block": block, "provenance": provenance}, indent=2, default=str))
    else:
        print(_render_inspect_text(block, provenance))
    return 0


# ── mm explain ────────────────────────────────────────────────────────────────


def _stage_bar(stages_hit: list[bool]) -> str:
    if _use_color():
        symbols = [_c("◉", _C_GREEN) if s else _c("◯", _C_DIM) for s in stages_hit]
    else:
        symbols = [f"[{'x' if s else ' '}]" for s in stages_hit]
    return " ".join(symbols)


def _cmd_explain(args: argparse.Namespace) -> int:
    ws = _workspace()
    limit = getattr(args, "limit", 10) or 10
    backend = getattr(args, "backend", "auto") or "auto"

    if backend == "hybrid":
        try:
            from mind_mem.hybrid_recall import HybridBackend

            cfg: dict[str, Any] = {}
            cfg_path = os.path.join(ws, "mind-mem.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, encoding="utf-8") as fh:
                    cfg = json.load(fh).get("recall", {})
            results = HybridBackend(cfg).search(ws, args.query, limit=limit)
        except Exception:
            from mind_mem.recall import recall as _recall

            results = _recall(ws, args.query, limit=limit)
    else:
        from mind_mem.recall import recall as _recall

        results = _recall(ws, args.query, limit=limit)

    trace_rows: list[dict[str, Any]] = []
    for rank, r in enumerate(results, 1):
        bm25 = r.get("score", r.get("bm25_score", 0.0))
        vec = r.get("vector_score", r.get("embed_score"))
        rrf = r.get("rrf_score")
        rerank = r.get("rerank_score", r.get("ce_score"))
        trace_rows.append(
            {
                "rank": rank,
                "block_id": r.get("_id", "?"),
                "bm25": round(float(bm25), 4),
                "vector": round(float(vec), 4) if vec is not None else None,
                "rrf": round(float(rrf), 4) if rrf is not None else None,
                "rerank": round(float(rerank), 4) if rerank is not None else None,
                "stages_hit": [True, vec is not None, rrf is not None, rerank is not None],
            }
        )

    from mind_mem.retrieval_graph import retrieval_diagnostics

    diag = retrieval_diagnostics(ws, last_n=20)

    fmt = getattr(args, "format", "text") or "text"
    if fmt == "json":
        print(json.dumps({"query": args.query, "results": trace_rows, "diagnostics": diag}, indent=2, default=str))
        return 0

    print(_c(f"Retrieval trace: {args.query!r}", _C_BOLD, _C_CYAN))
    print(_c("─" * 70, _C_DIM))
    header = f"  {'#':>4}  {'BLOCK':20}  {'BM25':>7}  {'VEC':>7}  {'RRF':>7}  {'RERANK':>7}  STAGES"
    print(_c(header, _C_BOLD))
    print(_c("─" * 70, _C_DIM))
    for row in trace_rows:
        bm25_s = f"{row['bm25']:7.4f}"
        vec_s = f"{row['vector']:7.4f}" if row["vector"] is not None else "      -"
        rrf_s = f"{row['rrf']:7.4f}" if row["rrf"] is not None else "      -"
        rer_s = f"{row['rerank']:7.4f}" if row["rerank"] is not None else "      -"
        bid = str(row["block_id"])[:20]
        print(f"  {row['rank']:>4}  {bid:20}  {bm25_s}  {vec_s}  {rrf_s}  {rer_s}  {_stage_bar(row['stages_hit'])}")

    print()
    print(_c("Diagnostics summary", _C_BOLD, _C_YELLOW))
    intent_dist = diag.get("intent_distribution", {})
    if intent_dist:
        top_intent = max(intent_dist, key=lambda k: intent_dist[k])
        print(f"  Intent: {_c(top_intent, _C_CYAN)} ({intent_dist[top_intent]} recent queries)")
    for stage, rate in sorted(diag.get("stage_rejection_rates", {}).items()):
        print(f"  {stage}: {rate:.1%} rejected")
    return 0


# ── mm trace ─────────────────────────────────────────────────────────────────

_TRACE_HEADER = "TIME              TOOL                           DURATION  STATUS  SIZE"


def _parse_log_lines(lines: list[str], *, tool_filter: Optional[str] = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if entry.get("event") != "mcp_tool_call":
            continue
        data = entry.get("data") or {}
        tool = data.get("tool") or entry.get("tool") or "?"
        if tool_filter and tool != tool_filter:
            continue
        rows.append(
            {
                "time": entry.get("ts", ""),
                "tool": str(tool),
                "duration_ms": data.get("duration_ms", data.get("latency_ms")),
                "success": data.get("success", True),
                "result_size": data.get("result_size"),
            }
        )
    return rows


def _render_trace_rows(rows: list[dict[str, Any]]) -> None:
    print(_c(_TRACE_HEADER, _C_BOLD))
    print(_c("─" * 70, _C_DIM))
    for r in rows:
        ts = str(r["time"])[:16]
        tool = str(r["tool"])[:29]
        dur = f"{int(r['duration_ms']):>6}ms" if r["duration_ms"] is not None else "      ?"
        size = str(r["result_size"]) if r["result_size"] is not None else "-"
        ok = r.get("success", True)
        status_str = _c("OK   ", _C_GREEN) if ok else _c("ERROR", _C_RED)
        print(f"{ts:16}  {tool:29}  {dur}  {status_str}  {size}")


def _cmd_trace(args: argparse.Namespace) -> int:
    last_n = getattr(args, "last", 20) or 20
    tool_filter: Optional[str] = getattr(args, "tool", None)
    live = getattr(args, "live", False)

    if live:
        log_file = os.environ.get("MIND_MEM_LOG_FILE", "")
        if log_file and os.path.isfile(log_file):
            import subprocess  # nosec B404 — subprocess is used with a fixed argument list

            proc = subprocess.Popen(["tail", "-f", "-n", str(last_n), log_file], stdout=subprocess.PIPE, text=True)  # nosec B603 B607 — fixed arg list; log_file is from MIND_MEM_LOG_FILE env var (operator-controlled), validated isfile above; shell=False (default)
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    rows = _parse_log_lines([line], tool_filter=tool_filter)
                    if rows:
                        _render_trace_rows(rows)
                        sys.stdout.flush()
            except KeyboardInterrupt:
                proc.terminate()
        else:
            import select

            while True:
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if ready:
                        line = sys.stdin.readline()
                        if not line:
                            break
                        rows = _parse_log_lines([line], tool_filter=tool_filter)
                        if rows:
                            _render_trace_rows(rows)
                            sys.stdout.flush()
                except KeyboardInterrupt:
                    break
        return 0

    log_file = os.environ.get("MIND_MEM_LOG_FILE", "")
    if log_file and os.path.isfile(log_file):
        with open(log_file, encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    elif not sys.stdin.isatty():
        all_lines = sys.stdin.readlines()
    else:
        all_lines = []

    rows = _parse_log_lines(all_lines, tool_filter=tool_filter)
    rows = rows[-last_n:]
    if not rows:
        print(_c("No mcp_tool_call log entries found.", _C_DIM))
        return 0
    _render_trace_rows(rows)
    return 0


# ---------------------------------------------------------------------------
# skill subcommands
# ---------------------------------------------------------------------------


def _cmd_skill_list(args: argparse.Namespace) -> int:
    from mind_mem.skill_opt.adapters import discover_all
    from mind_mem.skill_opt.config import load_config

    cfg = load_config(_workspace())
    specs = discover_all(cfg.resolve_sources())
    rows = [s.as_dict() for s in specs]
    print(json.dumps(rows, indent=2))
    return 0


def _cmd_skill_test(args: argparse.Namespace) -> int:
    import asyncio

    from mind_mem.skill_opt.adapters import discover_all
    from mind_mem.skill_opt.config import load_config
    from mind_mem.skill_opt.fleet_bridge import FleetBridge
    from mind_mem.skill_opt.test_runner import generate_test_cases, run_tests

    cfg = load_config(_workspace())
    specs = discover_all(cfg.resolve_sources())
    spec = next((s for s in specs if s.skill_id == args.skill_id), None)
    if spec is None:
        print(json.dumps({"error": f"Skill not found: {args.skill_id}"}))
        return 1
    fleet = FleetBridge(models=cfg.fleet_models.get("test_execution"))

    async def _run() -> list[dict[str, Any]]:
        cases = await generate_test_cases(spec, fleet, count=cfg.test_cases_per_skill)
        results = await run_tests(spec, cases, fleet)
        return [r.as_dict() for r in results]

    out = asyncio.run(_run())
    print(json.dumps(out, indent=2))
    return 0


def _cmd_skill_analyze(args: argparse.Namespace) -> int:
    import asyncio

    from mind_mem.skill_opt.adapters import discover_all
    from mind_mem.skill_opt.analyzer import aggregate_analysis, analyze_skill
    from mind_mem.skill_opt.config import load_config
    from mind_mem.skill_opt.fleet_bridge import FleetBridge
    from mind_mem.skill_opt.test_runner import generate_test_cases, run_tests

    cfg = load_config(_workspace())
    specs = discover_all(cfg.resolve_sources())
    spec = next((s for s in specs if s.skill_id == args.skill_id), None)
    if spec is None:
        print(json.dumps({"error": f"Skill not found: {args.skill_id}"}))
        return 1
    fleet = FleetBridge()

    async def _run() -> dict[str, Any]:
        cases = await generate_test_cases(spec, fleet, count=cfg.test_cases_per_skill)
        results = await run_tests(spec, cases, fleet)
        critiques = await analyze_skill(spec, results, fleet, min_critics=cfg.min_critics)
        return aggregate_analysis(critiques)

    out = asyncio.run(_run())
    print(json.dumps(out, indent=2))
    return 0


def _cmd_skill_optimize(args: argparse.Namespace) -> int:
    import asyncio
    import uuid

    from mind_mem.skill_opt.adapters import discover_all
    from mind_mem.skill_opt.analyzer import aggregate_analysis, analyze_skill
    from mind_mem.skill_opt.config import load_config
    from mind_mem.skill_opt.fleet_bridge import FleetBridge
    from mind_mem.skill_opt.history import HistoryStore
    from mind_mem.skill_opt.mutator import propose_mutations
    from mind_mem.skill_opt.test_runner import generate_test_cases, run_tests
    from mind_mem.skill_opt.validator import submit_to_governance, validate_mutation

    cfg = load_config(_workspace())
    specs = discover_all(cfg.resolve_sources())
    spec = next((s for s in specs if s.skill_id == args.skill_id), None)
    if spec is None:
        print(json.dumps({"error": f"Skill not found: {args.skill_id}"}))
        return 1
    fleet = FleetBridge()
    ws = cfg.governance_workspace or _workspace()
    db_path = os.path.join(ws, cfg.history_db_path)
    store = HistoryStore(db_path)
    run_id = f"R-{uuid.uuid4().hex[:12]}"

    async def _run() -> dict[str, Any]:
        store.start_run(run_id, spec.skill_id, spec.content_hash, cfg.as_dict())
        cases = await generate_test_cases(spec, fleet, count=cfg.test_cases_per_skill)
        results = await run_tests(spec, cases, fleet)
        critiques = await analyze_skill(spec, results, fleet, min_critics=cfg.min_critics)
        analysis = aggregate_analysis(critiques)
        mutations = await propose_mutations(spec, analysis, fleet, max_mutations=cfg.max_mutations_per_run)

        best_mutation = None
        best_validation = None
        for m in mutations:
            v = await validate_mutation(spec, m, cases, fleet, cfg)
            store.store_mutation(
                run_id,
                m.mutation_id,
                spec.skill_id,
                m.proposed_content,
                m.rationale,
                v.score_before,
                v.score_after,
            )
            if v.accepted and (best_validation is None or v.score_after > best_validation.score_after):
                best_mutation = m
                best_validation = v

        result: dict[str, Any] = {
            "run_id": run_id,
            "skill_id": spec.skill_id,
            "score_before": analysis.get("consensus_score", 0.0),
            "mutations_proposed": len(mutations),
            "best_accepted": None,
        }

        if best_mutation and best_validation:
            signal_id = submit_to_governance(best_mutation, best_validation, ws)
            store.update_mutation_status(best_mutation.mutation_id, "validated", signal_id)
            store.complete_run(run_id, "completed", best_validation.score_before, best_validation.score_after, True)
            result["best_accepted"] = {
                "mutation_id": best_mutation.mutation_id,
                "score_after": best_validation.score_after,
                "governance_signal": signal_id,
            }
        else:
            store.complete_run(run_id, "completed", analysis.get("consensus_score", 0.0), analysis.get("consensus_score", 0.0), False)
            result["best_accepted"] = None

        return result

    out = asyncio.run(_run())
    store.close()
    print(json.dumps(out, indent=2))
    return 0


def _cmd_skill_history(args: argparse.Namespace) -> int:
    from mind_mem.skill_opt.config import load_config
    from mind_mem.skill_opt.history import HistoryStore

    cfg = load_config(_workspace())
    ws = cfg.governance_workspace or _workspace()
    db_path = os.path.join(ws, cfg.history_db_path)
    store = HistoryStore(db_path)
    rows = store.get_run_history(args.skill_id, limit=args.limit)
    store.close()
    print(json.dumps(rows, indent=2, default=str))
    return 0


def _cmd_skill_score(args: argparse.Namespace) -> int:
    from mind_mem.skill_opt.config import load_config
    from mind_mem.skill_opt.history import HistoryStore

    cfg = load_config(_workspace())
    ws = cfg.governance_workspace or _workspace()
    db_path = os.path.join(ws, cfg.history_db_path)
    store = HistoryStore(db_path)
    score = store.get_latest_score(args.skill_id)
    store.close()
    print(json.dumps({"skill_id": args.skill_id, "latest_score": score}))
    return 0


# ---------------------------------------------------------------------------
# serve subcommand
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    from mind_mem.api.rest import run

    run(host=args.host, port=args.port, workspace=_workspace())
    return 0


def _cmd_audit_model(args: argparse.Namespace) -> int:
    from mind_mem.model_audit import audit_model, format_report_text

    try:
        report = audit_model(args.path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        out = report.to_dict()
        if not args.include_manifest:
            out.pop("manifest", None)
        print(json.dumps(out, indent=2))
    else:
        print(format_report_text(report, color=_use_color()))
        if args.include_manifest:
            print("\nmanifest (sha256):")
            for name, digest in sorted(report.manifest.items()):
                print(f"  {digest}  {name}")

    if args.manifest_out:
        out_path = os.path.expanduser(args.manifest_out)
        with open(out_path, "w") as f:
            for name, digest in sorted(report.manifest.items()):
                f.write(f"{digest}  {name}\n")
        print(f"\nmanifest written to {out_path}", file=sys.stderr)

    return 0 if report.passed else 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mm",
        description="Unified mind-mem CLI for non-MCP coding agents.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # recall
    p_recall = sub.add_parser("recall", help="Search memory and print JSON results.")
    p_recall.add_argument("query")
    p_recall.add_argument("--limit", type=int, default=10)
    p_recall.add_argument("--active-only", action="store_true")
    p_recall.set_defaults(func=_cmd_recall)

    # context
    p_ctx = sub.add_parser(
        "context",
        help="Recall + token-budget pack into a context snippet (JSON).",
    )
    p_ctx.add_argument("query")
    p_ctx.add_argument("--limit", type=int, default=20)
    p_ctx.add_argument("--max-tokens", type=int, default=2000)
    p_ctx.set_defaults(func=_cmd_context)

    # inject
    p_inj = sub.add_parser(
        "inject",
        help="Render a context snippet in the format a specific agent expects.",
    )
    p_inj.add_argument("query")
    p_inj.add_argument(
        "--agent",
        default="generic",
        help="Target agent (claude-code, codex, gemini, cursor, windsurf, aider, generic)",
    )
    p_inj.add_argument("--limit", type=int, default=10)
    p_inj.set_defaults(func=_cmd_inject)

    # status
    p_status = sub.add_parser("status", help="Print workspace status as JSON.")
    p_status.set_defaults(func=_cmd_status)

    # detect — list AI coding clients present on the machine
    p_detect = sub.add_parser(
        "detect",
        help="Auto-detect installed AI coding clients for this workspace.",
    )
    p_detect.set_defaults(func=_cmd_detect)

    # install — configure a single agent
    p_install = sub.add_parser(
        "install",
        help="Configure mind-mem for one named AI coding client.",
    )
    p_install.add_argument(
        "agent",
        help=(
            "Agent key: claude-code, codex, gemini, cursor, windsurf, "
            "aider, openclaw, nanoclaw, nemoclaw, continue, cline, roo, "
            "zed, copilot, cody, qodo."
        ),
    )
    p_install.add_argument("--dry-run", action="store_true")
    p_install.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config instead of non-destructive merge.",
    )
    p_install.set_defaults(func=_cmd_install)

    # install-all — auto-detect + configure everything
    p_install_all = sub.add_parser(
        "install-all",
        help="Auto-detect + configure every installed AI coding client.",
    )
    p_install_all.add_argument(
        "--agent",
        action="append",
        help=("Restrict installation to these named agents. Repeat flag for multiple. Default = auto-detect every installed client."),
    )
    p_install_all.add_argument("--dry-run", action="store_true")
    p_install_all.add_argument("--force", action="store_true")
    p_install_all.add_argument(
        "--no-mcp",
        action="store_true",
        help=(
            "Skip native MCP server registration. Default: write both "
            "the text hook AND the MCP registration for every agent "
            "that supports MCP (Codex, Gemini, Cursor, Windsurf, "
            "Continue, Cline, Roo, Zed)."
        ),
    )
    p_install_all.set_defaults(func=_cmd_install_all)

    # vault namespace
    p_vault = sub.add_parser("vault", help="Vault sync subcommands.")
    vsub = p_vault.add_subparsers(dest="vault_cmd", required=True)

    v_scan = vsub.add_parser("scan", help="Walk a vault and print parsed blocks.")
    v_scan.add_argument("vault_root")
    v_scan.add_argument(
        "--sync-dirs",
        nargs="*",
        help="Restrict scan to these subdirectories (relative).",
    )
    v_scan.set_defaults(func=_cmd_vault_scan)

    v_write = vsub.add_parser("write", help="Write a block to a vault.")
    v_write.add_argument("vault_root")
    v_write.add_argument("relative_path")
    v_write.add_argument("--id", required=True)
    v_write.add_argument("--type", default="note")
    v_write.add_argument("--title", default="")
    v_write.add_argument("--body", default="")
    v_write.add_argument("--overwrite", action="store_true")
    v_write.set_defaults(func=_cmd_vault_write)

    # skill namespace
    p_skill = sub.add_parser("skill", help="Self-improving skill optimization subcommands.")
    ssub = p_skill.add_subparsers(dest="skill_cmd", required=True)

    s_list = ssub.add_parser("list", help="List all discovered skills across systems.")
    s_list.set_defaults(func=_cmd_skill_list)

    s_test = ssub.add_parser("test", help="Generate + run synthetic tests for a skill.")
    s_test.add_argument("skill_id", help="Skill ID (e.g. openclaw:coding-agent, claude:code-reviewer)")
    s_test.set_defaults(func=_cmd_skill_test)

    s_analyze = ssub.add_parser("analyze", help="Cross-model critique of a skill.")
    s_analyze.add_argument("skill_id")
    s_analyze.set_defaults(func=_cmd_skill_analyze)

    s_optimize = ssub.add_parser("optimize", help="Full optimization loop: test → analyze → mutate → validate.")
    s_optimize.add_argument("skill_id")
    s_optimize.set_defaults(func=_cmd_skill_optimize)

    s_history = ssub.add_parser("history", help="Show optimization run history for a skill.")
    s_history.add_argument("skill_id")
    s_history.add_argument("--limit", type=int, default=10)
    s_history.set_defaults(func=_cmd_skill_history)

    s_score = ssub.add_parser("score", help="Show current score for a skill.")
    s_score.add_argument("skill_id")
    s_score.set_defaults(func=_cmd_skill_score)

    # serve — launch the REST API
    p_serve = sub.add_parser(
        "serve",
        help="Launch the mind-mem REST API server (requires mind-mem[api]).",
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8080, help="TCP port (default: 8080)")
    p_serve.set_defaults(func=_cmd_serve)

    # audit-model — static security scan of any local model checkpoint
    p_audit = sub.add_parser(
        "audit-model",
        help=(
            "Static security scan of a local model checkpoint. Flags remote-code "
            "hooks, unsafe pickle, tokenizer injection. Outputs SHA-256 manifest."
        ),
    )
    p_audit.add_argument(
        "path",
        help="Path to a local model directory (HF checkpoint layout).",
    )
    p_audit.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of text report."
    )
    p_audit.add_argument(
        "--include-manifest",
        action="store_true",
        help="Include SHA-256 per-file manifest in output.",
    )
    p_audit.add_argument(
        "--manifest-out",
        default="",
        help="Write SHA-256 manifest to this path (shasum-compatible format).",
    )
    p_audit.set_defaults(func=_cmd_audit_model)

    # ── inspect — full block fields + provenance tree ──────────────────────
    p_inspect = sub.add_parser(
        "inspect",
        help="Print full block fields and provenance tree for a block ID.",
    )
    p_inspect.add_argument("block_id", help="Block ID to inspect (e.g. D-001).")
    p_inspect.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="format",
        help="Output format: human-readable text (default) or structured JSON.",
    )
    p_inspect.set_defaults(func=_cmd_inspect)

    # ── explain — retrieval trace for a query ─────────────────────────────
    p_explain = sub.add_parser(
        "explain",
        help="Show per-stage retrieval scores (BM25 → vector → RRF → rerank) for a query.",
    )
    p_explain.add_argument("query", help="Search query to trace.")
    p_explain.add_argument("--limit", type=int, default=10, help="Number of results to show (default: 10).")
    p_explain.add_argument(
        "--backend",
        choices=["auto", "bm25", "hybrid"],
        default="auto",
        help="Retrieval backend: auto (default), bm25, or hybrid.",
    )
    p_explain.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="format",
        help="Output format: human-readable text (default) or structured JSON.",
    )
    p_explain.set_defaults(func=_cmd_explain)

    # ── trace — MCP call log tail ─────────────────────────────────────────
    p_trace = sub.add_parser(
        "trace",
        help="Display recent MCP tool calls parsed from structured JSON logs.",
    )
    p_trace.add_argument(
        "--live",
        action="store_true",
        help=("Stream new events in real time. Reads MIND_MEM_LOG_FILE if set, otherwise reads stdin."),
    )
    p_trace.add_argument(
        "--last",
        type=int,
        default=20,
        metavar="N",
        help="Show the last N MCP calls (default: 20).",
    )
    p_trace.add_argument(
        "--tool",
        default=None,
        metavar="TOOL_NAME",
        help="Filter output to a single tool name.",
    )
    p_trace.set_defaults(func=_cmd_trace)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
