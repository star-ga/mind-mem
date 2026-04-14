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
        result = install_config(
            args.agent, ws, dry_run=args.dry_run, force=args.force
        )
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
        ws, dry_run=args.dry_run, force=args.force, agents=agents
    )
    summary = {
        "workspace": ws,
        "agents": agents if agents is not None else detect_installed_agents(ws),
        "results": results,
        "summary": {
            "written": sum(1 for r in results if r.get("written")),
            "merged": sum(1 for r in results if r.get("merged")),
            "skipped": sum(1 for r in results if r.get("skipped")),
            "errored": sum(1 for r in results if "error" in r),
            "total": len(results),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["summary"]["errored"] == 0 else 1


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
    from mind_mem.skill_opt.scorer import aggregate_critiques
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
                run_id, m.mutation_id, spec.skill_id, m.proposed_content,
                m.rationale, v.score_before, v.score_after,
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
        help=(
            "Restrict installation to these named agents. Repeat flag for "
            "multiple. Default = auto-detect every installed client."
        ),
    )
    p_install_all.add_argument("--dry-run", action="store_true")
    p_install_all.add_argument("--force", action="store_true")
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

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
