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

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
