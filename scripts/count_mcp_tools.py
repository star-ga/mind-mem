#!/usr/bin/env python3
"""Count registered MCP tools and assert the count matches CLAUDE.md.

Audit A-3: the MCP surface count drifts between code, CLAUDE.md, and
the README every few releases. This script is the single source of
truth — CI runs it with ``--check`` and fails the build if the
recorded count doesn't match what's actually wired up.

Discovery model: a tool is a function decorated with ``@mcp_tool_observe``
that is registered onto a FastMCP instance via ``mcp.tool(fn)``. We
walk the package importing every ``register(mcp)`` entry point and
count the resulting registrations on a stub FastMCP. This catches
both the per-domain tool modules under ``mcp/tools/*`` and the
historical monolith in ``mcp_server.py``.

Usage:
    python scripts/count_mcp_tools.py            # print the count
    python scripts/count_mcp_tools.py --check N  # fail if != N
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _tool_source_files() -> list[Path]:
    """Every source file that can register MCP tools: the per-domain modules
    under ``mcp/tools/*`` and the historical monolith ``mcp_server.py``."""
    root = _project_root() / "src" / "mind_mem"
    files = sorted((root / "mcp" / "tools").glob("*.py"))
    monolith = root / "mcp_server.py"
    if monolith.exists():
        files.append(monolith)
    return files


def _tool_names(path: Path) -> list[str]:
    """Statically collect the argument name of every ``mcp.tool(<fn>)`` (or
    ``<x>.tool(<fn>)``) registration call in ``path``.

    Static AST parsing — NOT a runtime import — so the count is identical in
    any environment, including CI jobs that do not install the ``mcp`` extra
    (the runtime-import approach returned 0 there and red-lit the build)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError) as exc:  # pragma: no cover - defensive
        print(f"WARN: could not parse {path.name}: {exc}", file=sys.stderr)
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "tool" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                names.append(arg.id)
            elif isinstance(arg, ast.Attribute):
                names.append(arg.attr)
    return names


def count_tools() -> int:
    total = 0
    for path in _tool_source_files():
        total += len(_tool_names(path))
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        type=int,
        default=None,
        help="Expected count; non-zero exit if the discovered count differs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the discovered tool names.",
    )
    args = parser.parse_args(argv)

    n = count_tools()
    print(n)
    if args.check is not None and n != args.check:
        print(
            f"::error::mind-mem MCP tool count drift: expected {args.check}, got {n}.",
            file=sys.stderr,
        )
        return 1
    if args.verbose:
        names: list[str] = []
        for path in _tool_source_files():
            names.extend(_tool_names(path))
        for name in sorted(names):
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
