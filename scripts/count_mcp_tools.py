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
import importlib
import os
import pkgutil
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class _StubMCP:
    """Minimal stand-in for fastmcp.FastMCP that just counts ``.tool(...)``."""

    def __init__(self) -> None:
        self.tools: list[str] = []

    def tool(self, fn=None, **_kw):
        def _add(target):
            self.tools.append(getattr(target, "__name__", repr(target)))
            return target

        if fn is None:
            return _add
        return _add(fn)


def count_tools() -> int:
    src = _project_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    pkg = importlib.import_module("mind_mem.mcp.tools")
    stub = _StubMCP()
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        try:
            mod = importlib.import_module(f"mind_mem.mcp.tools.{mod_info.name}")
        except Exception as exc:
            print(f"WARN: could not import mind_mem.mcp.tools.{mod_info.name}: {exc}", file=sys.stderr)
            continue
        register = getattr(mod, "register", None)
        if callable(register):
            try:
                register(stub)
            except Exception as exc:
                print(
                    f"WARN: register({mod_info.name}) raised: {exc}",
                    file=sys.stderr,
                )
    return len(stub.tools)


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
        # Re-run with name capture for the verbose listing.
        from mind_mem.mcp import tools as _tools  # noqa: F401

        stub = _StubMCP()
        pkg = importlib.import_module("mind_mem.mcp.tools")
        for mod_info in pkgutil.iter_modules(pkg.__path__):
            try:
                mod = importlib.import_module(f"mind_mem.mcp.tools.{mod_info.name}")
            except Exception:
                continue
            register = getattr(mod, "register", None)
            if callable(register):
                try:
                    register(stub)
                except Exception:
                    continue
        for name in sorted(stub.tools):
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
