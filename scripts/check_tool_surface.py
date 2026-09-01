#!/usr/bin/env python3
"""Reachability, applied to the MCP tool surface.

The 5.0.0 sweep deleted 44 modules that no product code imported. The
generator of that debt was not the modules -- it was shipping surface before
consumers -- and the 98-tool MCP list is the most public instance of the same
habit. For an agent choosing what to call, a long tool list is a COST, not a
feature.

This does not (and cannot) measure what real clients call; there is no
telemetry and inventing one would be worse than the gap. What it measures is
whether each registered tool is *supported* anywhere a caller could learn it
from:

    tested       exercised by the test suite
    documented   named in docs/ or README
    trained      present in the 4b training corpus under train/

A tool that is none of the three is registered and unsupported: nothing
teaches it, nothing proves it, and nothing would notice if it broke. That is
the tool-surface twin of "no caller, no tick".

Usage:
    python3 scripts/check_tool_surface.py            # report
    python3 scripts/check_tool_surface.py --check N  # exit 1 if unsupported > N
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mind_mem"


def registered_tools() -> list[str]:
    """Every name passed to `mcp.tool(...)` across the tool modules."""
    names: set[str] = set()
    for path in (SRC / "mcp" / "tools").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        names.update(re.findall(r"mcp\.tool\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", text))
    return sorted(names)


def _corpus(paths: list[pathlib.Path], suffixes: tuple[str, ...]) -> str:
    out = []
    for base in paths:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix in suffixes and "__pycache__" not in path.parts:
                try:
                    out.append(path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
    return "\n".join(out)


def support_matrix() -> dict[str, dict[str, bool]]:
    tools = registered_tools()
    tested = _corpus([ROOT / "tests"], (".py",))
    documented = _corpus([ROOT / "docs"], (".md",)) + (
        (ROOT / "README.md").read_text(encoding="utf-8", errors="replace") if (ROOT / "README.md").exists() else ""
    )
    trained = _corpus([ROOT / "train"], (".py", ".md", ".json", ".jsonl"))

    matrix = {}
    for tool in tools:
        word = re.compile(rf"\b{re.escape(tool)}\b")
        matrix[tool] = {
            "tested": bool(word.search(tested)),
            "documented": bool(word.search(documented)),
            "trained": bool(word.search(trained)),
        }
    return matrix


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", type=int, metavar="N", help="exit 1 when more than N tools are unsupported")
    args = ap.parse_args(argv)

    matrix = support_matrix()
    unsupported = sorted(t for t, m in matrix.items() if not any(m.values()))
    undocumented = sorted(t for t, m in matrix.items() if not m["documented"])
    untested = sorted(t for t, m in matrix.items() if not m["tested"])

    print(f"registered tools: {len(matrix)}")
    print(f"  untested       : {len(untested)}")
    print(f"  undocumented   : {len(undocumented)}")
    print(f"  UNSUPPORTED    : {len(unsupported)}  (in none of tested/documented/trained)")
    if untested:
        print("\nuntested:")
        for t in untested:
            print(f"  - {t}")
    if undocumented:
        print("\nundocumented:")
        for t in undocumented:
            print(f"  - {t}")
    if unsupported:
        print("\nUNSUPPORTED (registered, and nothing teaches or proves it):")
        for t in unsupported:
            print(f"  ! {t}")

    if args.check is not None and len(unsupported) > args.check:
        print(f"\nFAIL: {len(unsupported)} unsupported tools > allowed {args.check}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
