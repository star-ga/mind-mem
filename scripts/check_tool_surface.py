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


#: Tool-shaped names a doc may use that are NOT MCP tools. Each needs a reason.
_NOT_A_TOOL: dict[str, str] = {
    # English connectives that survive the comma-run match.
    "and": "conjunction",
    "more": "as in 'and 62 more'",
    "the": "article",
    "total": "as in '(98 total)'",
    "full": "prose",
    "list": "prose",
    "see": "prose",
}


def ghost_tool_names(real: set[str]) -> dict[str, list[str]]:
    """Tool-shaped names the docs claim that the registry does not have.

    A doc naming a tool that does not exist is worse than an undocumented tool:
    it sends a caller to a surface that will fail, and it reads as authoritative.
    CLAUDE.md carried ELEVEN of these (create_snapshot, list_snapshots,
    restore_snapshot, briefing, cross_encoder_rerank, delete_memory,
    import_memory, audit_replay, tier_decay_apply, encrypt_status,
    alerts_subscribe) -- surfaces that were renamed or never shipped.

    Scoped deliberately to the curated TOOL LISTS, not to prose: matching every
    snake_case token in every document would flag ordinary function names and
    the gate would be turned off within a week. A "tool list" here is a
    comma-separated run of >= 6 tool-shaped names, which is the shape these
    lists actually take.
    """
    docs = [ROOT / "CLAUDE.md", ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))
    out: dict[str, list[str]] = {}

    # Scoped to lists under an explicit "MCP Tools" heading -- NOT a heuristic
    # scrape of every comma-run in the corpus. Two earlier attempts show why:
    # requiring an underscore on every element made the gate unable to fail at
    # all (the real lists mix `recall` with `hybrid_search`), and dropping that
    # requirement flagged parameter names in guardrails.md and edge types in the
    # 4b README. A gate that cries wolf gets switched off, and a gate that
    # cannot fire is worse than none, so the trigger is an explicit marker a
    # writer opts into rather than something inferred from prose.
    heading = re.compile(r"^#{1,6}\s+.*\bMCP Tools?\b.*$", re.I | re.M)
    ident = re.compile(r"\b[a-z][a-z0-9_]{2,}\b")

    for doc in docs:
        if not doc.is_file():
            continue
        text = doc.read_text(encoding="utf-8", errors="replace")
        named: set[str] = set()
        for match in heading.finditer(text):
            block = text[match.end() : match.end() + 1800]
            nxt = re.search(r"^#{1,6}\s", block, re.M)
            if nxt:
                block = block[: nxt.start()]
            # A run counts as a TOOL LIST only when most of it already names
            # real tools. Self-calibrating, and it settles every false positive
            # the earlier versions produced: a parameter list, a file-format
            # list, and an edge-type list all score ~0% and are ignored, while a
            # genuine list with one bad name scores high and the bad name is
            # reported. It also spares this file's own prose about which ghosts
            # USED to be listed, which is otherwise indistinguishable by shape.
            for run in re.finditer(r"(?:[a-z][a-z0-9_]{2,}\s*,\s*){3,}[a-z][a-z0-9_]{2,}", block):
                tokens = [t for t in ident.findall(run.group(0)) if t not in _NOT_A_TOOL]
                if not tokens:
                    continue
                hits = sum(1 for t in tokens if t in real)
                if hits / len(tokens) >= 0.6:
                    named |= set(tokens)
        ghosts = sorted(n for n in named - real if n not in _NOT_A_TOOL)
        if ghosts:
            out[doc.relative_to(ROOT).as_posix()] = ghosts
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", type=int, metavar="N", help="exit 1 when more than N tools are unsupported")
    ap.add_argument(
        "--check-doc-names",
        action="store_true",
        help="exit 1 when a doc tool-list names a tool that does not exist",
    )
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

    ghosts = ghost_tool_names(set(matrix))
    if ghosts:
        print("\nGHOST TOOL NAMES (documented, not registered):")
        for doc, names in ghosts.items():
            print(f"  {doc}: {', '.join(names)}")

    if args.check_doc_names and ghosts:
        total = sum(len(v) for v in ghosts.values())
        print(f"\nFAIL: {total} documented tool name(s) do not exist", file=sys.stderr)
        return 1

    ghosts = ghost_tool_names(set(matrix))
    if ghosts:
        print("\nGHOST TOOL NAMES (documented, not registered):")
        for doc, names in ghosts.items():
            print(f"  {doc}: {', '.join(names)}")

    if args.check_doc_names and ghosts:
        total = sum(len(v) for v in ghosts.values())
        print(f"\nFAIL: {total} documented tool name(s) do not exist", file=sys.stderr)
        return 1

    if args.check is not None and len(unsupported) > args.check:
        print(f"\nFAIL: {len(unsupported)} unsupported tools > allowed {args.check}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
