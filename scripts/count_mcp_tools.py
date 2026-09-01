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

Docs drift is the same bug one layer out. On 2026-08-28 the code count was
95 and CI enforced it, yet the README *badge* said 89, twenty-odd doc
locations said 89, and CLAUDE.md said 84 in one place and 95 in another.
``--check 95`` could not catch any of that, because it only ever compared
the code against a number passed on the command line. ``--check-docs``
closes that hole: it scans the public surfaces for any claim about the tool
count and fails on every one that disagrees with the discovered value.

Usage:
    python scripts/count_mcp_tools.py            # print the count
    python scripts/count_mcp_tools.py --check N  # fail if != N
    python scripts/count_mcp_tools.py --check-docs  # fail on doc/badge drift
"""

from __future__ import annotations

import argparse
import ast
import re
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
    """Number of DISTINCT tool names the server exposes.

    Summing per-file counts double-counts any name defined in two files, and
    one is: ``recall`` appears in both ``public.py`` and ``recall.py``. FastMCP
    registers it once (it logs "Component already exists: tool:recall"), so the
    sum reported 97 where the server exposes 96 — and the README was then
    written to match the wrong number, so ``--check-docs`` passed by agreeing
    with a doc that agreed with the bug. Counting the set is the same question
    the server answers.
    """
    names: set[str] = set()
    for path in _tool_source_files():
        names.update(_tool_names(path))
    return len(names)


# Surfaces a reader can reach: the shipped README, the docs tree, and the
# in-repo agent brief. A claim in any of them is a public claim.
_DOC_SURFACES = ("README.md", "CLAUDE.md", "docs/**/*.md")

# "95 MCP tools" / "95 tools" (prose), and "MCP_tools-95" / "Tools: 95"
# (shields.io badge label plus its alt text). Deliberately narrow: a bare
# integer near the word "tool" is too noisy to gate on.
# NOTE: the badge pattern must NOT be anchored with \b before "tools" -- the
# real badge reads "MCP_tools-89", and an underscore is a word character, so
# \btools- can never match it. That exact mistake let the badge drift to 89
# while the prose said 95.
_CLAIM_RE = re.compile(r"\b(\d{2,3})\s+(?:MCP\s+)?tools?\b", re.IGNORECASE)
_BADGE_RE = re.compile(r"tools?[-_:]\s*(\d{2,3})\b", re.IGNORECASE)

# A claim that names the release it describes ("19 MCP tools at v1.x") is a
# historical statement, not a claim about now.
#
# PROXIMITY IS REQUIRED (2026-08-31). This exemption used to fire on a version
# string ANYWHERE on the line, which silenced a live claim that had nothing to
# do with it:
#
#   "MIND-Mem exposes 89 MCP tools for integration with ... Zed (v3.1.0+)."
#
# The `v3.1.0+` qualifies ZED'S EDITOR SUPPORT and sits 137 characters from the
# count. The line-wide exemption made `--check-docs` print "all tool-count
# claims agree with 96" while docs/mcp-integration.md said 89 -- a gate
# reporting success over a file it had silently excused. A version now only
# exempts a claim it is actually ADJACENT to.
_LINE_VERSIONED = re.compile(r"\bv\d+(\.\d+|\.x)", re.IGNORECASE)

# How close a version stamp must be to a count to be read as qualifying it.
# "19 MCP tools at v1.x" is ~8 characters; the false positive above was 137.
_VERSION_PROXIMITY = 30


def _version_qualifies(line: str, match: "re.Match[str]") -> bool:
    """True when a version stamp is close enough to *match* to be describing it.

    Scoped to the claim, not the line: a version elsewhere in the same
    sentence describes something else.
    """
    lo = max(0, match.start() - _VERSION_PROXIMITY)
    hi = min(len(line), match.end() + _VERSION_PROXIMITY)
    return _LINE_VERSIONED.search(line, lo, hi) is not None


# "89 distinct tools" / "89 distinct tool names". A modifier between the count
# and the noun defeats _CLAIM_RE (which requires them adjacent, or separated only
# by "MCP"), and this is not hypothetical: README, docs/api-reference.md and
# docs/claude-desktop-setup.md all sat at "89 distinct tools / 90 registrations"
# long after the surface reached 98 -- README said it four lines under its own
# "### MCP Server (98 tools, 8 resources)" heading, and the gate reported
# agreement. Only this one spelling is added: widening to any modifier matches
# "17 AI development tools", which counts editors, not MCP tools.
_DISTINCT_RE = re.compile(r"\b(\d{2,3})\s+distinct\s+tools?\b", re.IGNORECASE)

# A parenthesised count in a heading: "### MCP Tools (97)". This form slipped
# past the prose and badge patterns entirely -- a live "(90)" claim sat in
# CLAUDE.md while both other checks reported agreement.
#
# "MCP" is REQUIRED. README has "### Tools (21)" heading a table of 21 tools in
# one subsection, which is a true statement about a subset; matching a bare
# "Tools (N)" against the full-surface count would fail CI on a correct line.
_HEADING_RE = re.compile(r"MCP\s+Tools?\s*\((\d{2,3})\)", re.IGNORECASE)

# A line describing a TRANSITION is a record of a past fix, not a claim:
# "CLAUDE.md drift cleared (`MCP Tools (81) -> (84)`)".
_LINE_TRANSITION = re.compile(r"(->|\u2192|\u2013>|=>)")

# Historical records must keep their original numbers or they stop being
# records: a release note that retroactively claims today's count is a lie
# about the release it documents. Only LIVING surfaces -- the ones a reader
# consults to learn what is true now -- are gated.
#
# Recognised as historical: the changelog, anything carrying a version stamp
# in its filename, and the record-type suffixes below.
_EXEMPT_NAMES = ("CHANGELOG.md", "docs/audit_response.md", "docs/security-audit-sow.md")
_EXEMPT_PREFIXES = ("docs/architecture_audit_", "docs/design/", "docs/review-docs-")
_EXEMPT_SUFFIXES = (
    "-release-notes.md",
    "-implementation-plan.md",
    "-decomposition-plan.md",
    "-self-audit.md",
    "-training-recipe.md",
    "-surface-reduction.md",
)
# A version stamp anywhere in the filename (v3.2.0, roadmap-v4, 4b-v2) means the
# document describes a specific past release.
_VERSION_STAMP = re.compile(r"(^|[-/])v\d+(\.\d+)*", re.IGNORECASE)
# A date stamp in the filename (2026_04, 2026-04-13) also means "record".
_DATE_STAMP = re.compile(r"\d{4}[-_]\d{2}")


def _is_historical(rel: str) -> bool:
    if rel in _EXEMPT_NAMES:
        return True
    if any(rel.startswith(x) for x in _EXEMPT_PREFIXES):
        return True
    if any(rel.endswith(x) for x in _EXEMPT_SUFFIXES):
        return True
    return bool(_VERSION_STAMP.search(rel) or _DATE_STAMP.search(rel))


def _doc_files() -> list[Path]:
    root = _project_root()
    out: list[Path] = []
    for pattern in _DOC_SURFACES:
        out.extend(sorted(root.glob(pattern)))
    return [p for p in out if not _is_historical(p.relative_to(root).as_posix())]


def check_docs(expected: int) -> list[str]:
    """Return one message per doc location whose tool-count claim is stale."""
    root = _project_root()
    bad: list[str] = []
    for path in _doc_files():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive
            print(f"WARN: could not read {path}: {exc}", file=sys.stderr)
            continue
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            if _LINE_TRANSITION.search(line):
                continue
            for regex in (_CLAIM_RE, _BADGE_RE, _HEADING_RE, _DISTINCT_RE):
                for match in regex.finditer(line):
                    # Checked PER CLAIM, not per line -- see _version_qualifies.
                    if _version_qualifies(line, match):
                        continue
                    found = int(match.group(1))
                    if found != expected:
                        bad.append(f"{rel}:{lineno}: claims {found} tools, actual is {expected} -- {match.group(0)!r}")
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        type=int,
        default=None,
        help="Expected count; non-zero exit if the discovered count differs.",
    )
    parser.add_argument(
        "--check-docs",
        action="store_true",
        help="Fail if any README/docs/CLAUDE.md tool-count claim or badge is stale.",
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
    if args.check_docs:
        stale = check_docs(n)
        if stale:
            print(
                f"::error::mind-mem doc tool-count drift: {len(stale)} stale claim(s).",
                file=sys.stderr,
            )
            for msg in stale:
                print(f"  {msg}", file=sys.stderr)
            return 1
        print(f"docs: all tool-count claims agree with {n}")
    if args.verbose:
        names: list[str] = []
        for path in _tool_source_files():
            names.extend(_tool_names(path))
        for name in sorted(names):
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
