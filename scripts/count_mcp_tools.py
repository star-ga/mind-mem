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


def version_qualifies_span(line: str, start: int, end: int) -> bool:
    """True when a version stamp is close enough to ``line[start:end]`` to describe it."""
    lo = max(0, start - _VERSION_PROXIMITY)
    hi = min(len(line), end + _VERSION_PROXIMITY)
    return _LINE_VERSIONED.search(line, lo, hi) is not None


def _version_qualifies(line: str, match: "re.Match[str]") -> bool:
    """True when a version stamp is close enough to *match* to be describing it.

    Scoped to the claim, not the line: a version elsewhere in the same
    sentence describes something else.
    """
    return version_qualifies_span(line, match.start(), match.end())


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

# ---------------------------------------------------------------------------
# Two shapes every pattern above is structurally blind to
# ---------------------------------------------------------------------------
#
# 1. A TABLE CELL states the count AFTER its label, in a different cell:
#        | MCP tools | 89 | N/A |        docs/comparison.md:37
#        | MCP Tools | 8  | 89  |        docs/migration-guide.md:13
#    Every pattern above wants the number adjacent to the word, so all three
#    reported agreement while these sat thirteen releases stale.
#
# 2. A WRAPPED CLAIM is split by a line break, because markdown reflows prose:
#        ...each of these tools can call MIND-Mem's 89 MCP
#        tools (recall, propose_update, ...)          docs/integrations.md:47
#    The scan is line-scoped, so "89 MCP" and "tools" never met.
#
# Both were found by `grep -n 89 docs/` after `--check-docs` printed
# "all tool-count claims agree with 102". A gate that reports success over a
# file it cannot read is worse than no gate.

# The row's FIRST cell is the label. "MCP" is required for the same reason
# _HEADING_RE requires it: a bare "| Tools |" row is usually a subset.
_TABLE_ROW_LABEL = re.compile(r"^\s*\|\s*(?:MCP\s+Tools?|Tools?\s*\(MCP\))\s*\|", re.IGNORECASE)
# The `|---|:---:|` rule that separates a markdown header from its body.
_TABLE_SEPARATOR = re.compile(r"^\s*\|(?:\s*:?-{2,}:?\s*\|)+\s*$")
# Which column is OURS. Both offending tables put someone else's count in the
# same row -- "| MCP Tools | 8 | 89 |" is mem-os then MIND-Mem -- so gating
# every numeric cell would fail the build on a TRUE statement about another
# product. The header row names the columns; when it names ours, only ours is
# a claim about our surface. A table with no such header has no other product
# in it, and every count in the row is then ours.
_OURS_HEADER = re.compile(r"MIND[-\s_]?Mem", re.IGNORECASE)
# A cell holding nothing but a count, optionally bolded: " 89 ", " **89** ".
# Two or three digits, exactly as everywhere else here.
_TABLE_CELL_COUNT = re.compile(r"^[\s*`]*(\d{2,3})[\s*`]*$")

# A continuation line must be PROSE. Joining a table row to the next row, or a
# heading to the paragraph under it, would fabricate claims that appear on
# neither line; a wrapped sentence is the only shape that legitimately spans
# the break.
_NOT_A_CONTINUATION = re.compile(r"^\s*(\||#{1,6}\s|[-*+>]\s|\d+[.)]\s|```)")


# Which surface a tool count is about. This module has exactly ONE authority,
# the LIVE registry, so a claim about what the WEIGHTS know is not its to
# judge: CLAUDE.md's "Knows all 84 / tools" is a true-or-false statement about
# the trained revision, whose count is 83, and answering it with 102 would be
# wrong in a new way. ``check_docs_alignment`` resolves both authorities and
# gates that claim against the right one -- so skipping it here is not an
# exemption, it is a referral, and the referral is verified by a test.
TRAINED_MARK = re.compile(r"\b(trained|training|weights|checkpoint|knows|corpus)\b", re.IGNORECASE)
LIVE_MARK = re.compile(r"\b(live|exposes|currently|current)\b", re.IGNORECASE)


def nearest_span(pattern: re.Pattern[str], text: str, start: int, end: int) -> int | None:
    """Distance from ``text[start:end]`` to the closest occurrence of *pattern*."""
    best: int | None = None
    for hit in pattern.finditer(text):
        if hit.end() <= start:
            gap = start - hit.end()
        elif hit.start() >= end:
            gap = hit.start() - end
        else:
            gap = 0
        if best is None or gap < best:
            best = gap
    return best


def is_trained_claim(line: str, start: int, end: int) -> bool:
    """True when the nearest scope marker to ``line[start:end]`` says "the weights"."""
    trained = nearest_span(TRAINED_MARK, line, start, end)
    live = nearest_span(LIVE_MARK, line, start, end)
    return trained is not None and (live is None or trained < live)


def _cells_with_offsets(line: str) -> list[tuple[int, int, str]]:
    """``(start, end, text)`` for each ``|``-delimited cell, offsets into *line*."""
    out: list[tuple[int, int, str]] = []
    pos = line.find("|")
    if pos < 0:
        return out
    while True:
        nxt = line.find("|", pos + 1)
        if nxt < 0:
            return out
        out.append((pos + 1, nxt, line[pos + 1 : nxt]))
        pos = nxt


def table_tool_claims(lines: list[str]) -> list[tuple[int, int, int, int]]:
    """``(lineno, start, end, value)`` for every tool count stated in a table cell.

    Shared with ``check_docs_alignment`` so the two gates cannot disagree about
    what a table claim is -- one matcher, two callers.
    """
    claims: list[tuple[int, int, int, int]] = []
    previous: list[tuple[int, int, str]] | None = None
    ours: int | None = None
    in_table = False
    for idx, line in enumerate(lines):
        if not line.strip():
            in_table, ours, previous = False, None, None
            continue
        cells = _cells_with_offsets(line)
        if _TABLE_SEPARATOR.match(line):
            in_table, ours = True, None
            for column, (_start, _end, text) in enumerate(previous or []):
                if _OURS_HEADER.search(text):
                    ours = column
                    break
            previous = None
            continue
        if in_table and cells and _TABLE_ROW_LABEL.match(line) and not _LINE_TRANSITION.search(line):
            for column, (start, _end, text) in enumerate(cells):
                if column == 0 or (ours is not None and column != ours):
                    continue
                cell = _TABLE_CELL_COUNT.match(text)
                if cell is not None:
                    claims.append((idx + 1, start + cell.start(1), start + cell.end(1), int(cell.group(1))))
        # A row can carry a scope marker of its own ("| MCP tools (trained) |"),
        # so table claims are filtered by the caller, which knows both counts.
        previous = cells
    return claims


def wrapped_line_pairs(lines: list[str]) -> list[tuple[int, str, int]]:
    """``(lineno, joined, boundary)`` for each prose line glued to its successor.

    *boundary* is the offset in *joined* one past the end of the first line, so
    a caller can map any span back to the real line that holds it. Claims that
    lie wholly inside one line are found by the ordinary per-line scan and must
    be dropped by the caller, or every claim would be reported twice.
    """
    out: list[tuple[int, str, int]] = []
    for idx in range(len(lines) - 1):
        first, second = lines[idx], lines[idx + 1]
        if not first.strip() or not second.strip():
            continue
        if first.rstrip().endswith("|") or _NOT_A_CONTINUATION.match(second):
            continue
        out.append((idx + 1, first + " " + second, len(first)))
    return out


def locate_in_pair(start: int, end: int, lineno: int, boundary: int) -> tuple[int, int, int]:
    """Map a span in a joined pair back to ``(lineno, start, end)`` in one real line."""
    if start >= boundary + 1:
        return lineno + 1, start - boundary - 1, end - boundary - 1
    return lineno, start, end


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


def _scan_line_claims(line: str, lineno: int) -> list[tuple[int, int, int, int, str]]:
    """``(lineno, start, end, value, excerpt)`` for every tool claim on one line."""
    found: list[tuple[int, int, int, int, str]] = []
    if _LINE_TRANSITION.search(line):
        return found
    for regex in (_CLAIM_RE, _BADGE_RE, _HEADING_RE, _DISTINCT_RE):
        for match in regex.finditer(line):
            # Checked PER CLAIM, not per line -- see _version_qualifies.
            if _version_qualifies(line, match) or is_trained_claim(line, match.start(), match.end()):
                continue
            found.append((lineno, match.start(1), match.end(1), int(match.group(1)), match.group(0)))
    return found


def scan_doc_claims(lines: list[str]) -> list[tuple[int, int, int, int, str]]:
    """Every tool-count claim in *lines*: per line, wrapped across a line, and in a table cell.

    A wrapped claim is deduplicated by the span it maps back to, so a claim
    that happens to sit wholly inside one of the two joined lines is reported
    once by the per-line pass and never again here.
    """
    claims: list[tuple[int, int, int, int, str]] = []
    for lineno, line in enumerate(lines, 1):
        claims.extend(_scan_line_claims(line, lineno))
    seen = {(lineno, start, end) for lineno, start, end, _value, _excerpt in claims}
    for lineno, joined, boundary in wrapped_line_pairs(lines):
        for _no, start, end, value, excerpt in _scan_line_claims(joined, lineno):
            span = locate_in_pair(start, end, lineno, boundary)
            if span in seen:
                continue
            seen.add(span)
            claims.append((span[0], span[1], span[2], value, excerpt))
    for lineno, start, end, value in table_tool_claims(lines):
        line = lines[lineno - 1]
        if (lineno, start, end) in seen or is_trained_claim(line, start, end) or version_qualifies_span(line, start, end):
            continue
        seen.add((lineno, start, end))
        claims.append((lineno, start, end, value, line.strip()))
    return claims


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
        for lineno, _start, _end, found, excerpt in scan_doc_claims(text.splitlines()):
            if found != expected:
                bad.append(f"{rel}:{lineno}: claims {found} tools, actual is {expected} -- {excerpt!r}")
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
