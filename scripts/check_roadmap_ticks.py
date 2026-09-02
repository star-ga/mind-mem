#!/usr/bin/env python3
# Copyright 2026 STARGA, Inc.
"""A tick must not contradict its own sentence, and a public roadmap must not
credit its sources — the roadmap honesty gate.

WHY THIS EXISTS. On 2026-09-01 an audit put five ``[x]`` boxes in
``ROADMAP.md`` under suspicion. Each was then checked against running code
rather than against the sentence beside it, and the result was not five out of
five -- it was four false ticks, only three of them from that list:

* **False, and confidently worded.** "Pluggable redaction layer" (the
  ``redaction`` flag is registered and has zero consumers; ``import
  mind_mem.v4.redaction`` raises ``ModuleNotFoundError``) and "Compliance
  export pipeline" (``mm export`` exits 2 with ``invalid choice``, and no
  ``--policy`` option exists anywhere in ``src/``). Nothing on the page hints
  at either; only running the code settles them, which is why rules 1-3 make
  no attempt to and ``tests/test_roadmap_hygiene.py`` pins them by name.
* **False, and self-refuting.** "Provenance-rich blocks" carried ``[x]`` while
  its own sentence said the ``provenance: off|recommended|required`` policy was
  **not added** -- and it is not: zero occurrences in ``src/``.
* **False, and found by the attribution sweep, not by the tick rules.**
  "TurboQuant-compressed prefix cache" (rule 1, ``placeholder``, below).
* **TRUE, with a sentence that libelled its own code.** "Auto-generated
  hierarchical index" said ``index.md`` / ``log.md`` autogen was **not wired**
  while ``memory_index.generate_index`` writes both and ``mm index`` runs it;
  "Vocabulary-bound fields" said the vocabularies were **not wired into
  ``validate_block``** while a reject-mode violation refuses a real
  ``propose_update`` and leaves ``SIGNALS.md`` byte-for-byte unchanged. Both
  ticks stayed; the sentences were corrected.

The lesson the gate is shaped by: **a self-refuting line is a defect either
way, but which half is wrong is not knowable from the page.** Two of these
five would have been "cleaned up" into open boxes by anyone who fixed the
contradiction in the cheap direction, silently retracting capabilities that
ship. So this file reports the contradiction and refuses to say which side
loses; the author must go and look.

Rules 1 and 2 are deliberately narrow: they catch only ticks that are
*self-refuting on the page*. Neither can tell you whether a confidently-worded
tick is true -- that is what the reachability gate
(``check_reachable_modules.py``) and the tool-surface gate
(``check_tool_surface.py``) are for. Cheap mechanical checks that always run
beat an expensive check nobody runs.

THE THREE RULES

1. **Self-refuting tick.** A ``[x]`` item whose own text carries a
   not-shipped marker: ``not wired``, ``not added``, ``not yet``,
   ``not shipped``, ``no such``, ``placeholder``. The first five are the
   phrases observed verbatim in the 2026-09-01 defects; the list is data
   (``NOT_SHIPPED_MARKERS``) so it can grow when a new phrasing shows up.

   ``placeholder`` was added 2026-09-01 with a *forward* yield of zero on
   ``ROADMAP.md`` and is honest about that: no line in the file says it today.
   It earns its place from the sixth false tick, found by the attribution
   sweep rather than by rule 1 -- "TurboQuant-compressed prefix cache" carried
   ``[x]`` while ``turbo_quant.py``'s own docstring called the format "a
   placeholder for a rotation + learned-codebook + residual-correction
   scheme". The roadmap said "ships", the module said "placeholder", and the
   only reason rule 1 could not see it is that the honest word lived in the
   source rather than on the page. It is in the list so that the next author
   who copies the module's own wording into the roadmap is caught.

   Scope is the checkbox line, plus the item's continuation lines. On the
   continuation lines, text from an honest-partial label onward
   (``**Remaining:**`` / ``Open half:``) is out of scope, because this file
   already uses that convention to ship a headline capability while naming a
   remaining sub-part.

   That carve-out is the one exemption in this file, so it is measured rather
   than asserted. Disabling it (``_strip_partial_clause`` made the identity
   function) and rescanning ``ROADMAP.md`` on 2026-09-01 moved the
   self-refuting count from 3 to 4. The single extra finding was line 1504,
   T-001 "Content-provenance tags on block writes" -- an entry that ships a
   real capability with 47 tests and then says, under ``**Remaining:**``, that
   "individual ingest producers do not yet / stamp the tag themselves". So the
   carve-out is load-bearing and it earns exactly one line of slack: without
   it the gate's only new catch is an honest caveat, and the cheapest way for
   an author to clear that finding would be to DELETE the caveat -- the exact
   opposite of what this gate exists to encourage. Re-run that experiment
   before widening or removing the carve-out; if it ever stops changing the
   count, delete it as dead weight.

   The carve-out deliberately does **not** apply to the checkbox line itself,
   so ``- [x] **X** -- **Remaining:** not wired`` still fails. Otherwise the
   label would be a one-word way to neuter the gate. ``tests/
   test_roadmap_ticks_gate.py`` pins both halves of that boundary.

2. **Tick under an Open heading.** A ``[x]`` item sitting in a section
   introduced by an ``Open`` heading. A shipped item belongs in the
   ``Shipped:`` list; leaving it under ``Open:`` makes both lists lie.

   Measured on the pre-fix file: three ``[x]`` items sat under one ``Open:``
   heading. Two were the self-refuting provenance/vocabulary lines rule 1 also
   caught; the third -- "Time-bounded and event-bounded recall" -- was a fully
   shipped capability filed in the wrong list, which rule 1 could never have
   seen because its sentence was true. That is the half of the defect this
   rule owns on its own. (The third self-refuting item, the hierarchical
   index, sat in an ordinary section and was caught by rule 1 alone, so the
   two rules genuinely cover different ground rather than double-reporting.)

   ``Open`` is matched with a trailing word boundary on purpose: this file
   mentions ``OpenAPI``, ``OpenTelemetry`` and other Open-prefixed names in
   prose, and one of them sits 300 lines above a run of 65 ``[x]`` items. A
   heading matcher without the boundary reports all 65.

3. **Prior-art attribution.** Any line carrying an arXiv identifier, a
   paper/preprint/DOI URL, a host-qualified source-repository URL, or an
   ``et al.`` author credit. This repository is PUBLIC and the house rule is
   that adopted external prior art is never credited in a public artifact --
   provenance lives in the private notes, and the public text says "recent
   research".

   The file itself stated that rule (``"Provenance (arxiv id, authors, exact
   tables) recorded privately ... public artifacts say 'recent scaling-law
   research' only"``) and then broke it in four places: three arXiv ids, two
   paper titles with authors and venue, and two third-party repository URLs.
   A stated norm nothing checks is a norm that drifts, which is why this rule
   is mechanical.

   Two scoping decisions, both deliberate:

   * The arXiv matcher requires an actual identifier (``arXiv:2504.19874``),
     never the bare word, so the policy sentence quoted above -- the one line
     in the file that *should* say "arxiv" -- is not flagged by the rule it
     describes.
   * There is NO carve-out for our own organisation's repositories. A
     ``github.com/<owner>/<repo>`` URL is reported whoever owns it, because a
     host-qualified URL cannot be distinguished from a citation by shape, and
     the file's existing practice already makes the exception unnecessary:
     every self-reference in ``ROADMAP.md`` is a bare slug
     (``star-ga/mind-nerve``, ``star-ga/mind-mem-4b``) or a relative path, and
     none is a URL. An exemption with no work to do is dead weight that only
     widens later.

   Known blind spot, named rather than pretended away: a bare ``owner/repo``
   slug with no host is not mechanically distinguishable from a source path
   (``src/mind_mem``, ``docs/design``), so an attribution written that way
   passes. Rule 3 catches the shapes a citation actually takes -- ids, URLs
   and author credits -- and the reviewer still owns the rest.

NOT A LICENCE TO SILENCE. There is no allowlist and no per-line suppression,
because either fix is honest and cheap: make the text true, or clear the box.

Usage:
    python3 scripts/check_roadmap_ticks.py             # report, exit 0
    python3 scripts/check_roadmap_ticks.py --check     # exit 1 on findings
    python3 scripts/check_roadmap_ticks.py --check FILE [FILE ...]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TARGETS: tuple[str, ...] = ("ROADMAP.md",)

# Phrases that mean "this capability does not exist yet". Observed verbatim in
# the 2026-09-01 defects; extend when a new phrasing is found in the wild.
NOT_SHIPPED_MARKERS: tuple[str, ...] = (
    "not wired",
    "not added",
    "not yet",
    "not shipped",
    "no such",
    # Zero yield on ROADMAP.md today; carried for the turbo_quant case where
    # the module's own docstring said "placeholder" and the roadmap said
    # "ships" -- see rule 1 in the module docstring.
    "placeholder",
)

# Labels under which an item may honestly name work it has NOT done. Only
# honoured on continuation lines -- see rule 1 above.
PARTIAL_LABELS: tuple[str, ...] = (
    "**remaining:**",
    "**open half:**",
    "remaining:",
    "open half:",
)

_CHECKBOX = re.compile(r"^(?P<indent>\s*)[-*+]\s+\[(?P<mark>[ xX])\]\s*(?P<text>.*)$")
_ATX_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
# A standalone bold label line: "**Shipped:**", "**Open (network gaps):**".
_BOLD_LABEL = re.compile(r"^\s*(?:\*\*|__).*(?:\*\*|__)\s*:?\s*$")
# "Open", "**Open:**", "__Open (gaps):__" -- the \b rejects OpenAPI/OpenTelemetry.
_OPEN_HEADING = re.compile(r"^\s{0,3}(?:#{1,6}\s+)?(?:\*\*|__)?\s*Open\b", re.IGNORECASE)

_MARKER_RE = re.compile("|".join(re.escape(m) for m in NOT_SHIPPED_MARKERS), re.IGNORECASE)

#: Rule 3. Each entry is ``(pattern, what it is)``; the second half becomes the
#: finding text, so a reader learns which shape tripped without reading this
#: file. Every pattern is anchored on a citation SHAPE -- an identifier, a
#: host-qualified URL, an author credit -- never on a bare topic word, so the
#: file can still discuss arXiv, GitHub or a paper's subject in prose.
PRIOR_ART_PATTERNS: tuple[tuple[str, str], ...] = (
    # "arXiv:2504.19874" / "arxiv: 2606.32032" -- an identifier, not the word.
    (r"arxiv\s*:\s*\d{4}\.\d{4,5}", "an arXiv identifier"),
    (r"\barxiv\.org\b", "an arXiv URL"),
    (r"\bdoi\.org/10\.\d{4,}", "a DOI URL"),
    (r"\bdoi\s*:\s*10\.\d{4,}", "a DOI"),
    (
        r"\b(?:openreview\.net|aclanthology\.org|semanticscholar\.org"
        r"|papers\.nips\.cc|proceedings\.mlr\.press|dl\.acm\.org)\b",
        "a paper-repository URL",
    ),
    # A host-qualified repo URL, scheme optional -- "github.com/owner/repo"
    # is a citation whether or not somebody typed the https://.
    (
        r"(?:https?://)?(?:www\.)?(?:github|gitlab|bitbucket)\.com"
        r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
        "a source-repository URL",
    ),
    (r"\bet\s+al\.", "an author credit"),
)

_ATTRIBUTION_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), what) for pat, what in PRIOR_ART_PATTERNS
)


@dataclass(frozen=True)
class Finding:
    """One rule violation, addressed to a file and a line."""

    path: str
    line: int
    rule: str
    detail: str
    text: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.detail}\n    {self.text.strip()[:160]}"


def _strip_partial_clause(text: str) -> str:
    """Return *text* truncated at the first honest-partial label."""
    lowered = text.lower()
    cut = len(text)
    for label in PARTIAL_LABELS:
        found = lowered.find(label)
        if found != -1:
            cut = min(cut, found)
    return text[:cut]


def _is_open_heading(line: str) -> bool:
    return bool(_OPEN_HEADING.match(line))


def _ends_open_region(line: str) -> bool:
    """A heading or a fresh bold label closes the Open region."""
    return bool(_ATX_HEADING.match(line)) or bool(_BOLD_LABEL.match(line))


def scan_attribution(lines: list[str], path: str = "<memory>") -> list[Finding]:
    """Apply rule 3 to every line of *lines*. Pure; no I/O.

    Deliberately line-oriented and blind to markdown structure: a citation is
    a leak in a heading, a blockquote, a checkbox, a table cell or a fenced
    code block alike, and the shape that identifies it survives all of them.
    """
    findings: list[Finding] = []
    for offset, raw in enumerate(lines):
        for pattern, what in _ATTRIBUTION_RES:
            hit = pattern.search(raw)
            if hit is None:
                continue
            findings.append(
                Finding(
                    path=path,
                    line=offset + 1,
                    rule="prior-art-attribution",
                    detail=(
                        f'carries {what} ("{hit.group(0)}") -- this repository is public, '
                        'so say "recent research" and keep the provenance in the private notes'
                    ),
                    text=raw,
                )
            )
            break  # one finding per line; fixing it means rewriting the clause
    return findings


def scan_lines(lines: list[str], path: str = "<memory>") -> list[Finding]:
    """Apply both rules to *lines*. Pure; no I/O."""
    findings: list[Finding] = []
    open_heading_line: int | None = None
    index = 0

    while index < len(lines):
        raw = lines[index]
        match = _CHECKBOX.match(raw)

        if match is None:
            if _is_open_heading(raw):
                open_heading_line = index + 1
            elif open_heading_line is not None and _ends_open_region(raw):
                open_heading_line = None
            index += 1
            continue

        line_no = index + 1
        ticked = match.group("mark") in ("x", "X")
        head_text = match.group("text")

        # Collect the item's continuation lines (indented, not a new item).
        continuations: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            nxt = lines[cursor]
            if not nxt.strip():
                break
            if _CHECKBOX.match(nxt) or _ATX_HEADING.match(nxt) or not nxt[:1].isspace():
                break
            continuations.append(nxt)
            cursor += 1

        if ticked:
            scanned = [head_text] + [_strip_partial_clause(c) for c in continuations]
            hit = next((m for m in (_MARKER_RE.search(s) for s in scanned) if m), None)
            if hit is not None:
                findings.append(
                    Finding(
                        path=path,
                        line=line_no,
                        rule="self-refuting-tick",
                        detail=(f'marked [x] but its own text says "{hit.group(0)}" -- make the text true or clear the box'),
                        text=head_text,
                    )
                )
            if open_heading_line is not None:
                findings.append(
                    Finding(
                        path=path,
                        line=line_no,
                        rule="tick-under-open-heading",
                        detail=(
                            f"marked [x] but sits under the Open heading on line {open_heading_line} -- "
                            "move it to the Shipped list or clear the box"
                        ),
                        text=head_text,
                    )
                )

        index = cursor

    findings.extend(scan_attribution(lines, path=path))
    findings.sort(key=lambda f: (f.line, f.rule))
    return findings


def scan_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    return scan_lines(text.splitlines(), path=str(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", default=None, help="Markdown files to scan (default: ROADMAP.md).")
    parser.add_argument("--check", action="store_true", help="Exit 1 when any finding is reported.")
    args = parser.parse_args(argv)

    targets = [Path(t) for t in (args.targets or DEFAULT_TARGETS)]
    missing = [t for t in targets if not t.is_file()]
    if missing:
        for t in missing:
            print(f"check_roadmap_ticks: no such file: {t}", file=sys.stderr)
        return 2

    findings: list[Finding] = []
    for target in targets:
        findings.extend(scan_file(target))

    scanned = ", ".join(str(t) for t in targets)
    if not findings:
        print(f"check_roadmap_ticks: OK -- no self-refuting ticks, no ticks under an Open heading, no prior-art attribution in {scanned}")
        return 0

    for finding in findings:
        print(finding.render())
    print(f"\ncheck_roadmap_ticks: {len(findings)} finding(s) in {scanned}")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
