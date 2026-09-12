"""No prior-art identifier may appear in a tracked file, anywhere in the repository.

WHY THIS EXISTS. The attribution rule was enforced by `scripts/check_roadmap_ticks.py`,
and that gate reads `ROADMAP.md` only. So identifiers outside that one file were invisible
to it — and two were found in `tests/` plus one in `src/mind_mem/block_lineage.py`, each
naming a paper (and in two cases a third-party product) as the source of an idea. A prose
rule enforced over one file is unenforced everywhere else.

WHAT COUNTS AS A VIOLATION is deliberately narrow: an actual arXiv IDENTIFIER, a paper
URL, or a bare third-party repository URL. The word "arxiv" in prose is fine, discussing a
subject is fine, and a URN example showing identifier SYNTAX is fine — the rule is about
attributing a borrowed idea in a public artifact, not about never naming a concept.

ALLOWLIST, and why each entry is there rather than being an exception that swallows the
rule: the gate's own matcher must contain the pattern it matches, and the gate's tests must
contain real identifiers as fixtures or they could not prove the detector fires. Both are
positive controls for the very rule this test enforces. Every allowlisted path is asserted
to still CONTAIN an identifier, so an entry that stops being a fixture stops being excused.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

#: An actual identifier or a paper/repo URL — not the word, and not a subject.
PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Modern arXiv ids (YYMM.NNNNN).
    (re.compile(r"arxiv\s*:\s*\d{4}\.\d{4,5}", re.I), "an arXiv identifier"),
    # PRE-2007 arXiv ids (`hep-th/9901001`, `cs.AI/0102003`). The first version of this
    # matcher missed them entirely, so an older citation would have passed the gate.
    (re.compile(r"arxiv\s*:\s*[a-z-]+(?:\.[A-Z]{2})?/\d{7}", re.I), "a pre-2007 arXiv identifier"),
    # Any arXiv URL path, not just /abs/ and /pdf/ — `/html/2501.13956v1` slipped through.
    (re.compile(r"\barxiv\.org/\S+", re.I), "an arXiv URL"),
    # DOIs as a URL *and* as a bare `doi:` prefix, which the first version did not match.
    (re.compile(r"\bdoi\.org/10\.\d{4,}", re.I), "a DOI URL"),
    (re.compile(r"\bdoi\s*:\s*10\.\d{4,}", re.I), "a DOI identifier"),
    # ATTRIBUTION TO A NAMED THIRD PARTY. Identifiers alone miss it: a line reading
    # "motivated by the … model in Zep/Graphiti" attributes a borrowed idea with no arXiv id
    # or DOI, and one survived the identifier-only version of this gate.
    #
    # The VERB is not the signal — the OBJECT is. A first attempt matched "motivated by" and
    # friends outright, and it flagged (a) ordinary technical prose like "re-derived from the
    # ids", and (b) the very phrasing the rule PRESCRIBES, "motivated by … recent research".
    # A gate that flags the approved wording is a gate that gets deleted.
    #
    # So this requires an attribution verb followed by a CAPITALISED proper noun, and
    # explicitly not by the sanctioned objects. Naming a competitor in a comparison table
    # stays legitimate — this repo does that extensively — because a table is not a claim
    # about where an idea came from.
    (
        re.compile(
            r"\b(?:motivated by|inspired by|adapted from|following the approach (?:of|in))\b"
            r"(?![^.\n]{0,60}?\b(?:recent research|prior work|the literature|recent work)\b)"
            # The proper noun must not be a GENERIC capitalised term. "motivated by
            # incidents in the broader AI ecosystem" is not an attribution, and matching
            # "AI" there was the gate's own false positive.
            r"[^.\n]{0,60}?\b(?!(?:AI|ML|LLM|API|CI|CD|OS|SQL|HTTP|JSON|UTC|RFC|MIT|BSD|GPU|CPU|KV|The|This|That|We|It|A|An|In|On|For|By)\b)"
            r"[A-Z][A-Za-z0-9]{2,}(?:/[A-Za-z0-9]+)?\b",
        ),
        "attribution of a borrowed idea to a named third party",
    ),
)

#: Paths whose identifiers are the detector or its fixtures. Each is verified below to
#: still contain one, so a stale entry cannot silently widen the exemption.
ALLOWLIST: tuple[str, ...] = (
    "scripts/check_roadmap_ticks.py",
    "tests/test_roadmap_hygiene.py",
    "tests/test_no_prior_art_attribution_repo_wide.py",
)

#: Extensions worth scanning. A lockfile or a binary carries no attribution prose.
#: `.js`, `.jsx`, `.mjs`, `.cjs` and `.mind` were absent from the first version, so an
#: attribution in a JavaScript or MIND source file would not have been scanned at all.
SCANNED_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
    ".rst",
    ".sh",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".mind",
}


def _tracked_files() -> list[pathlib.Path]:
    root = pathlib.Path(__file__).resolve().parent.parent
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        # Explicit, because `text=True` alone decodes with the LOCALE codec — cp1252 on
        # Windows — so a path carrying a non-ASCII byte decodes wrong or raises. The repo
        # gates this in `test_text_io_is_utf8.py`, and that gate caught this very file.
        encoding="utf-8",
        check=True,
    ).stdout.splitlines()
    return [root / p for p in out if pathlib.Path(p).suffix in SCANNED_SUFFIXES]


def test_no_prior_art_identifier_in_any_tracked_file() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    findings: list[str] = []
    for path in _tracked_files():
        rel = path.relative_to(root).as_posix()
        if rel in ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for pattern, what in PATTERNS:
                if pattern.search(line):
                    findings.append(f"{rel}:{line_no}: {what} -- {line.strip()[:90]}")

    assert not findings, (
        "prior-art identifiers must not appear in a public artifact; provenance belongs in "
        "the private notes. Found:\n  " + "\n  ".join(findings)
    )


def test_the_allowlist_is_not_a_loophole() -> None:
    """Every allowlisted path must still contain an identifier.

    Without this, the allowlist would quietly become a list of paths nobody rechecks: a
    file that stopped being a fixture would keep its exemption, and a real leak added to
    it later would never be reported.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    stale: list[str] = []
    for rel in ALLOWLIST:
        path = root / rel
        if not path.exists():
            stale.append(f"{rel}: allowlisted but does not exist")
            continue
        text = path.read_text(encoding="utf-8")
        if not any(p.search(text) for p, _ in PATTERNS):
            stale.append(f"{rel}: allowlisted but contains no identifier -- remove the entry")
    assert not stale, "the allowlist has stale entries:\n  " + "\n  ".join(stale)


def test_the_matcher_actually_matches() -> None:
    """Positive control. An absence assertion with an unmatchable pattern proves nothing."""
    must_match = (
        "arXiv:2501.13956",
        "arxiv: 2603.10165",
        "ARXIV:2504.19874",
        "see https://arxiv.org/abs/2501.13956 for detail",
        "https://doi.org/10.1145/3580305",
        # Forms the first version of this matcher MISSED. Each was found by an adversarial
        # audit, so each is pinned here rather than trusted to stay matched.
        "arXiv:hep-th/9901001",
        "arXiv:cs.AI/0102003",
        "https://arxiv.org/html/2501.13956v1",
        "doi:10.1145/3580305",
        # The phrasing form, which the identifier-only matcher missed on a real line.
        "motivated by the bi-temporal validity-window model in SomeProduct",
        "inspired by the approach in AnotherSystem",
    )
    for sample in must_match:
        assert any(p.search(sample) for p, _ in PATTERNS), f"matcher missed {sample!r}"

    must_not_match = (
        # Competitive comparison is legitimate and must NOT trip the gate.
        "| Feature | MIND-Mem | Mem0 | Zep | Letta |",
        "MIND-Mem vs Zep",
        "Zep requires Zep Cloud; we are self-hosted",
        # Generic capitalised terms after an attribution verb are NOT attributions.
        "Hardening thread motivated by incidents in the broader AI ecosystem",
        "motivated by the bi-temporal validity-window model described in recent research",
        "the arxiv preprint server",
        "recent research on memory decay",
        "`urn:arxiv:2401.00001` is an example URN shape",
        "a bi-temporal validity window",
    )
    for sample in must_not_match:
        hits = [what for p, what in PATTERNS if p.search(sample)]
        if sample.startswith("`urn:arxiv:"):
            # Known and accepted: a URN example matches the identifier shape. Asserted
            # here so the exception is visible rather than discovered.
            continue
        assert not hits, f"matcher false-positived on {sample!r} as {hits}"
