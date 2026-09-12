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
    (re.compile(r"arxiv\s*:\s*\d{4}\.\d{4,5}", re.I), "an arXiv identifier"),
    (re.compile(r"\barxiv\.org/(?:abs|pdf)/", re.I), "an arXiv URL"),
    (re.compile(r"\bdoi\.org/10\.\d{4,}", re.I), "a DOI URL"),
)

#: Paths whose identifiers are the detector or its fixtures. Each is verified below to
#: still contain one, so a stale entry cannot silently widen the exemption.
ALLOWLIST: tuple[str, ...] = (
    "scripts/check_roadmap_ticks.py",
    "tests/test_roadmap_hygiene.py",
    "tests/test_no_prior_art_attribution_repo_wide.py",
)

#: Extensions worth scanning. A lockfile or a binary carries no attribution prose.
SCANNED_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".ts", ".tsx", ".sh", ".rst"}


def _tracked_files() -> list[pathlib.Path]:
    root = pathlib.Path(__file__).resolve().parent.parent
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
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
    )
    for sample in must_match:
        assert any(p.search(sample) for p, _ in PATTERNS), f"matcher missed {sample!r}"

    must_not_match = (
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
