#!/usr/bin/env python3
"""Recompute every counted doc claim from its authority and fail on drift.

``scripts/count_mcp_tools.py`` proved the shape works for one number: give a
claim an authority, scan every public surface for it, fail the build when a
surface disagrees. Everything it does not cover kept drifting anyway. On
2026-09-01, with that gate green:

* ``README.md:24`` rendered a **tests-9,366** badge while the CI selector
  collected **9,701**. A green badge stating a false number is a defect, not
  cosmetics.
* ``docs/governance.md:52`` advertised "7,500+ tests across the suite".
* The live HuggingFace card said the weights know **84** MCP tools;
  ``train/HF_MODEL_CARD_v4.md`` said **96**. Two numbers for one fact, and --
  measured here for the first time -- *both wrong*: the distinct tool surface
  at the trained revision (``v4.1.1``, corpus ``v4.0.0``) is **83**. The 84 is
  the old double-counted registration total (``recall`` is registered in both
  ``public.py`` and ``recall.py``); the 96 matches no revision under any
  counting rule.
* ``docs/setup.md`` and ``docs/client-integrations.md`` each advertised an
  "89-tool surface" -- the hyphenated spelling ``count_mcp_tools`` cannot
  match, so that gate reported green over both for three releases.
* Three docs said ``mm install-all`` supports 8 MCP-aware clients; the
  registry has had 11 writers since ``copilot-cli``, ``grok-build`` and
  ``vibe`` landed.
* The README comparison matrix said 16 MIND kernels; ``mind/`` holds 26.

Each claim below names the command or file that decides its true value. Run
with ``--fix`` to rewrite the stale numbers in place.

Authorities
-----------
=====================  =====================================================
tests                  ``pytest tests/ --ignore=tests/integration
                       --collect-only -q -m "not stress"`` (the CI selector)
live tools             ``scripts/count_mcp_tools.count_tools()``
trained-on tools       the same counting rule applied to
                       ``git archive <TRAINED_REVISION> src/mind_mem/mcp``
clients                ``mind_mem.hook_installer.AGENT_REGISTRY``
MCP-aware clients      the same registry, entries with a non-empty ``mcp_fmt``
resources              ``mcp.resource(...)`` registrations in
                       ``src/mind_mem/mcp/resources.py``
MIND kernels           ``mind/*.mind``
version                ``__version__`` in ``src/mind_mem/__init__.py``
core deps              ``[project] dependencies`` in ``pyproject.toml``
=====================  =====================================================

An authority that cannot be computed exits **2**, never 0 with an empty
finding list: a verifier that died is not a verifier that passed.

Usage:
    python3 scripts/check_docs_alignment.py                  # report + exit 1 on drift
    python3 scripts/check_docs_alignment.py --fix            # rewrite stale numbers
    python3 scripts/check_docs_alignment.py --print          # just print the authorities
    python3 scripts/check_docs_alignment.py --check-live     # + the published hub card
    python3 scripts/check_docs_alignment.py --tests-collected 9701
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Importable both as ``python3 scripts/check_docs_alignment.py`` and as
# ``scripts.check_docs_alignment``; the repo root on sys.path gives the tool
# counter ONE module identity either way, so a test that monkeypatches it
# patches the object this module actually calls.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import count_mcp_tools as cmt  # noqa: E402  (path shim above must run first)
from scripts.alignment_authorities import (  # noqa: E402  (path shim above must run first)
    TRAINED_REVISION,
    AuthorityError,
    _project_root,
    client_count,
    collect_test_count,
    core_dependency_count,
    live_tool_count,
    mcp_client_count,
    mind_kernel_count,
    package_version,
    parse_collected,  # noqa: F401  re-exported: this module is the one entry point callers and tests import
    resource_count,
    trained_tool_count,
)

# Surfaces a reader can reach. ``CHANGELOG.md`` and every version/date-stamped
# document are records, not claims about now -- ``_is_historical_file`` decides
# (``cmt._is_historical`` plus the underscore-separator case it misses), so
# this gate and the tool-count gate excuse the same files for the same reason.
_DOC_SURFACES = ("README.md", "CLAUDE.md", "docs/**/*.md", "train/**/*.md")


# --------------------------------------------------------------------------
# Authorities -- the values live in scripts/alignment_authorities.py; the
# names are re-bound here so ``resolve_authorities`` calls THIS module's
# attributes and a test that patches one actually changes what runs.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Authorities:
    """The true value of every gated claim, each computed from its own source."""

    tests: int
    live_tools: int
    trained_tools: int
    clients: int
    mcp_clients: int
    resources: int
    mind_kernels: int
    version: str
    core_deps: int


def resolve_authorities(root: Path | None = None, tests_collected: int | None = None) -> Authorities:
    root = root or _project_root()
    return Authorities(
        tests=tests_collected if tests_collected is not None else collect_test_count(root),
        live_tools=live_tool_count(),
        trained_tools=trained_tool_count(root=root),
        clients=client_count(root),
        mcp_clients=mcp_client_count(root),
        resources=resource_count(root),
        mind_kernels=mind_kernel_count(root),
        version=package_version(root),
        core_deps=core_dependency_count(root),
    )


# --------------------------------------------------------------------------
# Claim patterns
# --------------------------------------------------------------------------

# A grouped integer as it appears in prose ("9,701"), in a shields.io path
# ("9%2C701"), or bare ("9701").
_GROUPED = r"\d{1,3}(?:(?:,|%2C)\d{3})*|\d+"

# Tool and client counts are two- or three-digit, exactly as
# ``count_mcp_tools`` has them: "a bare integer near the word tool is too
# noisy to gate on", and a one-digit rule turns "the builder sees 1 tool" into
# a finding. Resource counts are single-digit today, and the word "resources"
# is rare enough to carry the narrower number safely.
_SMALL = r"\d{2,3}"
_TINY = r"\d{1,3}"

_TESTS_PATTERNS = (
    # shields.io badge path: ...badge/tests-9%2C701-brightgreen...
    re.compile(rf"tests-(?P<n>{_GROUPED})(?P<plus>)-", re.IGNORECASE),
    # the badge's alt text, which drifts independently of the path
    re.compile(rf"Tests:\s*(?P<n>{_GROUPED})(?P<plus>)", re.IGNORECASE),
    # "7,500+ tests" -- the trailing "+" is part of the claim and is replaced
    # with it, so a fix produces "9,707 tests" and not "9,707+ tests".
    re.compile(rf"\b(?P<n>{_GROUPED})(?P<plus>\+?)\s+tests\b", re.IGNORECASE),
)

_CLIENT_PATTERNS = (
    re.compile(rf"clients-(?P<n>{_SMALL})(?P<plus>)-", re.IGNORECASE),
    re.compile(rf"Clients:\s*(?P<n>{_SMALL})(?P<plus>)", re.IGNORECASE),
    # "16 AI coding clients" (docs/client-integrations.md:3) -- one modifier
    # more than count_mcp_tools' shape allows, and it sat three releases stale.
    # Only the spellings actually present are added; a general "\w+ clients"
    # would match "16 of the 19 clients".
    re.compile(rf"\b(?P<n>{_SMALL})(?P<plus>)\s+(?:AI\s+)?(?:coding\s+)?clients\b", re.IGNORECASE),
)

# "Supports 8 MCP-aware clients" -- a capability claim with a mechanical
# authority (an AgentSpec with a non-empty ``mcp_fmt``), stale in three docs.
_MCP_CLIENT_PATTERNS = (re.compile(rf"\b(?P<n>{_TINY})(?P<plus>)\s+MCP-aware\s+clients?\b", re.IGNORECASE),)

# "16 MIND kernels" -- authority is the `mind/*.mind` directory. "MIND" is
# REQUIRED: docs/governance.md says "the 512 kernel", a product name whose
# 512 is not a count.
_KERNEL_PATTERNS = (re.compile(rf"\b(?P<n>{_TINY})(?P<plus>)\s+MIND\s+kernels?\b", re.IGNORECASE),)

_RESOURCE_PATTERNS = (re.compile(rf"\b(?P<n>{_TINY})(?P<plus>)\s+resources\b", re.IGNORECASE),)

_TOOL_PATTERNS = (
    re.compile(rf"\b(?P<n>{_SMALL})(?P<plus>)\s+(?:MCP\s+)?tools?\b", re.IGNORECASE),
    re.compile(rf"tools?[-_:]\s*(?P<n>{_SMALL})(?P<plus>)\b", re.IGNORECASE),
    re.compile(rf"MCP\s+Tools?\s*\((?P<n>{_SMALL})(?P<plus>)\)", re.IGNORECASE),
    re.compile(rf"\b(?P<n>{_SMALL})(?P<plus>)\s+distinct\s+tools?\b", re.IGNORECASE),
    # "trained against a 96-tool surface" -- the hyphenated adjective form,
    # which no pattern in count_mcp_tools can see. It is exactly the spelling
    # the stale trained-on count was hiding behind, and it was also hiding two
    # live "89-tool surface" claims in docs/ while that gate reported green.
    re.compile(rf"\b(?P<n>{_SMALL})(?P<plus>)-tool\b", re.IGNORECASE),
)

_VERSION_PATTERNS = (re.compile(r"(?:Current|Latest) release:?[^\n]{0,40}?\bv(?P<n>\d+\.\d+\.\d+)(?P<plus>)"),)

# A whole-suite test claim, as opposed to "18 tests" about one module. Both
# spellings appear in these docs and only the first is this gate's business.
#
# The floor is a fact about the corpus, not a tuning knob: the largest
# single-module test claim in docs/ is two digits ("43 tests" in
# docs/recompaction.md) and the suite has been four digits since v3.x. A claim
# below the floor is still suite-scale when the sentence says so -- "the suite
# has 900 tests" is caught by the scope words, not by its size.
_SUITE_SCALE_FLOOR = 1000
_SUITE_SCOPE = re.compile(r"\b(suite|full pytest|pytest matrix|entire test|whole test|test count)\b", re.IGNORECASE)

# Which tool count a claim is about. The model card and the 4b setup guide
# state BOTH -- what the weights know and what the server exposes -- so a
# per-file rule cannot separate them. Nearest marker wins; a line with no
# marker falls back to the file's default.
_TRAINED_MARK = re.compile(r"\b(trained|training|weights|checkpoint|knows|corpus)\b", re.IGNORECASE)
_LIVE_MARK = re.compile(r"\b(live|exposes|currently|current)\b", re.IGNORECASE)

# Files whose tool claims default to the TRAINED surface. Everything else
# defaults to the live surface.
_TRAINED_DEFAULT_PREFIXES = ("train/HF_MODEL_CARD", "huggingface.co/star-ga/mind-mem-4b")

# ``cmt._VERSION_STAMP`` reads "-v3.9.0" and "/v4" but not "_v3.9.0", so
# ``train/RETRAIN_v3.9.0.md`` -- a retrain plan for a shipped release -- was
# scanned as a live surface. Underscore is a filename separator here exactly
# as hyphen is.
_VERSION_STAMP_FILE = re.compile(r"(^|[-_/])v\d+(\.\d+)*", re.IGNORECASE)

# Published surfaces are claims about NOW whatever their filename says.
# ``train/HF_MODEL_CARD_v4.md`` is uploaded verbatim to the HuggingFace hub,
# so the "_v4" in its name must not excuse it the way it excuses
# ``train/RETRAIN_v3.9.0.md``.
_ALWAYS_LIVE_PREFIXES = ("train/HF_MODEL_CARD",)

# A CHANGELOG-shaped entry header makes everything under it a RECORD of that
# release: "## v3.1.4 (Released 2026-04-18) ... mm install-all now wires 17
# clients" is TRUE about v3.1.4, and rewriting it to 19 would falsify the
# history.
#
# The header must OPEN with the version. That is the whole discriminator, and
# it is load-bearing: a first cut exempted any scope that merely mentioned a
# version, which silently excused two live "89-tool surface" claims -- one
# under "## 5. All clients at once (v3.1.0 recommended)" and one in a
# paragraph opening "**Since v3.1.0, `mm install-all` writes TWO things**".
# Both describe the CURRENT product and both were three releases stale.
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_RECORD_HEADER = re.compile(r"^[*_\[]{0,2}v?\d+\.\d+(\.\d+)?", re.IGNORECASE)

# A line that OPENS with a past-tense release attribution and names a version
# is a record of that release: "Builds on **v3.9.0** - **81 tools**, 4000+
# tests, native MCP for 17 AI clients". Rewriting those to today's numbers
# would falsify what v3.9.0 shipped.
#
# "Since vX" and "As of vX" are deliberately NOT here. They scope FORWARD --
# "Since v3.1.0, `mm install-all` writes ... the full 89-tool surface" is a
# claim about the product now, and it was three releases stale.
_SCOPE_VERSION = re.compile(r"\bv\d+(\.\d+)+", re.IGNORECASE)
_RECORD_ATTRIBUTION = re.compile(
    r"^\s*(?:builds on|released in|shipped in|introduced in|landed in|v\d+(?:\.\d+)+\s+(?:added|introduced|shipped|landed))\b",
    re.IGNORECASE,
)


def _is_historical_file(rel: str) -> bool:
    if any(rel.startswith(prefix) for prefix in _ALWAYS_LIVE_PREFIXES):
        return False
    return cmt._is_historical(rel) or bool(_VERSION_STAMP_FILE.search(rel))


def _record_scopes(lines: list[str]) -> list[bool]:
    """For each line, whether it sits inside a release-record scope.

    Two scopes can open a record: the nearest preceding markdown heading, and
    the first line of the paragraph the claim is in (CLAUDE.md writes its
    release history as bare paragraphs opening ``**v4.1.1** (released
    ...)``, under a generic ``## Overview``). Either qualifies only when it
    OPENS with the version -- see ``_RECORD_HEADER``.
    """
    out: list[bool] = []
    heading_versioned = False
    para_versioned = False
    in_paragraph = False
    for line in lines:
        heading = _HEADING_RE.match(line)
        if heading is not None:
            heading_versioned = _RECORD_HEADER.match(heading.group(1).strip()) is not None
            in_paragraph = False
            para_versioned = False
            out.append(heading_versioned)
            continue
        if not line.strip():
            in_paragraph = False
            para_versioned = False
            out.append(heading_versioned)
            continue
        if not in_paragraph:
            in_paragraph = True
            para_versioned = _RECORD_HEADER.match(line.strip()) is not None
        out.append(heading_versioned or para_versioned)
    return out


@dataclass(frozen=True)
class Finding:
    surface: str
    lineno: int
    kind: str
    claimed: str
    actual: str
    excerpt: str
    start: int
    end: int

    def __str__(self) -> str:
        return f"{self.surface}:{self.lineno}: {self.kind} claims {self.claimed}, authority says {self.actual} -- {self.excerpt!r}"


def _parse_int(text: str) -> int:
    return int(text.replace(",", "").replace("%2C", "").replace("%2c", ""))


def _render_like(value: int, template: str) -> str:
    """Format *value* using the same digit grouping *template* used."""
    if "%2C" in template or "%2c" in template:
        return f"{value:,}".replace(",", "%2C")
    if "," in template:
        return f"{value:,}"
    return str(value)


def _doc_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for pattern in _DOC_SURFACES:
        out.extend(sorted(root.glob(pattern)))
    seen: set[Path] = set()
    keep: list[Path] = []
    for path in out:
        if path in seen:
            continue
        seen.add(path)
        if not _is_historical_file(path.relative_to(root).as_posix()):
            keep.append(path)
    return keep


def _nearest(pattern: re.Pattern[str], line: str, match: re.Match[str]) -> int | None:
    """Distance from *match* to the closest occurrence of *pattern* in *line*."""
    best: int | None = None
    for hit in pattern.finditer(line):
        if hit.end() <= match.start():
            gap = match.start() - hit.end()
        elif hit.start() >= match.end():
            gap = hit.start() - match.end()
        else:
            gap = 0
        if best is None or gap < best:
            best = gap
    return best


def _tool_scope(rel: str, line: str, match: re.Match[str]) -> str:
    """``"trained"`` or ``"live"`` for one tool-count claim."""
    trained = _nearest(_TRAINED_MARK, line, match)
    live = _nearest(_LIVE_MARK, line, match)
    if trained is not None and (live is None or trained < live):
        return "trained"
    if live is not None and (trained is None or live < trained):
        return "live"
    return "trained" if any(rel.startswith(p) for p in _TRAINED_DEFAULT_PREFIXES) else "live"


def _tool_expected(rel: str, line: str, match: re.Match[str], auth: Authorities) -> int | None:
    if _tool_scope(rel, line, match) == "trained":
        # NOT version-exempt. A version stamp next to a trained-on count is
        # the whole point of the sentence ("v4.0.0's 83 MCP tools"); excusing
        # it is what let the model card carry a wrong number through four
        # releases while the tool-count gate reported agreement.
        return auth.trained_tools
    if cmt._version_qualifies(line, match):
        # "19 MCP tools at v1.x" describes a past release, not the surface.
        return None
    return auth.live_tools


def scan_line(rel: str, lineno: int, line: str, auth: Authorities, historical: bool = False) -> list[Finding]:
    """Every stale numeric claim on *line*, plus the version claim.

    ``historical`` marks a line under a version-stamped heading or paragraph;
    its COUNT claims are records and are left alone. A "Current release: vX"
    line is present-tense by construction and is checked regardless.
    """
    findings: list[Finding] = []
    for pattern in _VERSION_PATTERNS:
        for match in pattern.finditer(line):
            if match.group("n") != auth.version:
                findings.append(
                    Finding(rel, lineno, "version", match.group("n"), auth.version, match.group(0), match.start("n"), match.end("n"))
                )
    if historical or cmt._LINE_TRANSITION.search(line):
        # "MCP Tools (81) -> (84)" records a past fix; it is not a claim.
        return findings
    if _RECORD_ATTRIBUTION.match(line) and _SCOPE_VERSION.search(line):
        return findings

    checks: list[tuple[str, tuple[re.Pattern[str], ...], Callable[[re.Match[str]], int | None]]] = [
        (
            "tests",
            _TESTS_PATTERNS,
            lambda m: (
                None
                if cmt._version_qualifies(line, m) or not (_parse_int(m.group("n")) >= _SUITE_SCALE_FLOOR or _SUITE_SCOPE.search(line))
                else auth.tests
            ),
        ),
        (
            "clients",
            _CLIENT_PATTERNS,
            lambda m: None if cmt._version_qualifies(line, m) else auth.clients,
        ),
        (
            "mcp_clients",
            _MCP_CLIENT_PATTERNS,
            lambda m: None if cmt._version_qualifies(line, m) else auth.mcp_clients,
        ),
        ("resources", _RESOURCE_PATTERNS, lambda m: auth.resources),
        (
            "mind_kernels",
            _KERNEL_PATTERNS,
            lambda m: None if cmt._version_qualifies(line, m) else auth.mind_kernels,
        ),
        ("tools", _TOOL_PATTERNS, lambda m: _tool_expected(rel, line, m, auth)),
    ]

    seen: set[tuple[int, int]] = set()
    for kind, patterns, expected_for in checks:
        for pattern in patterns:
            for match in pattern.finditer(line):
                span = (match.start("n"), match.end("plus"))
                if span in seen:
                    continue
                expected = expected_for(match)
                if expected is None:
                    continue
                seen.add(span)
                if _parse_int(match.group("n")) != expected:
                    findings.append(
                        Finding(
                            rel,
                            lineno,
                            kind,
                            match.group("n"),
                            _render_like(expected, match.group("n")),
                            match.group(0),
                            span[0],
                            span[1],
                        )
                    )
    return findings


def scan_docs(auth: Authorities, root: Path | None = None) -> list[Finding]:
    root = root or _project_root()
    findings: list[Finding] = []
    for path in _doc_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive
            print(f"WARN: could not read {rel}: {exc}", file=sys.stderr)
            continue
        lines = text.splitlines()
        scopes = _record_scopes(lines)
        for idx, line in enumerate(lines):
            findings.extend(scan_line(rel, idx + 1, line, auth, historical=scopes[idx]))
    findings.extend(scan_builder_default(auth, root))
    findings.extend(check_core_deps_badge(auth, root))
    return findings


# ``train/build_model_card.py`` renders the public HuggingFace card. Its
# ``MM_TRAINED_ON_TOOLS`` default is the number that gets published when the
# card is regenerated without an override -- a claim in code, gated like a
# claim in prose.
_BUILDER_DEFAULT_RE = re.compile(r'MM_TRAINED_ON_TOOLS"\s*,\s*"(?P<n>\d+)"')


def scan_builder_default(auth: Authorities, root: Path | None = None) -> list[Finding]:
    root = root or _project_root()
    path = root / "train" / "build_model_card.py"
    if not path.is_file():
        return []
    rel = path.relative_to(root).as_posix()
    out: list[Finding] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for match in _BUILDER_DEFAULT_RE.finditer(line):
            if int(match.group("n")) != auth.trained_tools:
                out.append(
                    Finding(
                        rel,
                        lineno,
                        "tools",
                        match.group("n"),
                        str(auth.trained_tools),
                        match.group(0),
                        match.start("n"),
                        match.end("n"),
                    )
                )
    return out


_CORE_DEPS_BADGE_RE = re.compile(r"core_deps-(?P<n>[A-Za-z0-9_]+)-")


def check_core_deps_badge(auth: Authorities, root: Path | None = None) -> list[Finding]:
    """The "core deps: zero" badge must agree with ``[project] dependencies``."""
    root = root or _project_root()
    readme = root / "README.md"
    if not readme.is_file():
        return []
    truth = "zero" if auth.core_deps == 0 else str(auth.core_deps)
    out: list[Finding] = []
    for lineno, line in enumerate(readme.read_text(encoding="utf-8").splitlines(), 1):
        for match in _CORE_DEPS_BADGE_RE.finditer(line):
            if match.group("n").lower() != truth:
                out.append(
                    Finding("README.md", lineno, "core_deps", match.group("n"), truth, match.group(0), match.start("n"), match.end("n"))
                )
    return out


# --------------------------------------------------------------------------
# The published model card (not a file in this repo)
# --------------------------------------------------------------------------

HF_CARD_URL = "https://huggingface.co/star-ga/mind-mem-4b/raw/main/README.md"


def check_live_hf_card(auth: Authorities, url: str = HF_CARD_URL, timeout: float = 30.0) -> list[Finding]:
    """Check the model card PUBLISHED on the hub, which no CI checkout contains.

    This is the surface that carried 84 while the tree carried 96 -- neither
    was reachable from the other, so neither could be gated by the other. It
    needs the network, so it is NOT part of the default run: the release
    preflight opts in with ``--check-live``, and a fetch failure raises
    ``AuthorityError`` (exit 2) rather than returning an empty list.
    """
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    if not url.startswith("https://huggingface.co/"):
        raise AuthorityError(f"refusing to fetch a non-HuggingFace URL: {url!r}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310 - scheme+host checked above
            text = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise AuthorityError(f"could not fetch the published model card ({url}): {exc}") from exc
    if not text.strip():
        raise AuthorityError(f"the published model card at {url} came back empty")
    rel = "huggingface.co/star-ga/mind-mem-4b/README.md"
    findings: list[Finding] = []
    lines = text.splitlines()
    scopes = _record_scopes(lines)
    for idx, line in enumerate(lines):
        findings.extend(scan_line(rel, idx + 1, line, auth, historical=scopes[idx]))
    return findings


# --------------------------------------------------------------------------
# Fix mode
# --------------------------------------------------------------------------

_FIXABLE = frozenset({"tests", "clients", "mcp_clients", "resources", "mind_kernels", "tools"})


def apply_fixes(findings: list[Finding], root: Path | None = None) -> tuple[int, list[Finding]]:
    """Rewrite every fixable stale number in place. Returns (fixed, skipped).

    Edits are applied by SPAN, rightmost first, so two claims on one line
    (a badge path and its alt text) cannot shift each other's offsets, and a
    number that also appears elsewhere in the same excerpt cannot be hit by
    accident.
    """
    root = root or _project_root()
    skipped = [f for f in findings if f.kind not in _FIXABLE]
    by_file: dict[str, list[Finding]] = {}
    for finding in findings:
        if finding.kind in _FIXABLE:
            by_file.setdefault(finding.surface, []).append(finding)
    fixed = 0
    for rel, items in by_file.items():
        path = root / rel
        if not path.is_file():
            # A surface that is not a file in this tree (the published hub
            # card) is re-uploaded, never rewritten in place.
            skipped.extend(items)
            continue
        text = path.read_text(encoding="utf-8")
        newline = "\r\n" if "\r\n" in text else "\n"
        lines = text.split(newline)
        for finding in sorted(items, key=lambda f: (f.lineno, f.start), reverse=True):
            idx = finding.lineno - 1
            line = lines[idx]
            if line[finding.start : finding.end].lstrip("+") != finding.claimed:
                # The line moved under us; refuse to guess.
                skipped.append(finding)
                continue
            lines[idx] = line[: finding.start] + finding.actual + line[finding.end :]
            fixed += 1
        path.write_text(newline.join(lines), encoding="utf-8")
    return fixed, skipped


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check every counted doc claim against its authority.")
    parser.add_argument("--fix", action="store_true", help="Rewrite stale numbers in place.")
    parser.add_argument("--print", dest="show", action="store_true", help="Print the resolved authorities and exit 0.")
    parser.add_argument(
        "--check-live",
        action="store_true",
        help="Also check the model card published on the HuggingFace hub (needs the network; release preflight only).",
    )
    parser.add_argument(
        "--tests-collected",
        type=int,
        default=None,
        help="Use this collected-test count instead of running pytest (CI passes the count it already collected with the CI selector).",
    )
    args = parser.parse_args(argv)

    try:
        auth = resolve_authorities(tests_collected=args.tests_collected)
    except AuthorityError as exc:
        # An unreachable authority is NOT a pass. Say so, loudly, and exit
        # non-zero -- an empty finding list here would read as "aligned".
        print(f"::error::mind-mem docs-alignment authority unavailable: {exc}", file=sys.stderr)
        return 2

    print("Authorities:")
    print(f"  tests (CI selector)              : {auth.tests}")
    print(f"  MCP tools (live)                 : {auth.live_tools}")
    print(f"  MCP tools (weights, {TRAINED_REVISION:<8})   : {auth.trained_tools}")
    print(f"  AI clients (AGENT_REGISTRY)      : {auth.clients}")
    print(f"  MCP-aware clients (mcp_fmt)      : {auth.mcp_clients}")
    print(f"  MCP resources                    : {auth.resources}")
    print(f"  MIND kernels (mind/*.mind)       : {auth.mind_kernels}")
    print(f"  package version                  : {auth.version}")
    print(f"  core dependencies                : {auth.core_deps}")
    if args.show:
        return 0

    findings = scan_docs(auth)
    live: list[Finding] = []
    if args.check_live:
        try:
            live = check_live_hf_card(auth)
        except AuthorityError as exc:
            print(f"::error::mind-mem published model card unreachable: {exc}", file=sys.stderr)
            return 2
        print(f"checked the published model card: {len(live)} stale claim(s)")
    if args.fix:
        fixed, skipped = apply_fixes(findings, _project_root())
        print(f"fixed {fixed} stale claim(s)")
        # The hub card is not a file here: it is re-uploaded, never rewritten.
        skipped += live
        for finding in skipped:
            print(f"  NOT auto-fixable ({finding.kind}): {finding}", file=sys.stderr)
        return 1 if skipped else 0

    findings += live
    if findings:
        print(f"::error::mind-mem docs alignment: {len(findings)} stale claim(s).", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        print("  run: python3 scripts/check_docs_alignment.py --fix", file=sys.stderr)
        return 1
    print("docs: every gated claim agrees with its authority")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
