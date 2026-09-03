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

A second sweep on 2026-09-02, with all of the above green, found the same
failure in three more places that had no authority at all:

* Four live docs enumerated the CI Python matrix as "3.10, 3.12, 3.13, 3.14",
  which has been missing 3.11 since that row was added, and
  ``docs/testing-guide.md`` then derived "12 CI jobs" from its own short list
  (the matrix is a 3 x 5 cross-product: 15). ``docs/troubleshooting.md`` still
  told readers to look for an ``allow-failure`` carve-out on the 3.14 rows
  that was deliberately removed.
* ``docs/ci-workflows.md`` enumerated the workflow directory in prose: it
  named a "Security Review" workflow that does not exist, gave Benchmark a
  push/PR trigger it had lost, and omitted two whole workflows.
* The model card advertised a "35-flag inventory" through four releases.
  ``ALL_V4_FLAGS`` held 38 at the trained revision and holds 52 today, so 35
  described no revision -- the 84-vs-96 failure again, in a different number.
  ``docs/mind-mem-4b-setup.md`` also still quoted the ``v4.0.0-base``
  archive's 109/109 eval for the weights ``main`` points at (111/111 main and
  22/22 held out, 133 total).

Each claim below names the command or file that decides its true value. Run
with ``--fix`` to rewrite the stale numbers in place.

Authorities
-----------
=====================  =====================================================
test functions         ``def test_*`` in every ``tests/**/test_*.py`` --
                       pytest's collection rule applied to SOURCE, never a
                       run, so the number cannot depend on which optional
                       extras the host has installed (it did through 5.0.1;
                       see ``alignment_authorities.static_test_count``)
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
v4 feature flags       ``ALL_V4_FLAGS``, live and at ``<TRAINED_REVISION>``
4b eval probes         the probe lists ``train/eval_harness.py`` and
                       ``train/eval_holdout.py`` actually bench
CI Python / OS / jobs  the ``test`` job's ``strategy.matrix`` in
                       ``.github/workflows/ci.yml`` (jobs = the cross-product)
workflow table         every file in ``.github/workflows`` and its ``name:``
supported Python       ``requires-python`` (the floor) and the
                       ``Programming Language :: Python ::`` classifiers (the
                       advertised range)
storage backends       ``init_workspace.SUPPORTED_BACKENDS``
module facts           the module's own line count, and the ``def test_*``
                       functions in ``tests/test_<stem>.py``
shipped vs experimental  the MCP tool registry: nothing under an
                       "Experimental" heading may name a registered module
=====================  =====================================================

An authority that cannot be computed exits **2**, never 0 with an empty
finding list: a verifier that died is not a verifier that passed. Two of them
read the TRAINED revision out of git history, so a shallow CI checkout exits 2
with a message naming ``fetch-depth: 0`` rather than reporting agreement.

Usage:
    python3 scripts/check_docs_alignment.py                  # report + exit 1 on drift
    python3 scripts/check_docs_alignment.py --fix            # rewrite stale numbers
    python3 scripts/check_docs_alignment.py --print          # just print the authorities
    python3 scripts/check_docs_alignment.py --check-live     # + the published hub card
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, replace
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
    ci_matrix,
    client_count,
    core_dependency_count,
    eval_probe_counts,
    live_flag_count,
    live_tool_count,
    mcp_client_count,
    mind_kernel_count,
    module_line_count,
    module_test_count,
    package_version,
    python_support,
    resource_count,
    static_test_count,
    storage_backends,
    trained_flag_count,
    trained_tool_count,
    workflow_inventory,
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
    live_flags: int
    trained_flags: int
    eval_main_probes: int
    eval_holdout_probes: int
    ci_python_versions: tuple[str, ...]
    ci_operating_systems: tuple[str, ...]
    workflows: tuple[tuple[str, str], ...]
    python_floor: str
    python_classifier_min: str
    python_classifier_max: str
    backends: tuple[str, ...]

    @property
    def ci_jobs(self) -> int:
        """The test matrix is a cross-product, so the job count is derived."""
        return len(self.ci_python_versions) * len(self.ci_operating_systems)

    @property
    def eval_total_probes(self) -> int:
        return self.eval_main_probes + self.eval_holdout_probes


def resolve_authorities(root: Path | None = None) -> Authorities:
    """Every authority, computed from the tree at *root* and nothing else.

    No caller may inject a value for a leg -- the previous signature took
    ``tests_collected`` so CI could pass the number it had just collected,
    and that hand-off was the seam through which the badge became its own
    authority once. Every leg is computed here or the gate exits 2.
    """
    root = root or _project_root()
    matrix = ci_matrix(root)
    eval_probes = eval_probe_counts(root)
    py_support = python_support(root)
    return Authorities(
        tests=static_test_count(root),
        live_tools=live_tool_count(),
        trained_tools=trained_tool_count(root=root),
        clients=client_count(root),
        mcp_clients=mcp_client_count(root),
        resources=resource_count(root),
        mind_kernels=mind_kernel_count(root),
        version=package_version(root),
        core_deps=core_dependency_count(root),
        live_flags=live_flag_count(root),
        trained_flags=trained_flag_count(root=root),
        eval_main_probes=eval_probes[0],
        eval_holdout_probes=eval_probes[1],
        ci_python_versions=matrix.python_versions,
        ci_operating_systems=matrix.operating_systems,
        workflows=tuple(sorted(workflow_inventory(root).items())),
        python_floor=py_support[0],
        python_classifier_min=py_support[1],
        python_classifier_max=py_support[2],
        backends=storage_backends(root),
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
    # shields.io badge path: ...badge/test_functions-10%2C617-brightgreen...
    re.compile(rf"test_functions-(?P<n>{_GROUPED})(?P<plus>)-", re.IGNORECASE),
    # the badge's alt text, which drifts independently of the path
    re.compile(rf"Test functions:\s*(?P<n>{_GROUPED})(?P<plus>)", re.IGNORECASE),
    # "7,500+ test functions" -- the trailing "+" is part of the claim and is
    # replaced with it, so a fix produces "10,617 test functions" and not
    # "10,617+ test functions".
    re.compile(rf"\b(?P<n>{_GROUPED})(?P<plus>\+?)\s+test\s+functions\b", re.IGNORECASE),
)

# The RETIRED spelling. "N tests" at suite scale was the shape of every claim
# through 5.0.1, and it named a runner's count -- which is a property of the
# machine that ran it (see ``alignment_authorities.static_test_count``). The
# authority now counts test FUNCTIONS in the tree, and no number this gate can
# compute makes "N tests" true on every machine. So the old shape is refused
# rather than re-numbered: rewriting the digits under an unchanged noun is how
# a true sentence becomes a false one with the gate green. Not auto-fixable on
# purpose -- the fix is to say what is measured. Module-scale claims ("18
# tests") are outside suite scale and untouched; the ``module_facts`` leg owns
# those.
_RETIRED_TESTS_PATTERNS = (
    re.compile(rf"tests-(?P<n>{_GROUPED})(?P<plus>)-", re.IGNORECASE),
    re.compile(rf"\bTests:\s*(?P<n>{_GROUPED})(?P<plus>)", re.IGNORECASE),
    re.compile(rf"\b(?P<n>{_GROUPED})(?P<plus>\+?)\s+tests\b(?!\s+functions)", re.IGNORECASE),
)
_RETIRED_TESTS_ACTUAL = "a 'test functions' claim (a runner's count depends on the machine; see scripts/alignment_authorities.py)"

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

# "38-flag inventory" / "38 v4 feature flags" / "52 flags + is_enabled". The
# model card advertised a **35-flag** inventory through four releases; the
# trained revision declared 38 and the tree now declares 52, so 35 was a
# number with no revision behind it -- the same failure as 84-vs-96, caught
# the same way. Scoped trained-vs-live by the same nearest-marker rule the
# tool count uses, because the card states both.
_FLAG_PATTERNS = (
    re.compile(rf"\b(?P<n>{_TINY})(?P<plus>)-flag\b", re.IGNORECASE),
    re.compile(rf"\b(?P<n>{_TINY})(?P<plus>)\s+(?:v4\s+)?feature\s+flags\b", re.IGNORECASE),
    re.compile(rf"\b(?P<n>{_TINY})(?P<plus>)\s+flags\b", re.IGNORECASE),
)

# "Total: 15 CI jobs" / "green across 15 OS x Python-version rows". Derived,
# not declared: the job count is len(os) * len(python-version) in ci.yml's
# `test` matrix, so a row added to either list moves it.
_CI_JOB_PATTERNS = (
    re.compile(rf"\b(?P<n>{_TINY})(?P<plus>)\s+CI\s+jobs?\b", re.IGNORECASE),
    re.compile(rf"\b(?P<n>{_TINY})(?P<plus>)\s+OS\s*[x\u00d7]\s*Python[\w-]*\s+rows\b", re.IGNORECASE),
)

# An enumeration of Python versions ("3.10, 3.12, 3.13, and 3.14",
# "3.10/3.12/3.13/3.14"). Four live docs enumerated FOUR versions while the
# matrix has run five since 3.11 was added, and one of them then derived a
# job count from its own short list.
_PYVER_LIST = re.compile(r"3\.\d{1,2}(?:\s*(?:,|/)\s*(?:and\s+)?3\.\d{1,2})+")

# The enumeration is only a claim about the matrix when the line -- or the
# section it sits under -- says so. Without this guard, "OOM kills on ubuntu
# 3.12/3.14" in a release record reads as a matrix claim; with it, a bare
# "- Python: 3.10, ..." under "## CI Matrix" still does.
_CI_SCOPE = re.compile(r"\b(CI|matrix|tested in|test matrix|supported)\b", re.IGNORECASE)

# "Python 3.10+" (nine live surfaces) and "Python 3.10-3.14 supported" (one).
# The floor is what pip enforces (``requires-python``); the range is what the
# index advertises (the classifiers). Neither had a gate, so both would have
# gone stale the day the floor moved -- and nine copies of a wrong minimum is
# nine users installing something that cannot run.
_PY_FLOOR_PATTERN = re.compile(r"Python\s+(?P<n>3\.\d{1,2})(?P<plus>\+)")
_PY_RANGE_PATTERN = re.compile(r"Python\s+(?P<lo>3\.\d{1,2})\s*[\u2013\u2014-]\s*(?P<hi>3\.\d{1,2})\s+supported")


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
#
# ONE definition, shared with ``count_mcp_tools``: that gate uses the same
# markers to REFER a trained-scope claim here instead of judging it with the
# live count. Two copies of this vocabulary would let the two gates disagree
# about which surface a sentence is talking about, and the claim would fall
# through the gap between them.
_TRAINED_MARK = cmt.TRAINED_MARK
_LIVE_MARK = cmt.LIVE_MARK

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


_nearest_span = cmt.nearest_span


def _nearest(pattern: re.Pattern[str], line: str, match: re.Match[str]) -> int | None:
    """Distance from *match* to the closest occurrence of *pattern* in *line*."""
    return _nearest_span(pattern, line, match.start(), match.end())


def _tool_scope_span(rel: str, line: str, start: int, end: int) -> str:
    """``"trained"`` or ``"live"`` for the tool-count claim at ``line[start:end]``."""
    trained = _nearest_span(_TRAINED_MARK, line, start, end)
    live = _nearest_span(_LIVE_MARK, line, start, end)
    if trained is not None and (live is None or trained < live):
        return "trained"
    if live is not None and (trained is None or live < trained):
        return "live"
    return "trained" if any(rel.startswith(p) for p in _TRAINED_DEFAULT_PREFIXES) else "live"


def _tool_scope(rel: str, line: str, match: re.Match[str]) -> str:
    """``"trained"`` or ``"live"`` for one tool-count claim."""
    return _tool_scope_span(rel, line, match.start(), match.end())


def _flag_expected(rel: str, line: str, match: re.Match[str], auth: Authorities) -> int | None:
    """Live or trained flag inventory, by the same nearest-marker rule as tools."""
    if _tool_scope(rel, line, match) == "trained":
        return auth.trained_flags
    if cmt._version_qualifies(line, match):
        return None
    return auth.live_flags


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
        ("flags", _FLAG_PATTERNS, lambda m: _flag_expected(rel, line, m, auth)),
        (
            "ci_jobs",
            _CI_JOB_PATTERNS,
            lambda m: None if cmt._version_qualifies(line, m) else auth.ci_jobs,
        ),
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
    findings.extend(_scan_python_support(rel, lineno, line, auth))
    findings.extend(_scan_retired_tests_spelling(rel, lineno, line, seen))
    return findings


def _scan_retired_tests_spelling(rel: str, lineno: int, line: str, seen: set[tuple[int, int]]) -> list[Finding]:
    """A suite-scale "N tests" claim, in any of the three shapes the gate used to renumber.

    Same qualifiers as the live leg -- a version-qualified record is history,
    and a number under the floor with no scope word is a module claim -- so
    this refuses exactly the sentences the old authority used to *rewrite*.
    """
    out: list[Finding] = []
    for pattern in _RETIRED_TESTS_PATTERNS:
        for match in pattern.finditer(line):
            span = (match.start("n"), match.end("plus"))
            if span in seen:
                continue
            if cmt._version_qualifies(line, match):
                continue
            if not (_parse_int(match.group("n")) >= _SUITE_SCALE_FLOOR or _SUITE_SCOPE.search(line)):
                continue
            seen.add(span)
            out.append(Finding(rel, lineno, "tests_spelling", match.group("n"), _RETIRED_TESTS_ACTUAL, match.group(0), span[0], span[1]))
    return out


def _scan_python_support(rel: str, lineno: int, line: str, auth: Authorities) -> list[Finding]:
    """ "Python 3.10+" against ``requires-python``; a range against the classifiers."""
    out: list[Finding] = []
    for match in _PY_FLOOR_PATTERN.finditer(line):
        if match.group("n") != auth.python_floor:
            out.append(
                Finding(
                    rel, lineno, "python_support", match.group("n"), auth.python_floor, match.group(0), match.start("n"), match.end("n")
                )
            )
    for match in _PY_RANGE_PATTERN.finditer(line):
        for group, expected in (("lo", auth.python_classifier_min), ("hi", auth.python_classifier_max)):
            if match.group(group) != expected:
                out.append(
                    Finding(
                        rel, lineno, "python_support", match.group(group), expected, match.group(0), match.start(group), match.end(group)
                    )
                )
    return out


def _ci_scopes(lines: list[str]) -> list[bool]:
    """For each line, whether it is talking about the CI matrix.

    Either the line itself says so, or the section heading above it does --
    ``docs/testing-guide.md`` writes the enumeration as a bare
    ``- Python: 3.10, ...`` bullet under ``## CI Matrix``, and a line-only
    rule would miss exactly the claim that was stale.
    """
    out: list[bool] = []
    heading_scoped = False
    for line in lines:
        heading = _HEADING_RE.match(line)
        if heading is not None:
            heading_scoped = _CI_SCOPE.search(heading.group(1)) is not None
            out.append(heading_scoped)
            continue
        out.append(heading_scoped or _CI_SCOPE.search(line) is not None)
    return out


def _render_version_list(versions: tuple[str, ...], template: str) -> str:
    """Re-render *versions* in the separator style the claim already used.

    A one-element authority (a matrix cut back to a single interpreter) takes
    no separator at all: the Oxford-comma branch would otherwise suggest
    ", and 3.12" as the replacement text.
    """
    if len(versions) == 1:
        return versions[0]
    if "/" in template:
        return "/".join(versions)
    if re.search(r",\s*and\s", template):
        return ", ".join(versions[:-1]) + ", and " + versions[-1]
    return ", ".join(versions)


def scan_ci_python_lists(rel: str, lines: list[str], auth: Authorities) -> list[Finding]:
    """Every enumeration of Python versions that disagrees with ci.yml's matrix.

    Compared as a SET: the claim is "these are the versions CI runs", and a
    doc that lists them in a different order is not wrong. A doc that omits
    3.11 -- as four of them did -- is.
    """
    expected = set(auth.ci_python_versions)
    scopes = _ci_scopes(lines)
    records = _record_scopes(lines)
    out: list[Finding] = []
    for idx, line in enumerate(lines):
        if not scopes[idx] or records[idx]:
            continue
        for match in _PYVER_LIST.finditer(line):
            claimed = re.findall(r"3\.\d{1,2}", match.group(0))
            if set(claimed) == expected:
                continue
            out.append(
                Finding(
                    rel,
                    idx + 1,
                    "ci_python",
                    match.group(0),
                    _render_version_list(auth.ci_python_versions, match.group(0)),
                    match.group(0),
                    match.start(),
                    match.end(),
                )
            )
    return out


def scan_table_tools(rel: str, lines: list[str], auth: Authorities, scopes: list[bool]) -> list[Finding]:
    """Tool counts stated in a table CELL, where the number follows its label.

    ``| MCP tools | 89 | N/A |`` states a claim no pattern in this module can
    see, because every one of them wants the count adjacent to the word. The
    cell matcher lives in ``count_mcp_tools`` and is shared, so the two gates
    cannot disagree about what a table claim is.
    """
    out: list[Finding] = []
    for lineno, start, end, value in cmt.table_tool_claims(lines):
        line = lines[lineno - 1]
        if scopes[lineno - 1] or cmt.version_qualifies_span(line, start, end):
            continue
        scope = _tool_scope_span(rel, line, start, end)
        expected = auth.trained_tools if scope == "trained" else auth.live_tools
        if value != expected:
            out.append(Finding(rel, lineno, "tools", str(value), str(expected), line.strip(), start, end))
    return out


def scan_text(rel: str, lines: list[str], auth: Authorities) -> list[Finding]:
    """Every stale claim in one document: per line, wrapped across a break, and in a table cell."""
    scopes = _record_scopes(lines)
    findings: list[Finding] = []
    for idx, line in enumerate(lines):
        findings.extend(scan_line(rel, idx + 1, line, auth, historical=scopes[idx]))
    # A claim the markdown reflow split in two ("...MIND-Mem's 89 MCP\ntools")
    # is invisible to every line-scoped pattern above. Rescan each prose line
    # glued to its successor, mapping each hit back to the real line that holds
    # the number, and drop anything the per-line pass already reported --
    # otherwise every claim inside a joined pair is counted twice.
    seen = {(f.lineno, f.start, f.end, f.kind) for f in findings}
    for lineno, joined, boundary in cmt.wrapped_line_pairs(lines):
        historical = scopes[lineno - 1] or scopes[lineno]
        for finding in scan_line(rel, lineno, joined, auth, historical=historical):
            real_lineno, start, end = cmt.locate_in_pair(finding.start, finding.end, lineno, boundary)
            key = (real_lineno, start, end, finding.kind)
            if key in seen:
                continue
            seen.add(key)
            findings.append(replace(finding, lineno=real_lineno, start=start, end=end))
    for finding in scan_table_tools(rel, lines, auth, scopes):
        if (finding.lineno, finding.start, finding.end, finding.kind) not in seen:
            findings.append(finding)
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
        findings.extend(scan_text(rel, lines, auth))
        findings.extend(scan_ci_python_lists(rel, lines, auth))
    findings.extend(scan_builder_default(auth, root))
    findings.extend(check_core_deps_badge(auth, root))
    findings.extend(check_backends_badge(auth, root))
    findings.extend(check_workflow_table(auth, root))
    findings.extend(check_ci_matrix_grid(auth, root))
    findings.extend(check_module_facts(auth, root))
    findings.extend(check_experimental_is_not_shipped(auth, root))
    findings.extend(check_eval_claims(auth, root))
    return findings


# ``docs/ci-workflows.md`` enumerates the workflow directory in prose. That
# table named a "Security Review" workflow that does not exist, gave Benchmark
# a push/PR trigger it had lost, and omitted two whole workflows -- a table
# cannot notice that about itself, so the directory checks it.
_WORKFLOW_DOC = "docs/ci-workflows.md"
_WORKFLOW_ROW = re.compile(r"^\|\s*(?P<name>[^|]+?)\s*\|\s*`(?P<file>[A-Za-z0-9_.-]+\.ya?ml)`\s*\|")


def check_workflow_table(auth: Authorities, root: Path | None = None) -> list[Finding]:
    """The workflow table must name every workflow file, and only real ones."""
    root = root or _project_root()
    path = root / _WORKFLOW_DOC
    if not path.is_file():
        return []
    truth = dict(auth.workflows)
    listed: dict[str, tuple[int, str]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, 1):
        row = _WORKFLOW_ROW.match(line)
        if row is not None:
            listed[row.group("file")] = (idx, row.group("name"))

    out: list[Finding] = []
    for filename, (lineno, name) in sorted(listed.items()):
        if filename not in truth:
            out.append(Finding(_WORKFLOW_DOC, lineno, "workflow", filename, "(no such workflow file)", filename, 0, 0))
        elif name != truth[filename]:
            out.append(Finding(_WORKFLOW_DOC, lineno, "workflow", name, truth[filename], lines[lineno - 1][:100], 0, 0))
    for filename in sorted(truth):
        if filename not in listed:
            out.append(
                Finding(_WORKFLOW_DOC, 0, "workflow", "(absent from the table)", f"{filename} = {truth[filename]!r}", filename, 0, 0)
            )
    return out


# ``docs/ci-workflows.md`` also draws the matrix as a GRID: the Python versions
# live in the header cells and the body rows carry only tick marks, so neither
# ``_PYVER_LIST`` (which needs a comma/slash-separated run) nor any count
# pattern can see it. Dropping a whole column from that grid left the gate
# green -- found by mutation, not by reading. The grid gets its own checker.
_CI_GRID_HEADER = re.compile(r"^\|\s*OS\s*\|(?P<cells>.+)\|\s*$")
_CI_GRID_VERSION_CELL = re.compile(r"^Python\s+(?P<v>\d+\.\d+)$")


def _table_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def check_ci_matrix_grid(auth: Authorities, root: Path | None = None) -> list[Finding]:
    """The OS × Python grid must be the matrix, and must be a full cross-product.

    Three separate things were wrong in the shipped grid and only the first is
    a number: it had no 3.11 column, and it showed macOS and Windows running
    two of the four versions it did list -- while ``ci.yml`` has always fanned
    every version out over every OS.
    """
    root = root or _project_root()
    path = root / _WORKFLOW_DOC
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[Finding] = []
    for idx, line in enumerate(lines):
        header = _CI_GRID_HEADER.match(line)
        if header is None:
            continue
        cells = _table_cells(header.group("cells"))
        versions = [m.group("v") for m in (_CI_GRID_VERSION_CELL.match(c) for c in cells) if m]
        if not versions:
            continue
        if set(versions) != set(auth.ci_python_versions):
            out.append(
                Finding(
                    _WORKFLOW_DOC,
                    idx + 1,
                    "ci_python",
                    ", ".join(versions),
                    ", ".join(auth.ci_python_versions),
                    line[:120],
                    0,
                    0,
                )
            )
        labels: dict[str, int] = {}
        for offset in range(idx + 2, len(lines)):  # +2 skips the |---| separator
            row = lines[offset]
            if not row.strip().startswith("|"):
                break
            cells = _table_cells(row.strip().strip("|"))
            if not cells or set(cells[0]) <= {"-", ":", " "}:
                continue
            labels[cells[0]] = offset + 1
            marks = [c for c in cells[1:] if c]
            if len(marks) != len(auth.ci_python_versions):
                out.append(
                    Finding(
                        _WORKFLOW_DOC,
                        offset + 1,
                        "ci_matrix",
                        f"{cells[0]}: {len(marks)} of {len(auth.ci_python_versions)} versions",
                        f"{cells[0]}: all {len(auth.ci_python_versions)} (the matrix is a full cross-product)",
                        row[:120],
                        0,
                        0,
                    )
                )
        claimed_os = {label.lower() for label in labels}
        actual_os = {name.split("-")[0].lower() for name in auth.ci_operating_systems}
        if claimed_os != actual_os:
            out.append(
                Finding(
                    _WORKFLOW_DOC,
                    idx + 1,
                    "ci_os",
                    ", ".join(sorted(claimed_os)) or "(no rows)",
                    ", ".join(sorted(actual_os)),
                    line[:120],
                    0,
                    0,
                )
            )
    return out


# "**Module:** `src/mind_mem/recompaction.py` (268 lines, 18 tests, 99%
# coverage)". The test count and the coverage were right; the line count was
# three stale. A number nobody recomputes drifts on the next edit, so the two
# with a cheap in-tree authority are recomputed here instead of trusted.
_MODULE_FACTS = re.compile(r"\*\*Module:\*\*\s*`(?P<path>[^`]+\.py)`\s*\((?P<lines>\d+)\s+lines,\s*(?P<tests>\d+)\s+tests")


def check_module_facts(auth: Authorities, root: Path | None = None) -> list[Finding]:
    """ "(N lines, M tests)" in a module doc header must match the module."""
    root = root or _project_root()
    out: list[Finding] = []
    for path in _doc_files(root):
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _MODULE_FACTS.finditer(line):
                module = match.group("path")
                if not (root / module).is_file():
                    out.append(Finding(rel, lineno, "module_facts", module, "(no such module)", match.group(0), 0, 0))
                    continue
                for group, actual in (
                    ("lines", module_line_count(module, root)),
                    ("tests", module_test_count(module, root)),
                ):
                    if int(match.group(group)) != actual:
                        out.append(
                            Finding(
                                rel,
                                lineno,
                                "module_facts",
                                match.group(group),
                                str(actual),
                                match.group(0),
                                match.start(group),
                                match.end(group),
                            )
                        )
    return out


# A CAPABILITY claim, not a count: `docs/status.md` filed model provenance under
# "Experimental (in-tree, behind feature flags) ... not yet shipped" while its
# three tools were registered unconditionally, counted in the 102 the README
# badge advertises, covered by 28 tests, and gated by their own workflow. The
# tool registry is the authority for what shipped, so a doc cannot call a
# registered surface unshipped.
_STATUS_DOC = "docs/status.md"
_EXPERIMENTAL_HEADING = re.compile(r"^(?P<hashes>\s{0,3}#{1,6})\s+(?P<title>.*)$")
_TOOL_MODULE_REF = re.compile(r"`(?P<path>src/mind_mem/mcp/tools/[A-Za-z0-9_]+\.py)`")


def check_experimental_is_not_shipped(auth: Authorities, root: Path | None = None) -> list[Finding]:
    """No component under an "Experimental"/"not yet shipped" heading may be live."""
    root = root or _project_root()
    path = root / _STATUS_DOC
    if not path.is_file():
        return []
    out: list[Finding] = []
    in_experimental = False
    level = 0
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        heading = _EXPERIMENTAL_HEADING.match(line)
        if heading is not None:
            depth = len(heading.group("hashes").strip())
            title = heading.group("title")
            if re.search(r"\b(experimental|planned|not yet shipped|roadmap)\b", title, re.IGNORECASE):
                in_experimental, level = True, depth
            elif in_experimental and depth <= level:
                in_experimental = False
            continue
        if not in_experimental:
            continue
        for match in _TOOL_MODULE_REF.finditer(line):
            module = root / match.group("path")
            if not module.is_file():
                continue
            names = sorted(cmt._tool_names(module))
            if names:
                out.append(
                    Finding(
                        _STATUS_DOC,
                        lineno,
                        "shipped_not_experimental",
                        match.group("path"),
                        f"registered and counted in the {auth.live_tools}-tool surface: {', '.join(names)}",
                        line[:120],
                        0,
                        0,
                    )
                )
    return out


# The 4b eval totals. Every per-category row in the model card already matched
# the harness; the TOTALS are what drifted -- ``docs/mind-mem-4b-setup.md``
# advertised 109/109, the score of the ``v4.0.0-base`` archive two revisions
# back, for the weights ``main`` points at today.
_EVAL_TOTAL_ROW = re.compile(r"\*\*Total (?P<which>main|holdout)\*\*\s*\|\s*\*\*(?P<n>\d+)\s*/\s*(?P<d>\d+)\*\*")
# Matched over the WHOLE file, not line by line: the 4b setup guide wraps
# "Eval score for the weights `main` currently points at (`v4.1.1`):" onto the
# line above its "**111/111**", and a per-line scan walked straight past the
# stale 109/109 -- found by mutating the doc back and watching the gate stay
# green, which is the only reason this is not still a per-line regex.
_EVAL_GRAND = re.compile(r"(?:Grand total|Eval score)[^:]{0,120}?:?\s*\**(?P<n>\d+)\s*/\s*(?P<d>\d+)")
_EVAL_HARNESS_PROBES = re.compile(r"Harness:[^\n]*?\*\*(?P<n>\d+) probes\*\*")
# "(`train/eval_holdout.py` -- 22 probes that do **not** appear verbatim...)".
# Anchored on the harness filename so it cannot drift onto a per-category
# "3 probes" elsewhere on the page.
_EVAL_HOLDOUT_PROBES = re.compile(r"eval_holdout\.py`?[^\n]{0,20}?(?P<n>\d+)\s+probes\b")


def check_eval_claims(auth: Authorities, root: Path | None = None) -> list[Finding]:
    """The published eval totals must equal the probe lists the harness benches."""
    root = root or _project_root()
    out: list[Finding] = []
    for rel in ("train/HF_MODEL_CARD_v4.md", "docs/mind-mem-4b-setup.md"):
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, expected_for in (
            (_EVAL_TOTAL_ROW, lambda m: auth.eval_main_probes if m.group("which") == "main" else auth.eval_holdout_probes),
            (_EVAL_GRAND, lambda m: auth.eval_total_probes),
            (_EVAL_HARNESS_PROBES, lambda m: auth.eval_main_probes),
            (_EVAL_HOLDOUT_PROBES, lambda m: auth.eval_holdout_probes),
        ):
            for match in pattern.finditer(text):
                expected = expected_for(match)
                for group in [g for g in ("n", "d") if g in match.groupdict()]:
                    if int(match.group(group)) == expected:
                        continue
                    line_start = text.rfind("\n", 0, match.start(group)) + 1
                    out.append(
                        Finding(
                            rel,
                            text.count("\n", 0, match.start(group)) + 1,
                            "eval_probes",
                            match.group(group),
                            str(expected),
                            match.group(0).replace("\n", " "),
                            match.start(group) - line_start,
                            match.end(group) - line_start,
                        )
                    )
    return out


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

# The shields.io path spells "|" as %7C: "backends-markdown_%7C_postgres-teal".
_BACKENDS_BADGE_RE = re.compile(r"backends-(?P<n>[A-Za-z0-9_%]+?)-[a-z]+\?")


def check_backends_badge(auth: Authorities, root: Path | None = None) -> list[Finding]:
    """The storage badge must list the backends ``--backend`` accepts."""
    root = root or _project_root()
    readme = root / "README.md"
    if not readme.is_file():
        return []
    truth = "_%7C_".join(auth.backends)
    out: list[Finding] = []
    for lineno, line in enumerate(readme.read_text(encoding="utf-8").splitlines(), 1):
        for match in _BACKENDS_BADGE_RE.finditer(line):
            claimed = {part.lower() for part in match.group("n").split("_%7C_")}
            if claimed != {name.lower() for name in auth.backends}:
                out.append(
                    Finding("README.md", lineno, "backends", match.group("n"), truth, match.group(0), match.start("n"), match.end("n"))
                )
    return out


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
    # Same scanner as the in-tree surfaces: the published card is where the
    # wrapped and table shapes hide too, and a hub-only matcher would drift.
    return scan_text(rel, text.splitlines(), auth)


# --------------------------------------------------------------------------
# Fix mode
# --------------------------------------------------------------------------

# Kinds whose finding carries a LINE-RELATIVE span, so the number can be
# rewritten in place. ``workflow``, ``ci_matrix`` and ``ci_os`` are structural
# -- a row is missing, or a grid is not a cross-product -- and there is no
# single number to substitute, so they are reported and never guessed at.
_FIXABLE = frozenset(
    {
        "tests",
        "clients",
        "mcp_clients",
        "resources",
        "mind_kernels",
        "tools",
        "flags",
        "ci_jobs",
        "ci_python",
        "eval_probes",
        "module_facts",
        "python_support",
        "backends",
    }
)


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
    args = parser.parse_args(argv)

    try:
        auth = resolve_authorities()
    except AuthorityError as exc:
        # An unreachable authority is NOT a pass. Say so, loudly, and exit
        # non-zero -- an empty finding list here would read as "aligned".
        print(f"::error::mind-mem docs-alignment authority unavailable: {exc}", file=sys.stderr)
        return 2

    print("Authorities:")
    print(f"  test functions (tests/**, static): {auth.tests}")
    print(f"  MCP tools (live)                 : {auth.live_tools}")
    print(f"  MCP tools (weights, {TRAINED_REVISION:<8})   : {auth.trained_tools}")
    print(f"  AI clients (AGENT_REGISTRY)      : {auth.clients}")
    print(f"  MCP-aware clients (mcp_fmt)      : {auth.mcp_clients}")
    print(f"  MCP resources                    : {auth.resources}")
    print(f"  MIND kernels (mind/*.mind)       : {auth.mind_kernels}")
    print(f"  package version                  : {auth.version}")
    print(f"  core dependencies                : {auth.core_deps}")
    print(f"  v4 feature flags (live)          : {auth.live_flags}")
    print(f"  v4 feature flags (weights)       : {auth.trained_flags}")
    print(f"  4b eval probes (main + holdout)  : {auth.eval_main_probes} + {auth.eval_holdout_probes} = {auth.eval_total_probes}")
    print(f"  CI Python versions               : {', '.join(auth.ci_python_versions)}")
    print(f"  CI operating systems             : {', '.join(auth.ci_operating_systems)}")
    print(f"  CI test jobs (cross-product)     : {auth.ci_jobs}")
    print(f"  GitHub workflows                 : {len(auth.workflows)}")
    print(f"  Python floor / classifier range  : {auth.python_floor}+ / {auth.python_classifier_min}-{auth.python_classifier_max}")
    print(f"  storage backends                 : {', '.join(auth.backends)}")
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
