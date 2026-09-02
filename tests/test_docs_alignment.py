# Copyright 2026 STARGA, Inc.
"""Regression tests for the doc-alignment gate (scripts/check_docs_alignment.py).

``count_mcp_tools.py`` proved one number can be held to an authority. These
tests cover the rest, and they exist in this shape because the FIRST cut of
this gate got two things wrong that only a positive control could show:

* a version mentioned anywhere in a heading or paragraph exempted everything
  under it, which silently excused two live "89-tool surface" claims;
* the model card and the 4b setup guide state BOTH the trained-on count and
  the live count, so a per-file rule cannot separate them.

Every negative assertion below ("this is not flagged") is paired with the
positive control that proves the same code path CAN flag a true positive.
"""

from __future__ import annotations

import dataclasses
import functools
import re
import subprocess  # nosec B404 - fixed argv, no shell
import sys
from pathlib import Path

import pytest

from scripts import alignment_authorities as aa
from scripts import check_docs_alignment as cda

ROOT = Path(__file__).resolve().parent.parent


def make_authorities(**overrides) -> cda.Authorities:
    """Authorities with fixed, obviously-distinct values.

    Every number differs from every other so a leg that reads the wrong
    authority produces a finding instead of an accidental pass.
    """
    base = dict(
        tests=9707,
        live_tools=102,
        trained_tools=83,
        clients=19,
        mcp_clients=11,
        resources=8,
        mind_kernels=26,
        version="5.0.1",
        core_deps=0,
        live_flags=52,
        trained_flags=38,
        eval_main_probes=111,
        eval_holdout_probes=22,
        ci_python_versions=("3.10", "3.11", "3.12", "3.13", "3.14"),
        ci_operating_systems=("ubuntu-latest", "macos-latest", "windows-latest"),
        workflows=(("ci.yml", "CI"), ("release.yml", "Release")),
        python_floor="3.10",
        python_classifier_min="3.10",
        python_classifier_max="3.14",
        backends=("markdown", "postgres", "encrypted"),
    )
    base.update(overrides)
    return cda.Authorities(**base)


def scan(line: str, *, rel: str = "docs/example.md", auth=None, historical: bool = False):
    return cda.scan_line(rel, 1, line, auth or make_authorities(), historical=historical)


@functools.lru_cache(maxsize=1)
def _real_authorities() -> cda.Authorities:
    """The tree's own authorities, resolved once (git archive is not free).

    The test-count leg comes from the README badge for the reason
    ``TestRepositoryIsAligned`` documents: CI checks the badge against the
    collector once, rather than 15 times inside the suite.
    """
    return cda.resolve_authorities(ROOT, tests_collected=readme_tests_badge())


def real_auth_kwargs() -> dict:
    """Every real authority as kwargs, so a test can move exactly one."""
    return dataclasses.asdict(_real_authorities())


def _lines(rel: str) -> list[str]:
    return (ROOT / rel).read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# The authorities themselves
# ---------------------------------------------------------------------------


class TestAuthorities:
    def test_parse_collected_reads_the_selector_tail(self):
        out = "tests/test_a.py::test_one\n\n9701/10034 tests collected (333 deselected) in 13.19s\n"
        assert cda.parse_collected(out) == 9701

    def test_parse_collected_reads_the_undeselected_form(self):
        assert cda.parse_collected("42 tests collected in 0.10s\n") == 42

    def test_a_collector_that_said_nothing_is_an_error_not_a_zero(self):
        """A verifier that died is not a verifier that passed.

        Returning 0 here would make every four-digit badge look stale and,
        worse, would make an empty finding list readable as "aligned".
        """
        with pytest.raises(cda.AuthorityError):
            cda.parse_collected("INTERNALERROR> boom\n")

    def test_trained_tool_count_is_measured_from_git_history(self):
        """83, and measured -- not the 96 the card asserted or the hub's 84."""
        trained = cda.trained_tool_count()
        assert trained == 83, "the v4.1.1 tree registers 83 DISTINCT tool names"
        assert trained != cda.live_tool_count(), "trained-on and live must be separate authorities"

    def test_an_absent_revision_fails_loud(self):
        with pytest.raises(cda.AuthorityError) as exc:
            cda.trained_tool_count(revision="v0.0.0-does-not-exist")
        assert "not in this checkout" in str(exc.value)

    def test_client_count_is_the_registry_length(self):
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from mind_mem.hook_installer import AGENT_REGISTRY
        finally:
            sys.path.remove(str(ROOT / "src"))
        assert cda.client_count() == len(AGENT_REGISTRY)

    def test_mcp_client_count_is_the_registry_entries_with_a_writer(self):
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from mind_mem.hook_installer import AGENT_REGISTRY
        finally:
            sys.path.remove(str(ROOT / "src"))
        expected = sum(1 for spec in AGENT_REGISTRY.values() if getattr(spec, "mcp_fmt", ""))
        assert cda.mcp_client_count() == expected
        assert cda.mcp_client_count() < cda.client_count(), "not every client speaks MCP"

    def test_mind_kernel_count_is_the_shipped_directory(self):
        assert cda.mind_kernel_count() == len(list((ROOT / "mind").glob("*.mind")))

    def test_resource_count_matches_the_registrations(self):
        source = (ROOT / "src" / "mind_mem" / "mcp" / "resources.py").read_text(encoding="utf-8")
        assert cda.resource_count() == source.count("mcp.resource(")

    def test_version_authority_is_the_package_version(self):
        init = (ROOT / "src" / "mind_mem" / "__init__.py").read_text(encoding="utf-8")
        assert f'__version__ = "{cda.package_version()}"' in init


# ---------------------------------------------------------------------------
# Positive controls: the scanner can see a true positive
# ---------------------------------------------------------------------------


class TestTestCountClaims:
    BADGE = '<img src="https://img.shields.io/badge/tests-{n}-brightgreen?style=flat-square" alt="Tests: {alt}">'

    def test_a_stale_badge_is_caught(self):
        findings = scan(self.BADGE.format(n="9%2C366", alt="9,366"))
        kinds = {(f.kind, f.claimed) for f in findings}
        assert ("tests", "9%2C366") in kinds, "the shields path must be gated"
        assert ("tests", "9,366") in kinds, "the alt text drifts independently and must be gated too"
        assert {f.actual for f in findings} == {"9%2C707", "9,707"}, "the fix must keep each spelling's grouping"

    def test_a_correct_badge_is_not_flagged(self):
        assert scan(self.BADGE.format(n="9%2C707", alt="9,707")) == []

    def test_a_suite_scale_prose_claim_is_caught(self):
        findings = scan("- **CI** on every push and PR - full pytest matrix (7,500+ tests across the suite).")
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("tests", "7,500", "9,707")]

    def test_the_trailing_plus_is_part_of_the_claim_it_replaces(self):
        """ "7,500+ tests" must become "9,707 tests", never "9,707+ tests"."""
        line = "full pytest matrix (7,500+ tests across the suite)."
        finding = scan(line)[0]
        assert line[finding.start : finding.end] == "7,500+"

    def test_a_module_scale_claim_is_not_gated(self):
        """ "18 tests" is about one module; the suite gate has no business there."""
        assert scan("Recompaction ships with 18 tests covering the merge path.") == []

    def test_but_a_small_number_the_sentence_calls_suite_wide_IS_gated(self):
        """The floor is a size heuristic, not an escape hatch -- scope words win."""
        findings = scan("The whole test suite is 900 tests today.")
        assert [(f.kind, f.claimed) for f in findings] == [("tests", "900")]


class TestToolCountScoping:
    def test_a_stale_live_claim_is_caught(self):
        findings = scan("MIND-Mem exposes 89 MCP tools over stdio.")
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("tools", "89", "102")]

    def test_the_hyphenated_form_is_caught(self):
        """ "89-tool surface" -- the spelling count_mcp_tools cannot see, and the
        spelling two live docs were stale in while that gate reported green."""
        findings = scan("the client gets the full 89-tool surface.")
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("tools", "89", "102")]

    def test_a_trained_on_claim_is_measured_against_the_trained_revision(self):
        findings = scan("These weights were trained against a 96-tool surface.", rel="train/HF_MODEL_CARD_v4.md")
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("tools", "96", "83")]

    def test_a_live_claim_inside_the_model_card_is_still_live(self):
        card = "train/HF_MODEL_CARD_v4.md"
        assert scan("The live server now exposes 102 tools.", rel=card) == []
        findings = scan("The live server now exposes 83 tools.", rel=card)
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("tools", "83", "102")]

    def test_a_swapped_pair_is_caught_both_ways(self):
        """The failure that produced 84-vs-96: the two numbers traded places.

        Neither direction may pass, or the gate would accept a card claiming
        the weights know the live surface.
        """
        card = "train/HF_MODEL_CARD_v4.md"
        assert scan("v4 knows all 83 MCP tools from v3.x", rel=card) == []
        swapped = scan("v4 knows all 102 MCP tools from v3.x", rel=card)
        assert [(f.claimed, f.actual) for f in swapped] == [("102", "83")]

    def test_the_model_card_defaults_to_trained_with_no_marker_at_all(self):
        """No "live"/"trained" word on the line -- the file's default decides."""
        card = "train/HF_MODEL_CARD_v4.md"
        assert [f.actual for f in scan("The surface below is 96 MCP tools.", rel=card)] == ["83"]
        assert scan("The surface below is 96 MCP tools.") == [] or [f.actual for f in scan("The surface below is 96 MCP tools.")] == [
            "102"
        ], "outside the card the same sentence is a live claim"

    def test_a_one_digit_count_near_the_word_tool_is_too_noisy_to_gate(self):
        assert scan("the existing builder sees 1 tool, not 81.") == []


class TestRecordScopes:
    """A release record must keep its own numbers, and only a release record."""

    RELEASE_SECTION = [
        "## v3.1.4 (Released 2026-04-18)",
        "",
        "`mm install-all` now wires 17 clients. Fixes the Windows path round-trip.",
    ]
    LIVE_SECTION = [
        "## 5. All clients at once (v3.1.0 recommended)",
        "",
        "This installs the native MCP server entry (full 89-tool surface).",
    ]

    def _scan_block(self, lines):
        auth = make_authorities()
        scopes = cda._record_scopes(lines)
        out = []
        for idx, line in enumerate(lines):
            out.extend(cda.scan_line("docs/roadmap.md", idx + 1, line, auth, historical=scopes[idx]))
        return out

    def test_a_release_heading_protects_its_own_counts(self):
        assert self._scan_block(self.RELEASE_SECTION) == []

    def test_a_heading_that_merely_mentions_a_version_does_not(self):
        """The exact regression the first cut of this rule shipped."""
        findings = self._scan_block(self.LIVE_SECTION)
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("tools", "89", "102")]

    def test_a_paragraph_opening_with_a_version_is_a_record(self):
        block = [
            "**v4.1.1** (released 2026-06-14) - Postgres backend parity.",
            "SQLite/markdown default unchanged (5566 tests, no regression).",
        ]
        assert self._scan_block(block) == []

    def test_a_paragraph_merely_containing_a_version_is_not(self):
        block = [
            "**Since v3.1.0, `mm install-all` writes TWO things per client:**",
            "2. A native MCP server entry so the client gets the full 89-tool surface.",
        ]
        assert [(f.kind, f.claimed) for f in self._scan_block(block)] == [("tools", "89")]

    def test_a_past_tense_attribution_line_is_a_record(self):
        assert scan("Builds on **v3.9.0** - **81 tools**, 4000+ tests, native MCP for 17 AI clients,") == []

    def test_a_forward_scoped_since_line_is_not(self):
        """The real sentence, from docs/client-integrations.md.

        ``cmt._version_qualifies`` still excuses a version within 30 characters
        of the claim -- that inherited rule is unchanged here -- so the case
        that matters is the one where the version introduces the sentence and
        the claim lands well past it.
        """
        line = "Since v3.1.0, `mm install-all` writes a native MCP server entry so the client gets the full 89-tool surface."
        assert [(f.kind, f.claimed) for f in scan(line)] == [("tools", "89")]

    def test_a_transition_line_is_a_record_of_a_fix(self):
        assert scan("CLAUDE.md drift cleared (`MCP Tools (81) -> (102)`).") == []


class TestOtherClaims:
    def test_a_stale_client_badge_is_caught(self):
        findings = scan('<img src="https://img.shields.io/badge/clients-16-blueviolet" alt="AI Clients: 16">')
        assert {(f.kind, f.claimed, f.actual) for f in findings} == {("clients", "16", "19")}

    def test_the_extra_modifier_spelling_is_caught(self):
        """ "16 AI coding clients" -- one modifier past count_mcp_tools' shape."""
        findings = scan("MIND-Mem works with **16 AI coding clients** out of the box.")
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("clients", "16", "19")]

    def test_a_stale_mcp_aware_client_claim_is_caught(self):
        findings = scan("Supports 8 MCP-aware clients: `codex` (TOML), `zed` (JSON).")
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("mcp_clients", "8", "11")]

    def test_a_stale_resource_claim_is_caught(self):
        findings = scan("The server also publishes 6 resources.")
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("resources", "6", "8")]

    def test_a_stale_mind_kernel_claim_is_caught(self):
        findings = scan("Ships **16 MIND kernels** with an FFI bridge.")
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("mind_kernels", "16", "26")]

    def test_a_version_qualified_kernel_record_keeps_its_number(self):
        assert scan("- **16 MIND kernels at v1.0.3** (26 ship today) -- Native C99 kernels") == []

    def test_a_product_name_is_not_a_kernel_count(self):
        """docs/governance.md: "MIND-Mem is one consumer of the 512 kernel".

        512 is the product, not a count -- which is why the pattern requires
        the word MIND between the number and "kernel".
        """
        assert scan("MIND-Mem is one consumer of the 512 kernel; the same kernel runs in:") == []

    def test_a_stale_release_line_is_caught(self):
        findings = scan("Current release: **v4.9.9** - see CHANGELOG")
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("version", "4.9.9", "5.0.1")]

    def test_a_release_line_is_checked_even_inside_a_record_scope(self):
        """ "Current release" is present tense by construction.

        The paragraph it lives in opens with the version it names, so the
        record-scope rule would otherwise excuse the one claim that must never
        be excused.
        """
        findings = scan("Current release: **v4.9.9** - a restoration release.", historical=True)
        assert [(f.kind, f.claimed) for f in findings] == [("version", "4.9.9")]

    def test_the_core_deps_badge_must_agree_with_pyproject(self, tmp_path):
        (tmp_path / "README.md").write_text(
            '<img src="https://img.shields.io/badge/core_deps-zero-brightgreen" alt="Zero Core Dependencies">\n',
            encoding="utf-8",
        )
        assert cda.check_core_deps_badge(make_authorities(core_deps=0), tmp_path) == []
        stale = cda.check_core_deps_badge(make_authorities(core_deps=3), tmp_path)
        assert [(f.kind, f.claimed, f.actual) for f in stale] == [("core_deps", "zero", "3")]


class TestHistoricalFiles:
    def test_a_published_card_is_never_excused_by_its_filename(self):
        """`_v4` in the name must not exempt a file that is uploaded to the hub."""
        assert not cda._is_historical_file("train/HF_MODEL_CARD_v4.md")

    def test_an_underscore_versioned_plan_is_a_record(self):
        """`train/RETRAIN_v3.9.0.md` -- underscore is a separator too."""
        assert cda._is_historical_file("train/RETRAIN_v3.9.0.md")

    def test_a_live_doc_is_not_a_record(self):
        assert not cda._is_historical_file("docs/governance.md")


class TestFixMode:
    def test_fix_rewrites_by_span_rightmost_first(self, tmp_path):
        doc = tmp_path / "docs" / "x.md"
        doc.parent.mkdir(parents=True)
        doc.write_text(
            '<img src="https://img.shields.io/badge/tests-9%2C366-x" alt="Tests: 9,366">\n',
            encoding="utf-8",
        )
        auth = make_authorities()
        findings = [f for f in cda.scan_line("docs/x.md", 1, doc.read_text(encoding="utf-8").rstrip("\n"), auth)]
        assert len(findings) == 2, "both the path and the alt text must be found"
        fixed, skipped = cda.apply_fixes(findings, tmp_path)
        assert (fixed, skipped) == (2, [])
        assert doc.read_text(encoding="utf-8") == '<img src="https://img.shields.io/badge/tests-9%2C707-x" alt="Tests: 9,707">\n'

    def test_fix_refuses_a_finding_whose_line_moved(self, tmp_path):
        doc = tmp_path / "docs" / "x.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("the suite has 900 tests\n", encoding="utf-8")
        stale = cda.Finding("docs/x.md", 1, "tests", "8888", "9707", "8888 tests", 0, 4)
        fixed, skipped = cda.apply_fixes([stale], tmp_path)
        assert fixed == 0 and skipped == [stale]
        assert doc.read_text(encoding="utf-8") == "the suite has 900 tests\n"


# ---------------------------------------------------------------------------
# The repository itself
# ---------------------------------------------------------------------------


_BADGE_TESTS_RE = re.compile(r"badge/tests-([\d%C2c]+)-")


def readme_tests_badge() -> int:
    m = _BADGE_TESTS_RE.search((ROOT / "README.md").read_text(encoding="utf-8"))
    assert m is not None, "README must carry a tests badge"
    return cda._parse_int(m.group(1))


class TestRepositoryIsAligned:
    """Every gated claim in the tree agrees with its authority.

    The test-count leg is injected from the README badge rather than collected
    here: running ``pytest --collect-only`` inside the suite would add one full
    collection per matrix cell (15 of them) for a check the ``version-check``
    CI job already performs once against the live selector. So this asserts
    that every test-count claim in the tree agrees with the BADGE, and CI
    asserts the badge agrees with the collector.
    """

    def test_no_claim_disagrees_with_its_authority(self):
        auth = cda.resolve_authorities(tests_collected=readme_tests_badge())
        findings = cda.scan_docs(auth, ROOT)
        assert findings == [], "stale claims:\n" + "\n".join(str(f) for f in findings)

    def test_the_scan_actually_read_something(self):
        """An empty finding list is only evidence when the search happened."""
        files = cda._doc_files(ROOT)
        rels = {p.relative_to(ROOT).as_posix() for p in files}
        assert len(files) > 40, f"only {len(files)} surfaces scanned"
        for required in ("README.md", "docs/governance.md", "train/HF_MODEL_CARD_v4.md"):
            assert required in rels, f"{required} must be scanned"

    def test_the_cli_exits_zero_on_an_aligned_tree(self):
        proc = subprocess.run(  # nosec B603
            [sys.executable, "scripts/check_docs_alignment.py", "--tests-collected", str(readme_tests_badge())],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_an_unreachable_authority_exits_two_not_zero(self, monkeypatch):
        """A dead authority must not read as a clean bill of health."""

        def boom(*_a, **_k):
            raise cda.AuthorityError("git is gone")

        monkeypatch.setattr(cda, "trained_tool_count", boom)
        assert cda.main(["--tests-collected", "1"]) == 2


class TestPublishedModelCard:
    """The hub copy is the surface no CI checkout contains.

    It carried **84** while the tree carried 96 and the truth was 83; neither
    file could see the other, so neither could gate the other. These tests run
    OFFLINE -- the fetch is stubbed -- because a unit test must not depend on
    huggingface.co being up; the live fetch is the release preflight's job
    (``--check-live``).
    """

    CARD = "\n".join(
        [
            "# mind-mem-4b v4.1.1",
            "",
            "v4 knows all 84 MCP tools from v3.x, plus the following v4 surfaces:",
            "",
        ]
    )

    def _stub_fetch(self, monkeypatch, body: str):
        import urllib.request

        class _Response:
            def __init__(self, payload: bytes):
                self._payload = payload

            def read(self):
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response(body.encode("utf-8")))

    def test_the_hub_cards_stale_count_is_caught(self, monkeypatch):
        self._stub_fetch(monkeypatch, self.CARD)
        findings = cda.check_live_hf_card(make_authorities())
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("tools", "84", "83")]
        assert findings[0].surface.startswith("huggingface.co/"), "the finding must name the published surface"

    def test_a_corrected_hub_card_passes(self, monkeypatch):
        self._stub_fetch(monkeypatch, self.CARD.replace("84 MCP tools", "83 MCP tools"))
        assert cda.check_live_hf_card(make_authorities()) == []

    def test_an_empty_body_is_an_error_not_a_clean_card(self, monkeypatch):
        """An empty fetch must never read as "no stale claims"."""
        self._stub_fetch(monkeypatch, "   \n")
        with pytest.raises(cda.AuthorityError):
            cda.check_live_hf_card(make_authorities())

    def test_a_network_failure_is_an_error_not_a_pass(self, monkeypatch):
        import urllib.error
        import urllib.request

        def boom(*_a, **_k):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        with pytest.raises(cda.AuthorityError):
            cda.check_live_hf_card(make_authorities())

    def test_only_the_hub_is_fetched(self):
        with pytest.raises(cda.AuthorityError) as exc:
            cda.check_live_hf_card(make_authorities(), url="http://example.invalid/README.md")
        assert "refusing to fetch" in str(exc.value)


class TestMutationTwin:
    """A gate that cannot fail is not a gate.

    Each case disables one leg and asserts the protective assertion above now
    fails. Nothing here weakens the shipped gate; the monkeypatches are local
    to the test.
    """

    def test_disabling_the_scanner_makes_the_repo_assertion_vacuous(self, monkeypatch):
        monkeypatch.setattr(cda, "scan_line", lambda *a, **k: [])
        monkeypatch.setattr(cda, "scan_builder_default", lambda *a, **k: [])
        monkeypatch.setattr(cda, "check_core_deps_badge", lambda *a, **k: [])
        auth = cda.resolve_authorities(tests_collected=readme_tests_badge())
        # With the scanner neutered a KNOWN-BAD tree would still report clean,
        # which is exactly what the real assertion must be able to detect.
        assert cda.scan_docs(auth, ROOT) == []
        assert cda.scan_line("docs/x.md", 1, "the suite has 900 tests", auth) == []

    def test_the_trained_authority_is_load_bearing(self):
        """Point the trained leg at the live count and the card goes red."""
        card = "train/HF_MODEL_CARD_v4.md"
        good = make_authorities()
        assert scan("v4 knows all 83 MCP tools from v3.x", rel=card, auth=good) == []
        mutated = make_authorities(trained_tools=102)
        assert scan("v4 knows all 83 MCP tools from v3.x", rel=card, auth=mutated) != []

    def test_the_record_scope_is_load_bearing(self):
        """Force ``historical`` off and the release record goes red."""
        line = "`mm install-all` now wires 17 clients."
        assert scan(line, historical=True) == []
        assert [(f.kind, f.claimed) for f in scan(line, historical=False)] == [("clients", "17")]


# ---------------------------------------------------------------------------
# CI shape: the workflow directory is the authority for what CI runs
# ---------------------------------------------------------------------------

_CI_STUB = """name: CI

on:
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.11"]

  docs:
    runs-on: ubuntu-latest
"""


def write_ci(tmp_path: Path, body: str) -> Path:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(body, encoding="utf-8")
    return tmp_path


class TestCIMatrixAuthority:
    def test_the_real_matrix_is_a_full_cross_product(self):
        matrix = aa.ci_matrix(ROOT)
        assert matrix.job_count == len(matrix.python_versions) * len(matrix.operating_systems)
        assert "3.11" in matrix.python_versions, "3.11 has been a matrix row since it was added"
        assert set(matrix.operating_systems) == {"ubuntu-latest", "macos-latest", "windows-latest"}

    def test_only_the_named_jobs_matrix_is_read(self, tmp_path):
        """A version PINNED by another job must not be read as a matrix row.

        ``ci.yml`` holds several jobs that pin 3.12; a whole-file scan would
        report a Python list the matrix never ran.
        """
        root = write_ci(tmp_path, _CI_STUB)
        matrix = aa.ci_matrix(root)
        assert matrix.python_versions == ("3.10", "3.11")
        assert matrix.operating_systems == ("ubuntu-latest", "macos-latest")
        assert matrix.job_count == 4

    def test_a_missing_job_fails_loud(self, tmp_path):
        root = write_ci(tmp_path, _CI_STUB)
        with pytest.raises(aa.AuthorityError) as exc:
            aa.ci_matrix(root, job="nope")
        assert "no job named" in str(exc.value)

    def test_a_shape_the_parser_cannot_read_is_an_error_not_a_default(self, tmp_path):
        """A block-style matrix must stop the gate, not silently return nothing."""
        body = _CI_STUB.replace(
            "        os: [ubuntu-latest, macos-latest]\n",
            "        os:\n          - ubuntu-latest\n          - macos-latest\n",
        )
        root = write_ci(tmp_path, body)
        with pytest.raises(aa.AuthorityError) as exc:
            aa.ci_matrix(root)
        assert "matrix list" in str(exc.value)


class TestWorkflowInventoryAuthority:
    def test_every_shipped_workflow_is_named(self):
        inventory = aa.workflow_inventory(ROOT)
        assert inventory["ci.yml"] == "CI"
        assert len(inventory) == len(list((ROOT / ".github" / "workflows").glob("*.yml")))

    def test_a_workflow_without_a_name_is_an_error(self, tmp_path):
        root = write_ci(tmp_path, "on:\n  push:\njobs:\n  x:\n    runs-on: ubuntu-latest\n")
        with pytest.raises(aa.AuthorityError) as exc:
            aa.workflow_inventory(root)
        assert "no top-level 'name:'" in str(exc.value)

    def test_an_empty_directory_is_an_error_not_an_empty_inventory(self, tmp_path):
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        with pytest.raises(aa.AuthorityError):
            aa.workflow_inventory(tmp_path)


class TestFlagCountAuthority:
    def test_live_and_trained_counts_are_both_measured_and_differ(self):
        live = aa.live_flag_count(ROOT)
        trained = aa.trained_flag_count(root=ROOT)
        assert live > 0 and trained > 0
        assert live != trained, "if these ever coincide, pick a claim the card can still get wrong"
        assert 35 not in (live, trained), "35 was the asserted number this gate exists to catch"

    def test_a_sequence_that_is_not_a_literal_is_an_error(self):
        with pytest.raises(aa.AuthorityError) as exc:
            aa._count_flag_literal("ALL_V4_FLAGS = tuple(_load())", "<stub>")
        assert "not a literal sequence" in str(exc.value)

    def test_an_absent_assignment_is_an_error(self):
        with pytest.raises(aa.AuthorityError):
            aa._count_flag_literal("X = (1, 2)", "<stub>")


class TestFlagClaims:
    def test_a_stale_live_flag_claim_is_caught(self):
        found = scan("feature_flags.py — 35 flags + is_enabled/require_enabled")
        assert [(f.kind, f.claimed, f.actual) for f in found] == [("flags", "35", "52")]

    def test_a_correct_live_flag_claim_is_not_flagged(self):
        assert scan("feature_flags.py — 52 flags + is_enabled/require_enabled") == []

    def test_the_hyphenated_inventory_form_is_caught(self):
        found = scan("`FeatureDisabledError`, 35-flag inventory, startup rejection", rel="train/HF_MODEL_CARD_v4.md")
        assert [(f.kind, f.claimed, f.actual) for f in found] == [("flags", "35", "38")]

    def test_a_trained_marker_selects_the_trained_authority(self):
        line = "the 38 v4 feature flags the trained revision declared"
        assert scan(line, rel="docs/mind-mem-4b-setup.md") == []
        # Positive control: the LIVE number on the same trained-scoped line is
        # a finding, so the scoping is doing work rather than passing both.
        found = scan("the 52 v4 feature flags the trained revision declared", rel="docs/mind-mem-4b-setup.md")
        assert [(f.kind, f.claimed, f.actual) for f in found] == [("flags", "52", "38")]


class TestCIJobClaims:
    def test_a_stale_job_count_is_caught(self):
        found = scan("- Total: 12 CI jobs")
        assert [(f.kind, f.claimed, f.actual) for f in found] == [("ci_jobs", "12", "15")]

    def test_the_osxpython_rows_spelling_is_caught(self):
        found = scan("CI matrix fully green across 12 OS\u00d7Python-version rows.")
        assert [(f.kind, f.claimed, f.actual) for f in found] == [("ci_jobs", "12", "15")]

    def test_the_true_job_count_passes(self):
        assert scan("- Total: 15 CI jobs (a full cross-product)") == []

    def test_a_release_record_keeps_its_own_job_count(self):
        line = "CI matrix fully green across 12 OS\u00d7Python-version rows."
        assert scan(line, historical=True) == []
        assert scan(line, historical=False) != []


class TestCIPythonListClaims:
    def scan_lines(self, lines, rel="docs/example.md", auth=None):
        return cda.scan_ci_python_lists(rel, lines, auth or make_authorities())

    def test_a_list_missing_a_version_is_caught(self):
        found = self.scan_lines(["Python 3.10, 3.12, 3.13, and 3.14 are tested in CI."])
        assert [(f.kind, f.claimed, f.actual) for f in found] == [
            ("ci_python", "3.10, 3.12, 3.13, and 3.14", "3.10, 3.11, 3.12, 3.13, and 3.14")
        ]

    def test_the_slash_style_is_re_rendered_in_its_own_style(self):
        found = self.scan_lines(["- **CI**: Runs tests on Python 3.10/3.12/3.13/3.14 across Ubuntu"])
        assert [f.actual for f in found] == ["3.10/3.11/3.12/3.13/3.14"]

    def test_order_is_not_the_claim(self):
        assert self.scan_lines(["tested in CI: 3.14, 3.13, 3.12, 3.11, 3.10"]) == []

    def test_a_bare_bullet_under_a_ci_heading_is_still_gated(self):
        lines = ["## CI Matrix", "", "- Python: 3.10, 3.12, 3.13, 3.14"]
        found = self.scan_lines(lines)
        assert [(f.lineno, f.claimed) for f in found] == [(3, "3.10, 3.12, 3.13, 3.14")]

    def test_an_enumeration_with_no_ci_scope_is_left_alone(self):
        """A release note about two rows is not a claim about the matrix."""
        lines = ["## Overview", "", "closes the OOM kills on ubuntu 3.12/3.14 from stress tests"]
        assert self.scan_lines(lines) == []
        # Positive control: the same enumeration under a CI heading IS gated,
        # so the guard is scoping rather than disabling the check.
        assert self.scan_lines(["## CI Matrix", "", "closes the OOM kills on ubuntu 3.12/3.14"]) != []

    def test_a_single_version_is_not_an_enumeration(self):
        assert self.scan_lines(["Core requires only Python 3.10+ stdlib; CI covers more."]) == []

    def test_fix_rewrites_the_whole_enumeration_in_place(self, tmp_path):
        """The finding's span covers the whole list, so --fix can replace it."""
        (tmp_path / "docs").mkdir(parents=True)
        doc = tmp_path / "docs" / "faq.md"
        doc.write_text("Python 3.10, 3.12, 3.13, and 3.14 are tested in CI.\n", encoding="utf-8")
        findings = cda.scan_ci_python_lists("docs/faq.md", doc.read_text(encoding="utf-8").splitlines(), make_authorities())
        fixed, skipped = cda.apply_fixes(findings, tmp_path)
        assert (fixed, skipped) == (1, [])
        assert doc.read_text(encoding="utf-8") == "Python 3.10, 3.11, 3.12, 3.13, and 3.14 are tested in CI.\n"

    def test_a_one_interpreter_matrix_suggests_no_separator(self):
        """A matrix cut to one row must not suggest ", and 3.12" as the fix."""
        auth = make_authorities(ci_python_versions=("3.12",))
        found = self.scan_lines(["tested in CI: 3.10, 3.11, and 3.12"], auth=auth)
        assert [f.actual for f in found] == ["3.12"]


class TestWorkflowTableCheck:
    def make_doc(self, tmp_path: Path, rows: str) -> Path:
        doc = tmp_path / "docs"
        doc.mkdir(parents=True)
        (doc / "ci-workflows.md").write_text(
            "# CI Workflows\n\n| Workflow | File | Trigger | Description |\n|---|---|---|---|\n" + rows,
            encoding="utf-8",
        )
        return tmp_path

    def test_a_workflow_absent_from_the_table_is_caught(self, tmp_path):
        root = self.make_doc(tmp_path, "| CI | `ci.yml` | push | tests |\n")
        found = cda.check_workflow_table(make_authorities(), root)
        assert [(f.kind, f.actual) for f in found] == [("workflow", "release.yml = 'Release'")]

    def test_a_row_for_a_file_that_does_not_exist_is_caught(self, tmp_path):
        rows = (
            "| CI | `ci.yml` | push | tests |\n"
            "| Release | `release.yml` | tags | ship |\n"
            "| Security Review | `security-review.yml` | push | scan |\n"
        )
        root = self.make_doc(tmp_path, rows)
        found = cda.check_workflow_table(make_authorities(), root)
        assert [(f.claimed, f.actual) for f in found] == [("security-review.yml", "(no such workflow file)")]

    def test_a_renamed_workflow_is_caught(self, tmp_path):
        rows = "| Continuous Integration | `ci.yml` | push | tests |\n| Release | `release.yml` | tags | ship |\n"
        root = self.make_doc(tmp_path, rows)
        found = cda.check_workflow_table(make_authorities(), root)
        assert [(f.claimed, f.actual) for f in found] == [("Continuous Integration", "CI")]

    def test_an_accurate_table_is_clean(self, tmp_path):
        rows = "| CI | `ci.yml` | push | tests |\n| Release | `release.yml` | tags | ship |\n"
        root = self.make_doc(tmp_path, rows)
        assert cda.check_workflow_table(make_authorities(), root) == []

    def test_the_shipped_table_agrees_with_the_shipped_directory(self):
        auth = cda.Authorities(**{**real_auth_kwargs(), "workflows": tuple(sorted(aa.workflow_inventory(ROOT).items()))})
        assert cda.check_workflow_table(auth, ROOT) == []


class TestEvalProbeAuthority:
    def test_the_probe_totals_are_derived_from_the_harness(self):
        main, holdout = aa.eval_probe_counts(ROOT)
        assert (main, holdout) == (111, 22)

    def test_a_harness_with_nothing_benched_is_an_error(self, tmp_path):
        (tmp_path / "PROBES.py").write_text("QUESTIONS = [1, 2]\n", encoding="utf-8")
        with pytest.raises(aa.AuthorityError) as exc:
            aa._probe_total(tmp_path / "PROBES.py")
        assert "counting rule broke" in str(exc.value)


class TestEvalClaims:
    def write_card(self, tmp_path: Path, body: str) -> Path:
        (tmp_path / "train").mkdir(parents=True)
        (tmp_path / "train" / "HF_MODEL_CARD_v4.md").write_text(body, encoding="utf-8")
        return tmp_path

    def test_a_stale_main_total_is_caught(self, tmp_path):
        root = self.write_card(tmp_path, "| **Total main** | **109 / 109** | 100% |\n")
        found = cda.check_eval_claims(make_authorities(), root)
        assert [(f.kind, f.claimed, f.actual) for f in found] == [
            ("eval_probes", "109", "111"),
            ("eval_probes", "109", "111"),
        ]

    def test_a_stale_grand_total_is_caught(self, tmp_path):
        root = self.write_card(tmp_path, "**Grand total: 131 / 131 = 100%**\n")
        assert [f.actual for f in cda.check_eval_claims(make_authorities(), root)] == ["133", "133"]

    def test_the_harness_probe_line_is_gated(self, tmp_path):
        root = self.write_card(tmp_path, "Harness: `train/eval_harness.py` — **109 probes** (95 v3.x + 14)\n")
        assert [(f.claimed, f.actual) for f in cda.check_eval_claims(make_authorities(), root)] == [("109", "111")]

    def test_the_holdout_probe_sentence_is_gated(self, tmp_path):
        root = self.write_card(tmp_path, "Held-out paraphrase eval (`train/eval_holdout.py` — 19 probes that\n")
        assert [(f.claimed, f.actual) for f in cda.check_eval_claims(make_authorities(), root)] == [("19", "22")]

    def test_a_per_category_probe_count_elsewhere_is_not_gated(self, tmp_path):
        """ "3 probes, 95 % threshold" is about one category, not the holdout set."""
        root = self.write_card(tmp_path, "Transform-hash has 3 probes at a 95% threshold.\n")
        assert cda.check_eval_claims(make_authorities(), root) == []

    def test_the_true_totals_pass(self, tmp_path):
        body = (
            "Harness: `train/eval_harness.py` — **111 probes**\n"
            "| **Total main** | **111 / 111** | 100% |\n"
            "| **Total holdout** | **22 / 22** | 100% |\n"
            "**Grand total: 133 / 133 = 100%**\n"
        )
        root = self.write_card(tmp_path, body)
        assert cda.check_eval_claims(make_authorities(), root) == []

    def test_the_shipped_card_agrees_with_the_shipped_harness(self):
        main, holdout = aa.eval_probe_counts(ROOT)
        auth = cda.Authorities(**{**real_auth_kwargs(), "eval_main_probes": main, "eval_holdout_probes": holdout})
        assert cda.check_eval_claims(auth, ROOT) == []


class TestNewLegsAreLoadBearing:
    """Mutation twins for the legs added alongside the CI/flag/eval authorities.

    Each one moves the AUTHORITY, not the doc, and asserts the tree goes red.
    A leg whose authority can be wrong without anything failing is decoration.
    """

    def test_the_ci_matrix_authority_is_load_bearing(self):
        good = cda.Authorities(**real_auth_kwargs())
        assert cda.scan_ci_python_lists("docs/faq.md", _lines("docs/faq.md"), good) == []
        mutated = cda.Authorities(**{**real_auth_kwargs(), "ci_python_versions": ("3.10", "3.12", "3.13", "3.14")})
        assert cda.scan_ci_python_lists("docs/faq.md", _lines("docs/faq.md"), mutated) != []

    def test_the_job_count_is_derived_not_declared(self):
        auth = cda.Authorities(**{**real_auth_kwargs(), "ci_operating_systems": ("ubuntu-latest",)})
        assert auth.ci_jobs == 5
        assert scan("- Total: 15 CI jobs", auth=auth) != []

    def test_the_trained_flag_authority_is_load_bearing(self):
        card = "train/HF_MODEL_CARD_v4.md"
        line = "`FeatureDisabledError`, 38-flag inventory, startup rejection"
        assert scan(line, rel=card) == []
        assert scan(line, rel=card, auth=make_authorities(trained_flags=52)) != []

    def test_the_workflow_inventory_is_load_bearing(self):
        mutated = cda.Authorities(
            **{**real_auth_kwargs(), "workflows": tuple(sorted(aa.workflow_inventory(ROOT).items())) + (("ghost.yml", "Ghost"),)}
        )
        found = cda.check_workflow_table(mutated, ROOT)
        assert [f.actual for f in found] == ["ghost.yml = 'Ghost'"]

    def test_the_eval_authority_is_load_bearing(self):
        main, holdout = aa.eval_probe_counts(ROOT)
        mutated = cda.Authorities(**{**real_auth_kwargs(), "eval_main_probes": main + 1, "eval_holdout_probes": holdout})
        assert cda.check_eval_claims(mutated, ROOT) != []


class TestCIMatrixGrid:
    """The grid in ``docs/ci-workflows.md`` -- the shape no count pattern sees.

    Every case here was found by MUTATION: dropping a whole column from that
    grid left the gate green, because the versions live in header cells and the
    body rows carry only tick marks.
    """

    GOOD = (
        "| OS | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 | Python 3.14 |\n"
        "|----|:--:|:--:|:--:|:--:|:--:|\n"
        "| Ubuntu | x | x | x | x | x |\n"
        "| macOS | x | x | x | x | x |\n"
        "| Windows | x | x | x | x | x |\n"
    )

    def make(self, tmp_path: Path, grid: str) -> Path:
        (tmp_path / "docs").mkdir(parents=True)
        (tmp_path / "docs" / "ci-workflows.md").write_text("# CI Workflows\n\n## CI Matrix\n\n" + grid, encoding="utf-8")
        return tmp_path

    def test_an_accurate_grid_is_clean(self, tmp_path):
        assert cda.check_ci_matrix_grid(make_authorities(), self.make(tmp_path, self.GOOD)) == []

    def test_a_dropped_version_column_is_caught(self, tmp_path):
        grid = self.GOOD.replace("| Python 3.11 ", "").replace("| x | x | x | x | x |", "| x | x | x | x |")
        found = cda.check_ci_matrix_grid(make_authorities(), self.make(tmp_path, grid))
        # The header loses a column AND every row then carries one mark too
        # few, so both legs fire -- that is the grid being checked as a grid,
        # not one regex happening to match.
        assert [f.kind for f in found] == ["ci_python", "ci_matrix", "ci_matrix", "ci_matrix"]
        assert found[0].actual == "3.10, 3.11, 3.12, 3.13, 3.14"

    def test_a_partial_cross_product_row_is_caught(self, tmp_path):
        grid = self.GOOD.replace("| macOS | x | x | x | x | x |", "| macOS | | x | | x | |")
        found = cda.check_ci_matrix_grid(make_authorities(), self.make(tmp_path, grid))
        assert [f.kind for f in found] == ["ci_matrix"]
        assert "2 of 5" in found[0].claimed

    def test_a_missing_os_row_is_caught(self, tmp_path):
        grid = self.GOOD.replace("| Windows | x | x | x | x | x |\n", "")
        found = cda.check_ci_matrix_grid(make_authorities(), self.make(tmp_path, grid))
        assert [f.kind for f in found] == ["ci_os"]
        assert found[0].actual == "macos, ubuntu, windows"

    def test_a_table_that_is_not_the_matrix_is_ignored(self, tmp_path):
        other = "| Workflow | File |\n|---|---|\n| CI | `ci.yml` |\n"
        assert cda.check_ci_matrix_grid(make_authorities(), self.make(tmp_path, other)) == []

    def test_the_shipped_grid_agrees_with_the_shipped_matrix(self):
        auth = cda.Authorities(**real_auth_kwargs())
        assert cda.check_ci_matrix_grid(auth, ROOT) == []


class TestEvalHeadlineWrapping:
    """A per-line eval scan walked past the stale 109/109 -- it was wrapped."""

    def test_a_headline_wrapped_onto_the_next_line_is_still_gated(self, tmp_path):
        (tmp_path / "train").mkdir(parents=True)
        (tmp_path / "train" / "HF_MODEL_CARD_v4.md").write_text(
            "Eval score for the weights `main` currently points at (`v4.1.1`):\n**109/109 = 100%** on the un-softened harness.\n",
            encoding="utf-8",
        )
        found = cda.check_eval_claims(make_authorities(), tmp_path)
        assert [(f.lineno, f.claimed, f.actual) for f in found] == [(2, "109", "133"), (2, "109", "133")]

    def test_the_reported_span_is_line_relative_so_fix_can_use_it(self, tmp_path):
        (tmp_path / "train").mkdir(parents=True)
        card = tmp_path / "train" / "HF_MODEL_CARD_v4.md"
        card.write_text("Grand total: 131 / 131 = 100%\n", encoding="utf-8")
        findings = cda.check_eval_claims(make_authorities(), tmp_path)
        fixed, skipped = cda.apply_fixes(findings, tmp_path)
        assert (fixed, skipped) == (2, [])
        assert card.read_text(encoding="utf-8") == "Grand total: 133 / 133 = 100%\n"


class TestFixModeCoversTheNewKinds:
    def test_fix_rewrites_a_stale_flag_count(self, tmp_path):
        (tmp_path / "docs").mkdir(parents=True)
        doc = tmp_path / "docs" / "x.md"
        doc.write_text("feature_flags.py — 35 flags + is_enabled\n", encoding="utf-8")
        findings = cda.scan_line("docs/x.md", 1, doc.read_text(encoding="utf-8").rstrip("\n"), make_authorities())
        fixed, skipped = cda.apply_fixes(findings, tmp_path)
        assert (fixed, skipped) == (1, [])
        assert doc.read_text(encoding="utf-8") == "feature_flags.py — 52 flags + is_enabled\n"

    def test_a_structural_workflow_finding_is_never_guessed_at(self, tmp_path):
        (tmp_path / "docs").mkdir(parents=True)
        doc = tmp_path / "docs" / "ci-workflows.md"
        doc.write_text("| CI | `ci.yml` | push | tests |\n", encoding="utf-8")
        findings = cda.check_workflow_table(make_authorities(), tmp_path)
        before = doc.read_text(encoding="utf-8")
        fixed, skipped = cda.apply_fixes(findings, tmp_path)
        assert fixed == 0
        assert [f.kind for f in skipped] == ["workflow"]
        assert doc.read_text(encoding="utf-8") == before


class TestModuleFactsAuthority:
    def test_the_line_count_is_the_file(self):
        rel = "src/mind_mem/recompaction.py"
        assert aa.module_line_count(rel, ROOT) == len((ROOT / rel).read_text(encoding="utf-8").splitlines())

    def test_the_test_count_is_the_test_functions(self):
        assert aa.module_test_count("src/mind_mem/recompaction.py", ROOT) == 18

    def test_a_module_with_no_test_file_fails_loud(self):
        with pytest.raises(aa.AuthorityError) as exc:
            aa.module_test_count("src/mind_mem/__init__.py", ROOT)
        assert "no test file" in str(exc.value)

    def test_a_test_file_with_no_tests_is_an_error_not_a_zero(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "widget.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_widget.py").write_text("def helper():\n    pass\n", encoding="utf-8")
        with pytest.raises(aa.AuthorityError) as exc:
            aa.module_test_count("src/widget.py", tmp_path)
        assert "counting rule broke" in str(exc.value)


class TestModuleFactsClaims:
    def build(self, tmp_path: Path, header: str) -> Path:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "widget.py").write_text("\n".join(f"line {i}" for i in range(10)) + "\n", encoding="utf-8")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_widget.py").write_text("def test_a():\n    pass\n\n\ndef test_b():\n    pass\n", encoding="utf-8")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "widget.md").write_text(header, encoding="utf-8")
        return tmp_path

    def test_a_stale_line_count_is_caught(self, tmp_path):
        root = self.build(tmp_path, "**Module:** `src/widget.py` (268 lines, 2 tests, 99% coverage)\n")
        found = cda.check_module_facts(make_authorities(), root)
        assert [(f.kind, f.claimed, f.actual) for f in found] == [("module_facts", "268", "10")]

    def test_a_stale_test_count_is_caught(self, tmp_path):
        root = self.build(tmp_path, "**Module:** `src/widget.py` (10 lines, 7 tests, 99% coverage)\n")
        assert [(f.claimed, f.actual) for f in cda.check_module_facts(make_authorities(), root)] == [("7", "2")]

    def test_an_accurate_header_is_clean(self, tmp_path):
        root = self.build(tmp_path, "**Module:** `src/widget.py` (10 lines, 2 tests, 99% coverage)\n")
        assert cda.check_module_facts(make_authorities(), root) == []

    def test_a_header_naming_a_module_that_does_not_exist_is_caught(self, tmp_path):
        root = self.build(tmp_path, "**Module:** `src/ghost.py` (10 lines, 2 tests, 99% coverage)\n")
        assert [f.actual for f in cda.check_module_facts(make_authorities(), root)] == ["(no such module)"]

    def test_fix_rewrites_a_stale_line_count(self, tmp_path):
        root = self.build(tmp_path, "**Module:** `src/widget.py` (268 lines, 2 tests, 99% coverage)\n")
        fixed, skipped = cda.apply_fixes(cda.check_module_facts(make_authorities(), root), root)
        assert (fixed, skipped) == (1, [])
        assert (root / "docs" / "widget.md").read_text(encoding="utf-8").startswith("**Module:** `src/widget.py` (10 lines,")

    def test_the_shipped_module_headers_agree_with_their_modules(self):
        assert cda.check_module_facts(cda.Authorities(**real_auth_kwargs()), ROOT) == []


class TestPythonSupportClaims:
    """ "Python 3.10+" appears on nine live surfaces and had no authority."""

    def test_the_authority_is_packaging_metadata(self):
        floor, low, high = aa.python_support(ROOT)
        assert floor == low, "the requires-python floor and the lowest classifier must agree"
        assert tuple(int(p) for p in high.split(".")) >= tuple(int(p) for p in low.split("."))

    def test_a_stale_floor_is_caught(self):
        found = scan("**Requirements:** Python 3.9+, FastMCP 2.0+ (for MCP server).")
        assert [(f.kind, f.claimed, f.actual) for f in found] == [("python_support", "3.9", "3.10")]

    def test_the_true_floor_passes(self):
        assert scan("- Python: 3.10+") == []

    def test_a_stale_range_endpoint_is_caught(self):
        found = scan("Python 3.10–3.13 supported. No required native dependencies")
        assert [(f.claimed, f.actual) for f in found] == [("3.13", "3.14")]

    def test_the_true_range_passes(self):
        assert scan("Python 3.10–3.14 supported. No required native dependencies") == []

    def test_a_release_record_keeps_its_own_floor(self):
        line = "**Requirements:** Python 3.9+"
        assert scan(line, historical=True) == []
        assert scan(line, historical=False) != []

    def test_fix_rewrites_a_stale_floor(self, tmp_path):
        (tmp_path / "docs").mkdir(parents=True)
        doc = tmp_path / "docs" / "x.md"
        doc.write_text("Requires Python 3.9+ and nothing else.\n", encoding="utf-8")
        findings = cda.scan_line("docs/x.md", 1, "Requires Python 3.9+ and nothing else.", make_authorities())
        fixed, skipped = cda.apply_fixes(findings, tmp_path)
        assert (fixed, skipped) == (1, [])
        assert doc.read_text(encoding="utf-8") == "Requires Python 3.10+ and nothing else.\n"

    def test_the_python_authority_is_load_bearing(self):
        """Move the floor and the nine live surfaces go red."""
        auth = cda.Authorities(**real_auth_kwargs())
        assert cda.scan_docs(auth, ROOT) == []
        mutated = cda.Authorities(**{**real_auth_kwargs(), "python_floor": "3.11"})
        findings = cda.scan_docs(mutated, ROOT)
        assert len(findings) >= 5
        assert {f.kind for f in findings} == {"python_support"}


class TestBackendsBadge:
    """The badge undersold the product: `encrypted` is a `--backend` choice."""

    def test_the_authority_is_the_init_workspace_tuple(self):
        assert aa.storage_backends(ROOT) == ("markdown", "postgres", "encrypted")

    def test_a_computed_tuple_is_an_error_not_a_guess(self, tmp_path):
        (tmp_path / "src" / "mind_mem").mkdir(parents=True)
        (tmp_path / "src" / "mind_mem" / "init_workspace.py").write_text("SUPPORTED_BACKENDS = tuple(_discover())\n", encoding="utf-8")
        with pytest.raises(aa.AuthorityError) as exc:
            aa.storage_backends(tmp_path)
        assert "not a literal sequence" in str(exc.value)

    def test_a_badge_missing_a_backend_is_caught(self, tmp_path):
        (tmp_path / "README.md").write_text(
            '<img src="https://img.shields.io/badge/backends-markdown_%7C_postgres-teal?style=flat-square" alt="x">\n',
            encoding="utf-8",
        )
        found = cda.check_backends_badge(make_authorities(), tmp_path)
        assert [(f.kind, f.claimed, f.actual) for f in found] == [
            ("backends", "markdown_%7C_postgres", "markdown_%7C_postgres_%7C_encrypted")
        ]

    def test_a_badge_naming_a_backend_that_does_not_exist_is_caught(self, tmp_path):
        (tmp_path / "README.md").write_text(
            '<img src="https://img.shields.io/badge/backends-markdown_%7C_redis-teal?style=flat-square" alt="x">\n',
            encoding="utf-8",
        )
        assert [f.claimed for f in cda.check_backends_badge(make_authorities(), tmp_path)] == ["markdown_%7C_redis"]

    def test_the_true_badge_passes(self, tmp_path):
        (tmp_path / "README.md").write_text(
            '<img src="https://img.shields.io/badge/backends-markdown_%7C_postgres_%7C_encrypted-teal?style=flat-square" alt="x">\n',
            encoding="utf-8",
        )
        assert cda.check_backends_badge(make_authorities(), tmp_path) == []

    def test_the_shipped_badge_agrees_with_the_shipped_tuple(self):
        assert cda.check_backends_badge(cda.Authorities(**real_auth_kwargs()), ROOT) == []

    def test_the_backends_authority_is_load_bearing(self):
        mutated = cda.Authorities(**{**real_auth_kwargs(), "backends": ("markdown", "postgres")})
        assert cda.check_backends_badge(mutated, ROOT) != []


class TestExperimentalIsNotShipped:
    """A capability claim with an authority: the tool registry says what shipped.

    ``docs/status.md`` filed model provenance under "Experimental ... not yet
    shipped" while ``audit_model_tool``/``sign_model_tool``/``verify_model_tool``
    were registered unconditionally and counted in the tool badge.
    """

    def build(self, tmp_path: Path, status_body: str, tool_body: str) -> Path:
        (tmp_path / "docs").mkdir(parents=True)
        (tmp_path / "docs" / "status.md").write_text(status_body, encoding="utf-8")
        (tmp_path / "src" / "mind_mem" / "mcp" / "tools").mkdir(parents=True)
        (tmp_path / "src" / "mind_mem" / "mcp" / "tools" / "widget.py").write_text(tool_body, encoding="utf-8")
        return tmp_path

    # ``count_mcp_tools._tool_names`` reads ``mcp.tool(<fn>)`` REGISTRATION
    # calls, not decorators -- the same rule the 102 comes from, so this stub
    # is registered by exactly the definition of "registered" the badge uses.
    REGISTERED = "def widget_tool(x):\n    return x\n\n\ndef register(mcp):\n    mcp.tool(widget_tool)\n"
    UNREGISTERED = "def widget_helper(x):\n    return x\n"

    def test_a_registered_module_under_experimental_is_caught(self, tmp_path):
        status = "## Experimental (in-tree, behind feature flags)\n\n| A | `src/mind_mem/mcp/tools/widget.py` | not yet shipped |\n"
        found = cda.check_experimental_is_not_shipped(make_authorities(), self.build(tmp_path, status, self.REGISTERED))
        assert [f.kind for f in found] == ["shipped_not_experimental"]
        assert "widget_tool" in found[0].actual

    def test_a_module_registering_nothing_is_left_alone(self, tmp_path):
        status = "## Experimental (in-tree, behind feature flags)\n\n| A | `src/mind_mem/mcp/tools/widget.py` | not yet shipped |\n"
        assert cda.check_experimental_is_not_shipped(make_authorities(), self.build(tmp_path, status, self.UNREGISTERED)) == []

    def test_the_same_module_under_an_implemented_heading_is_fine(self, tmp_path):
        status = "## Implemented now (operational, tested)\n\n| A | `src/mind_mem/mcp/tools/widget.py` | ships |\n"
        assert cda.check_experimental_is_not_shipped(make_authorities(), self.build(tmp_path, status, self.REGISTERED)) == []

    def test_the_experimental_scope_closes_at_the_next_heading(self, tmp_path):
        status = (
            "## Experimental (in-tree, behind feature flags)\n\n| A | `nothing` | x |\n\n"
            "## Implemented now\n\n| B | `src/mind_mem/mcp/tools/widget.py` | ships |\n"
        )
        assert cda.check_experimental_is_not_shipped(make_authorities(), self.build(tmp_path, status, self.REGISTERED)) == []
        # Positive control: move the row back under the experimental heading and
        # the same code path fires, so the scope is scoping rather than muting.
        moved = "## Experimental (in-tree, behind feature flags)\n\n| B | `src/mind_mem/mcp/tools/widget.py` | x |\n"
        assert cda.check_experimental_is_not_shipped(make_authorities(), self.build(tmp_path / "second", moved, self.REGISTERED)) != []

    def test_the_shipped_status_page_makes_no_unshipped_claim_about_a_live_tool(self):
        assert cda.check_experimental_is_not_shipped(cda.Authorities(**real_auth_kwargs()), ROOT) == []
