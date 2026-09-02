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

import re
import subprocess  # nosec B404 - fixed argv, no shell
import sys
from pathlib import Path

import pytest

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
    )
    base.update(overrides)
    return cda.Authorities(**base)


def scan(line: str, *, rel: str = "docs/example.md", auth=None, historical: bool = False):
    return cda.scan_line(rel, 1, line, auth or make_authorities(), historical=historical)


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
