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

The test-count leg changed shape once more on 2026-09-03, and the tests that
covered the old shape were REPLACED rather than deleted. The authority had
been ``pytest --collect-only`` with the CI selector, and this module went to
some length to make that honest (``collected_or_skip`` refused to compare the
badge against a machine that dropped modules at collection). It was still a
property of the machine: CI, installing ``[test]`` alone, collected 11,662 on
a commit the workstation collected 11,726 on, and ``--fix`` could not write a
number both would accept. ``alignment_authorities.static_test_count`` now
counts ``def test_*`` functions from source and imports nothing;
:class:`TestStaticTestCount` covers its counting rule and
:class:`TestTheAuthorityIsEnvironmentIndependent` proves the number does not
move when ``psycopg`` and ``sqlite_vec`` are hidden from the import system --
the equality that the collection-based authority could never satisfy. The
retired "N tests" spelling is refused at suite scale
(:class:`TestRetiredTestsSpelling`), because renumbering it under the new
authority would state a runner's count that no machine measured.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import re
import subprocess  # nosec B404 - fixed argv, no shell
import sys
from pathlib import Path

import pytest

from scripts import alignment_authorities as aa
from scripts import check_docs_alignment as cda
from scripts import count_mcp_tools as cmt

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

    Every leg is computed by ``resolve_authorities`` from the tree; nothing
    here injects a value. The test-count leg is a static count of the source,
    so it is the same number on this machine as on every CI row, and no
    ``skip`` is needed to keep the comparison honest.
    """
    return cda.resolve_authorities(ROOT)


def real_auth_kwargs() -> dict:
    """Every real authority as kwargs, so a test can move exactly one."""
    return dataclasses.asdict(_real_authorities())


def _lines(rel: str) -> list[str]:
    return (ROOT / rel).read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# The authorities themselves
# ---------------------------------------------------------------------------


_PYPROJECT_PYTEST = (
    "[tool.pytest.ini_options]\n"
    'testpaths = ["tests"]\n'
    'python_files = ["test_*.py"]\n'
    'python_classes = ["Test*"]\n'
    'python_functions = ["test_*"]\n'
)


def _tree(tmp_path: Path, files: dict[str, str], pyproject: str = _PYPROJECT_PYTEST) -> Path:
    """A throwaway repo root with the given test files and pytest options."""
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    for rel, body in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return tmp_path


class TestStaticTestCount:
    """The counting rule is pytest's collection rule applied to source."""

    def test_module_level_functions_and_test_class_methods_are_counted(self):
        src = "def test_a(): pass\n\nclass TestX:\n    def test_b(self): pass\n    def test_c(self): pass\n"
        assert aa.test_functions_in_source(src) == 3

    def test_nested_test_classes_are_counted(self):
        src = "class TestOuter:\n    class TestInner:\n        def test_deep(self): pass\n"
        assert aa.test_functions_in_source(src) == 1

    def test_helpers_non_test_classes_and_nested_defs_are_not(self):
        """Positive control is the test above: same walker, counting shapes it should."""
        src = (
            "def helper(): pass\n"
            "class Fixture:\n    def test_not_collected(self): pass\n"
            "def test_outer():\n    def test_inner(): pass\n"
            "async def test_async(): pass\n"
        )
        assert aa.test_functions_in_source(src) == 2, "test_outer and test_async; nothing else is collected by pytest"

    def test_the_tree_count_sums_every_matching_file_including_integration(self, tmp_path):
        root = _tree(
            tmp_path,
            {
                "tests/test_a.py": "def test_1(): pass\n",
                "tests/integration/test_b.py": "def test_2(): pass\ndef test_3(): pass\n",
                "tests/helpers.py": "def test_ignored_file(): pass\n",
                "tests/conftest.py": "def test_not_a_test_file(): pass\n",
            },
        )
        assert aa.static_test_count(root) == 3

    def test_the_file_rule_is_read_from_pyproject_not_hard_coded(self, tmp_path):
        """Change ``python_files`` and the count follows -- the rule is pytest's, by construction."""
        root = _tree(
            tmp_path,
            {"tests/a_spec.py": "def test_1(): pass\n", "tests/test_b.py": "def test_2(): pass\n"},
            pyproject=_PYPROJECT_PYTEST.replace('python_files = ["test_*.py"]', 'python_files = ["*_spec.py"]'),
        )
        assert aa.static_test_count(root) == 1

    def test_a_tree_with_nothing_to_count_is_an_error_not_a_zero(self, tmp_path):
        """A verifier that died is not a verifier that passed.

        Returning 0 here would make every four-digit badge look stale and,
        worse, would make an empty finding list readable as "aligned".
        """
        with pytest.raises(cda.AuthorityError):
            aa.static_test_count(_tree(tmp_path, {}))
        with pytest.raises(cda.AuthorityError):
            aa.static_test_count(_tree(tmp_path, {"tests/test_empty.py": "def helper(): pass\n"}))

    def test_an_unparseable_test_file_is_an_error_not_a_short_count(self, tmp_path):
        root = _tree(tmp_path, {"tests/test_ok.py": "def test_1(): pass\n", "tests/test_bad.py": "def (:\n"})
        with pytest.raises(cda.AuthorityError, match="could not parse"):
            aa.static_test_count(root)

    def test_the_real_tree_is_counted_without_importing_a_test_module(self, monkeypatch):
        """The authority reads files; it must never execute them.

        A counting rule that imported test modules would inherit every
        ``importorskip`` in the suite -- the exact dependence on the host this
        authority exists to remove. ``sys.modules`` is snapshotted around the
        call; a test module appearing in it is the failure.
        """
        before = set(sys.modules)
        count = aa.static_test_count(ROOT)
        assert count > 1000, "the suite has been four digits since v3.x"
        imported = {name for name in set(sys.modules) - before if name.startswith("test_") or name.startswith("tests.")}
        assert imported == set(), f"the static authority imported test modules: {sorted(imported)[:5]}"


class TestAuthorities:
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
    BADGE = '<img src="https://img.shields.io/badge/test_functions-{n}-brightgreen?style=flat-square" alt="Test functions: {alt}">'

    def test_a_stale_badge_is_caught(self):
        findings = scan(self.BADGE.format(n="9%2C366", alt="9,366"))
        kinds = {(f.kind, f.claimed) for f in findings}
        assert ("tests", "9%2C366") in kinds, "the shields path must be gated"
        assert ("tests", "9,366") in kinds, "the alt text drifts independently and must be gated too"
        assert {f.actual for f in findings} == {"9%2C707", "9,707"}, "the fix must keep each spelling's grouping"

    def test_a_correct_badge_is_not_flagged(self):
        assert scan(self.BADGE.format(n="9%2C707", alt="9,707")) == []

    def test_a_suite_scale_prose_claim_is_caught(self):
        findings = scan("- **CI** on every push and PR - full pytest matrix (7,500+ test functions across the suite).")
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("tests", "7,500", "9,707")]

    def test_the_trailing_plus_is_part_of_the_claim_it_replaces(self):
        """ "7,500+ test functions" must become "9,707 test functions", never "9,707+ ..."."""
        line = "full pytest matrix (7,500+ test functions across the suite)."
        finding = scan(line)[0]
        assert line[finding.start : finding.end] == "7,500+"

    def test_a_module_scale_claim_is_not_gated(self):
        """ "18 test functions" is about one module; the suite gate has no business there."""
        assert scan("Recompaction ships with 18 test functions covering the merge path.") == []

    def test_but_a_small_number_the_sentence_calls_suite_wide_IS_gated(self):
        """The floor is a size heuristic, not an escape hatch -- scope words win."""
        findings = scan("The whole test suite is 900 test functions today.")
        assert [(f.kind, f.claimed) for f in findings] == [("tests", "900")]


class TestRetiredTestsSpelling:
    """ "N tests" at suite scale named a runner's count. It is refused, not renumbered.

    The number the authority computes is a count of functions in the tree; a
    sentence saying "N tests" reads as what pytest reported, and that depends
    on the machine. Renumbering it would keep the gate green over a claim the
    authority does not measure -- which is the failure this whole gate exists
    to catch.
    """

    OLD_BADGE = '<img src="https://img.shields.io/badge/tests-{n}-brightgreen?style=flat-square" alt="Tests: {alt}">'

    def test_the_old_badge_is_refused_in_both_spellings(self):
        findings = scan(self.OLD_BADGE.format(n="9%2C707", alt="9,707"))
        assert [(f.kind, f.claimed) for f in findings] == [("tests_spelling", "9%2C707"), ("tests_spelling", "9,707")]

    def test_it_is_refused_even_when_the_number_equals_the_authority(self):
        """The finding is about the noun, not the digits."""
        findings = scan("full pytest matrix (9,707 tests across the suite).")
        assert [(f.kind, f.claimed) for f in findings] == [("tests_spelling", "9,707")]
        assert "test functions" in findings[0].actual

    def test_it_is_not_auto_fixable(self, tmp_path):
        doc = tmp_path / "docs" / "x.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("the suite has 9,707 tests\n", encoding="utf-8")
        findings = cda.scan_line("docs/x.md", 1, "the suite has 9,707 tests", make_authorities())
        fixed, skipped = cda.apply_fixes(findings, tmp_path)
        assert (fixed, [f.kind for f in skipped]) == (0, ["tests_spelling"])
        assert doc.read_text(encoding="utf-8") == "the suite has 9,707 tests\n", "a refused claim must not be rewritten"

    def test_a_module_scale_old_spelling_is_left_alone(self):
        """Negative control: module claims are the module_facts leg's business."""
        assert scan("Recompaction ships with 18 tests covering the merge path.") == []

    def test_a_version_qualified_record_is_left_alone(self):
        """Positive control for the qualifier: the same sentence, unqualified, is refused."""
        assert scan("(9,701 tests in v4.1)") == []
        assert [f.kind for f in scan("the suite has 9,701 tests")] == ["tests_spelling"]

    def test_the_new_spelling_does_not_trip_the_retired_pattern(self):
        assert [f.kind for f in scan("full pytest matrix (9,707 test functions across the suite).")] == []


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
            '<img src="https://img.shields.io/badge/test_functions-9%2C366-x" alt="Test functions: 9,366">\n',
            encoding="utf-8",
        )
        auth = make_authorities()
        findings = [f for f in cda.scan_line("docs/x.md", 1, doc.read_text(encoding="utf-8").rstrip("\n"), auth)]
        assert len(findings) == 2, "both the path and the alt text must be found"
        fixed, skipped = cda.apply_fixes(findings, tmp_path)
        assert (fixed, skipped) == (2, [])
        expected = '<img src="https://img.shields.io/badge/test_functions-9%2C707-x" alt="Test functions: 9,707">\n'
        assert doc.read_text(encoding="utf-8") == expected

    def test_fix_refuses_a_finding_whose_line_moved(self, tmp_path):
        doc = tmp_path / "docs" / "x.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("the suite has 900 test functions\n", encoding="utf-8")
        stale = cda.Finding("docs/x.md", 1, "tests", "8888", "9707", "8888 test functions", 0, 4)
        fixed, skipped = cda.apply_fixes([stale], tmp_path)
        assert fixed == 0 and skipped == [stale]
        assert doc.read_text(encoding="utf-8") == "the suite has 900 test functions\n"


# ---------------------------------------------------------------------------
# The repository itself
# ---------------------------------------------------------------------------


_BADGE_TESTS_RE = re.compile(r"badge/test_functions-([\d%C2c]+)-")


def readme_tests_badge() -> int:
    m = _BADGE_TESTS_RE.search((ROOT / "README.md").read_text(encoding="utf-8"))
    assert m is not None, "README must carry a tests badge"
    return cda._parse_int(m.group(1))


class TestRepositoryIsAligned:
    """Every gated claim in the tree agrees with its authority.

    The test-count leg used to be injected FROM the README badge, with a
    docstring saying "CI asserts the badge agrees with the collector". CI did
    not: ``grep -n check_docs_alignment .github/workflows/*.yml`` returned
    nothing, so the badge was the only authority for itself and agreed with
    itself no matter what it said -- it read 10,550 while the selector
    collected 11,137. The repair collected the count here and in CI, and
    that was a second defect: the two machines collected different numbers
    (11,726 against 11,662 on one commit) because collection is a property
    of the environment. The number is now a static count of the source, so
    this test and CI's ``version-check`` compute the same value from the
    same files with nothing injected by either.
    """

    def test_no_claim_disagrees_with_its_authority(self):
        auth = cda.resolve_authorities(ROOT)
        findings = cda.scan_docs(auth, ROOT)
        assert findings == [], "stale claims:\n" + "\n".join(str(f) for f in findings)

    def test_the_badge_is_not_its_own_authority(self):
        """The README badge equals the count of test functions in the tree."""
        assert readme_tests_badge() == aa.static_test_count(ROOT)

    def test_the_checker_takes_no_injected_count(self):
        """The seam through which the badge once fed itself is gone.

        ``resolve_authorities`` computes every leg; a keyword that let a
        caller supply the test count is exactly what made the badge its own
        authority, and the CLI flag that fed it from a YAML step is the same
        seam one layer up.
        """
        assert "tests_collected" not in inspect.signature(cda.resolve_authorities).parameters
        assert "--tests-collected" not in inspect.getsource(cda.main)

    def test_the_scan_actually_read_something(self):
        """An empty finding list is only evidence when the search happened."""
        files = cda._doc_files(ROOT)
        rels = {p.relative_to(ROOT).as_posix() for p in files}
        assert len(files) > 40, f"only {len(files)} surfaces scanned"
        for required in ("README.md", "docs/governance.md", "train/HF_MODEL_CARD_v4.md"):
            assert required in rels, f"{required} must be scanned"

    def test_the_cli_exits_zero_on_an_aligned_tree(self):
        proc = subprocess.run(  # nosec B603
            [sys.executable, "scripts/check_docs_alignment.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_an_unreachable_authority_exits_two_not_zero(self, monkeypatch):
        """A dead authority must not read as a clean bill of health."""

        def boom(*_a, **_k):
            raise cda.AuthorityError("git is gone")

        monkeypatch.setattr(cda, "trained_tool_count", boom)
        assert cda.main([]) == 2


# ---------------------------------------------------------------------------
# The two shapes every line-scoped pattern was structurally blind to
# ---------------------------------------------------------------------------


def _write_doc(tmp_path: Path, name: str, body: str) -> Path:
    doc = tmp_path / "docs" / name
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(body, encoding="utf-8")
    return doc


def _cmt_claims(body: str) -> list[tuple[int, int]]:
    """``(lineno, value)`` for every tool claim ``count_mcp_tools`` can see."""
    return [(lineno, value) for lineno, _s, _e, value, _x in cmt.scan_doc_claims(body.splitlines())]


class TestTableCellToolClaims:
    """``| MCP tools | 89 | N/A |`` -- the count stated AFTER its label.

    Every pattern in ``count_mcp_tools`` wants the number adjacent to the word,
    so all four reported "all tool-count claims agree with 102" while three
    live docs said 89. These are the fixture positive controls: a doc
    containing the row FAILS the gate.
    """

    COMPARISON = "\n".join(
        [
            "| Feature | MIND-Mem | LangMem |",
            "|---------|----------|---------|",
            "| MCP tools | 89 | N/A |",
            "",
        ]
    )
    MIGRATION = "\n".join(
        [
            "| Feature | mem-os | MIND-Mem |",
            "|---------|--------|----------|",
            "| MCP Tools | 8 | 89 |",
            "",
        ]
    )

    def test_a_stale_cell_fails_the_tool_gate(self):
        assert _cmt_claims(self.COMPARISON) == [(3, 89)]

    def test_a_current_cell_passes(self):
        """The positive control's negative twin: 102 in the same shape is clean."""
        assert _cmt_claims(self.COMPARISON.replace("89", "102")) == [(3, 102)]

    def test_the_gate_reports_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cmt, "_project_root", lambda: tmp_path)
        _write_doc(tmp_path, "comparison.md", self.COMPARISON)
        bad = cmt.check_docs(102)
        assert len(bad) == 1 and "docs/comparison.md:3" in bad[0] and "claims 89" in bad[0]

    def test_a_two_number_row_gates_only_our_column(self):
        """``| MCP Tools | 8 | 89 |`` is mem-os then MIND-Mem.

        Gating every numeric cell would fail the build on a TRUE statement
        about another product; the header row says which column is ours.
        """
        assert _cmt_claims(self.MIGRATION) == [(3, 89)]

    def test_a_competitor_column_is_not_our_claim(self):
        """Positive control: our own cell in the SAME row is still gated."""
        body = self.MIGRATION.replace("| MCP Tools | 8 | 89 |", "| MCP Tools | 45 | 89 |")
        assert _cmt_claims(body) == [(3, 89)]

    def test_a_table_with_no_header_of_ours_gates_every_count(self):
        body = "| Item | Count |\n|------|-------|\n| MCP tools | 89 |\n"
        assert _cmt_claims(body) == [(3, 89)]

    def test_a_row_outside_a_table_is_not_a_claim(self):
        """No separator row, no table: a bare pipe line is prose, not a cell."""
        assert _cmt_claims("| MCP tools | 89 |\n") == []

    def test_a_transition_row_is_a_record_not_a_claim(self):
        body = "| Item | MIND-Mem |\n|------|----------|\n| MCP tools | 89 -> 102 |\n"
        assert _cmt_claims(body) == []

    def test_the_alignment_gate_sees_the_same_cell(self):
        findings = cda.scan_text("docs/comparison.md", self.COMPARISON.splitlines(), make_authorities())
        assert [(f.lineno, f.kind, f.claimed, f.actual) for f in findings] == [(3, "tools", "89", "102")]


class TestWrappedClaims:
    """A claim a markdown reflow split across the line break.

    ``docs/integrations.md`` said "MIND-Mem's 89 MCP\\ntools" -- the number and
    its noun on different lines, so no line-scoped matcher could ever see it.
    """

    WRAPPED = "**What this means**: each of these tools can call MIND-Mem's 89 MCP\ntools (recall, propose_update, scan) the same way.\n"

    def test_a_wrapped_claim_fails_the_tool_gate(self):
        assert _cmt_claims(self.WRAPPED) == [(1, 89)]

    def test_a_current_wrapped_claim_passes(self):
        assert _cmt_claims(self.WRAPPED.replace("89", "102")) == [(1, 102)]

    def test_the_number_is_reported_on_the_line_that_holds_it(self):
        """The fix rewrites by span, so the span must name the real line."""
        claims = cmt.scan_doc_claims(self.WRAPPED.splitlines())
        lineno, start, end, value, _excerpt = claims[0]
        assert (lineno, value) == (1, 89)
        assert self.WRAPPED.splitlines()[lineno - 1][start:end] == "89"

    def test_a_claim_wholly_inside_one_line_is_reported_once(self):
        body = "the server exposes 89 MCP tools today\nand that is the whole story.\n"
        assert _cmt_claims(body) == [(1, 89)]

    def test_a_table_row_is_never_glued_to_the_next_row(self):
        """Joining structure fabricates claims that exist on neither line."""
        body = "| Rows | 89 |\n| tools | many |\n"
        assert _cmt_claims(body) == []

    def test_a_heading_is_never_glued_to_the_paragraph_under_it(self):
        body = "Released in 2026 with 89\n## tools and other things\n"
        assert _cmt_claims(body) == []

    def test_the_alignment_gate_fixes_a_wrapped_number_in_place(self, tmp_path):
        doc = _write_doc(tmp_path, "integrations.md", self.WRAPPED)
        findings = cda.scan_text("docs/integrations.md", self.WRAPPED.splitlines(), make_authorities())
        assert [(f.lineno, f.claimed, f.actual) for f in findings] == [(1, "89", "102")]
        fixed, skipped = cda.apply_fixes(findings, tmp_path)
        assert (fixed, skipped) == (1, [])
        assert doc.read_text(encoding="utf-8").splitlines()[0].endswith("MIND-Mem's 102 MCP")


class TestTrainedClaimsAreReferredNotJudged:
    """``count_mcp_tools`` knows only the LIVE count, so a claim about the
    WEIGHTS is not its to answer.

    CLAUDE.md's "Knows all 84 / tools" is wrapped, so widening the matcher made
    it visible for the first time -- and answering it with 102 would be wrong
    in a new way, because the trained revision registers 83. It is skipped
    there and gated HERE, against the authority that can decide it.
    """

    TRAINED = "Qwen3.5-4B retrained for v4.0.0 (v4 weights revision). Knows all 84\ntools, v4 surfaces (cognitive kernel).\n"

    def test_the_live_gate_refers_it_rather_than_judging_it(self):
        assert _cmt_claims(self.TRAINED) == []

    def test_the_alignment_gate_judges_it_against_the_trained_count(self):
        findings = cda.scan_text("CLAUDE.md", self.TRAINED.splitlines(), make_authorities())
        assert [(f.kind, f.claimed, f.actual) for f in findings] == [("tools", "84", "83")]

    def test_a_live_claim_in_the_same_file_is_still_the_live_gates(self):
        """Positive control: the referral is scoped to the marker, not the file."""
        assert _cmt_claims("the server currently exposes 89 MCP\ntools.\n") == [(1, 89)]


class _HideImports:
    """A ``sys.meta_path`` finder that makes the named top-level packages unimportable.

    Raises ``ModuleNotFoundError`` exactly as an absent package would, so
    code under it sees what a CI row without the extra sees.
    """

    def __init__(self, *names: str) -> None:
        self.names = frozenset(names)

    def find_spec(self, name: str, path: object = None, target: object = None) -> None:
        if name.split(".")[0] in self.names:
            raise ModuleNotFoundError(f"No module named {name!r} (hidden for the environment-independence proof)", name=name)
        return None


class TestTheAuthorityIsEnvironmentIndependent:
    """THE deliverable for the static authority: the same number with the extras hidden.

    The collection-based authority this replaced produced 11,726 on a
    workstation with every extra and 11,662 on CI rows without ``psycopg``
    (four Postgres modules importorskip at module level and were dropped
    whole). A static count cannot see an import, so hiding the extras must
    change nothing -- and the assertion is only evidence beside the positive
    control that the hiding actually works in this process.
    """

    HIDDEN = ("psycopg", "psycopg_pool", "pgvector", "sqlite_vec")

    def test_the_count_is_identical_with_psycopg_and_sqlite_vec_hidden(self, monkeypatch):
        import importlib

        with_everything = aa.static_test_count(ROOT)

        hider = _HideImports(*self.HIDDEN)
        monkeypatch.setattr(sys, "meta_path", [hider, *sys.meta_path])
        for name in self.HIDDEN:
            monkeypatch.delitem(sys.modules, name, raising=False)
        # Positive control: the hidden packages really are unimportable now.
        for name in ("psycopg", "sqlite_vec"):
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(name)

        hidden = aa.static_test_count(ROOT)
        assert hidden == with_everything, (
            f"the test-count authority moved when the extras were hidden ({with_everything} -> {hidden}); "
            "it is reading the environment, not the tree"
        )

    def test_the_hider_would_have_changed_a_collection(self, tmp_path):
        """Positive control for the proof's method, not just its subject.

        A module-level ``importorskip`` on a hidden name drops the module
        at collection -- the exact mechanism that made the old authority
        environment-dependent. The static count of the same file is one
        either way.
        """
        src = "import pytest\n\npytest.importorskip('psycopg')\n\n\ndef test_needs_the_extra():\n    pass\n"
        (tmp_path / "test_dropped.py").write_text(src, encoding="utf-8")
        assert aa.test_functions_in_source(src) == 1
        proc = subprocess.run(  # nosec B603 - fixed argv, no shell
            [
                sys.executable,
                "-c",
                (
                    "import sys, pytest\n"
                    "class H:\n"
                    "    def find_spec(self, name, path=None, target=None):\n"
                    "        if name.split('.')[0] == 'psycopg':\n"
                    "            raise ModuleNotFoundError(name, name=name)\n"
                    "sys.meta_path.insert(0, H())\n"
                    f"raise SystemExit(pytest.main([{str(tmp_path)!r}, '--collect-only', '-q', '-p', 'no:cacheprovider']))\n"
                ),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )
        assert "1 test collected" not in proc.stdout, proc.stdout
        assert "no tests collected" in proc.stdout or "0 tests collected" in proc.stdout or "skipped" in proc.stdout, proc.stdout


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
        # The table-cell leg is a SECOND finding producer, not a caller of
        # scan_line, so neutering scan_line alone no longer empties the scan --
        # which is itself the proof that the new leg is load-bearing.
        monkeypatch.setattr(cda, "scan_table_tools", lambda *a, **k: [])
        auth = cda.resolve_authorities(ROOT)
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


# ---------------------------------------------------------------------------
# Generated doc blocks: the CLI verb index and the HTTP route table
#
# Both surfaces were documented by hand and both had drifted past the point
# where a reader could trust them: 32 of the 51 `mm` verbs had no line in
# docs/cli-reference.md -- including `mm bind`, the command
# governance_gate.py names in EVERY drift refusal, so the gate's own remedy
# was undocumented -- and 9 of the 11 stdlib-transport routes were absent
# from docs/rest-api.md entirely.
#
# Hand-editing them again would buy one correct day. These render the tables
# FROM the argparse tree and FROM http_transport.ROUTES, and the tests below
# require the committed docs to equal the render, so the next verb or route
# either updates its documentation or fails the build.
# ---------------------------------------------------------------------------

CLI_MARKER = "cli-verb-index"
ROUTE_MARKER = "http-transport-routes"


def _begin(marker: str) -> str:
    return f"<!-- BEGIN GENERATED: {marker} — regenerate with tests/test_docs_alignment.py -->"


def _end(marker: str) -> str:
    return f"<!-- END GENERATED: {marker} -->"


def generated_block(text: str, marker: str) -> str:
    """The body between *marker*'s sentinels, or "" when absent."""
    begin, end = _begin(marker), _end(marker)
    if begin not in text or end not in text:
        return ""
    return text.split(begin, 1)[1].split(end, 1)[0]


def render_cli_verb_index() -> str:
    """Every `mm` subcommand and its own argparse help, as a table."""
    import argparse

    from mind_mem.mm_cli import build_parser

    parser = build_parser()
    subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(subparsers) == 1, f"expected one subparser group, found {len(subparsers)}"
    lines = ["", "| Command | What it does |", "| --- | --- |"]
    for action in subparsers[0]._choices_actions:
        help_text = " ".join((action.help or "").split()).replace("|", r"\|")
        lines.append(f"| `mm {action.dest}` | {help_text} |")
    lines.append("")
    return "\n".join(lines)


def render_route_table() -> str:
    """Every route the stdlib HTTP transport serves, with its declarations."""
    from mind_mem.http_transport import CONTENT, ROUTES

    lines = [
        "",
        "| Method | Path | Serves block content | Mutates state |",
        "| --- | --- | --- | --- |",
    ]
    for route in ROUTES:
        path = route.path if route.takes != "tail" else f"{route.path}{{tail}}"
        content = "yes" if route.verdict == CONTENT else "no"
        lines.append(f"| `{route.method}` | `{path}` | {content} | {'yes' if route.mutates else 'no'} |")
    lines.append("")
    return "\n".join(lines)


class TestGeneratedCliVerbIndex:
    DOC = ROOT / "docs" / "cli-reference.md"

    def test_the_renderer_sees_a_real_parser(self) -> None:
        """Positive control: a renderer over an empty parser documents nothing
        and would make every assertion below vacuously true."""
        rendered = render_cli_verb_index()
        assert rendered.count("\n| `mm ") >= 40, rendered
        assert "| `mm recall` |" in rendered

    def test_the_committed_index_matches_the_parser(self) -> None:
        block = generated_block(self.DOC.read_text(encoding="utf-8"), CLI_MARKER)
        assert block, f"{self.DOC} has no generated verb index — the sentinels are gone"
        assert block == render_cli_verb_index(), (
            "docs/cli-reference.md's verb index has drifted from mm_cli.build_parser(). "
            "Replace the block between the sentinels with:\n" + render_cli_verb_index()
        )

    def test_every_verb_the_parser_accepts_is_in_the_doc(self) -> None:
        """The census, taken a second way, so a renderer that agreed with
        itself could not hide a dropped verb."""
        import argparse

        from mind_mem.mm_cli import build_parser

        parser = build_parser()
        subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]
        verbs = set(subparsers.choices)
        assert verbs, "positive control: the parser registered no subcommands"
        # Scoped to the generated block, NOT the page. A whole-document
        # substring search was satisfied by this section's own INTRO PROSE
        # (which names `mm bind` as an example), so renaming the verb's row
        # to a typo left the census green -- measured, in the mutation run
        # that was supposed to prove this test worked.
        block = generated_block(self.DOC.read_text(encoding="utf-8"), CLI_MARKER)
        assert block, "the generated block is gone; the census would pass on prose"
        rows = {line.split("`")[1] for line in block.splitlines() if line.startswith("| `mm ")}
        missing = sorted(f"mm {v}" for v in verbs if f"mm {v}" not in rows)
        assert not missing, f"verbs the CLI accepts and the generated index never names: {missing}"

    def test_bind_is_documented(self) -> None:
        """The verb every drift refusal tells the operator to run.

        ``governance_gate`` names ``mm bind`` in the message it prints when
        it stops a write. Until 5.0.2 the CLI reference did not contain the
        string: the product's own remedy pointed at an undocumented command.
        """
        block = generated_block(self.DOC.read_text(encoding="utf-8"), CLI_MARKER)
        assert "| `mm bind` |" in block, "mm bind has no ROW in the generated index (prose mentioning it does not count)"


class TestGeneratedRouteTable:
    DOC = ROOT / "docs" / "rest-api.md"

    def test_the_renderer_sees_real_routes(self) -> None:
        rendered = render_route_table()
        assert rendered.count("\n| `") >= 8, rendered
        assert "`/status`" in rendered

    def test_the_committed_table_matches_the_route_tuple(self) -> None:
        block = generated_block(self.DOC.read_text(encoding="utf-8"), ROUTE_MARKER)
        assert block, f"{self.DOC} has no generated route table — the sentinels are gone"
        assert block == render_route_table(), (
            "docs/rest-api.md's route table has drifted from http_transport.ROUTES. "
            "Replace the block between the sentinels with:\n" + render_route_table()
        )

    def test_every_handler_is_reachable_from_a_documented_route(self) -> None:
        """Census from the handlers, not the table, so a route dropped from
        ``ROUTES`` cannot hide behind a table that agrees with it."""
        import inspect

        from mind_mem import http_transport

        handlers = {name for name, obj in vars(http_transport).items() if name.startswith("_handle_") and inspect.isfunction(obj)}
        assert handlers, "positive control: no _handle_* functions found"
        routed = {route.handler.__name__ for route in http_transport.ROUTES}
        assert handlers == routed, (
            f"handlers with no route: {sorted(handlers - routed)}; routes naming an absent handler: {sorted(routed - handlers)}"
        )
        block = generated_block(self.DOC.read_text(encoding="utf-8"), ROUTE_MARKER)
        assert block, "the generated block is gone; the census would pass on prose"
        for route in http_transport.ROUTES:
            path = route.path if route.takes != "tail" else f"{route.path}{{tail}}"
            assert f"| `{route.method}` | `{path}` |" in block, f"{route.name} is served but has no ROW in docs/rest-api.md"


class TestArchitectureNamesEveryLedger:
    """``docs/architecture.md`` must name each ledger it claims to describe.

    MEASURED before the fix: the file contained ZERO occurrences of
    ``hash_chain_v2``, ``evidence_chain``, ``audit_sidecar`` and
    ``served_ledger`` — four distinct artifacts, one phrase ("audit chain")
    used for whichever, and a reader with no way to tell which one a claim
    was about. Naming them is not documentation polish: the product's
    differentiator is what those files guarantee, and three of them
    guarantee different things.
    """

    DOC = ROOT / "docs" / "architecture.md"

    def test_every_ledger_check_is_described(self) -> None:
        from mind_mem.verify_cli import LEDGER_CHECKS

        assert LEDGER_CHECKS, "positive control: the authority is empty"
        # A ROW, not a mention. The first cut of this test searched the whole
        # page, and the section's own opening paragraph lists all four names
        # while explaining that they used to be missing -- so unnaming a row
        # left the gate green. Measured in a mutation run; fixed by anchoring
        # on the table's own shape.
        rows = [line for line in self.DOC.read_text(encoding="utf-8").splitlines() if line.startswith("| `")]
        assert rows, "positive control: the ledger table has no rows at all"
        named = {line.split("`")[1] for line in rows}
        missing = sorted(name for name in LEDGER_CHECKS if name not in named)
        assert not missing, f"ledgers verify_cli walks with no row in docs/architecture.md: {missing}"

    def test_every_ledger_names_its_artifact_path(self) -> None:
        """Naming the row is half of it; the reader needs the file.

        Without this, a doc could satisfy the test above with a bullet list
        of row names and still leave "which file is the audit chain?"
        unanswered — the exact question the section exists for.
        """
        doc = self.DOC.read_text(encoding="utf-8")
        for artifact in (
            "memory/hash_chain_v2.db",
            "memory/evidence_chain.jsonl",
            ".mind-mem-audit/chain.jsonl",
            ".mind-mem-ledger/served.jsonl",
        ):
            assert artifact in doc, f"{artifact} is walked by verify_workspace and named nowhere in the architecture doc"

    def test_positive_control_the_scan_can_see_an_absence(self) -> None:
        """Proof the two assertions above are measurements.

        A substring search that always succeeded would pass them on any
        file at all. This name is not in the document and must be reported
        missing by the same method.
        """
        rows = [line for line in self.DOC.read_text(encoding="utf-8").splitlines() if line.startswith("| `")]
        named = {line.split("`")[1] for line in rows}
        assert "a_fifth_ledger_that_does_not_exist" not in named


class TestCompiledKernelClaim:
    """ "Scoring kernels compiled from MIND source" is not true of any wheel.

    MEASURED, three ways, and the README asserted the opposite in one place
    while denying it in another 1500 lines later:

    * ``lib/kernels.c`` opens "mind-mem scoring kernels — C99 reference
      implementations … the C equivalents of the MIND tensor kernels".
      The optional ``libmindmem.so`` is built from THAT, by ``gcc``.
    * ``pyproject.toml`` ships ``*.mind`` sources (``package-data``) and
      ``mind/*.mind`` (``data-files``) and no shared object at all, so no
      wheel carries a compiled kernel to begin with.
    * ``mind_kernels.py`` is the authoritative scoring path, and
      ``mind_ffi`` falls back to it and reports the backend.

    The claim is not banned forever — it is banned until it is true. The
    gate is conditional on the artifact: ship a compiled kernel in the
    wheel and the phrase is allowed in the same commit.
    """

    PHRASES = ("compiled from MIND source", "kernels compiled from MIND")
    SURFACES = ("README.md", "CLAUDE.md", "ROADMAP.md")

    @staticmethod
    def _wheel_ships_a_compiled_kernel() -> bool:
        """Whether ``pyproject`` puts a shared object in the wheel."""
        try:
            import tomllib as toml
        except ModuleNotFoundError:  # pragma: no cover - python 3.10 only
            import tomli as toml  # type: ignore[no-redef]

        data = toml.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        setuptools = data.get("tool", {}).get("setuptools", {})
        globs: list[str] = []
        for patterns in setuptools.get("package-data", {}).values():
            globs.extend(patterns)
        for patterns in setuptools.get("data-files", {}).values():
            globs.extend(patterns)
        return any(g.endswith((".so", ".dylib", ".dll")) for g in globs)

    def _docs(self) -> dict[str, str]:
        paths = [ROOT / name for name in self.SURFACES]
        paths.extend(sorted((ROOT / "docs").rglob("*.md")))
        return {str(p.relative_to(ROOT)): p.read_text(encoding="utf-8") for p in paths if p.is_file()}

    def test_the_scan_covers_the_public_surface(self) -> None:
        """Positive control: an empty corpus finds no claim and says clean."""
        docs = self._docs()
        assert len(docs) > 20, f"only {len(docs)} public docs scanned — the walk is wrong"
        assert "README.md" in docs

    def test_the_scan_can_find_a_phrase_that_is_present(self) -> None:
        """Positive control for the METHOD, not the corpus.

        A substring search over the wrong text, or over text read with the
        wrong encoding, reports zero for everything. This phrase IS in the
        README, so a scanner that cannot see it cannot be trusted to see
        the forbidden one either.
        """
        docs = self._docs()
        hits = [rel for rel, body in docs.items() if "Q16.16" in body]
        assert "README.md" in hits, hits

    def test_no_public_doc_claims_a_compiled_mind_kernel(self) -> None:
        if self._wheel_ships_a_compiled_kernel():
            pytest.skip("a compiled kernel now ships in the wheel; the claim has become true")
        offenders = [f"{rel}: {phrase!r}" for rel, body in self._docs().items() for phrase in self.PHRASES if phrase in body]
        assert not offenders, (
            "no wheel ships a compiled kernel (pyproject ships .mind SOURCES; libmindmem.so is built "
            f"from lib/kernels.c, in C99), so this claim is false where it stands: {offenders}"
        )

    def test_the_readme_states_what_actually_ships(self) -> None:
        """The claim was removed, not merely softened.

        Deleting a false sentence and leaving a hole is how the next writer
        re-adds it. The README now says what the wheel carries, so there is
        a true sentence occupying the space.
        """
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "lib/kernels.c" in readme, "the README no longer says where the native library comes from"
        assert "mind_kernels.py" in readme, "the README no longer names the authoritative scoring path"

    def test_the_troubleshooting_row_agrees_with_the_kernel_inventory(self) -> None:
        """The README told readers ALL the .mind files are INI configs.

        ``docs/MIND_CONFIG_VS_MIND_LANG.md`` — linked from that very row —
        says 18 of 26 are INI and 8 are MIND-language tensor source. The two
        documents contradicted each other, with the wrong one doing the
        reassuring. Counted from the files rather than from either doc.
        """
        sources = sorted((ROOT / "mind").glob("*.mind"))
        assert sources, "positive control: no .mind files found at all"
        lang = [p for p in sources if any(line.lstrip().startswith("fn ") for line in p.read_text(encoding="utf-8").splitlines())]
        assert lang, "positive control: the MIND-language detector matched nothing"
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "the `.mind` files are INI configs, not yet MIND-language source" not in readme, (
            f"{len(lang)} of {len(sources)} mind/*.mind files ARE MIND-language source; the README says none are"
        )
        assert f"{len(sources) - len(lang)} are INI-style config" in readme and f"{len(lang)} are MIND-language" in readme, (
            f"the README's split must match the files: {len(sources) - len(lang)} INI / {len(lang)} MIND-language"
        )


class TestServedLedgerDefault:
    """The docs said `false`; the function has said on since 5.0.2.

    ``served_ledger.ledger_enabled`` returns True unless a workspace opts
    out with a literal ``false``. The configuration table said the default
    was ``false`` and that "only a literal ``true`` turns it on" — the exact
    inverse — and ``ROADMAP.md``, ``verify_cli``, ``replay_check`` and
    ``accountability_views`` all repeated the opt-in story in prose. Six
    documents describing a behaviour the code stopped having.

    The authority is the FUNCTION, evaluated here on a real empty config
    rather than read out of a comment, so the prose cannot drift from it
    again without this going red.
    """

    OPT_IN_PHRASES = ("opt-in", "opt in", "Off by default", "default OFF", "default-OFF", "off by default")
    PROSE_FILES = (
        "src/mind_mem/verify_cli.py",
        "src/mind_mem/replay_check.py",
        "src/mind_mem/accountability_views.py",
    )

    @staticmethod
    def default_is_on(tmp_path: Path) -> bool:
        """``ledger_enabled`` on a workspace whose config sets nothing."""
        from mind_mem.served_ledger import ledger_enabled

        (tmp_path / "mind-mem.json").write_text("{}", encoding="utf-8")
        return ledger_enabled(str(tmp_path))

    def test_the_authority_is_computed_not_quoted(self, tmp_path: Path) -> None:
        assert self.default_is_on(tmp_path) is True, (
            "ledger_enabled says OFF for an empty config — the docs below are now wrong the other way"
        )

    def test_the_probe_can_see_the_off_case(self, tmp_path: Path) -> None:
        """Positive control: a probe that always answered True would make
        every assertion here vacuous."""
        from mind_mem.served_ledger import ledger_enabled

        (tmp_path / "mind-mem.json").write_text('{"served_ledger": {"enabled": false}}', encoding="utf-8")
        assert ledger_enabled(str(tmp_path)) is False

    def test_the_configuration_table_cell_matches_the_function(self, tmp_path: Path) -> None:
        rows = [
            line
            for line in (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("| `served_ledger.enabled`")
        ]
        assert len(rows) == 1, f"expected exactly one config row for served_ledger.enabled, found {len(rows)}"
        expected = "`true`" if self.default_is_on(tmp_path) else "`false`"
        cells = [c.strip() for c in rows[0].strip("|").split("|")]
        assert cells[2] == expected, f"docs say default {cells[2]}, ledger_enabled computes {expected}"

    def test_no_shipped_docstring_still_calls_it_opt_in(self, tmp_path: Path) -> None:
        """The prose the machine consumer reads back in a verdict.

        ``verify_cli`` printed "served-recall ledger disabled (opt-in)" into
        the report a CI gate reads. A message that describes the wrong
        default is not a comment — it is output.
        """
        if not self.default_is_on(tmp_path):
            pytest.skip("the default is off again; the opt-in wording would be correct")
        offenders = [
            f"{rel}: {phrase!r}"
            for rel in self.PROSE_FILES
            for phrase in self.OPT_IN_PHRASES
            if phrase in (ROOT / rel).read_text(encoding="utf-8")
        ]
        assert not offenders, f"the ledger is on by default; these still describe it as opt-in: {offenders}"

    def test_the_phrase_scan_can_see_a_phrase_that_is_present(self) -> None:
        """Positive control for the scanner in the test above.

        Without it, "no offenders" could equally mean the paths are wrong,
        the files are empty, or the phrase list never matched anything.
        """
        body = (ROOT / "src" / "mind_mem" / "verify_cli.py").read_text(encoding="utf-8")
        assert "served_ledger" in body, "the scanner is reading the wrong file"
        seeded = body.replace("served_ledger", "Off by default", 1)
        assert any(phrase in seeded for phrase in self.OPT_IN_PHRASES), "the phrase list matches nothing at all"

    def test_the_roadmap_records_the_flip(self, tmp_path: Path) -> None:
        if not self.default_is_on(tmp_path):
            pytest.skip("the default is off again")
        roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
        assert "default ON since 5.0.2" in roadmap, "RA.1 still describes the ledger as default OFF"
