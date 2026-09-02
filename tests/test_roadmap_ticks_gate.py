# Copyright 2026 STARGA, Inc.
"""Tests for the roadmap-tick honesty gate (``scripts/check_roadmap_ticks.py``).

Every rule here is written as a POSITIVE CONTROL first: before any test is
allowed to assert that a document is clean, another test proves the same
detector reports a violation that really is present. An "assert no findings"
test on its own is worthless -- it passes just as happily when the detector has
been deleted.

The load-bearing test is :class:`TestTheGateHasTeeth`. It takes the real
``ROADMAP.md``, injects one synthetic violation of each rule, and requires the
gate to go red. Neutering the gate -- emptying ``NOT_SHIPPED_MARKERS``,
loosening ``_CHECKBOX``, widening the honest-partial carve-out to the checkbox
line, or dropping the Open-heading rule -- turns those tests red, which is the
whole point of checking them in.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_roadmap_ticks.py"
ROADMAP = REPO_ROOT / "ROADMAP.md"


def _load_module() -> Any:
    """Import the gate script by path, registered so dataclasses resolve."""
    spec = importlib.util.spec_from_file_location("_check_roadmap_ticks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses looks the class's module up in sys.modules, so a path
    # import that skips this registration dies on the @dataclass line.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_module()


def _rules(text: str) -> list[str]:
    return sorted(f.rule for f in gate.scan_lines(text.splitlines()))


# ---------------------------------------------------------------------------
# Rule 1 -- a tick must not contradict its own sentence
# ---------------------------------------------------------------------------


class TestSelfRefutingTick:
    """POSITIVE CONTROLS for the self-refuting-tick rule."""

    @pytest.mark.parametrize(
        "marker",
        ["not wired", "not added", "not yet", "not shipped", "no such"],
    )
    def test_every_marker_is_caught_on_a_ticked_line(self, marker: str) -> None:
        # Positive control, one per marker: deleting any entry from
        # NOT_SHIPPED_MARKERS turns exactly one of these red.
        findings = gate.scan_lines([f"- [x] **Thing** - the detector chain is {marker}."])
        assert [f.rule for f in findings] == ["self-refuting-tick"]
        assert marker in findings[0].detail

    def test_the_marker_match_is_case_insensitive(self) -> None:
        findings = gate.scan_lines(["- [x] **Thing** - NOT WIRED anywhere."])
        assert [f.rule for f in findings] == ["self-refuting-tick"]

    def test_a_marker_on_a_continuation_line_is_caught(self) -> None:
        text = "- [x] **Thing** - ships everywhere.\n  The flag is not wired to a consumer.\n"
        assert _rules(text) == ["self-refuting-tick"]

    def test_an_unticked_item_may_say_anything(self) -> None:
        # The whole point of an open box is that it can be honest.
        text = "- [ ] **Thing** - not wired, not added, not yet, not shipped, no such module.\n"
        assert gate.scan_lines(text.splitlines()) == []

    def test_a_confident_tick_with_no_marker_is_clean(self) -> None:
        # NEGATIVE case, and it is only meaningful because the parametrized
        # positive control above proves this detector fires at all.
        text = "- [x] **Thing** - ships in `thing.py`, covered by 14 tests.\n"
        assert gate.scan_lines(text.splitlines()) == []


class TestHonestPartialCarveOut:
    """The one exemption in the gate, pinned from both sides."""

    def test_a_remaining_clause_on_a_continuation_line_is_exempt(self) -> None:
        # Mirrors ROADMAP.md T-001: a real capability plus a named remaining
        # sub-part. Flagging it would reward deleting the caveat.
        text = (
            "- [x] **T-001: Content-provenance tags** - shipped with 47 tests.\n"
            "  **Remaining:** individual ingest producers do not yet\n"
            "  stamp the tag themselves.\n"
        )
        assert gate.scan_lines(text.splitlines()) == []

    def test_the_carve_out_does_not_reach_the_checkbox_line(self) -> None:
        # Otherwise "**Remaining:**" would be a one-word gate bypass.
        text = "- [x] **Thing** - **Remaining:** the detector chain is not wired.\n"
        assert _rules(text) == ["self-refuting-tick"]

    def test_text_before_the_label_is_still_in_scope(self) -> None:
        text = "- [x] **Thing** - ships.\n  The flag is not wired. **Remaining:** docs.\n"
        assert _rules(text) == ["self-refuting-tick"]


# ---------------------------------------------------------------------------
# Rule 2 -- a tick must not sit under an Open heading
# ---------------------------------------------------------------------------


class TestTickUnderOpenHeading:
    """POSITIVE CONTROLS for the tick-under-open-heading rule."""

    @pytest.mark.parametrize(
        "heading",
        ["**Open:**", "Open:", "### Open", "**Open (genuine gaps):**", "__Open:__"],
    )
    def test_a_tick_under_an_open_heading_is_caught(self, heading: str) -> None:
        text = f"{heading}\n\n- [x] **Thing** - ships in `thing.py`.\n"
        assert _rules(text) == ["tick-under-open-heading"]

    def test_an_unticked_item_under_open_is_correct(self) -> None:
        text = "**Open:**\n\n- [ ] **Thing** - deferred.\n"
        assert gate.scan_lines(text.splitlines()) == []

    def test_a_shipped_label_closes_the_open_region(self) -> None:
        text = "**Open:**\n\n- [ ] **A** - deferred.\n\n**Shipped:**\n\n- [x] **B** - ships.\n"
        assert gate.scan_lines(text.splitlines()) == []

    def test_a_heading_closes_the_open_region(self) -> None:
        text = "**Open:**\n\n- [ ] **A** - deferred.\n\n### F. Next section\n\n- [x] **B** - ships.\n"
        assert gate.scan_lines(text.splitlines()) == []

    @pytest.mark.parametrize("word", ["OpenAPI", "OpenTelemetry", "OpenClaw"])
    def test_an_open_prefixed_name_is_not_an_open_heading(self, word: str) -> None:
        # The \\b in _OPEN_HEADING. Without it, one "**OpenAPI specs:**"
        # heading turns every later [x] in the file into a finding.
        text = f"**{word} specs:**\n\n- [x] **Thing** - ships in `thing.py`.\n"
        assert gate.scan_lines(text.splitlines()) == []

    def test_but_a_bare_open_heading_still_fires(self) -> None:
        # Positive control for the test directly above: proves the boundary
        # guard narrowed the match instead of disabling the rule.
        text = "**Open specs:**\n\n- [x] **Thing** - ships in `thing.py`.\n"
        assert _rules(text) == ["tick-under-open-heading"]


class TestBothRulesCanFireOnOneLine:
    def test_a_self_refuting_tick_under_open_reports_twice(self) -> None:
        # This is the exact shape of the three 2026-09-01 defects.
        text = "**Open:**\n\n- [x] **Thing** - controlled vocabularies not wired into `validate_block`.\n"
        assert _rules(text) == ["self-refuting-tick", "tick-under-open-heading"]


# ---------------------------------------------------------------------------
# The gate has teeth on the real file
# ---------------------------------------------------------------------------


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


class TestTheGateHasTeeth:
    """Mutation controls: break what the gate guards, watch it go red."""

    def test_the_committed_roadmap_is_clean(self) -> None:
        # Only meaningful alongside the two mutation tests below.
        result = _run("--check", str(ROADMAP))
        assert result.returncode == 0, result.stdout + result.stderr

    def test_injecting_a_self_refuting_tick_turns_the_real_file_red(self, tmp_path: Path) -> None:
        mutated = tmp_path / "ROADMAP.md"
        mutated.write_text(
            ROADMAP.read_text(encoding="utf-8") + "\n- [x] **Injected** - the detector chain is not wired. Tracked.\n",
            encoding="utf-8",
        )
        result = _run("--check", str(mutated))
        assert result.returncode == 1, result.stdout + result.stderr
        assert "self-refuting-tick" in result.stdout

    def test_injecting_a_tick_under_open_turns_the_real_file_red(self, tmp_path: Path) -> None:
        mutated = tmp_path / "ROADMAP.md"
        mutated.write_text(
            ROADMAP.read_text(encoding="utf-8") + "\n**Open:**\n\n- [x] **Injected** - ships. Tracked.\n",
            encoding="utf-8",
        )
        result = _run("--check", str(mutated))
        assert result.returncode == 1, result.stdout + result.stderr
        assert "tick-under-open-heading" in result.stdout

    def test_the_five_retracted_items_stay_retracted(self) -> None:
        """Regression lock on the 2026-09-01 audit.

        Two items carried a confidently-worded false tick that no mechanical
        rule can catch -- the gate above would not notice them coming back, so
        they are pinned here by name.
        """
        text = ROADMAP.read_text(encoding="utf-8")
        for label in ("Pluggable redaction layer", "Compliance export pipeline", "Provenance-rich blocks"):
            ticked = [ln for ln in text.splitlines() if label in ln and ln.lstrip().startswith("- [x]")]
            assert ticked == [], f"{label} is ticked again: {ticked}"


class TestCommandLineContract:
    def test_check_exits_zero_when_clean(self, tmp_path: Path) -> None:
        doc = tmp_path / "clean.md"
        doc.write_text("- [x] **Thing** - ships in `thing.py`.\n", encoding="utf-8")
        result = _run("--check", str(doc))
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_report_mode_exits_zero_but_still_prints(self, tmp_path: Path) -> None:
        doc = tmp_path / "dirty.md"
        doc.write_text("- [x] **Thing** - not wired.\n", encoding="utf-8")
        result = _run(str(doc))
        assert result.returncode == 0
        assert "self-refuting-tick" in result.stdout

    def test_a_missing_file_exits_two_rather_than_passing(self, tmp_path: Path) -> None:
        # A gate that cannot read its target must never report success.
        result = _run("--check", str(tmp_path / "absent.md"))
        assert result.returncode == 2
        assert "no such file" in result.stderr

    def test_the_default_target_is_the_roadmap(self) -> None:
        result = _run("--check")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ROADMAP.md" in result.stdout
