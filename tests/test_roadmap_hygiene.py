# Copyright 2026 STARGA, Inc.
"""Public-artifact hygiene for ``ROADMAP.md`` — rule 3 and the retraction locks.

``tests/test_roadmap_ticks_gate.py`` pins the two mechanical tick rules in
``scripts/check_roadmap_ticks.py``. This file pins what was added on
2026-09-01 alongside them:

* **rule 3, prior-art attribution** — this repository is PUBLIC, and adopted
  external prior art is never credited in a public artifact. ``ROADMAP.md``
  stated that rule in its own prose and then broke it in six places: three
  arXiv identifiers, two paper titles with authors and venue, and two
  third-party repository URLs. A norm nothing checks is a norm that drifts;
* the **``placeholder`` marker** on rule 1, which exists because of a sixth
  false tick the attribution sweep uncovered rather than rule 1;
* the **retraction locks** — four items whose confident wording no mechanical
  rule can refute, pinned by name so they cannot quietly come back ticked;
* the **anti-over-correction locks** — two items that were *accused* of being
  false ticks, verified against running code, and found to be REAL. Retracting
  a true capability to make a sweep look tidy is the same defect with the sign
  flipped, so those ticks are pinned too.

Every assertion about the real file is paired with a control. An
"``ROADMAP.md`` is clean" test on its own passes just as happily when the
detector was deleted, so each one has either a synthetic positive control for
the same detector or a mutation of the real file that must go red.
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
    spec = importlib.util.spec_from_file_location("_check_roadmap_hygiene", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_module()


def _rules(text: str) -> list[str]:
    return sorted(f.rule for f in gate.scan_lines(text.splitlines()))


def _roadmap_text() -> str:
    return ROADMAP.read_text(encoding="utf-8")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Rule 3 -- POSITIVE CONTROLS, one per citation shape
# ---------------------------------------------------------------------------

#: ``(line, the shape it is)``. Deleting any entry from PRIOR_ART_PATTERNS
#: turns exactly one of these red.
_LEAKS: tuple[tuple[str, str], ...] = (
    ("  (arXiv:2504.19874) to cached KV embeddings for 6x memory reduction.", "an arXiv identifier"),
    ('> **Paper:** "Train Any Agent Simply by Talking" (arxiv: 2603.10165)', "an arXiv identifier"),
    ("  See https://arxiv.org/abs/2504.19874 for the derivation.", "an arXiv URL"),
    ("  Published at https://doi.org/10.1145/3641519 — see table 3.", "a DOI URL"),
    ("  The method (doi:10.1145/3641519) reports a 2x speedup.", "a DOI"),
    ("  Reviewed on openreview.net before we adopted the quantile rule.", "a paper-repository URL"),
    ("  Prior art: https://github.com/yale-nlp/RLMF (the reference impl).", "a source-repository URL"),
    ("- [x] **GitNexus** (`github.com/h4ckf0r0day/GitNexus`) — indexer", "a source-repository URL"),
    ("  Ported from gitlab.com/some-lab/some-project, MIT licensed.", "a source-repository URL"),
    ('  Faithful Uncertainty Expression in LLMs"* (Liu et al., Yale NLP)', "an author credit"),
)


class TestPriorArtAttributionIsCaught:
    """POSITIVE CONTROLS for rule 3 — the shapes a citation actually takes."""

    @pytest.mark.parametrize(("line", "shape"), _LEAKS, ids=[s for _, s in _LEAKS])
    def test_every_citation_shape_is_reported(self, line: str, shape: str) -> None:
        findings = gate.scan_attribution([line])
        assert [f.rule for f in findings] == ["prior-art-attribution"], line
        assert shape in findings[0].detail

    def test_a_leak_is_reported_through_the_public_entry_point_too(self) -> None:
        # scan_attribution is the detector; scan_lines is what the CLI calls.
        # A rule wired into the detector but not into scan_lines would ship a
        # green gate over a dirty file -- "imported is not wired".
        assert _rules("  (arXiv:2504.19874) to cached KV embeddings.\n") == ["prior-art-attribution"]

    def test_the_match_is_case_insensitive(self) -> None:
        assert _rules("  ARXIV:2504.19874 is the reference.\n") == ["prior-art-attribution"]

    def test_a_leak_outside_a_checkbox_is_still_reported(self) -> None:
        # The tick rules only look at list items. A citation in a blockquote,
        # a heading or a code fence is the same leak.
        for line in (
            "> **Paper:** something (arXiv:2603.10165)",
            "## Related work (arXiv:2603.10165)",
            "    git clone https://github.com/yale-nlp/RLMF",
        ):
            assert _rules(line + "\n") == ["prior-art-attribution"], line

    def test_one_finding_per_line_even_when_several_shapes_collide(self) -> None:
        # The 2026-09-01 worst case carried an id AND a repo URL on one line.
        # Reporting it once keeps the count equal to the number of clauses an
        # author has to rewrite.
        line = "  arXiv:2606.32032, github.com/yale-nlp/RLMF). Distinction worth stealing:"
        assert len(gate.scan_attribution([line])) == 1


class TestRuleThreeDoesNotOverReach:
    """NEGATIVE cases. Each is only meaningful because the controls above
    prove this detector fires at all."""

    @pytest.mark.parametrize(
        "line",
        [
            # The file's own no-public-attribution policy sentence. If rule 3
            # flagged this, the one line that states the norm would be the
            # first casualty of enforcing it.
            "> Provenance (arxiv id, authors, exact tables) recorded privately in",
            '  say "recent scaling-law research" only.',
            # Domain vocabulary, not a citation.
            "- [x] CI runs on GitHub Actions; see `.github/workflows/ci.yml`",
            "  The GitLab mirror is not maintained.",
            # Self-reference the way this file already writes it.
            "  see `star-ga/mind-nerve`'s ROADMAP for the emit-shared work",
            "- [x] HF upload — `star-ga/mind-mem-4b` v4 revision is the GA pointer",
            # Ordinary source paths, which a bare owner/repo matcher would eat.
            "  `src/mind_mem/turbo_quant.py` has zero consumers.",
            "  docs/design/eval-set-ground-truth.md describes the build.",
            # A version-ish number that is not an arXiv id.
            "  Released 2026-04-13; see CHANGELOG 5.0.1 for the ladder.",
        ],
    )
    def test_a_non_citation_line_is_clean(self, line: str) -> None:
        assert gate.scan_attribution([line]) == []

    def test_the_bare_word_arxiv_needs_an_identifier_to_fire(self) -> None:
        # Positive control for the policy-sentence case directly above: proves
        # the matcher was narrowed to identifiers, not disabled.
        assert gate.scan_attribution(["  arxiv id"]) == []
        assert [f.rule for f in gate.scan_attribution(["  arXiv:2504.19874"])] == ["prior-art-attribution"]


class TestNoCarveOutForOurOwnRepositories:
    """The one place rule 3 could have grown an exemption, pinned shut.

    A host-qualified repository URL is reported whoever owns it. The file's
    existing practice makes the exception unnecessary -- every self-reference
    in ``ROADMAP.md`` is a bare slug or a relative path -- and an exemption
    with no work to do is dead weight that only widens later.
    """

    def test_our_own_org_url_is_reported_like_any_other(self) -> None:
        findings = gate.scan_attribution(["  Client code lives at https://github.com/star-ga/mind-mem."])
        assert [f.rule for f in findings] == ["prior-art-attribution"]

    def test_but_the_bare_slug_form_the_file_actually_uses_is_clean(self) -> None:
        assert gate.scan_attribution(["  see `star-ga/mind-nerve`'s ROADMAP"]) == []

    def test_the_roadmap_still_references_our_repos_by_slug(self) -> None:
        # Proves the test above is describing the real file, not a fiction:
        # the no-carve-out decision costs the file nothing today.
        text = _roadmap_text()
        assert "star-ga/" in text
        assert "github.com/star-ga" not in text


# ---------------------------------------------------------------------------
# Rule 1 -- the "placeholder" marker
# ---------------------------------------------------------------------------


class TestPlaceholderMarker:
    def test_a_ticked_item_calling_itself_a_placeholder_is_caught(self) -> None:
        # The turbo_quant shape: the module docstring said "placeholder", the
        # roadmap said "ships".
        text = "- [x] **Quantized prefix cache** — the format is a placeholder for a real scheme.\n"
        assert _rules(text) == ["self-refuting-tick"]

    def test_an_open_box_may_call_itself_a_placeholder(self) -> None:
        text = "- [ ] **Quantized prefix cache** — the format is a placeholder. Tracked.\n"
        assert gate.scan_lines(text.splitlines()) == []

    def test_the_marker_is_registered_in_the_data_list(self) -> None:
        # Removing it from NOT_SHIPPED_MARKERS turns the first test red too;
        # this one names the reason, so the failure is self-explaining.
        assert "placeholder" in gate.NOT_SHIPPED_MARKERS


# ---------------------------------------------------------------------------
# The real file: clean, and provably so
# ---------------------------------------------------------------------------


class TestTheRealRoadmapIsAttributionClean:
    def test_the_committed_roadmap_carries_no_attribution(self) -> None:
        # Only meaningful alongside the mutation control below.
        findings = gate.scan_attribution(_roadmap_text().splitlines(), path=str(ROADMAP))
        assert findings == [], "\n".join(f.render() for f in findings)

    @pytest.mark.parametrize(("line", "shape"), _LEAKS, ids=[s for _, s in _LEAKS])
    def test_injecting_each_shape_into_the_real_file_turns_it_red(self, line: str, shape: str, tmp_path: Path) -> None:
        # MUTATION CONTROL: the file is clean because the leaks were removed,
        # not because the scan cannot see this file.
        mutated = tmp_path / "ROADMAP.md"
        mutated.write_text(_roadmap_text() + "\n" + line + "\n", encoding="utf-8")
        result = _run("--check", str(mutated))
        assert result.returncode == 1, result.stdout + result.stderr
        assert "prior-art-attribution" in result.stdout
        assert shape in result.stdout

    def test_the_policy_sentence_that_the_rule_mechanises_survives(self) -> None:
        """Rule 3 exists because this sentence was true and unenforced.

        Erasing the norm is the one repair that would clear every rule-3
        finding forever while making the file less honest, so both halves of
        it are pinned: the rule as stated, and the wording it prescribes in
        place of a citation. A bare ``"no-public-attribution" in text`` check
        is too weak -- the phrase survives on wrapped continuation lines even
        after the statement itself has been rewritten away.
        """
        text = _roadmap_text()
        assert "per the no-public-attribution rule" in text
        assert 'say "recent scaling-law research" only' in text

    def test_the_cli_reports_the_third_rule_in_its_clean_banner(self) -> None:
        result = _run("--check", str(ROADMAP))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "prior-art attribution" in result.stdout


# ---------------------------------------------------------------------------
# Retraction locks and anti-over-correction locks
# ---------------------------------------------------------------------------


def _ticked_lines(label: str) -> list[str]:
    return [ln for ln in _roadmap_text().splitlines() if label in ln and ln.lstrip().startswith("- [x]")]


def _open_lines(label: str) -> list[str]:
    return [ln for ln in _roadmap_text().splitlines() if label in ln and ln.lstrip().startswith("- [ ]")]


class TestRetractedItemsStayRetracted:
    """Four false ticks whose wording no mechanical rule can refute.

    Rules 1-3 cannot catch a confidently-worded lie -- "ships under the
    redaction module" reads exactly like a true sentence. These are pinned by
    name, with the verification that retired each one recorded beside it.
    """

    @pytest.mark.parametrize(
        ("label", "why"),
        [
            (
                "Pluggable redaction layer",
                "no v4/redaction.py, no pre-write detector chain in src/, flag has zero consumers",
            ),
            (
                "Compliance export pipeline",
                "no `mm export` verb, no --policy option anywhere in src/, flag has zero consumers",
            ),
            (
                "Provenance-rich blocks",
                "the five fields exist; the off|recommended|required policy has zero occurrences in src/",
            ),
            (
                "Quantized prefix cache",
                "prefix_cache caches responses not embeddings; turbo_quant is a placeholder with no consumers",
            ),
        ],
    )
    def test_the_item_is_not_ticked(self, label: str, why: str) -> None:
        assert _ticked_lines(label) == [], f"{label} is ticked again ({why}): {_ticked_lines(label)}"

    @pytest.mark.parametrize(
        "label",
        [
            "Pluggable redaction layer",
            "Compliance export pipeline",
            "Provenance-rich blocks",
            "Quantized prefix cache",
        ],
    )
    def test_the_item_is_still_present_as_an_open_box(self, label: str) -> None:
        # POSITIVE CONTROL for the assertion above, and deletion discipline in
        # test form: "not ticked" also passes when somebody deleted the line.
        # Retracting a tick must keep the capability on the roadmap.
        assert _open_lines(label), f"{label} vanished from ROADMAP.md instead of being retracted"


class TestVerifiedRealItemsStayTicked:
    """Two items accused of being false ticks that verification found REAL.

    Both carried self-refuting *text* over working *code*: the code was right
    and the sentence was wrong. The honest repair is to fix the sentence, and
    the failure mode this class guards is the tidy-looking one -- clearing a
    box for a capability that ships.
    """

    @pytest.mark.parametrize(
        ("label", "evidence"),
        [
            (
                "Auto-generated hierarchical index",
                "memory_index.generate_index writes index.md + log.md; `mm index` runs it; 14 tests",
            ),
            (
                "Vocabulary-bound fields",
                "v4/vocabulary enforced by block_metadata.validate_block on the propose_update door",
            ),
        ],
    )
    def test_the_item_is_ticked(self, label: str, evidence: str) -> None:
        assert _ticked_lines(label), f"{label} was retracted, but it ships ({evidence})"

    def test_the_capability_behind_the_index_tick_actually_runs(self) -> None:
        # The tick is only honest while the code is. This is the cheapest
        # standing proof that the roadmap sentence is backed by a call path.
        from mind_mem.memory_index import INDEX_FILENAME, LOG_FILENAME, generate_index

        assert callable(generate_index)
        assert INDEX_FILENAME == "index.md"
        assert LOG_FILENAME == "log.md"

    def test_the_vocabulary_tick_names_the_enforcing_symbol(self) -> None:
        # A tick that names its enforcing symbol can be checked by the next
        # reader in one grep; one that says "ships" cannot.
        line = _ticked_lines("Vocabulary-bound fields")[-1]
        assert "validate_block" in line
        assert "propose_update" in line
