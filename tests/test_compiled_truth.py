"""Tests for mind-mem compiled truth pages (compiled_truth.py)."""

import os

import pytest

from mind_mem.compiled_truth import (
    VALID_CONFIDENCE_LEVELS,
    VALID_ENTITY_TYPES,
    CompiledTruthPage,
    EvidenceEntry,
    add_evidence,
    detect_contradictions,
    format_truth_page,
    load_truth_page,
    parse_truth_page,
    recompile_truth,
    save_truth_page,
    scan_for_promotable_facts,
    supersede_evidence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace with required directories."""
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "entities", "compiled"), exist_ok=True)
    os.makedirs(os.path.join(ws, "memory"), exist_ok=True)
    return ws


def _make_page(
    entity_id: str = "PRJ-test",
    entity_type: str = "project",
    compiled_section: str = "Test project is a test.",
    evidence: list[EvidenceEntry] | None = None,
    last_compiled: str = "2026-04-10T00:00:00+00:00",
    version: int = 1,
) -> CompiledTruthPage:
    return CompiledTruthPage(
        entity_id=entity_id,
        entity_type=entity_type,
        compiled_section=compiled_section,
        evidence_entries=evidence or [],
        last_compiled=last_compiled,
        version=version,
    )


def _make_entry(
    timestamp: str = "2026-04-10T12:00:00+00:00",
    source: str = "memory/2026-04-10.md",
    observation: str = "Observed something important.",
    confidence: str = "high",
    superseded: bool = False,
) -> EvidenceEntry:
    return EvidenceEntry(
        timestamp=timestamp,
        source=source,
        observation=observation,
        confidence=confidence,
        superseded=superseded,
    )


# ---------------------------------------------------------------------------
# EvidenceEntry dataclass tests
# ---------------------------------------------------------------------------


class TestEvidenceEntry:
    def test_defaults(self):
        entry = EvidenceEntry(
            timestamp="2026-01-01T00:00:00+00:00",
            source="session-1",
            observation="A fact.",
            confidence="high",
        )
        assert entry.superseded is False

    def test_frozen(self):
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.superseded = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CompiledTruthPage dataclass tests
# ---------------------------------------------------------------------------


class TestCompiledTruthPage:
    def test_fields(self):
        page = _make_page()
        assert page.entity_id == "PRJ-test"
        assert page.entity_type == "project"
        assert page.version == 1
        assert page.evidence_entries == []

    def test_frozen(self):
        page = _make_page()
        with pytest.raises(AttributeError):
            page.version = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# format_truth_page / parse_truth_page round-trip
# ---------------------------------------------------------------------------


class TestFormatParse:
    def test_round_trip_empty_evidence(self):
        page = _make_page()
        md = format_truth_page(page)
        restored = parse_truth_page(md)
        assert restored.entity_id == page.entity_id
        assert restored.entity_type == page.entity_type
        assert restored.compiled_section == page.compiled_section
        assert restored.version == page.version
        assert restored.evidence_entries == []

    def test_round_trip_with_evidence(self):
        entries = [
            _make_entry(timestamp="2026-04-09T10:00:00+00:00", observation="First obs."),
            _make_entry(
                timestamp="2026-04-10T10:00:00+00:00",
                observation="Second obs.",
                confidence="medium",
                superseded=True,
            ),
        ]
        page = _make_page(evidence=entries)
        md = format_truth_page(page)
        restored = parse_truth_page(md)

        assert len(restored.evidence_entries) == 2
        assert restored.evidence_entries[0].observation == "First obs."
        assert restored.evidence_entries[0].superseded is False
        assert restored.evidence_entries[1].observation == "Second obs."
        assert restored.evidence_entries[1].confidence == "medium"
        assert restored.evidence_entries[1].superseded is True

    def test_format_contains_sections(self):
        page = _make_page(compiled_section="The compiled truth here.")
        md = format_truth_page(page)
        assert "## Current Understanding" in md
        assert "## Evidence Trail" in md
        assert "The compiled truth here." in md
        assert "entity_id: PRJ-test" in md

    def test_parse_missing_frontmatter_raises(self):
        with pytest.raises(ValueError, match="frontmatter"):
            parse_truth_page("# No frontmatter here\n\nJust text.")

    def test_parse_missing_required_field_raises(self):
        md = "---\nentity_id: X\n---\n\n# X\n"
        with pytest.raises(ValueError, match="entity_type"):
            parse_truth_page(md)


# ---------------------------------------------------------------------------
# load_truth_page / save_truth_page
# ---------------------------------------------------------------------------


class TestLoadSave:
    def test_save_and_load(self, workspace):
        page = _make_page(entity_id="PRJ-roundtrip")
        path = save_truth_page(workspace, page)
        assert os.path.isfile(path)
        assert path.endswith("PRJ-roundtrip.md")

        loaded = load_truth_page(workspace, "PRJ-roundtrip")
        assert loaded is not None
        assert loaded.entity_id == "PRJ-roundtrip"
        assert loaded.version == page.version

    def test_load_nonexistent_returns_none(self, workspace):
        assert load_truth_page(workspace, "DOES-NOT-EXIST") is None

    def test_save_creates_directory(self, tmp_path):
        ws = str(tmp_path / "fresh")
        page = _make_page(entity_id="PRJ-newdir")
        path = save_truth_page(ws, page)
        assert os.path.isfile(path)

    def test_save_with_evidence_round_trips(self, workspace):
        entry = _make_entry(observation="Saved evidence.")
        page = add_evidence(_make_page(entity_id="PRJ-ev"), entry)
        save_truth_page(workspace, page)
        loaded = load_truth_page(workspace, "PRJ-ev")
        assert loaded is not None
        assert len(loaded.evidence_entries) == 1
        assert loaded.evidence_entries[0].observation == "Saved evidence."


# ---------------------------------------------------------------------------
# add_evidence
# ---------------------------------------------------------------------------


class TestAddEvidence:
    def test_appends_entry(self):
        page = _make_page()
        entry = _make_entry()
        new_page = add_evidence(page, entry)
        assert len(new_page.evidence_entries) == 1
        assert new_page.evidence_entries[0] is entry

    def test_original_unchanged(self):
        page = _make_page()
        entry = _make_entry()
        add_evidence(page, entry)
        assert len(page.evidence_entries) == 0

    def test_invalid_confidence_raises(self):
        page = _make_page()
        entry = _make_entry(confidence="very_high")
        with pytest.raises(ValueError, match="confidence"):
            add_evidence(page, entry)

    def test_multiple_appends(self):
        page = _make_page()
        for i in range(5):
            page = add_evidence(page, _make_entry(observation=f"Obs {i}"))
        assert len(page.evidence_entries) == 5
        assert page.evidence_entries[4].observation == "Obs 4"


# ---------------------------------------------------------------------------
# supersede_evidence
# ---------------------------------------------------------------------------


class TestSupersedeEvidence:
    def test_marks_superseded(self):
        page = _make_page(evidence=[_make_entry()])
        new_page = supersede_evidence(page, 0, "outdated")
        assert new_page.evidence_entries[0].superseded is True

    def test_original_unchanged(self):
        page = _make_page(evidence=[_make_entry()])
        supersede_evidence(page, 0, "outdated")
        assert page.evidence_entries[0].superseded is False

    def test_index_out_of_range_raises(self):
        page = _make_page(evidence=[_make_entry()])
        with pytest.raises(IndexError):
            supersede_evidence(page, 5, "bad index")

    def test_negative_index_raises(self):
        page = _make_page(evidence=[_make_entry()])
        with pytest.raises(IndexError):
            supersede_evidence(page, -1, "negative")

    def test_already_superseded_returns_same(self):
        entry = _make_entry(superseded=True)
        page = _make_page(evidence=[entry])
        result = supersede_evidence(page, 0, "already done")
        assert result is page


# ---------------------------------------------------------------------------
# recompile_truth
# ---------------------------------------------------------------------------


class TestRecompileTruth:
    def test_increments_version(self):
        page = _make_page(version=3, evidence=[_make_entry()])
        new_page = recompile_truth(page)
        assert new_page.version == 4

    def test_updates_last_compiled(self):
        page = _make_page(evidence=[_make_entry()])
        new_page = recompile_truth(page)
        assert new_page.last_compiled != page.last_compiled
        assert "T" in new_page.last_compiled  # ISO format

    def test_excludes_superseded(self):
        entries = [
            _make_entry(observation="Active fact.", superseded=False),
            _make_entry(observation="Old fact.", superseded=True),
        ]
        page = _make_page(evidence=entries)
        new_page = recompile_truth(page)
        assert "Active fact." in new_page.compiled_section
        assert "Old fact." not in new_page.compiled_section

    def test_empty_evidence_produces_placeholder(self):
        page = _make_page(evidence=[])
        new_page = recompile_truth(page)
        assert "No active evidence" in new_page.compiled_section

    def test_original_unchanged(self):
        page = _make_page(version=1, evidence=[_make_entry()])
        recompile_truth(page)
        assert page.version == 1


# ---------------------------------------------------------------------------
# detect_contradictions
# ---------------------------------------------------------------------------


class TestDetectContradictions:
    def test_no_contradictions_in_consistent_evidence(self):
        entries = [
            _make_entry(observation="The system uses Python 3.12."),
            _make_entry(observation="Tests are written with pytest."),
        ]
        page = _make_page(evidence=entries)
        assert detect_contradictions(page) == []

    def test_negation_asymmetry_detected(self):
        entries = [
            _make_entry(observation="The module uses caching for the query engine."),
            _make_entry(observation="The module does not use caching for the query engine."),
        ]
        page = _make_page(evidence=entries)
        contradictions = detect_contradictions(page)
        assert len(contradictions) >= 1
        assert "egation" in contradictions[0][2]  # "Negation asymmetry"

    def test_antonym_detected(self):
        entries = [
            _make_entry(observation="The build process will increase the cache size."),
            _make_entry(observation="The build process will decrease the cache size."),
        ]
        page = _make_page(evidence=entries)
        contradictions = detect_contradictions(page)
        assert len(contradictions) >= 1
        assert "ntonym" in contradictions[0][2]  # "Antonym pair"

    def test_superseded_entries_excluded(self):
        entries = [
            _make_entry(observation="The module uses caching for the query engine.", superseded=True),
            _make_entry(observation="The module does not use caching for the query engine."),
        ]
        page = _make_page(evidence=entries)
        assert detect_contradictions(page) == []

    def test_empty_evidence(self):
        page = _make_page(evidence=[])
        assert detect_contradictions(page) == []


# ---------------------------------------------------------------------------
# scan_for_promotable_facts
# ---------------------------------------------------------------------------


class TestScanForPromotableFacts:
    def test_finds_repeated_fact(self, workspace):
        fact = "The mind-mem package uses hybrid BM25 and vector search for recall."
        for day in ("2026-04-08.md", "2026-04-09.md", "2026-04-10.md"):
            path = os.path.join(workspace, "memory", day)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Daily Log\n\n{fact}\n")

        results = scan_for_promotable_facts(workspace, min_mentions=3)
        assert len(results) >= 1
        assert results[0]["mentions"] >= 3
        assert len(results[0]["sources"]) >= 3

    def test_below_threshold_excluded(self, workspace):
        fact = "This fact appears only twice in the logs."
        for day in ("2026-04-08.md", "2026-04-09.md"):
            path = os.path.join(workspace, "memory", day)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Daily Log\n\n{fact}\n")

        results = scan_for_promotable_facts(workspace, min_mentions=3)
        assert len(results) == 0

    def test_no_memory_dir_returns_empty(self, tmp_path):
        results = scan_for_promotable_facts(str(tmp_path), min_mentions=1)
        assert results == []

    def test_short_sentences_ignored(self, workspace):
        for day in ("2026-04-08.md", "2026-04-09.md", "2026-04-10.md"):
            path = os.path.join(workspace, "memory", day)
            with open(path, "w", encoding="utf-8") as f:
                f.write("Short.\n")

        results = scan_for_promotable_facts(workspace, min_mentions=1)
        assert len(results) == 0

    def test_results_sorted_by_mentions(self, workspace):
        fact_a = "Fact A appears in many daily memory log files for testing."
        fact_b = "Fact B appears in even more daily memory log files for testing."
        for day in ("2026-04-07.md", "2026-04-08.md", "2026-04-09.md"):
            path = os.path.join(workspace, "memory", day)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{fact_a}\n")

        for day in ("2026-04-06.md", "2026-04-07.md", "2026-04-08.md", "2026-04-09.md"):
            path = os.path.join(workspace, "memory", day)
            with open(path, "a") as f:
                f.write(f"\n{fact_b}\n")

        results = scan_for_promotable_facts(workspace, min_mentions=3)
        assert len(results) >= 2
        assert results[0]["mentions"] >= results[1]["mentions"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_valid_confidence_levels(self):
        assert "high" in VALID_CONFIDENCE_LEVELS
        assert "medium" in VALID_CONFIDENCE_LEVELS
        assert "low" in VALID_CONFIDENCE_LEVELS

    def test_valid_entity_types(self):
        assert "project" in VALID_ENTITY_TYPES
        assert "person" in VALID_ENTITY_TYPES
        assert "tool" in VALID_ENTITY_TYPES
        assert "topic" in VALID_ENTITY_TYPES
