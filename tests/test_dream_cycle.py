#!/usr/bin/env python3
"""Tests for dream_cycle.py — autonomous memory enrichment passes."""

import os
import time
from datetime import datetime, timedelta

import pytest

from mind_mem.dream_cycle import (
    BrokenCitation,
    ConsolidationCandidate,
    DreamCycleReport,
    EntityProposal,
    StaleBlock,
    _format_report_markdown,
    _normalize_line,
    pass_citation_repair,
    pass_consolidation,
    pass_entity_discovery,
    pass_integrity_summary,
    pass_stale_detection,
    run_dream_cycle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path):
    """Create a minimal workspace directory structure."""
    for d in ("memory", "entities", "decisions", "tasks", "intelligence"):
        (tmp_path / d).mkdir(exist_ok=True)
    return str(tmp_path)


def _today_str():
    return datetime.now().strftime("%Y-%m-%d")


def _days_ago_str(days):
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Pass 1: Entity Discovery
# ---------------------------------------------------------------------------


class TestEntityDiscovery:
    """Tests for pass_entity_discovery (Pass 1)."""

    def test_discovers_github_url(self, tmp_path):
        """GitHub repo URLs in recent logs yield entity proposals."""
        ws = _make_workspace(tmp_path)
        today = _today_str()
        log = tmp_path / "memory" / f"{today}.md"
        log.write_text("Check https://github.com/star-ga/newproject for details.\n")

        proposals = pass_entity_discovery(ws)
        slugs = [p.slug for p in proposals]
        assert "newproject" in slugs
        match = next(p for p in proposals if p.slug == "newproject")
        assert match.entity_type == "project"
        assert match.source_pattern == "github_repo"
        assert match.source_file == f"{today}.md"

    def test_discovers_at_mention(self, tmp_path):
        """@handle mentions in recent logs yield person proposals."""
        ws = _make_workspace(tmp_path)
        today = _today_str()
        log = tmp_path / "memory" / f"{today}.md"
        log.write_text("Thanks @alice-dev for the review.\n")

        proposals = pass_entity_discovery(ws)
        slugs = [p.slug for p in proposals]
        assert "alice-dev" in slugs
        match = next(p for p in proposals if p.slug == "alice-dev")
        assert match.entity_type == "person"

    def test_skips_already_tracked_entities(self, tmp_path):
        """Entities already in entities/*.md are not proposed."""
        ws = _make_workspace(tmp_path)
        today = _today_str()
        log = tmp_path / "memory" / f"{today}.md"
        log.write_text("Using https://github.com/star-ga/mind-mem daily.\n")
        entities = tmp_path / "entities" / "projects.md"
        entities.write_text("[PRJ-mind-mem] mind-mem project\n")

        proposals = pass_entity_discovery(ws)
        slugs = [p.slug for p in proposals]
        assert "mind-mem" not in slugs

    def test_skips_old_logs(self, tmp_path):
        """Logs older than lookback_days are not scanned."""
        ws = _make_workspace(tmp_path)
        old_date = _days_ago_str(30)
        log = tmp_path / "memory" / f"{old_date}.md"
        log.write_text("Old project: https://github.com/org/ancient-repo\n")

        proposals = pass_entity_discovery(ws, lookback_days=7)
        assert len(proposals) == 0

    def test_empty_workspace(self, tmp_path):
        """Workspace with no memory/ dir returns empty list."""
        ws = str(tmp_path)
        proposals = pass_entity_discovery(ws)
        assert proposals == []

    def test_deduplicates_across_files(self, tmp_path):
        """Same entity in multiple daily logs is proposed only once."""
        ws = _make_workspace(tmp_path)
        for i in range(3):
            date = _days_ago_str(i)
            log = tmp_path / "memory" / f"{date}.md"
            log.write_text("Always using @bob-dev for reviews.\n")

        proposals = pass_entity_discovery(ws)
        slugs = [p.slug for p in proposals]
        assert slugs.count("bob-dev") == 1


# ---------------------------------------------------------------------------
# Pass 2: Citation Repair
# ---------------------------------------------------------------------------


class TestCitationRepair:
    """Tests for pass_citation_repair (Pass 2)."""

    def test_detects_broken_decision_reference(self, tmp_path):
        """References to nonexistent decision IDs are reported."""
        ws = _make_workspace(tmp_path)
        decisions = tmp_path / "decisions" / "DECISIONS.md"
        decisions.write_text("[D-20260101-001]\nStatement: Valid decision\nStatus: active\n\nSee also D-20260101-999 for context.\n")

        broken = pass_citation_repair(ws)
        cited_ids = [b.cited_id for b in broken]
        assert "D-20260101-999" in cited_ids

    def test_valid_references_not_reported(self, tmp_path):
        """References to existing block IDs are not flagged."""
        ws = _make_workspace(tmp_path)
        decisions = tmp_path / "decisions" / "DECISIONS.md"
        decisions.write_text(
            "[D-20260101-001]\nStatement: First decision\nStatus: active\n\n"
            "[D-20260101-002]\nStatement: Depends on D-20260101-001\nStatus: active\n"
        )

        broken = pass_citation_repair(ws)
        # D-20260101-001 is defined, so reference from D-002 should be valid
        cited_ids = [b.cited_id for b in broken]
        assert "D-20260101-001" not in cited_ids

    def test_broken_task_reference(self, tmp_path):
        """Broken T- references in entity files are detected."""
        ws = _make_workspace(tmp_path)
        entities = tmp_path / "entities" / "projects.md"
        entities.write_text("[PRJ-myproject]\nRelated task: T-20260101-999\n")
        # No tasks file, so T-20260101-999 is broken
        broken = pass_citation_repair(ws)
        cited_ids = [b.cited_id for b in broken]
        assert "T-20260101-999" in cited_ids

    def test_empty_workspace(self, tmp_path):
        """Workspace with no files yields no broken citations."""
        ws = _make_workspace(tmp_path)
        broken = pass_citation_repair(ws)
        assert broken == []

    def test_citation_includes_line_number(self, tmp_path):
        """BrokenCitation includes the correct line number."""
        ws = _make_workspace(tmp_path)
        decisions = tmp_path / "decisions" / "DECISIONS.md"
        decisions.write_text("# Decisions\n\n[D-20260101-001]\nStatement: Valid\nSee D-20260101-888 here\n")

        broken = pass_citation_repair(ws)
        match = next(b for b in broken if b.cited_id == "D-20260101-888")
        assert match.line_number == 5


# ---------------------------------------------------------------------------
# Pass 3: Stale Block Detection
# ---------------------------------------------------------------------------


class TestStaleDetection:
    """Tests for pass_stale_detection (Pass 3)."""

    def test_detects_old_block(self, tmp_path):
        """Blocks with old dates in stale files are flagged."""
        ws = _make_workspace(tmp_path)
        decisions = tmp_path / "decisions" / "DECISIONS.md"
        decisions.write_text("[D-20240101-001]\nStatement: Ancient decision\nStatus: active\n")
        # Set file mtime to 60 days ago
        old_time = time.time() - (60 * 86400)
        os.utime(str(decisions), (old_time, old_time))

        stale = pass_stale_detection(ws, stale_days=30)
        block_ids = [s.block_id for s in stale]
        assert "D-20240101-001" in block_ids

    def test_recent_blocks_not_stale(self, tmp_path):
        """Blocks in recently modified files are not flagged."""
        ws = _make_workspace(tmp_path)
        today_compact = datetime.now().strftime("%Y%m%d")
        decisions = tmp_path / "decisions" / "DECISIONS.md"
        decisions.write_text(f"[D-{today_compact}-001]\nStatement: Fresh decision\nStatus: active\n")

        stale = pass_stale_detection(ws, stale_days=30)
        assert len(stale) == 0

    def test_no_decisions_file(self, tmp_path):
        """Missing decisions file returns empty list."""
        ws = _make_workspace(tmp_path)
        stale = pass_stale_detection(ws)
        assert stale == []


# ---------------------------------------------------------------------------
# Pass 4: Consolidation
# ---------------------------------------------------------------------------


class TestConsolidation:
    """Tests for pass_consolidation (Pass 4)."""

    def test_detects_repeated_fact(self, tmp_path):
        """Fact appearing in 3+ daily logs is a consolidation candidate."""
        ws = _make_workspace(tmp_path)
        fact = "The deployment pipeline uses Docker containers for isolation."
        for i in range(4):
            date = _days_ago_str(i)
            log = tmp_path / "memory" / f"{date}.md"
            log.write_text(f"# {date}\n\n{fact}\n")

        candidates = pass_consolidation(ws, lookback_days=7, min_occurrences=3)
        assert len(candidates) >= 1
        texts = [c.fact_text for c in candidates]
        assert any(fact in t for t in texts)

    def test_unique_lines_not_consolidated(self, tmp_path):
        """Lines appearing only once are not consolidation candidates."""
        ws = _make_workspace(tmp_path)
        for i in range(5):
            date = _days_ago_str(i)
            log = tmp_path / "memory" / f"{date}.md"
            log.write_text(f"# {date}\n\nUnique fact number {i} about something.\n")

        candidates = pass_consolidation(ws, lookback_days=7, min_occurrences=3)
        assert len(candidates) == 0

    def test_short_lines_ignored(self, tmp_path):
        """Lines shorter than 20 chars (normalized) are skipped."""
        ws = _make_workspace(tmp_path)
        short_line = "Yes, confirmed."
        for i in range(5):
            date = _days_ago_str(i)
            log = tmp_path / "memory" / f"{date}.md"
            log.write_text(f"{short_line}\n")

        candidates = pass_consolidation(ws, lookback_days=7, min_occurrences=3)
        assert len(candidates) == 0

    def test_normalization_matches_variants(self, tmp_path):
        """Markdown formatting variants of the same line are consolidated."""
        ws = _make_workspace(tmp_path)
        variants = [
            "The system uses PostgreSQL for persistence layer.",
            "**The system uses PostgreSQL for persistence layer.**",
            "  The system uses PostgreSQL for persistence layer.  ",
        ]
        for i, variant in enumerate(variants):
            date = _days_ago_str(i)
            log = tmp_path / "memory" / f"{date}.md"
            log.write_text(f"# {date}\n\n{variant}\n")

        candidates = pass_consolidation(ws, lookback_days=7, min_occurrences=3)
        assert len(candidates) >= 1

    def test_empty_memory_dir(self, tmp_path):
        """No memory/ dir returns empty list."""
        ws = str(tmp_path)
        candidates = pass_consolidation(ws)
        assert candidates == []


# ---------------------------------------------------------------------------
# Pass 5: Integrity Summary
# ---------------------------------------------------------------------------


class TestIntegritySummary:
    """Tests for pass_integrity_summary (Pass 5) and report formatting."""

    def test_writes_report_file(self, tmp_path):
        """Summary report is written to memory/dream-cycle-YYYY-MM-DD.md."""
        ws = _make_workspace(tmp_path)
        report = DreamCycleReport(
            timestamp="2026-04-10T03:00:00",
            workspace=ws,
            entity_proposals=(EntityProposal("project", "foo", "github_repo", "excerpt", "2026-04-10.md"),),
        )

        pass_integrity_summary(ws, report, dry_run=False)
        today = _today_str()
        report_path = tmp_path / "memory" / f"dream-cycle-{today}.md"
        assert report_path.exists()
        on_disk = report_path.read_text()
        assert "PRJ-foo" in on_disk
        assert "Dream Cycle Report" in on_disk

    def test_dry_run_does_not_write(self, tmp_path):
        """In dry-run mode, no file is written but content is returned."""
        ws = _make_workspace(tmp_path)
        report = DreamCycleReport(
            timestamp="2026-04-10T03:00:00",
            workspace=ws,
        )

        content = pass_integrity_summary(ws, report, dry_run=True)
        assert "Dream Cycle Report" in content
        # No file should exist
        today = _today_str()
        report_path = tmp_path / "memory" / f"dream-cycle-{today}.md"
        assert not report_path.exists()

    def test_format_includes_all_sections(self):
        """Report markdown includes all 4 pass sections."""
        report = DreamCycleReport(
            timestamp="2026-04-10T03:00:00",
            workspace="/tmp/test",
            broken_citations=(BrokenCitation("decisions/DECISIONS.md", "D-20260101-999", 5, "context"),),
            stale_blocks=(StaleBlock("D-20240101-001", "decisions/DECISIONS.md", "2024-01-01", 460),),
            consolidation_candidates=(ConsolidationCandidate("Repeated fact here", 4, ("2026-04-07.md", "2026-04-08.md")),),
            errors=("Pass 1 failed: test error",),
        )

        md = _format_report_markdown(report)
        assert "## Pass 1: Entity Discovery" in md
        assert "## Pass 2: Citation Repair" in md
        assert "D-20260101-999" in md
        assert "## Pass 3: Stale Block Detection" in md
        assert "D-20240101-001" in md
        assert "## Pass 4: Consolidation" in md
        assert "Repeated fact here" in md
        assert "## Errors" in md
        assert "Pass 1 failed: test error" in md


# ---------------------------------------------------------------------------
# DreamCycleReport dataclass
# ---------------------------------------------------------------------------


class TestDreamCycleReport:
    """Tests for the DreamCycleReport dataclass."""

    def test_total_findings(self):
        """total_findings sums all finding categories."""
        report = DreamCycleReport(
            timestamp="2026-04-10",
            workspace="/tmp/test",
            entity_proposals=(
                EntityProposal("project", "a", "github_repo", "x", "f.md"),
                EntityProposal("tool", "b", "cli_tool", "x", "f.md"),
            ),
            broken_citations=(BrokenCitation("f.md", "D-001", 1, "ctx"),),
            stale_blocks=(
                StaleBlock("D-002", "f.md", "2024-01-01", 100),
                StaleBlock("T-003", "t.md", "2024-02-01", 70),
            ),
            consolidation_candidates=(ConsolidationCandidate("fact", 3, ("a.md", "b.md", "c.md")),),
        )
        assert report.total_findings == 6

    def test_empty_report(self):
        """Empty report has zero findings."""
        report = DreamCycleReport(
            timestamp="2026-04-10",
            workspace="/tmp/test",
        )
        assert report.total_findings == 0

    def test_frozen(self):
        """Report is immutable."""
        report = DreamCycleReport(
            timestamp="2026-04-10",
            workspace="/tmp/test",
        )
        with pytest.raises(AttributeError):
            report.timestamp = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Full cycle (run_dream_cycle)
# ---------------------------------------------------------------------------


class TestRunDreamCycle:
    """Tests for the main run_dream_cycle entry point."""

    def test_full_cycle_empty_workspace(self, tmp_path):
        """Full cycle on an empty workspace produces zero findings."""
        ws = _make_workspace(tmp_path)
        report = run_dream_cycle(ws, dry_run=True)
        assert isinstance(report, DreamCycleReport)
        assert report.total_findings == 0
        assert len(report.errors) == 0

    def test_full_cycle_with_findings(self, tmp_path):
        """Full cycle detects entities and writes a summary."""
        ws = _make_workspace(tmp_path)

        # Add a daily log with an entity
        today = _today_str()
        log = tmp_path / "memory" / f"{today}.md"
        log.write_text("New tool: @alice-dev mentioned https://github.com/org/newrepo\n")

        # Add a broken citation
        decisions = tmp_path / "decisions" / "DECISIONS.md"
        decisions.write_text("[D-20260101-001]\nStatement: Decision\nStatus: active\n\nDepends on D-20260101-999\n")

        report = run_dream_cycle(ws, dry_run=False)
        assert report.total_findings > 0
        # At least entities and broken citations
        assert len(report.entity_proposals) >= 1
        assert len(report.broken_citations) >= 1
        # Summary file should exist
        report_path = tmp_path / "memory" / f"dream-cycle-{today}.md"
        assert report_path.exists()

    def test_full_cycle_dry_run(self, tmp_path):
        """Dry-run does not write the summary file."""
        ws = _make_workspace(tmp_path)
        today = _today_str()
        log = tmp_path / "memory" / f"{today}.md"
        log.write_text("Using @some-person for reviews.\n")

        report = run_dream_cycle(ws, dry_run=True)
        assert isinstance(report, DreamCycleReport)
        report_path = tmp_path / "memory" / f"dream-cycle-{today}.md"
        assert not report_path.exists()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestNormalizeLine:
    """Tests for the _normalize_line helper."""

    def test_strips_markdown_header(self):
        assert _normalize_line("## My Header") == "my header"

    def test_strips_formatting(self):
        assert _normalize_line("**bold** and *italic*") == "bold and italic"

    def test_strips_date_prefix(self):
        assert _normalize_line("2026-04-10: Some note") == "some note"

    def test_collapses_whitespace(self):
        assert _normalize_line("  lots   of   space  ") == "lots of space"

    def test_empty_string(self):
        assert _normalize_line("") == ""
