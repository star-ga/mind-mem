"""Tests for validate_py.py — workspace integrity validator."""

from __future__ import annotations

import json
import os
import sys

import pytest

# Ensure scripts/ is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

from validate_py import Validator  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def empty_workspace(tmp_path):
    """Workspace with mind-mem.json but no data files."""
    (tmp_path / "mind-mem.json").write_text(json.dumps({"version": "1.1.0"}))
    return str(tmp_path)


@pytest.fixture
def minimal_workspace(tmp_path):
    """Workspace with all required structure for a passing validation."""
    (tmp_path / "mind-mem.json").write_text(json.dumps({"version": "1.1.0"}))

    # Create directories
    for d in ["decisions", "tasks", "entities", "intelligence",
              "intelligence/proposed", "summaries/weekly", "memory", "maintenance"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    # File structure requirements
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        "# Decisions\n\n"
        "[D-20260101-001]\n"
        "Date: 2026-01-01\n"
        "Status: active\n"
        "Scope: global\n"
        "Statement: Test decision\n"
        "Rationale: Testing\n"
        "Supersedes: none\n"
        "Tags: test\n"
        "Sources: manual\n"
    )
    (tmp_path / "tasks" / "TASKS.md").write_text(
        "# Tasks\n\n"
        "[T-20260101-001]\n"
        "Date: 2026-01-01\n"
        "Title: Test task\n"
        "Status: todo\n"
        "Priority: P1\n"
        "Project: test\n"
        "Due: 2026-02-01\n"
        "Owner: user\n"
        "Context: test\n"
        "Next: complete it\n"
        "Dependencies: none\n"
        "Sources: manual\n"
        "History: created\n"
    )
    (tmp_path / "entities" / "projects.md").write_text(
        "[PRJ-test]\nName: Test Project\n"
    )
    (tmp_path / "entities" / "people.md").write_text(
        "[PER-alice]\nName: Alice\n"
    )
    (tmp_path / "entities" / "tools.md").write_text(
        "[TOOL-pytest]\nName: pytest\n"
    )
    (tmp_path / "entities" / "incidents.md").write_text(
        "[INC-20260101-test]\n"
        "Date: 2026-01-01\n"
        "Title: Test incident\n"
        "Impact: low\n"
        "Summary: test\n"
        "RootCause: testing\n"
        "Fix: fixed\n"
        "Prevention: tests\n"
        "Sources: manual\n"
    )
    (tmp_path / "memory" / "maint-state.json").write_text("{}")
    (tmp_path / "memory" / "intel-state.json").write_text("{}")
    (tmp_path / "summaries" / "weekly" / "2026-W01.md").write_text("# Week 1\n")

    # MEMORY.md with protocol header
    (tmp_path / "MEMORY.md").write_text("# Memory Protocol v1.0\n\nTest memory.\n")

    # Intelligence files
    for f in ["SIGNALS.md", "CONTRADICTIONS.md", "DRIFT.md",
              "IMPACT.md", "BRIEFINGS.md", "AUDIT.md"]:
        (tmp_path / "intelligence" / f).write_text(f"# {f}\n")

    return str(tmp_path)


# ── Validator basic ──────────────────────────────────────────────────


class TestValidatorBasic:
    def test_init(self, empty_workspace):
        v = Validator(empty_workspace)
        assert v.checks == 0
        assert v.passed == 0
        assert v.issues == 0
        assert v.warnings == 0
        assert v.lines == []

    def test_pass_increments(self, empty_workspace):
        v = Validator(empty_workspace)
        v.pass_("test check")
        assert v.checks == 1
        assert v.passed == 1
        assert v.issues == 0

    def test_fail_increments(self, empty_workspace):
        v = Validator(empty_workspace)
        v.fail("test failure")
        assert v.checks == 1
        assert v.passed == 0
        assert v.issues == 1

    def test_warn_increments(self, empty_workspace):
        v = Validator(empty_workspace)
        v.warn("test warning")
        assert v.warnings == 1
        assert v.checks == 0  # warnings don't count as checks

    def test_log_appends(self, empty_workspace):
        v = Validator(empty_workspace)
        v.log("line 1")
        v.log("line 2")
        assert v.lines == ["line 1", "line 2"]

    def test_section(self, empty_workspace):
        v = Validator(empty_workspace)
        v.section("Test Section")
        assert "=== Test Section ===" in v.lines[-1]


# ── file_exists ──────────────────────────────────────────────────────


class TestFileExists:
    def test_existing_file(self, empty_workspace):
        v = Validator(empty_workspace)
        # mind-mem.json exists in the workspace
        assert v.file_exists("mind-mem.json") is True
        assert v.passed == 1

    def test_missing_file(self, empty_workspace):
        v = Validator(empty_workspace)
        assert v.file_exists("nonexistent.txt") is False
        assert v.issues == 1

    def test_custom_label(self, empty_workspace):
        v = Validator(empty_workspace)
        v.file_exists("nonexistent.txt", label="Custom Label")
        assert "Custom Label MISSING" in v.lines[-1]


# ── run() — no workspace ─────────────────────────────────────────────


class TestRunNoWorkspace:
    def test_no_mind_mem_json(self, tmp_path):
        """Workspace without mind-mem.json should return 1."""
        v = Validator(str(tmp_path))
        result = v.run()
        assert result == 1


# ── run() — empty workspace ──────────────────────────────────────────


class TestRunEmptyWorkspace:
    def test_empty_workspace_has_issues(self, empty_workspace):
        v = Validator(empty_workspace)
        result = v.run()
        assert result == 1  # Missing required files
        assert v.issues > 0

    def test_writes_report(self, empty_workspace):
        v = Validator(empty_workspace)
        v.run()
        report_path = os.path.join(empty_workspace, "maintenance", "validation-report.txt")
        assert os.path.isfile(report_path)


# ── run() — minimal valid workspace ─────────────────────────────────


class TestRunMinimalWorkspace:
    def test_minimal_workspace_passes(self, minimal_workspace):
        v = Validator(minimal_workspace)
        result = v.run()
        assert result == 0  # All checks pass
        assert v.issues == 0

    def test_all_sections_run(self, minimal_workspace):
        v = Validator(minimal_workspace)
        v.run()
        section_headers = [l for l in v.lines if l.startswith("===")]
        assert len(section_headers) >= 7  # Sections 0-6

    def test_report_contains_totals(self, minimal_workspace):
        v = Validator(minimal_workspace)
        v.run()
        report = "\n".join(v.lines)
        assert "TOTAL:" in report
        assert "checks" in report
        assert "passed" in report


# ── _check_file_structure ────────────────────────────────────────────


class TestCheckFileStructure:
    def test_missing_decisions(self, empty_workspace):
        v = Validator(empty_workspace)
        v._check_file_structure()
        assert v.issues > 0

    def test_missing_memory_md(self, empty_workspace):
        v = Validator(empty_workspace)
        v._check_file_structure()
        fails = [l for l in v.lines if "MEMORY.md MISSING" in l]
        assert len(fails) == 1

    def test_memory_md_without_protocol_header(self, empty_workspace):
        """MEMORY.md without Protocol v1.0 header should fail."""
        mem_path = os.path.join(empty_workspace, "MEMORY.md")
        with open(mem_path, "w") as f:
            f.write("# Just some notes\n")
        v = Validator(empty_workspace)
        v._check_file_structure()
        fails = [l for l in v.lines if "Protocol v1.0 header" in l]
        assert len(fails) == 1


# ── _check_decisions ─────────────────────────────────────────────────


class TestCheckDecisions:
    def test_valid_decision_passes(self, minimal_workspace):
        v = Validator(minimal_workspace)
        v._check_decisions()
        assert v.issues == 0

    def test_bad_decision_id(self, tmp_path):
        (tmp_path / "mind-mem.json").write_text("{}")
        (tmp_path / "decisions").mkdir()
        (tmp_path / "decisions" / "DECISIONS.md").write_text(
            "[BAD-ID]\nDate: 2026-01-01\nStatus: active\n"
            "Scope: global\nStatement: x\nRationale: x\n"
            "Supersedes: none\nTags: x\nSources: x\n"
        )
        v = Validator(str(tmp_path))
        v._check_decisions()
        assert v.issues > 0

    def test_invalid_scope(self, tmp_path):
        (tmp_path / "mind-mem.json").write_text("{}")
        (tmp_path / "decisions").mkdir()
        (tmp_path / "decisions" / "DECISIONS.md").write_text(
            "[D-20260101-001]\nDate: 2026-01-01\nStatus: active\n"
            "Scope: invalid-scope\nStatement: x\nRationale: x\n"
            "Supersedes: none\nTags: x\nSources: x\n"
        )
        v = Validator(str(tmp_path))
        v._check_decisions()
        fails = [l for l in v.lines if "invalid Scope" in l]
        assert len(fails) == 1


# ── _check_tasks ─────────────────────────────────────────────────────


class TestCheckTasks:
    def test_valid_task_passes(self, minimal_workspace):
        v = Validator(minimal_workspace)
        v._check_tasks()
        assert v.issues == 0

    def test_invalid_priority(self, tmp_path):
        (tmp_path / "mind-mem.json").write_text("{}")
        (tmp_path / "tasks").mkdir()
        (tmp_path / "tasks" / "TASKS.md").write_text(
            "[T-20260101-001]\nDate: 2026-01-01\nTitle: x\n"
            "Status: todo\nPriority: P9\nProject: x\n"
            "Due: 2026-02-01\nOwner: user\nContext: x\n"
            "Next: x\nDependencies: none\nSources: x\nHistory: x\n"
        )
        v = Validator(str(tmp_path))
        v._check_tasks()
        fails = [l for l in v.lines if "invalid Priority" in l]
        assert len(fails) == 1


# ── _check_entities ──────────────────────────────────────────────────


class TestCheckEntities:
    def test_valid_entities_pass(self, minimal_workspace):
        v = Validator(minimal_workspace)
        v._check_entities()
        assert v.issues == 0


# ── _check_provenance ────────────────────────────────────────────────


class TestCheckProvenance:
    def test_valid_provenance(self, minimal_workspace):
        v = Validator(minimal_workspace)
        v._check_provenance()
        assert v.issues == 0

    def test_missing_sources(self, tmp_path):
        (tmp_path / "mind-mem.json").write_text("{}")
        (tmp_path / "decisions").mkdir()
        (tmp_path / "decisions" / "DECISIONS.md").write_text(
            "[D-20260101-001]\nDate: 2026-01-01\nStatus: active\n"
            "Scope: global\nStatement: x\nRationale: x\n"
            "Supersedes: none\nTags: x\n"  # No Sources field
        )
        v = Validator(str(tmp_path))
        v._check_provenance()
        assert v.issues > 0


# ── _check_cross_refs ────────────────────────────────────────────────


class TestCheckCrossRefs:
    def test_all_refs_resolve(self, minimal_workspace):
        v = Validator(minimal_workspace)
        v._check_cross_refs()
        assert v.issues == 0

    def test_dangling_ref(self, minimal_workspace):
        """Reference to nonexistent ID should fail."""
        # Add a decision that references a nonexistent task
        dec_path = os.path.join(minimal_workspace, "decisions", "DECISIONS.md")
        with open(dec_path, "a") as f:
            f.write(
                "\n[D-20260101-002]\n"
                "Date: 2026-01-01\n"
                "Status: active\n"
                "Scope: global\n"
                "Statement: See T-20261231-999 for details\n"
                "Rationale: testing\n"
                "Supersedes: none\n"
                "Tags: test\n"
                "Sources: manual\n"
            )
        v = Validator(minimal_workspace)
        v._check_cross_refs()
        fails = [l for l in v.lines if "MISSING" in l and "T-20261231-999" in l]
        assert len(fails) == 1


# ── _check_intelligence ──────────────────────────────────────────────


class TestCheckIntelligence:
    def test_all_intel_files_present(self, minimal_workspace):
        v = Validator(minimal_workspace)
        v._check_intelligence()
        assert v.issues == 0
        assert v.warnings == 0

    def test_missing_intel_files_warn(self, empty_workspace):
        v = Validator(empty_workspace)
        v._check_intelligence()
        # Should have warnings for missing intelligence files
        assert v.warnings > 0
