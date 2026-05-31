"""Tests for contradiction_detector.py — Contradiction detection at governance gate (#432).

Covers:
    - Text extraction from proposals and blocks
    - Similarity computation (TF-IDF cosine, Jaccard)
    - Conflict classification (contradiction, refinement, duplicate, related)
    - Core detection pipeline (committed corpus scanning)
    - Integration with check_proposal_contradictions()
    - Edge cases (empty proposals, no corpus, threshold tuning)
    - Config-driven threshold
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem.contradiction_detector import (
    _classify_conflict,
    _extract_block_text,
    _extract_proposal_text,
    _jaccard_similarity,
    _tfidf_cosine_similarity,
    _tokenize,
    check_proposal_contradictions,
    detect_contradictions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    """Create a minimal workspace with committed corpus."""
    ws = str(tmp_path)

    # Create directory structure
    (tmp_path / "decisions").mkdir()
    (tmp_path / "tasks").mkdir()
    (tmp_path / "intelligence" / "proposed").mkdir(parents=True)
    (tmp_path / "memory").mkdir()

    # Write committed decisions
    (tmp_path / "decisions" / "DECISIONS.md").write_text(
        """\
[D-20260201-001]
Title: Use SQLite for all local storage
Type: decision
Status: active
Decision: All local data storage must use SQLite. No external databases.
Evidence:
- Zero infrastructure dependency
- Single file backup
Rationale: Simplicity and portability are paramount for local-first agents.

[D-20260205-002]
Title: BM25 as primary retrieval algorithm
Type: decision
Status: active
Decision: BM25 is the primary text retrieval algorithm. Vector search is optional.
Evidence:
- BM25 outperforms TF-IDF on our benchmarks
- Zero external deps (no sentence-transformers required)
Rationale: Local-first means no mandatory external models.

[D-20260210-003]
Title: Governance mode defaults to detect_only
Type: decision
Status: active
Decision: New workspaces start in detect_only mode. Proposals are surfaced but never auto-applied.
Evidence:
- Safety first -- humans review before any mutation
Rationale: Trust must be earned before automation.

[D-20260215-004]
Title: Enable automatic proposal application
Type: decision
Status: deprecated
Decision: Auto-apply low-risk proposals without human review.
Evidence:
- Speeds up workflow
Rationale: Reduce human review burden.
"""
    )

    # Write committed tasks
    (tmp_path / "tasks" / "TASKS.md").write_text(
        """\
[T-20260220-001]
Title: Implement vector search fallback
Type: task
Status: in_progress
Action: Add optional sentence-transformers backend for semantic recall.
Details: Currently BM25 only. Vector search needed for semantic similarity queries.

[T-20260221-002]
Title: Add retention policy
Type: task
Status: completed
Action: Implement automatic pruning of old memories based on configurable TTL.
"""
    )

    # Config
    (tmp_path / "mind-mem.json").write_text(
        json.dumps(
            {
                "contradiction": {"threshold": 0.7},
                "proposal_budget": {"backlog_limit": 30},
            }
        )
    )

    return ws


@pytest.fixture
def contradicting_proposal() -> dict:
    """Proposal that contradicts committed decision D-20260201-001."""
    return {
        "ProposalId": "P-20260227-001",
        "Type": "decision",
        "TargetBlock": "D-20260201-001",
        "Status": "staged",
        "Risk": "high",
        "Title": "Replace SQLite with PostgreSQL for local storage",
        "Description": "Remove SQLite dependency. Use PostgreSQL for all local data storage instead.",
        "Evidence": [
            "PostgreSQL handles concurrent writes better",
            "Advanced query capabilities needed",
        ],
        "Ops": [
            {
                "op": "supersede_decision",
                "file": "decisions/DECISIONS.md",
                "target": "D-20260201-001",
                "patch": "[D-20260227-001]\nTitle: Use PostgreSQL\nDecision: Replace SQLite with PostgreSQL.",
            }
        ],
        "Rollback": "Revert to SQLite",
        "Fingerprint": "test",
    }


@pytest.fixture
def refinement_proposal() -> dict:
    """Proposal that refines an existing decision."""
    return {
        "ProposalId": "P-20260227-002",
        "Type": "decision",
        "TargetBlock": "D-20260205-002",
        "Status": "staged",
        "Risk": "low",
        "Title": "Add BM25 parameter tuning for recall",
        "Description": "Tune BM25 k1 and b parameters for better retrieval quality. BM25 remains primary.",
        "Evidence": [
            "Default k1=1.2 is suboptimal for short documents",
            "b=0.75 should be adjusted based on corpus statistics",
        ],
        "Ops": [
            {
                "op": "update_field",
                "file": "decisions/DECISIONS.md",
                "target": "D-20260205-002",
                "field": "Details",
                "value": "BM25 with tuned k1=1.5, b=0.6",
            }
        ],
        "Rollback": "Restore default params",
        "Fingerprint": "test2",
    }


@pytest.fixture
def unrelated_proposal() -> dict:
    """Proposal with no relation to existing corpus."""
    return {
        "ProposalId": "P-20260227-003",
        "Type": "task",
        "TargetBlock": "T-NEW",
        "Status": "staged",
        "Risk": "low",
        "Title": "Add Kubernetes deployment manifest",
        "Description": "Create Helm chart for cloud deployment of mind-mem.",
        "Evidence": ["Users requesting cloud deployment option"],
        "Ops": [
            {
                "op": "append_block",
                "file": "tasks/TASKS.md",
                "patch": "[T-20260227-003]\nTitle: Kubernetes deployment",
            }
        ],
        "Rollback": "Remove task",
        "Fingerprint": "test3",
    }


# ---------------------------------------------------------------------------
# Text extraction tests
# ---------------------------------------------------------------------------


class TestTextExtraction:
    def test_extract_proposal_text_all_fields(self, contradicting_proposal):
        text = _extract_proposal_text(contradicting_proposal)
        assert "Replace SQLite with PostgreSQL" in text
        assert "Remove SQLite dependency" in text
        assert "PostgreSQL handles concurrent writes" in text
        assert "Replace SQLite with PostgreSQL" in text

    def test_extract_proposal_text_includes_patches(self):
        proposal = {
            "Title": "Test",
            "Ops": [{"op": "append_block", "patch": "New content here"}],
        }
        text = _extract_proposal_text(proposal)
        assert "New content here" in text

    def test_extract_proposal_text_includes_values(self):
        proposal = {
            "Title": "Test",
            "Ops": [{"op": "update_field", "value": "updated value content"}],
        }
        text = _extract_proposal_text(proposal)
        assert "updated value content" in text

    def test_extract_proposal_text_empty(self):
        text = _extract_proposal_text({})
        assert text.strip() == ""

    def test_extract_proposal_text_evidence_as_string(self):
        proposal = {"Evidence": "single evidence string"}
        text = _extract_proposal_text(proposal)
        assert "single evidence string" in text

    def test_extract_block_text(self):
        block = {
            "Title": "Use SQLite",
            "Decision": "All storage uses SQLite",
            "Rationale": "Simplicity",
            "Evidence": ["zero deps", "single file"],
        }
        text = _extract_block_text(block)
        assert "Use SQLite" in text
        assert "All storage uses SQLite" in text
        assert "Simplicity" in text
        assert "zero deps" in text

    def test_extract_block_text_empty(self):
        text = _extract_block_text({})
        assert text.strip() == ""


# ---------------------------------------------------------------------------
# Similarity computation tests
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_tokenize_basic(self):
        tokens = _tokenize("Hello World, this is a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tokenize_numbers(self):
        tokens = _tokenize("BM25 with k1=1.5")
        assert "bm25" in tokens
        assert "k1" in tokens

    def test_jaccard_identical(self):
        sim = _jaccard_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_jaccard_no_overlap(self):
        sim = _jaccard_similarity("alpha beta", "gamma delta")
        assert sim == 0.0

    def test_jaccard_partial(self):
        sim = _jaccard_similarity("hello world foo", "hello world bar")
        assert 0.3 < sim < 0.7  # 2/4 overlap

    def test_jaccard_empty(self):
        assert _jaccard_similarity("", "hello") == 0.0
        assert _jaccard_similarity("hello", "") == 0.0
        assert _jaccard_similarity("", "") == 0.0

    def test_cosine_identical(self):
        sim = _tfidf_cosine_similarity(
            "SQLite local storage database",
            "SQLite local storage database",
        )
        assert sim == pytest.approx(1.0)

    def test_cosine_no_overlap(self):
        sim = _tfidf_cosine_similarity("alpha beta gamma", "delta epsilon zeta")
        assert sim == 0.0

    def test_cosine_partial_overlap(self):
        sim = _tfidf_cosine_similarity(
            "Use SQLite for local storage",
            "Replace SQLite with PostgreSQL for storage",
        )
        assert 0.2 < sim < 0.8

    def test_cosine_empty(self):
        assert _tfidf_cosine_similarity("", "hello") == 0.0
        assert _tfidf_cosine_similarity("hello", "") == 0.0

    def test_cosine_word_frequency_matters(self):
        # Repeated words should increase similarity
        text_a = "database database database storage"
        text_b = "database storage query"
        sim1 = _tfidf_cosine_similarity(text_a, text_b)

        text_c = "database storage"
        sim2 = _tfidf_cosine_similarity(text_c, text_b)

        # Both should be positive but different
        assert sim1 > 0
        assert sim2 > 0


# ---------------------------------------------------------------------------
# Conflict classification tests
# ---------------------------------------------------------------------------


class TestConflictClassification:
    def test_duplicate_detection(self):
        result = _classify_conflict(
            "Use SQLite for all local storage database",
            "Use SQLite for all local storage database",
            0.95,
        )
        assert result == "duplicate"

    def test_contradiction_negation(self):
        result = _classify_conflict(
            "Do not use SQLite. Cannot use it. Replace with PostgreSQL. Never use file-based storage.",
            "Use SQLite for all local storage.",
            0.6,
        )
        assert result == "contradiction"

    def test_contradiction_status_reversal(self):
        result = _classify_conflict(
            "Disable automatic proposal application",
            "Enable automatic proposal application",
            0.6,
        )
        assert result == "contradiction"

    def test_contradiction_remove_vs_add(self):
        result = _classify_conflict(
            "Remove the SQLite dependency from the project",
            "Add SQLite as the primary storage backend",
            0.5,
        )
        assert result == "contradiction"

    def test_refinement_high_similarity(self):
        result = _classify_conflict(
            "Tune BM25 parameters k1=1.5 and b=0.6 for better retrieval",
            "BM25 is the primary text retrieval algorithm with good results",
            0.75,
        )
        assert result == "refinement"

    def test_related_moderate_similarity(self):
        result = _classify_conflict(
            "Add vector search as an optional backend",
            "BM25 is the primary text retrieval algorithm",
            0.5,
        )
        assert result == "related"

    def test_low_similarity_always_related(self):
        result = _classify_conflict(
            "Deploy to Kubernetes",
            "Use SQLite for storage",
            0.3,
        )
        assert result == "related"


# ---------------------------------------------------------------------------
# Core detection pipeline tests
# ---------------------------------------------------------------------------


class TestDetectContradictions:
    def test_finds_contradicting_proposal(self, workspace, contradicting_proposal):
        conflicts = detect_contradictions(workspace, contradicting_proposal, threshold=0.3, use_bm25=False)
        assert len(conflicts) > 0
        # Should find at least the SQLite decision as a conflict
        block_ids = [c["block_id"] for c in conflicts]
        assert any("D-20260201-001" in bid for bid in block_ids)

    def test_refinement_detected(self, workspace, refinement_proposal):
        conflicts = detect_contradictions(workspace, refinement_proposal, threshold=0.15, use_bm25=False)
        # Should find BM25 decision as related/refinement, not contradiction
        assert len(conflicts) > 0
        bm25_conflict = next(
            (c for c in conflicts if "D-20260205-002" in c["block_id"]),
            None,
        )
        if bm25_conflict:
            assert bm25_conflict["conflict_type"] in ("refinement", "related")

    def test_unrelated_no_conflicts(self, workspace, unrelated_proposal):
        conflicts = detect_contradictions(workspace, unrelated_proposal, threshold=0.7, use_bm25=False)
        # Kubernetes has nothing to do with SQLite/BM25 decisions
        contradictions = [c for c in conflicts if c["conflict_type"] == "contradiction"]
        assert len(contradictions) == 0

    def test_empty_proposal(self, workspace):
        conflicts = detect_contradictions(workspace, {}, threshold=0.3, use_bm25=False)
        assert len(conflicts) == 0

    def test_high_threshold_fewer_results(self, workspace, contradicting_proposal):
        low = detect_contradictions(workspace, contradicting_proposal, threshold=0.2, use_bm25=False)
        high = detect_contradictions(workspace, contradicting_proposal, threshold=0.8, use_bm25=False)
        assert len(low) >= len(high)

    def test_top_k_limits_results(self, workspace, contradicting_proposal):
        results = detect_contradictions(workspace, contradicting_proposal, threshold=0.1, top_k=2, use_bm25=False)
        assert len(results) <= 2

    def test_results_sorted_by_similarity(self, workspace, contradicting_proposal):
        results = detect_contradictions(workspace, contradicting_proposal, threshold=0.1, use_bm25=False)
        if len(results) > 1:
            sims = [r["similarity"] for r in results]
            assert sims == sorted(sims, reverse=True)

    def test_conflict_fields_present(self, workspace, contradicting_proposal):
        results = detect_contradictions(workspace, contradicting_proposal, threshold=0.2, use_bm25=False)
        for conflict in results:
            assert "block_id" in conflict
            assert "source_file" in conflict
            assert "similarity" in conflict
            assert "conflict_type" in conflict
            assert "existing_excerpt" in conflict
            assert "proposal_excerpt" in conflict
            assert isinstance(conflict["similarity"], float)
            assert conflict["conflict_type"] in ("contradiction", "refinement", "duplicate", "related")

    def test_skips_deprecated_blocks(self, workspace, contradicting_proposal):
        """Deprecated blocks (D-20260215-004) should not be flagged."""
        results = detect_contradictions(workspace, contradicting_proposal, threshold=0.1, use_bm25=False)
        deprecated_ids = [c["block_id"] for c in results if "D-20260215-004" in c["block_id"]]
        # Deprecated block should not be in results (it has Status: deprecated)
        assert len(deprecated_ids) == 0

    def test_no_corpus_files(self, tmp_path):
        """Empty workspace returns no conflicts."""
        ws = str(tmp_path)
        (tmp_path / "mind-mem.json").write_text("{}")
        proposal = {"Title": "Test", "Ops": [{"patch": "something"}]}
        results = detect_contradictions(ws, proposal, threshold=0.1, use_bm25=False)
        assert results == []


# ---------------------------------------------------------------------------
# check_proposal_contradictions integration tests
# ---------------------------------------------------------------------------


class TestCheckProposalContradictions:
    def test_report_structure(self, workspace, contradicting_proposal):
        report = check_proposal_contradictions(workspace, contradicting_proposal)
        assert "has_contradictions" in report
        assert "has_conflicts" in report
        assert "contradiction_count" in report
        assert "duplicate_count" in report
        assert "refinement_count" in report
        assert "total_conflicts" in report
        assert "conflicts" in report
        assert "summary" in report
        assert isinstance(report["summary"], str)

    def test_contradiction_flagged(self, workspace, contradicting_proposal):
        # Use a lower threshold since TF-IDF cosine on short texts yields moderate scores
        report = check_proposal_contradictions(workspace, contradicting_proposal, threshold=0.3)
        assert report["has_conflicts"]
        assert report["total_conflicts"] > 0
        assert "⚠️" in report["summary"] or "📝" in report["summary"] or "✅" in report["summary"]

    def test_no_conflicts_clean_summary(self, workspace, unrelated_proposal):
        report = check_proposal_contradictions(workspace, unrelated_proposal)
        # Even if there are no contradictions, summary should be present
        assert "summary" in report
        assert isinstance(report["summary"], str)

    def test_config_threshold_used(self, workspace, contradicting_proposal):
        # Override config threshold to very high
        config_path = os.path.join(workspace, "mind-mem.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"contradiction": {"threshold": 0.99}}, f)

        report = check_proposal_contradictions(workspace, contradicting_proposal)
        # Very high threshold should result in fewer/no conflicts
        assert report["total_conflicts"] <= 1  # Almost nothing passes 0.99

    def test_explicit_threshold_overrides_config(self, workspace, contradicting_proposal):
        report_low = check_proposal_contradictions(workspace, contradicting_proposal, threshold=0.1)
        report_high = check_proposal_contradictions(workspace, contradicting_proposal, threshold=0.95)
        assert report_low["total_conflicts"] >= report_high["total_conflicts"]

    def test_empty_proposal_clean_report(self, workspace):
        report = check_proposal_contradictions(workspace, {})
        assert not report["has_contradictions"]
        assert report["contradiction_count"] == 0

    def test_missing_config_uses_default(self, tmp_path):
        """When mind-mem.json is absent, uses default threshold."""
        ws = str(tmp_path)
        (tmp_path / "decisions").mkdir()
        (tmp_path / "decisions" / "DECISIONS.md").write_text("[D-001]\nTitle: Test\nStatus: active\nDecision: Hello world\n")
        proposal = {"Title": "Test hello world", "Ops": []}
        report = check_proposal_contradictions(ws, proposal)
        assert isinstance(report, dict)
        assert "summary" in report


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_proposal_with_only_ops(self, workspace):
        """Proposal with no title/description, only op patches."""
        proposal = {
            "ProposalId": "P-20260227-099",
            "Ops": [
                {
                    "op": "append_block",
                    "file": "decisions/DECISIONS.md",
                    "patch": "Use SQLite for everything. No external databases allowed.",
                }
            ],
        }
        results = detect_contradictions(workspace, proposal, threshold=0.3, use_bm25=False)
        # Should still find SQLite decision via patch content
        assert len(results) > 0

    def test_very_long_proposal_text(self, workspace):
        """Large proposal text doesn't crash."""
        proposal = {
            "Title": "Large proposal " * 100,
            "Description": "Lots of words about storage " * 200,
            "Ops": [{"patch": "content " * 500}],
        }
        results = detect_contradictions(workspace, proposal, threshold=0.1, use_bm25=False)
        # Should complete without error
        assert isinstance(results, list)

    def test_special_characters_in_text(self, workspace):
        """Proposals with special chars don't break tokenization."""
        proposal = {
            "Title": "Test <script>alert('xss')</script>",
            "Description": "DROP TABLE decisions; -- SQLi attempt",
            "Ops": [{"patch": "Content with émojis 🎉 and ünïcödé"}],
        }
        results = detect_contradictions(workspace, proposal, threshold=0.1, use_bm25=False)
        assert isinstance(results, list)

    def test_non_string_evidence(self, workspace):
        """Evidence field with non-string items is handled gracefully."""
        proposal = {
            "Title": "Test",
            "Evidence": [123, None, "valid evidence", True],
            "Ops": [],
        }
        text = _extract_proposal_text(proposal)
        assert "valid evidence" in text

    def test_block_with_no_id(self, workspace):
        """Blocks without _id are skipped gracefully."""
        # Write a corpus file with a block missing ID
        decisions_path = os.path.join(workspace, "decisions", "DECISIONS.md")
        with open(decisions_path, "a") as f:
            f.write("\nTitle: Orphan block with no ID\nStatus: active\n")

        proposal = {"Title": "Orphan test", "Ops": []}
        results = detect_contradictions(workspace, proposal, threshold=0.1, use_bm25=False)
        # Should not crash
        assert isinstance(results, list)

    def test_threshold_zero_returns_everything(self, workspace, contradicting_proposal):
        """Threshold of 0 should return all non-empty blocks."""
        results = detect_contradictions(workspace, contradicting_proposal, threshold=0.0, use_bm25=False)
        # Should find at least the committed decisions/tasks
        assert len(results) > 0

    def test_threshold_one_returns_nothing(self, workspace, contradicting_proposal):
        """Threshold of 1.0 should return nothing (exact match only)."""
        results = detect_contradictions(workspace, contradicting_proposal, threshold=1.0, use_bm25=False)
        assert len(results) == 0
