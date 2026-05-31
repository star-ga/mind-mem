"""Tests for the v3.9 persona-aware recall projection."""

from __future__ import annotations

import pytest
from mind_mem.personas import (
    DEFAULT_PERSONA,
    PERSONAS,
    PersonaError,
    apply_persona,
    project_block,
)

# ---------------------------------------------------------------------------
# Fixture blocks
# ---------------------------------------------------------------------------


@pytest.fixture
def block_full():
    return {
        "_id": "D-20260503-001",
        "Subject": "Auth middleware rewrite",
        "Statement": "The new middleware swaps session tokens for short-lived JWTs.",
        "Status": "active",
        "Tags": ["auth", "security"],
        "score": 0.87,
        "axis_scores": {"importance": 0.9, "recency": 0.7},
        "AuditHash": "a" * 64,
        "TransformHash": "b" * 64,
    }


@pytest.fixture
def block_minimal():
    return {
        "id": "T-20260101-099",
        "content": "Implement password reset.\nSecond paragraph that should be dropped from brief.",
    }


# ---------------------------------------------------------------------------
# project_block
# ---------------------------------------------------------------------------


class TestProjectBlock:
    def test_unknown_persona_raises(self, block_full) -> None:
        with pytest.raises(PersonaError, match="unknown persona"):
            project_block(block_full, "casual")  # type: ignore[arg-type]

    def test_brief_keeps_id_score_subject(self, block_full) -> None:
        out = project_block(block_full, "brief")
        assert out["id"] == "D-20260503-001"
        assert out["score"] == 0.87
        assert out["subject"] == "Auth middleware rewrite"
        # Brief omits everything else.
        assert "Statement" not in out
        assert "Tags" not in out

    def test_brief_truncates_long_subject(self) -> None:
        long_subject = "x" * 500
        out = project_block({"_id": "X-1", "Subject": long_subject}, "brief")
        assert len(out["subject"]) <= 121  # 120 + ellipsis

    def test_brief_uses_first_line_of_content(self, block_minimal) -> None:
        out = project_block(block_minimal, "brief")
        assert out["subject"] == "Implement password reset."

    def test_brief_no_subject_returns_placeholder(self) -> None:
        out = project_block({"_id": "X-1"}, "brief")
        assert out["subject"] == "(no subject)"

    def test_detailed_is_identity(self, block_full) -> None:
        out = project_block(block_full, "detailed")
        assert out == block_full
        # Must be a copy, not the same object.
        assert out is not block_full

    def test_technical_promotes_governance_fields(self, block_full) -> None:
        out = project_block(block_full, "technical")
        # axis_scores already at top — preserved.
        assert out["axis_scores"] == {"importance": 0.9, "recency": 0.7}
        # AuditHash promoted to provenance_hash
        assert out["provenance_hash"] == "a" * 64
        # Status promoted to governance_state
        assert out["governance_state"] == "active"
        # TransformHash promoted lowercase
        assert out["transform_hash"] == "b" * 64

    def test_technical_preserves_existing_keys_when_present(self) -> None:
        block = {
            "_id": "X-1",
            "governance_state": "draft",
            "Status": "active",
            "provenance_hash": "z" * 64,
            "AuditHash": "a" * 64,
        }
        out = project_block(block, "technical")
        # governance_state already set — must NOT be overwritten by Status
        assert out["governance_state"] == "draft"
        assert out["provenance_hash"] == "z" * 64

    def test_technical_block_id_falls_through(self) -> None:
        block = {"id": "T-1", "content": "x"}
        out = project_block(block, "technical")
        # Identity-ish: the technical projection only adds, never removes.
        assert out["id"] == "T-1"
        assert out["content"] == "x"


# ---------------------------------------------------------------------------
# apply_persona — list level
# ---------------------------------------------------------------------------


class TestApplyPersona:
    def test_default_when_none(self, block_full, block_minimal) -> None:
        out = apply_persona([block_full, block_minimal], None)
        # default == "detailed" == identity copy
        assert out[0] == block_full
        assert out[1] == block_minimal

    def test_default_when_empty_string(self, block_full) -> None:
        out = apply_persona([block_full], "")
        assert out[0] == block_full

    def test_brief_projects_each(self, block_full, block_minimal) -> None:
        out = apply_persona([block_full, block_minimal], "brief")
        assert all(set(b.keys()) == {"id", "score", "subject"} for b in out)

    def test_unknown_persona_raises(self, block_full) -> None:
        with pytest.raises(PersonaError):
            apply_persona([block_full], "verbose")

    def test_does_not_mutate_input(self, block_full) -> None:
        original = dict(block_full)
        apply_persona([block_full], "brief")
        assert block_full == original  # untouched

    def test_personas_constant_matches_default(self) -> None:
        assert DEFAULT_PERSONA in PERSONAS
        assert set(PERSONAS) == {"brief", "detailed", "technical"}
