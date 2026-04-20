# Copyright 2026 STARGA, Inc.
"""Frozen dataclasses for the skill optimization pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillSpec:
    """Normalized representation of any skill/agent/prompt file."""

    skill_id: str
    system: str
    source_path: str
    format: str
    name: str
    description: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            h = hashlib.sha256(self.content.encode()).hexdigest()
            object.__setattr__(self, "content_hash", h)

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "system": self.system,
            "source_path": self.source_path,
            "format": self.format,
            "name": self.name,
            "description": self.description,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TestCase:
    """A synthetic test case derived from a skill definition."""

    # Pytest sees any class named ``Test*`` as a test class by default and
    # emits ``PytestCollectionWarning`` because dataclasses declare
    # ``__init__``. This attribute is pytest's documented opt-out.
    __test__ = False

    test_id: str
    skill_id: str
    category: str
    prompt: str
    expected_behavior: str
    rubric: tuple[str, ...] = ()
    difficulty: str = "medium"

    def as_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "skill_id": self.skill_id,
            "category": self.category,
            "prompt": self.prompt,
            "expected_behavior": self.expected_behavior,
            "rubric": list(self.rubric),
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True)
class TestResult:
    """Output from running a test case against a model."""

    test_id: str
    skill_id: str
    model: str
    output: str
    latency_ms: float
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "skill_id": self.skill_id,
            "model": self.model,
            "output": self.output,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class CritiqueReport:
    """One model's critique of a test result."""

    critic_model: str
    test_id: str
    scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    failure_modes: tuple[str, ...] = ()
    improvement_suggestions: tuple[str, ...] = ()
    raw_response: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "critic_model": self.critic_model,
            "test_id": self.test_id,
            "scores": self.scores,
            "overall_score": self.overall_score,
            "failure_modes": list(self.failure_modes),
            "improvement_suggestions": list(self.improvement_suggestions),
        }


@dataclass(frozen=True)
class Mutation:
    """A proposed rewrite of a skill/prompt."""

    mutation_id: str
    skill_id: str
    original_hash: str
    proposed_content: str
    rationale: str
    failure_modes_addressed: tuple[str, ...] = ()
    critic_consensus: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "skill_id": self.skill_id,
            "original_hash": self.original_hash,
            "rationale": self.rationale,
            "failure_modes_addressed": list(self.failure_modes_addressed),
            "critic_consensus": self.critic_consensus,
            "proposed_hash": hashlib.sha256(self.proposed_content.encode()).hexdigest(),
        }


@dataclass(frozen=True)
class SkillScore:
    """Aggregated score for a skill version."""

    skill_id: str
    content_hash: str
    overall: float
    by_category: dict[str, float] = field(default_factory=dict)
    by_rubric: dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "content_hash": self.content_hash,
            "overall": self.overall,
            "by_category": self.by_category,
            "by_rubric": self.by_rubric,
            "sample_size": self.sample_size,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a mutation against the original."""

    mutation_id: str
    skill_id: str
    score_before: float
    score_after: float
    improved: bool
    regression_categories: tuple[str, ...] = ()
    critic_votes: dict[str, bool] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        if not self.improved:
            return False
        if self.regression_categories:
            return False
        yes = sum(1 for v in self.critic_votes.values() if v)
        return yes >= len(self.critic_votes) * 2 / 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "skill_id": self.skill_id,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "improved": self.improved,
            "accepted": self.accepted,
            "regression_categories": list(self.regression_categories),
            "critic_votes": self.critic_votes,
        }
