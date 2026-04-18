# Copyright 2026 STARGA, Inc.
"""Configuration for the skill optimization subsystem."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_FLEET_MODELS: dict[str, list[str]] = {
    "test_execution": ["grok-4-1-fast-reasoning", "mistral-large-latest"],
    "critique": [
        "deepseek-reasoner",
        "sonar-pro",
        "glm-5",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    ],
    "mutation": ["grok-4-1-fast-reasoning", "mistral-large-latest"],
}

DEFAULT_SKILL_SOURCES: dict[str, str] = {
    "openclaw": "~/.openclaw/skills",
    "openclaw_agent": "~/.agent-bot/skills",
    "claude_agents": "~/.claude/agents",
    "codex_skills": "~/.codex/skills",
    "codex_memories": "~/.codex/memories",
    "gemini": "~/.gemini",
}

ORCHESTRATOR_PATH = os.path.expanduser("~/.claude/plugins/marketplaces/claude-code-ultimate/multi-llm-orchestrator")

ENV_PATH = os.path.expanduser("~/.claude-ultimate/.env")


@dataclass(frozen=True)
class SkillOptConfig:
    """Typed configuration for the skill_opt subsystem."""

    enabled: bool = False
    fleet_models: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_FLEET_MODELS))
    min_critics: int = 3
    improvement_threshold: float = 0.05
    regression_threshold: float = 0.10
    max_mutations_per_run: int = 3
    test_cases_per_skill: int = 5
    skill_sources: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SKILL_SOURCES))
    auto_optimize_on_drift: bool = False
    history_db_path: str = ".mind-mem-skill-opt/history.db"
    governance_workspace: str = ""

    def resolve_sources(self) -> dict[str, str]:
        return {k: os.path.expanduser(v) for k, v in self.skill_sources.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fleet_models": self.fleet_models,
            "min_critics": self.min_critics,
            "improvement_threshold": self.improvement_threshold,
            "regression_threshold": self.regression_threshold,
            "max_mutations_per_run": self.max_mutations_per_run,
            "test_cases_per_skill": self.test_cases_per_skill,
            "skill_sources": self.skill_sources,
            "auto_optimize_on_drift": self.auto_optimize_on_drift,
            "history_db_path": self.history_db_path,
            "governance_workspace": self.governance_workspace,
        }


def load_config(workspace: str) -> SkillOptConfig:
    """Load skill_opt config from mind-mem.json in workspace."""
    cfg_path = os.path.join(workspace, "mind-mem.json")
    if not os.path.isfile(cfg_path):
        return SkillOptConfig()
    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("skill_opt", {})
    if not isinstance(raw, dict):
        return SkillOptConfig()
    return SkillOptConfig(
        enabled=bool(raw.get("enabled", False)),
        fleet_models=raw.get("fleet_models", dict(DEFAULT_FLEET_MODELS)),
        min_critics=int(raw.get("min_critics", 3)),
        improvement_threshold=float(raw.get("improvement_threshold", 0.05)),
        regression_threshold=float(raw.get("regression_threshold", 0.10)),
        max_mutations_per_run=int(raw.get("max_mutations_per_run", 3)),
        test_cases_per_skill=int(raw.get("test_cases_per_skill", 5)),
        skill_sources=raw.get("skill_sources", dict(DEFAULT_SKILL_SOURCES)),
        auto_optimize_on_drift=bool(raw.get("auto_optimize_on_drift", False)),
        history_db_path=str(raw.get("history_db_path", ".mind-mem-skill-opt/history.db")),
        governance_workspace=str(raw.get("governance_workspace", "")),
    )
