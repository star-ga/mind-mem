# Copyright 2026 STARGA, Inc.
"""Configuration for the skill optimization subsystem."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_FLEET_MODELS: dict[str, list[str]] = {
    # A STARTING DEFAULT, not a verified-live roster. Nothing in this package
    # probes whether a listed seat is reachable, correctly named or billed, so
    # do not read this list as a health claim: upstream vendors rename and
    # retire model IDs on their own schedule, and an ID that has gone stale
    # since this list was last touched fails only at request time.
    #
    # That failure is quiet. A dead seat comes back as a FleetResponse with
    # ``error`` set, and the consumers drop errored responses rather than
    # raising, so a fleet where EVERY seat is dead is indistinguishable from a
    # fleet that simply found nothing to say — a run over dead seats still
    # exits 0 with a zero consensus score. Treat a suspiciously empty critique
    # set as a possible transport failure, not as a verdict on the skill.
    #
    # Override per workspace with ``skill_opt.fleet_models`` in
    # ``mind-mem.json`` (see :func:`load_config`) rather than editing this
    # list, which lives inside the installed package. A key here that has no
    # entry in ``FLEET_MODELS`` in ``fleet_bridge.py`` is silently skipped
    # when the bridge builds its providers, so the two must stay in sync.
    "test_execution": ["grok-4.3", "mistral-large-latest"],
    "critique": [
        "deepseek-v4-pro",
        "sonar-pro",
        "glm-5.1",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    ],
    "mutation": ["grok-4.3", "mistral-large-latest"],
}

DEFAULT_SKILL_SOURCES: dict[str, str] = {
    "openclaw": "~/.openclaw/skills",
    "openclaw_agent": "~/.agent-bot/skills",
    "claude_agents": "~/.claude/agents",
    "codex_skills": "~/.codex/skills",
    "codex_memories": "~/.codex/memories",
    "gemini": "~/.gemini",
}

#: Environment variable that relocates the multi-LLM orchestrator package.
#:
#: The default below is a development-machine layout: a plugin directory that
#: exists outside the installed ``mind_mem`` package and ships with nothing.
#: It is the ONLY source of fleet access (``fleet_bridge._load_orchestrator``
#: puts it on ``sys.path`` and imports ``providers`` / ``config`` from it), so
#: on any machine without that exact directory the whole ``skill_opt`` feature
#: is unreachable. Making it an environment variable is what keeps the
#: remediation OUTSIDE the installed package — an operator points this at
#: their own checkout instead of editing site-packages source.
ORCHESTRATOR_PATH_ENV = "MIND_MEM_ORCHESTRATOR_PATH"

#: Fallback used when :data:`ORCHESTRATOR_PATH_ENV` is unset or empty.
DEFAULT_ORCHESTRATOR_PATH = "~/.claude/plugins/marketplaces/claude-code-ultimate/multi-llm-orchestrator"


def resolve_orchestrator_path(env: Mapping[str, str] | None = None) -> str:
    """Resolve the orchestrator directory from *env* (default ``os.environ``).

    An empty or missing variable falls back to
    :data:`DEFAULT_ORCHESTRATOR_PATH` — an exported-but-empty value must not
    resolve the import root to the current working directory, which would put
    whatever the process happens to be sitting in ahead of every other entry
    on ``sys.path``. The result is ``expanduser``-ed so an operator can write
    ``~/...`` exactly as the default does.
    """
    source = (os.environ if env is None else env).get(ORCHESTRATOR_PATH_ENV) or DEFAULT_ORCHESTRATOR_PATH
    return os.path.expanduser(source)


#: Resolved once at import time, because ``fleet_bridge`` binds this name with
#: ``from .config import ORCHESTRATOR_PATH``; mutating this module attribute
#: later would not reach that binding. Set :data:`ORCHESTRATOR_PATH_ENV`
#: before importing ``mind_mem``.
ORCHESTRATOR_PATH = resolve_orchestrator_path()


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
