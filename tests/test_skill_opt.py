# Copyright 2026 STARGA, Inc.
"""Tests for the skill_opt subpackage."""

from __future__ import annotations

import json
import os

import pytest

from mind_mem.skill_opt._types import (
    CritiqueReport,
    SkillSpec,
    TestCase,
    ValidationResult,
)
from mind_mem.skill_opt.adapters import (
    ClaudeAgentAdapter,
    OpenClawSkillAdapter,
    _parse_yaml_frontmatter,
    adapter_for_spec,
    discover_all,
)
from mind_mem.skill_opt.config import SkillOptConfig, load_config
from mind_mem.skill_opt.history import HistoryStore
from mind_mem.skill_opt.scorer import (
    aggregate_critiques,
    build_critique_prompt,
    classify_skill,
)

# ── Types ───────────────────────────────────────────────────────


class TestSkillSpec:
    def test_frozen(self):
        s = SkillSpec(
            skill_id="test:a",
            system="test",
            source_path="/tmp/test.md",
            format="agent-md",
            name="test",
            description="desc",
            content="hello",
        )
        assert s.content_hash
        with pytest.raises(AttributeError):
            s.name = "changed"  # type: ignore[misc]

    def test_content_hash_auto(self):
        s = SkillSpec(
            skill_id="x",
            system="x",
            source_path="/x",
            format="x",
            name="x",
            description="x",
            content="hello world",
        )
        import hashlib

        assert s.content_hash == hashlib.sha256(b"hello world").hexdigest()

    def test_as_dict(self):
        s = SkillSpec(
            skill_id="test:b",
            system="test",
            source_path="/b",
            format="skill-md",
            name="b",
            description="d",
            content="c",
        )
        d = s.as_dict()
        assert d["skill_id"] == "test:b"
        assert "content" not in d


class TestTestCase:
    def test_frozen_tuple_rubric(self):
        tc = TestCase(
            test_id="t1",
            skill_id="s1",
            category="correctness",
            prompt="do something",
            expected_behavior="good output",
            rubric=("accuracy", "format"),
        )
        assert tc.rubric == ("accuracy", "format")
        assert tc.as_dict()["rubric"] == ["accuracy", "format"]


class TestValidationResult:
    def test_accepted_requires_improvement_and_votes(self):
        v = ValidationResult(
            mutation_id="m1",
            skill_id="s1",
            score_before=0.5,
            score_after=0.6,
            improved=True,
            critic_votes={"a": True, "b": True, "c": False},
        )
        assert v.accepted is True

    def test_rejected_when_not_improved(self):
        v = ValidationResult(
            mutation_id="m2",
            skill_id="s2",
            score_before=0.5,
            score_after=0.52,
            improved=False,
            critic_votes={"a": True, "b": True},
        )
        assert v.accepted is False

    def test_rejected_when_regression(self):
        v = ValidationResult(
            mutation_id="m3",
            skill_id="s3",
            score_before=0.5,
            score_after=0.6,
            improved=True,
            regression_categories=("safety",),
            critic_votes={"a": True, "b": True},
        )
        assert v.accepted is False

    def test_rejected_when_minority_votes(self):
        v = ValidationResult(
            mutation_id="m4",
            skill_id="s4",
            score_before=0.5,
            score_after=0.6,
            improved=True,
            critic_votes={"a": True, "b": False, "c": False},
        )
        assert v.accepted is False


# ── Config ──────────────────────────────────────────────────────


class TestConfig:
    def test_defaults(self):
        cfg = SkillOptConfig()
        assert cfg.enabled is False
        assert cfg.min_critics == 3
        assert "openclaw" in cfg.skill_sources
        assert cfg.improvement_threshold == 0.05

    def test_load_missing_file(self):
        cfg = load_config("/nonexistent/path")
        assert cfg == SkillOptConfig()

    def test_load_from_json(self, tmp_path):
        data = {"skill_opt": {"enabled": True, "min_critics": 5}}
        cfg_path = tmp_path / "mind-mem.json"
        cfg_path.write_text(json.dumps(data))
        cfg = load_config(str(tmp_path))
        assert cfg.enabled is True
        assert cfg.min_critics == 5

    def test_resolve_sources(self):
        cfg = SkillOptConfig(skill_sources={"test": "~/some/path"})
        resolved = cfg.resolve_sources()
        assert resolved["test"] == os.path.expanduser("~/some/path")


# ── Adapters ────────────────────────────────────────────────────


class TestYamlFrontmatter:
    def test_parse_basic(self):
        text = "---\nname: test\ndescription: hello\n---\n\n# Body"
        meta, body = _parse_yaml_frontmatter(text)
        assert meta["name"] == "test"
        assert meta["description"] == "hello"
        assert body.startswith("# Body")

    def test_parse_list(self):
        text = '---\ntools: ["Read", "Grep", "Glob"]\n---\nbody'
        meta, body = _parse_yaml_frontmatter(text)
        assert meta["tools"] == ["Read", "Grep", "Glob"]

    def test_parse_json_metadata(self):
        text = '---\nmetadata: {"moltbot":{"emoji":"X"}}\n---\nbody'
        meta, body = _parse_yaml_frontmatter(text)
        assert meta["metadata"]["moltbot"]["emoji"] == "X"

    def test_no_frontmatter(self):
        text = "Just plain text"
        meta, body = _parse_yaml_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_bool_values(self):
        text = "---\nenabled: true\ndisabled: false\n---\nbody"
        meta, _ = _parse_yaml_frontmatter(text)
        assert meta["enabled"] is True
        assert meta["disabled"] is False


class TestOpenClawAdapter:
    def test_discover_and_parse(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test skill\n---\n\n# My Skill")
        adapter = OpenClawSkillAdapter()
        paths = adapter.discover(str(tmp_path))
        assert len(paths) == 1
        spec = adapter.parse(paths[0])
        assert spec.skill_id == "openclaw:my-skill"
        assert spec.name == "my-skill"
        assert spec.system == "openclaw"
        assert spec.format == "skill-md"

    def test_can_handle(self):
        adapter = OpenClawSkillAdapter()
        assert adapter.can_handle("/some/path/SKILL.md") is True
        assert adapter.can_handle("/some/path/README.md") is False


class TestClaudeAgentAdapter:
    def test_discover_and_parse(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "reviewer.md").write_text(
            '---\nname: reviewer\ndescription: Code reviewer\ntools: ["Read", "Grep"]\nmodel: sonnet\n---\n\nYou are a reviewer.'
        )
        adapter = ClaudeAgentAdapter()
        paths = adapter.discover(str(agents_dir))
        assert len(paths) == 1
        spec = adapter.parse(paths[0])
        assert spec.skill_id == "claude:reviewer"
        assert spec.metadata["tools"] == ["Read", "Grep"]
        assert spec.metadata["model"] == "sonnet"


class TestDiscoverAll:
    def test_empty_sources(self):
        specs = discover_all({})
        assert specs == []

    def test_nonexistent_paths(self):
        specs = discover_all({"openclaw": "/nonexistent/skills"})
        assert specs == []


class TestAdapterForSpec:
    def test_known_format(self):
        spec = SkillSpec(
            skill_id="x",
            system="x",
            source_path="/x",
            format="skill-md",
            name="x",
            description="x",
            content="x",
        )
        adapter = adapter_for_spec(spec)
        assert adapter.format_id == "skill-md"

    def test_unknown_format(self):
        spec = SkillSpec(
            skill_id="x",
            system="x",
            source_path="/x",
            format="unknown",
            name="x",
            description="x",
            content="x",
        )
        with pytest.raises(ValueError, match="No adapter"):
            adapter_for_spec(spec)


# ── Scorer ──────────────────────────────────────────────────────


class TestClassifySkill:
    def test_coding(self):
        assert classify_skill("code-reviewer", "Reviews code") == "coding"

    def test_tool(self):
        assert classify_skill("github", "CLI tool for repos") == "tool"

    def test_security(self):
        assert classify_skill("pentest", "Vulnerability scanning") == "security"

    def test_process(self):
        assert classify_skill("planner", "Architecture design") == "process"

    def test_default(self):
        assert classify_skill("random", "does stuff") == "coding"


class TestAggregateCritiques:
    def test_empty(self):
        score = aggregate_critiques("s1", "h1", [])
        assert score.overall == 0.0

    def test_basic_aggregation(self):
        critiques = [
            CritiqueReport(
                critic_model="m1",
                test_id="t1",
                scores={"accuracy": 0.8, "safety": 0.9},
                overall_score=0.85,
            ),
            CritiqueReport(
                critic_model="m2",
                test_id="t1",
                scores={"accuracy": 0.7, "safety": 0.8},
                overall_score=0.75,
            ),
        ]
        score = aggregate_critiques("s1", "h1", critiques)
        assert score.overall == 0.8
        assert score.sample_size == 2
        assert 0.74 < score.by_rubric["accuracy"] < 0.76
        assert 0.84 < score.by_rubric["safety"] < 0.86


class TestBuildCritiquePrompt:
    def test_contains_rubric(self):
        from mind_mem.skill_opt.scorer import RUBRICS

        rubric = RUBRICS["coding"]
        prompt = build_critique_prompt("skill content", "test prompt", "model output", rubric)
        assert "correctness" in prompt
        assert "safety" in prompt
        assert "JSON only" in prompt


# ── History ─────────────────────────────────────────────────────


class TestHistoryStore:
    def test_run_lifecycle(self, tmp_path):
        db = str(tmp_path / "test.db")
        store = HistoryStore(db)

        store.start_run("R-001", "s1", "hash1")
        store.complete_run("R-001", "completed", 0.5, 0.7, True)

        history = store.get_run_history("s1")
        assert len(history) == 1
        assert history[0]["run_id"] == "R-001"
        assert history[0]["status"] == "completed"

        score = store.get_latest_score("s1")
        assert score == 0.7

        store.close()

    def test_mutation_storage(self, tmp_path):
        db = str(tmp_path / "test2.db")
        store = HistoryStore(db)

        store.start_run("R-002", "s2", "hash2")
        store.store_mutation(
            "R-002",
            "M-001",
            "s2",
            "new content",
            "because reasons",
            score_before=0.5,
            score_after=0.7,
        )

        m = store.get_mutation("M-001")
        assert m is not None
        assert m["rationale"] == "because reasons"

        pending = store.get_pending_mutations("s2")
        assert len(pending) == 1

        store.update_mutation_status("M-001", "validated", "SIG-001")
        pending = store.get_pending_mutations("s2")
        assert len(pending) == 0

        store.close()

    def test_empty_score(self, tmp_path):
        db = str(tmp_path / "test3.db")
        store = HistoryStore(db)
        assert store.get_latest_score("nonexistent") is None
        store.close()


# ── CLI ─────────────────────────────────────────────────────────


class TestCLI:
    def test_skill_list_parser(self):
        from mind_mem.mm_cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["skill", "list"])
        assert args.skill_cmd == "list"

    def test_skill_test_parser(self):
        from mind_mem.mm_cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["skill", "test", "openclaw:coding-agent"])
        assert args.skill_id == "openclaw:coding-agent"

    def test_skill_optimize_parser(self):
        from mind_mem.mm_cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["skill", "optimize", "claude:code-reviewer"])
        assert args.skill_id == "claude:code-reviewer"

    def test_skill_history_parser(self):
        from mind_mem.mm_cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["skill", "history", "s1", "--limit", "5"])
        assert args.skill_id == "s1"
        assert args.limit == 5
