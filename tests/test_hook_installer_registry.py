# Copyright 2026 STARGA, Inc.
"""Tests for the registry-driven hook installer (v3.0.0)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem.hook_installer import (
    AGENT_REGISTRY,
    AgentSpec,
    detect_installed_agents,
    install_all,
    install_config,
)


class TestAgentRegistry:
    def test_registry_has_all_known_clients(self) -> None:
        expected = {
            "claude-code", "codex", "gemini", "cursor", "windsurf", "aider",
            "openclaw", "nanoclaw", "nemoclaw",
            "continue", "cline", "roo", "zed", "copilot", "cody", "qodo",
        }
        assert set(AGENT_REGISTRY.keys()) >= expected, (
            f"missing: {expected - set(AGENT_REGISTRY.keys())}"
        )

    def test_every_spec_has_required_fields(self) -> None:
        for name, spec in AGENT_REGISTRY.items():
            assert spec.name == name
            assert spec.description
            assert spec.config_fmt
            assert spec.path_tmpl


class TestInstallConfigForEveryAgent:
    @pytest.mark.parametrize("agent", sorted(AGENT_REGISTRY.keys()))
    def test_dry_run_returns_content_for_every_agent(
        self, agent: str, tmp_path: Path
    ) -> None:
        path = AGENT_REGISTRY[agent].expand_path(str(tmp_path))
        mtime_before = os.path.getmtime(path) if os.path.isfile(path) else None
        size_before = os.path.getsize(path) if os.path.isfile(path) else None
        result = install_config(agent, str(tmp_path), dry_run=True)
        assert result["agent"] == agent
        assert result["written"] is False
        assert result["content"]
        # dry_run must not mutate the target file (whether or not it pre-existed)
        if mtime_before is not None:
            assert os.path.getmtime(path) == mtime_before
            assert os.path.getsize(path) == size_before


class TestInstallNonDestructive:
    def test_text_block_preserves_user_content(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursorrules"
        path.write_text("# my existing rules\nsome user content\n")
        install_config("cursor", str(tmp_path), dry_run=False)
        body = path.read_text()
        assert "my existing rules" in body
        assert "some user content" in body
        assert "# mind-mem" in body

    def test_text_block_idempotent(self, tmp_path: Path) -> None:
        install_config("cursor", str(tmp_path))
        install_config("cursor", str(tmp_path))
        install_config("cursor", str(tmp_path))
        body = (tmp_path / ".cursorrules").read_text()
        assert body.count("# mind-mem") == 1

    def test_json_merge_preserves_user_keys(self, tmp_path: Path) -> None:
        continue_dir = tmp_path / ".continue"
        continue_dir.mkdir()
        # pre-existing user config
        path = continue_dir / "config.json"
        path.write_text(json.dumps({"models": [{"title": "my-llm"}], "theme": "dark"}))
        # set HOME so json-continue target resolves into tmp_path
        import mind_mem.hook_installer as hi

        spec = hi.AGENT_REGISTRY["continue"]
        # override path_tmpl to tmp-local
        hi.AGENT_REGISTRY["continue"] = AgentSpec(
            name=spec.name, description=spec.description,
            config_fmt=spec.config_fmt,
            path_tmpl=str(path),
            content_tmpl=spec.content_tmpl,
            detect_paths=spec.detect_paths, detect_binaries=spec.detect_binaries,
            always_offer=spec.always_offer,
        )
        try:
            install_config("continue", str(tmp_path), dry_run=False)
            loaded = json.loads(path.read_text())
            assert loaded["theme"] == "dark"
            assert loaded["models"][0]["title"] == "my-llm"
            assert "systemMessage" in loaded
            assert "mind-mem" in loaded["systemMessage"]
        finally:
            hi.AGENT_REGISTRY["continue"] = spec


class TestClawFamily:
    def test_openclaw_nanoclaw_nemoclaw_share_hook_shape(
        self, tmp_path: Path
    ) -> None:
        # All three claw variants route through _merge_openclaw_hooks.
        for agent in ("openclaw", "nanoclaw", "nemoclaw"):
            result = install_config(agent, str(tmp_path), dry_run=True)
            content = json.loads(result["content"])
            hooks = content["hooks"]["internal"]["entries"]["mind-mem"]
            assert hooks["enabled"] is True
            assert hooks["workspace"] == str(tmp_path)
            assert "commands" in hooks


class TestCopilotAlwaysOffer:
    def test_copilot_flagged_always_offer(self) -> None:
        assert AGENT_REGISTRY["copilot"].always_offer is True


class TestDetectInstalledAgents:
    def test_detection_always_includes_copilot(self, tmp_path: Path) -> None:
        detected = detect_installed_agents(str(tmp_path))
        assert "copilot" in detected  # always_offer=True

    def test_detection_returns_list_of_registered_names(
        self, tmp_path: Path
    ) -> None:
        detected = detect_installed_agents(str(tmp_path))
        for name in detected:
            assert name in AGENT_REGISTRY


class TestInstallAll:
    def test_install_all_explicit_list_dry_run(self, tmp_path: Path) -> None:
        results = install_all(
            str(tmp_path),
            dry_run=True,
            agents=["cursor", "windsurf", "copilot"],
        )
        assert len(results) == 3
        assert {r["agent"] for r in results} == {"cursor", "windsurf", "copilot"}
        for r in results:
            assert r["written"] is False

    def test_install_all_writes_all_requested(self, tmp_path: Path) -> None:
        results = install_all(
            str(tmp_path),
            agents=["cursor", "windsurf", "aider", "copilot", "cline"],
        )
        assert all(r["written"] or r["error"] for r in results if "error" in r or r["written"])
        # Corresponding files exist
        assert (tmp_path / ".cursorrules").is_file()
        assert (tmp_path / ".windsurfrules").is_file()
        assert (tmp_path / ".aider.conf.yml").is_file()
        assert (tmp_path / ".github" / "copilot-instructions.md").is_file()
        assert (tmp_path / ".clinerules").is_file()

    def test_install_all_error_isolation(self, tmp_path: Path) -> None:
        # One bad agent shouldn't stop the others.
        results = install_all(
            str(tmp_path),
            agents=["cursor", "definitely-not-an-agent", "windsurf"],
        )
        errors = [r for r in results if "error" in r]
        successes = [r for r in results if r.get("written")]
        assert len(errors) == 1
        assert errors[0]["agent"] == "definitely-not-an-agent"
        assert len(successes) == 2
