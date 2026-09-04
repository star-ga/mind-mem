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
    install_mcp_config,
    mcp_server_spec,
)


class TestAgentRegistry:
    def test_registry_has_all_known_clients(self) -> None:
        expected = {
            "claude-code",
            "codex",
            "gemini",
            "cursor",
            "windsurf",
            "aider",
            "openclaw",
            "nanoclaw",
            "nemoclaw",
            "continue",
            "cline",
            "roo",
            "zed",
            "copilot",
            "cody",
            "qodo",
        }
        assert set(AGENT_REGISTRY.keys()) >= expected, f"missing: {expected - set(AGENT_REGISTRY.keys())}"

    def test_every_spec_has_required_fields(self) -> None:
        for name, spec in AGENT_REGISTRY.items():
            assert spec.name == name
            assert spec.description
            assert spec.config_fmt
            assert spec.path_tmpl


class TestInstallConfigForEveryAgent:
    @pytest.mark.parametrize("agent", sorted(AGENT_REGISTRY.keys()))
    def test_dry_run_returns_content_for_every_agent(self, agent: str, tmp_path: Path) -> None:
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
        path.write_text("# my existing rules\nsome user content\n", encoding="utf-8")
        install_config("cursor", str(tmp_path), dry_run=False)
        body = path.read_text(encoding="utf-8")
        assert "my existing rules" in body
        assert "some user content" in body
        assert "# mind-mem" in body

    def test_text_block_idempotent(self, tmp_path: Path) -> None:
        install_config("cursor", str(tmp_path))
        install_config("cursor", str(tmp_path))
        install_config("cursor", str(tmp_path))
        body = (tmp_path / ".cursorrules").read_text(encoding="utf-8")
        assert body.count("# mind-mem") == 1

    def test_json_merge_preserves_user_keys(self, tmp_path: Path) -> None:
        continue_dir = tmp_path / ".continue"
        continue_dir.mkdir()
        # pre-existing user config
        path = continue_dir / "config.json"
        path.write_text(json.dumps({"models": [{"title": "my-llm"}], "theme": "dark"}), encoding="utf-8")
        # set HOME so json-continue target resolves into tmp_path
        import mind_mem.hook_installer as hi

        spec = hi.AGENT_REGISTRY["continue"]
        # override path_tmpl to tmp-local
        hi.AGENT_REGISTRY["continue"] = AgentSpec(
            name=spec.name,
            description=spec.description,
            config_fmt=spec.config_fmt,
            path_tmpl=str(path),
            content_tmpl=spec.content_tmpl,
            detect_paths=spec.detect_paths,
            detect_binaries=spec.detect_binaries,
            always_offer=spec.always_offer,
        )
        try:
            install_config("continue", str(tmp_path), dry_run=False)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            assert loaded["theme"] == "dark"
            assert loaded["models"][0]["title"] == "my-llm"
            assert "systemMessage" in loaded
            assert "mind-mem" in loaded["systemMessage"]
        finally:
            hi.AGENT_REGISTRY["continue"] = spec


class TestClawFamily:
    def test_openclaw_nanoclaw_nemoclaw_share_hook_shape(self, tmp_path: Path) -> None:
        # All three claw variants route through _merge_openclaw_hooks.
        for agent in ("openclaw", "nanoclaw", "nemoclaw"):
            result = install_config(agent, str(tmp_path), dry_run=True)
            content = json.loads(result["content"])
            hooks = content["hooks"]["internal"]["entries"]["mind-mem"]
            assert hooks["enabled"] is True
            assert hooks["workspace"] == str(tmp_path)
            assert "commands" in hooks

    def test_claw_inject_command_is_a_runnable_mm_invocation(self, tmp_path: Path) -> None:
        # Regression: the claw inject hook shipped
        # ``mm inject --agent openclaw --workspace <ws>``, which fails on
        # BOTH counts on every fire — ``mm inject`` has no ``--workspace``
        # flag (argparse rejects it: "unrecognized arguments") and
        # ``openclaw`` is a client-registry name, not a KNOWN mm agent
        # (raises UnknownAgentError; valid tags live in KNOWN_AGENTS). So
        # openclaw received zero memory injection until this was fixed.
        # The installed command must be a real ``mm inject`` invocation:
        # a KNOWN_AGENTS tag and no ``--workspace`` (the workspace
        # resolves from the MIND_MEM_WORKSPACE env / entry workspace).
        from mind_mem.agent_bridge import KNOWN_AGENTS

        for agent in ("openclaw", "nanoclaw", "nemoclaw"):
            result = install_config(agent, str(tmp_path), dry_run=True)
            content = json.loads(result["content"])
            cmd = content["hooks"]["internal"]["entries"]["mind-mem"]["commands"]["inject"]
            assert "--workspace" not in cmd, f"{agent} inject hook uses --workspace, which `mm inject` rejects: {cmd!r}"
            parts = cmd.split()
            assert "--agent" in parts, f"{agent} inject hook missing --agent: {cmd!r}"
            tag = parts[parts.index("--agent") + 1]
            assert tag in KNOWN_AGENTS, (
                f"{agent} inject hook uses --agent {tag!r}, not a KNOWN mm agent "
                f"({', '.join(KNOWN_AGENTS)}) → UnknownAgentError on every fire"
            )


class TestClaudeCodeHookFormat:
    """v1.9.2: hook entries must use Claude Code's nested shape.

    Regression: pre-v1.9.2 wrote bare ``{"command": "..."}`` entries,
    which Claude Code accepted silently but blocked at runtime with
    "Can't edit settings.json directly — it's a protected file" style
    errors on every Stop / PostToolUse. The fix installs the nested
    ``{"matcher": "", "hooks": [{"type": "command", ...}]}`` shape
    and also replaces two commands (``mm capture --stdin``,
    ``mm vault status``) that aren't real CLI subcommands yet.
    """

    def test_claude_code_hooks_use_nested_matcher_shape(self, tmp_path: Path) -> None:
        result = install_config("claude-code", str(tmp_path), dry_run=True)
        content = json.loads(result["content"])
        for event in ("SessionStart", "Stop"):
            assert event in content["hooks"], f"{event} missing from hooks"
            entries = content["hooks"][event]
            assert isinstance(entries, list) and entries, f"{event} empty"
            for entry in entries:
                # Reject the legacy flat shape outright.
                assert "hooks" in entry, (
                    f"{event} entry uses flat {{command: ...}} shape; required nested {{matcher, hooks: [...]}} shape. Got: {entry!r}"
                )
                assert entry.get("matcher", None) is not None, f"{event} entry missing 'matcher' field"
                assert isinstance(entry["hooks"], list), f"{event} entry 'hooks' must be a list"
                for inner in entry["hooks"]:
                    assert inner.get("type") == "command"
                    assert inner.get("command"), "inner command is empty"

    def test_no_broken_mm_subcommands_installed(self, tmp_path: Path) -> None:
        # ``mm capture --stdin`` and ``mm vault status`` are not real
        # CLI subcommands in the current ``mm`` release. Installing a
        # hook that always errors blocks every future Stop/PostToolUse.
        result = install_config("claude-code", str(tmp_path), dry_run=True)
        text = result["content"]
        assert "mm capture" not in text, (
            "PostToolUse hook points at mm capture which is not a real CLI subcommand yet — will block every tool call"
        )
        assert "mm vault status" not in text, (
            "Stop hook points at mm vault status which is not a real CLI subcommand — `mm vault` only has {scan, write}"
        )

    def test_install_is_idempotent_across_both_shapes(self, tmp_path: Path) -> None:
        # Pre-populate settings.json with the LEGACY flat shape a
        # pre-v1.9.2 installer would have written. Running the new
        # installer must NOT duplicate and MUST migrate to the nested
        # shape in place.

        legacy = {
            "hooks": {
                "SessionStart": [{"command": "mm status"}],
                "Stop": [{"command": "mm status"}],
            }
        }
        settings_path = AGENT_REGISTRY["claude-code"].expand_path(str(tmp_path))
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(legacy, f)

        result = install_config("claude-code", str(tmp_path), dry_run=False)
        assert result["written"] is True
        loaded = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        # After migration: the legacy flat entry is upgraded in place (never
        # duplicated), and 5.0.2's second SessionStart command is appended.
        # Counted per event rather than assumed uniform -- the installer
        # wants two commands on SessionStart and one on Stop, and a test
        # that asserted "one everywhere" would have to be relaxed rather
        # than corrected every time a hook is added.
        expected = {"SessionStart": 2, "Stop": 1}
        for event, count in expected.items():
            assert len(loaded["hooks"][event]) == count, loaded["hooks"][event]
            for entry in loaded["hooks"][event]:
                assert "hooks" in entry
                assert "command" not in entry  # legacy flat removed

        # Running install again should be a no-op (or at least not
        # add a duplicate).
        install_config("claude-code", str(tmp_path), dry_run=False)
        loaded2 = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        for event, count in expected.items():
            assert len(loaded2["hooks"][event]) == count, f"{event} duplicated on re-install"

    def test_session_start_installs_the_query_free_resume_verb(self, tmp_path: Path) -> None:
        """The hook that puts memory in front of a session, not a status blob.

        ``mm inject`` needs a positional query a SessionStart hook cannot
        supply, which is why this event pointed at ``mm status`` and carried
        a note promising a query-free verb. ``mm resume-on-start`` is that
        verb; if it stops being installed, the SessionStart hook is back to
        injecting nothing but workspace counters.
        """
        result = install_config("claude-code", str(tmp_path), dry_run=True)
        content = json.loads(result["content"])
        commands = [inner.get("command") for entry in content["hooks"]["SessionStart"] for inner in entry.get("hooks", [])]
        assert "mm resume-on-start" in commands, commands
        assert "mm status" in commands, commands

    def test_every_installed_command_is_a_verb_the_cli_accepts(self, tmp_path: Path) -> None:
        """The census that would have caught ``mm capture`` and
        ``mm vault status`` before they shipped into a hook that fired on
        every tool call. Read off the parser rather than a hardcoded list,
        so a verb renamed in the CLI fails here instead of at a user's
        session start."""
        import argparse

        from mind_mem.mm_cli import build_parser

        parser = build_parser()
        subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]
        verbs = set(subparsers.choices)
        assert "resume-on-start" in verbs, "positive control: the parser has no such verb"

        result = install_config("claude-code", str(tmp_path), dry_run=True)
        content = json.loads(result["content"])
        for event, entries in content["hooks"].items():
            for entry in entries:
                for inner in entry.get("hooks", []):
                    command = inner.get("command", "")
                    assert command.startswith("mm "), command
                    verb = command.split()[1]
                    assert verb in verbs, f"{event} hook runs `{command}`; `mm {verb}` is not a CLI verb"


class TestNativeMCPConfig:
    """v3.1.0 MCP server registration writers."""

    @pytest.mark.parametrize(
        "agent",
        ["codex", "gemini", "cursor", "windsurf", "continue", "cline", "roo", "zed"],
    )
    def test_mcp_dry_run_for_every_supported_agent(self, agent: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Redirect the real per-client MCP path into tmp so we don't
        # touch real config files on the developer's machine.
        spec = AGENT_REGISTRY[agent]
        # Move the per-client mcp_path into tmp_path
        relocated = AgentSpec(
            name=spec.name,
            description=spec.description,
            config_fmt=spec.config_fmt,
            path_tmpl=spec.path_tmpl,
            content_tmpl=spec.content_tmpl,
            detect_paths=spec.detect_paths,
            detect_binaries=spec.detect_binaries,
            always_offer=spec.always_offer,
            mcp_fmt=spec.mcp_fmt,
            mcp_path_tmpl=str(tmp_path / f"{agent}_mcp.json"),
        )
        monkeypatch.setitem(AGENT_REGISTRY, agent, relocated)
        result = install_mcp_config(agent, str(tmp_path), dry_run=True)
        assert result["agent"] == agent
        assert result["written"] is False
        assert "mind-mem" in result["content"]

    def test_mcp_skip_for_unsupported_agent(self, tmp_path: Path) -> None:
        result = install_mcp_config("aider", str(tmp_path))
        assert result["skipped"] is True
        assert "does not support native MCP" in result["reason"]

    def test_mcp_writer_preserves_user_keys(self, tmp_path: Path) -> None:
        # Pre-populate a Cursor-style mcp.json with user content.
        path = tmp_path / "mcp.json"
        path.write_text(
            json.dumps(
                {
                    "mcpServers": {"my-tool": {"command": "node", "args": ["x.js"]}},
                    "schemaVersion": 1,
                }
            ),
            encoding="utf-8",
        )

        # Relocate cursor's mcp_path into tmp.
        spec = AGENT_REGISTRY["cursor"]
        relocated = AgentSpec(
            name=spec.name,
            description=spec.description,
            config_fmt=spec.config_fmt,
            path_tmpl=spec.path_tmpl,
            content_tmpl=spec.content_tmpl,
            detect_paths=spec.detect_paths,
            detect_binaries=spec.detect_binaries,
            always_offer=spec.always_offer,
            mcp_fmt=spec.mcp_fmt,
            mcp_path_tmpl=str(path),
        )
        original = AGENT_REGISTRY["cursor"]
        AGENT_REGISTRY["cursor"] = relocated
        try:
            install_mcp_config("cursor", str(tmp_path))
            loaded = json.loads(path.read_text(encoding="utf-8"))
            # User's existing mcpServer survives.
            assert loaded["mcpServers"]["my-tool"]["command"] == "node"
            # mind-mem now alongside it.
            assert "mind-mem" in loaded["mcpServers"]
            # User's other top-level keys survive.
            assert loaded["schemaVersion"] == 1
        finally:
            AGENT_REGISTRY["cursor"] = original


class TestMcpServerSpecHelper:
    def test_returns_command_args_env(self) -> None:
        spec = mcp_server_spec("/tmp/ws")
        assert "command" in spec
        assert "args" in spec and len(spec["args"]) == 1
        assert spec["env"] == {"MIND_MEM_WORKSPACE": "/tmp/ws"}


class TestCopilotAlwaysOffer:
    def test_copilot_flagged_always_offer(self) -> None:
        assert AGENT_REGISTRY["copilot"].always_offer is True


class TestDetectInstalledAgents:
    def test_detection_always_includes_copilot(self, tmp_path: Path) -> None:
        detected = detect_installed_agents(str(tmp_path))
        assert "copilot" in detected  # always_offer=True

    def test_detection_returns_list_of_registered_names(self, tmp_path: Path) -> None:
        detected = detect_installed_agents(str(tmp_path))
        for name in detected:
            assert name in AGENT_REGISTRY


class TestInstallAll:
    def test_install_all_explicit_list_dry_run(self, tmp_path: Path) -> None:
        # Opt out of MCP writes so we get one result per agent (hook phase only).
        results = install_all(
            str(tmp_path),
            dry_run=True,
            agents=["cursor", "windsurf", "copilot"],
            include_mcp=False,
        )
        assert len(results) == 3
        assert {r["agent"] for r in results} == {"cursor", "windsurf", "copilot"}
        for r in results:
            assert r["written"] is False

    def test_install_all_with_mcp_emits_hook_plus_mcp_phase(self, tmp_path: Path) -> None:
        # Default: include_mcp=True → two results per MCP-aware agent.
        results = install_all(
            str(tmp_path),
            dry_run=True,
            agents=["gemini", "aider", "copilot"],
        )
        # 3 agents × 2 phases = 6 result dicts.
        assert len(results) == 6
        phases = sorted({r.get("phase", "?") for r in results})
        assert phases == ["hook", "mcp"]
        # aider + copilot have no mcp_fmt → their MCP phase is a skip-result.
        mcp_results = [r for r in results if r.get("phase") == "mcp"]
        skipped_no_mcp = [r for r in mcp_results if r.get("reason") == "agent does not support native MCP"]
        assert {r["agent"] for r in skipped_no_mcp} == {"aider", "copilot"}

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
        # One bad agent shouldn't stop the others. include_mcp=False keeps
        # results count at one per agent so the error count assertion is
        # unambiguous.
        results = install_all(
            str(tmp_path),
            agents=["cursor", "definitely-not-an-agent", "windsurf"],
            include_mcp=False,
        )
        errors = [r for r in results if "error" in r]
        successes = [r for r in results if r.get("written")]
        assert len(errors) == 1
        assert errors[0]["agent"] == "definitely-not-an-agent"
        assert len(successes) == 2
