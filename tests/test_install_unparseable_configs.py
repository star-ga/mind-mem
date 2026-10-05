# Copyright 2026 STARGA, Inc.
"""The installer never rewrites a client config it cannot parse.

Regression: ``install_config`` / ``install_mcp_config`` treated a config that
was not strict JSON (JSONC with comments, a typo, a non-object root) as ``{}``
and then wrote the mind-mem entry over it, destroying the user's settings for
Cursor, Windsurf, Zed, Gemini, Continue, Cline, Roo, Copilot CLI and the
JSON hook configs (Claude Code, OpenClaw family, Zed, Continue).

Every case below asserts the file is byte-identical afterwards, nothing was
written, no backup was taken, and the result carries a reason plus the
snippet to paste by hand. The positive controls prove a valid file IS written
(with a backup), so "unchanged" is not passing because nothing ever writes;
``TestGuardIsLoadBearing`` swaps the guard for the old lenient loader and
shows the same file is then destroyed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from mind_mem import hook_installer
from mind_mem.client_config_io import BACKUP_INFIX
from mind_mem.hook_installer import AGENT_REGISTRY, install_config, install_mcp_config

JSONC = b'// user settings - hand edited\n{\n  "theme": "dark", // keep\n  "mcpServers": {"other": {"command": "x"}},\n}\n'
INVALID = b'{"theme": "dark", "mcpServers": {"other": '
NON_OBJECT = b'["not", "an", "object"]\n'
BAD_UTF8 = b'{"theme": "\xff\xfe"}\n'

UNSAFE = {"jsonc": JSONC, "invalid": INVALID, "non-object": NON_OBJECT, "bad-utf8": BAD_UTF8}

MCP_JSON_AGENTS = sorted(n for n, s in AGENT_REGISTRY.items() if s.mcp_fmt in hook_installer._MCP_WRITERS_JSON)
HOOK_JSON_AGENTS = sorted(n for n, s in AGENT_REGISTRY.items() if s.config_fmt in hook_installer._JSON_MERGERS)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    assert os.path.expanduser("~") == str(h)
    return h


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    w = tmp_path / "ws"
    w.mkdir()
    return w


def _seed(path: str, data: bytes) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _backups(p: Path) -> list[Path]:
    return sorted(p.parent.glob(p.name + BACKUP_INFIX + "*"))


def _assert_refused(result: dict[str, Any], p: Path, before: bytes, marker: str = "mind-mem") -> None:
    assert p.read_bytes() == before, "config was modified"
    assert result["written"] is False
    assert result["skipped"] is True
    assert result["merged"] is False
    assert result.get("reason"), "a refusal must say why"
    assert marker in result["content"], "the snippet to paste must be returned"
    assert _backups(p) == [], "a refused file must not produce a backup"
    leftovers = [q.name for q in p.parent.iterdir() if q.name.startswith(".mind-mem-")]
    assert leftovers == [], leftovers


def test_the_agent_lists_are_not_empty() -> None:
    """Guard against a parametrize that silently selects nothing."""
    assert {"cursor", "windsurf", "zed", "gemini"} <= set(MCP_JSON_AGENTS)
    assert {"claude-code", "openclaw", "zed", "continue"} <= set(HOOK_JSON_AGENTS)


class TestMcpConfigRefusesUnsafeFiles:
    @pytest.mark.parametrize("force", [False, True], ids=["default", "force"])
    @pytest.mark.parametrize("kind", sorted(UNSAFE))
    @pytest.mark.parametrize("agent", MCP_JSON_AGENTS)
    def test_left_byte_identical(self, agent: str, kind: str, force: bool, home: Path, ws: Path) -> None:
        p = _seed(AGENT_REGISTRY[agent].expand_mcp_path(str(ws)), UNSAFE[kind])
        result = install_mcp_config(agent, str(ws), force=force)
        _assert_refused(result, p, UNSAFE[kind])

    def test_dry_run_reports_the_refusal_too(self, home: Path, ws: Path) -> None:
        p = _seed(AGENT_REGISTRY["cursor"].expand_mcp_path(str(ws)), JSONC)
        result = install_mcp_config("cursor", str(ws), dry_run=True)
        _assert_refused(result, p, JSONC)
        assert "JSONC" in result["reason"]
        assert json.loads(result["content"])["mcpServers"]["mind-mem"]["command"]


class TestHookConfigRefusesUnsafeFiles:
    @pytest.mark.parametrize("force", [False, True], ids=["default", "force"])
    @pytest.mark.parametrize("kind", sorted(UNSAFE))
    @pytest.mark.parametrize("agent", HOOK_JSON_AGENTS)
    def test_left_byte_identical(self, agent: str, kind: str, force: bool, home: Path, ws: Path) -> None:
        p = _seed(AGENT_REGISTRY[agent].expand_path(str(ws)), UNSAFE[kind])
        result = install_config(agent, str(ws), force=force)
        # The snippet is exactly what a fresh install would write.
        merger = hook_installer._JSON_MERGERS[AGENT_REGISTRY[agent].config_fmt]
        expected = json.dumps(merger({}, str(ws))[0], indent=2)
        _assert_refused(result, p, UNSAFE[kind], marker=expected)


class TestUnreadableTextConfigs:
    def test_mcp_toml_unreadable_is_not_blanked(self, home: Path, ws: Path) -> None:
        p = _seed(AGENT_REGISTRY["codex"].expand_mcp_path(str(ws)), BAD_UTF8)
        result = install_mcp_config("codex", str(ws))
        _assert_refused(result, p, BAD_UTF8)

    def test_text_block_unreadable_is_not_blanked(self, home: Path, ws: Path) -> None:
        p = _seed(AGENT_REGISTRY["cursor"].expand_path(str(ws)), BAD_UTF8)
        result = install_config("cursor", str(ws))
        _assert_refused(result, p, BAD_UTF8, marker="mind-mem")


class TestValidFilesStillWrite:
    """Positive controls: the guard must not turn every install into a no-op."""

    def test_mcp_write_keeps_siblings_and_backs_up(self, home: Path, ws: Path) -> None:
        before = json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "x"}}}).encode()
        p = _seed(AGENT_REGISTRY["cursor"].expand_mcp_path(str(ws)), before)
        result = install_mcp_config("cursor", str(ws))
        assert result["written"] is True
        after = json.loads(p.read_text(encoding="utf-8"))
        assert after["theme"] == "dark"
        assert after["mcpServers"]["other"] == {"command": "x"}
        assert "mind-mem" in after["mcpServers"]
        backups = _backups(p)
        assert len(backups) == 1 and backups[0].read_bytes() == before
        assert result["backup"] == str(backups[0])

    def test_hook_write_keeps_siblings_and_backs_up(self, home: Path, ws: Path) -> None:
        before = json.dumps({"model": "keep-me", "env": {"A": "1"}}).encode()
        p = _seed(AGENT_REGISTRY["claude-code"].expand_path(str(ws)), before)
        result = install_config("claude-code", str(ws))
        assert result["written"] is True
        after = json.loads(p.read_text(encoding="utf-8"))
        assert after["model"] == "keep-me" and after["env"] == {"A": "1"}
        assert "SessionStart" in after["hooks"]
        assert [b.read_bytes() for b in _backups(p)] == [before]

    def test_new_file_is_created_without_a_backup(self, home: Path, ws: Path) -> None:
        p = Path(AGENT_REGISTRY["windsurf"].expand_mcp_path(str(ws)))
        assert not p.exists()
        result = install_mcp_config("windsurf", str(ws))
        assert result["written"] is True and "backup" not in result
        assert "mind-mem" in json.loads(p.read_text(encoding="utf-8"))["mcpServers"]
        assert _backups(p) == []

    def test_empty_file_counts_as_new(self, home: Path, ws: Path) -> None:
        p = _seed(AGENT_REGISTRY["zed"].expand_mcp_path(str(ws)), b"  \n")
        result = install_mcp_config("zed", str(ws))
        assert result["written"] is True
        assert "mind-mem" in json.loads(p.read_text(encoding="utf-8"))["context_servers"]

    def test_unchanged_rerun_writes_no_new_backup(self, home: Path, ws: Path) -> None:
        install_mcp_config("cursor", str(ws))
        p = Path(AGENT_REGISTRY["cursor"].expand_mcp_path(str(ws)))
        second = install_mcp_config("cursor", str(ws))
        assert second["skipped"] is True and second["written"] is False
        assert _backups(p) == []

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_symlinked_config_is_written_through(self, home: Path, ws: Path, tmp_path: Path) -> None:
        real = tmp_path / "dotfiles" / "mcp.json"
        real.parent.mkdir()
        real.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
        link = Path(AGENT_REGISTRY["cursor"].expand_mcp_path(str(ws)))
        link.parent.mkdir(parents=True)
        link.symlink_to(real)
        install_mcp_config("cursor", str(ws))
        assert link.is_symlink()
        assert "mind-mem" in json.loads(real.read_text(encoding="utf-8"))["mcpServers"]


class TestGuardIsLoadBearing:
    """Mutation twin: restore the pre-fix lenient loader and the file is destroyed.

    If this test ever fails, the byte-identity assertions above are no longer
    exercising the guard (for example because the installer stopped reaching
    the parse step), and they would pass vacuously.
    """

    def test_lenient_loader_overwrites_a_jsonc_config(self, home: Path, ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def lenient(path: str) -> dict[str, Any] | None:
            if not os.path.isfile(path):
                return None
            try:
                with open(path, encoding="utf-8") as fh:
                    loaded = json.load(fh)
            except (OSError, ValueError):
                return {}
            return loaded if isinstance(loaded, dict) else {}

        p = _seed(AGENT_REGISTRY["cursor"].expand_mcp_path(str(ws)), JSONC)
        monkeypatch.setattr(hook_installer, "read_json_object", lenient)
        result = install_mcp_config("cursor", str(ws))
        assert result["written"] is True
        assert p.read_bytes() != JSONC
        assert "theme" not in json.loads(p.read_text(encoding="utf-8"))
