# Copyright 2026 STARGA, Inc.
"""OpenCode (1.x and 2.x) integration: detection, MCP merge, AGENTS.md."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind_mem import hook_installer
from mind_mem.hook_installer import (
    AGENT_REGISTRY,
    detect_installed_agents,
    install_all,
    install_config,
    install_mcp_config,
    mcp_server_spec,
)
from mind_mem.opencode_config import merge_opencode_mcp


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``~`` at an empty tmp dir and hide every real binary."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    monkeypatch.setattr(hook_installer._shutil, "which", lambda _name: None)
    return h


def _cfg(home: Path) -> Path:
    return home / ".config" / "opencode" / "opencode.json"


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(path.name + ".bak-mind-mem-*"))


class TestRegistry:
    def test_opencode_is_registered_with_mcp(self) -> None:
        spec = AGENT_REGISTRY["opencode"]
        assert spec.mcp_fmt == "mcp-json-opencode"
        assert spec.mcp_path_tmpl.endswith("/.config/opencode/opencode.json")
        assert spec.path_tmpl.endswith("/.config/opencode/AGENTS.md")
        assert "opencode" in spec.detect_binaries


class TestDetection:
    def test_not_detected_without_signals(self, home: Path) -> None:
        assert "opencode" not in detect_installed_agents(str(home / "ws"))

    def test_detected_by_config_dir(self, home: Path) -> None:
        (home / ".config" / "opencode").mkdir(parents=True)
        assert "opencode" in detect_installed_agents(str(home / "ws"))

    def test_detected_by_v2_install_dir(self, home: Path) -> None:
        (home / ".opencode" / "bin").mkdir(parents=True)
        assert "opencode" in detect_installed_agents(str(home / "ws"))

    def test_detected_by_binary(self, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            hook_installer._shutil,
            "which",
            lambda name: "/usr/bin/opencode" if name == "opencode" else None,
        )
        assert "opencode" in detect_installed_agents(str(home / "ws"))


class TestMcpFreshInstall:
    def test_creates_v1_compatible_entry(self, home: Path) -> None:
        ws = str(home / "ws")
        result = install_mcp_config("opencode", ws)
        assert result["written"] is True
        assert result["dialect"] == "v1-compatible"
        assert "backup" not in result  # nothing existed to back up
        loaded = json.loads(_cfg(home).read_text(encoding="utf-8"))
        entry = loaded["mcp"]["mind-mem"]
        srv = mcp_server_spec(ws)
        assert entry == {
            "type": "local",
            "command": [srv["command"], *srv["args"]],
            "environment": {"MIND_MEM_WORKSPACE": ws},
            "enabled": True,
        }

    def test_prefers_existing_jsonc_file(self, home: Path) -> None:
        jsonc = _cfg(home).with_suffix(".jsonc")
        _write(jsonc, {"model": "x/y"})
        result = install_mcp_config("opencode", str(home / "ws"))
        assert Path(result["path"]) == jsonc
        assert not _cfg(home).exists()
        loaded = json.loads(jsonc.read_text(encoding="utf-8"))
        assert loaded["model"] == "x/y"
        assert "mind-mem" in loaded["mcp"]

    def test_dry_run_writes_nothing(self, home: Path) -> None:
        result = install_mcp_config("opencode", str(home / "ws"), dry_run=True)
        assert result["written"] is False
        assert "mind-mem" in result["content"]
        assert not _cfg(home).exists()


class TestMcpMerge:
    def test_preserves_user_keys_and_servers_and_backs_up(self, home: Path) -> None:
        user = {
            "$schema": "https://opencode.ai/config.json",
            "model": "anthropic/some-model",
            "provider": {"x": {"options": {"baseURL": "http://localhost"}}},
            "mcp": {"context7": {"type": "remote", "url": "https://mcp.example/mcp"}},
        }
        _write(_cfg(home), user)
        original = _cfg(home).read_text(encoding="utf-8")

        result = install_mcp_config("opencode", str(home / "ws"))

        assert result["written"] is True
        loaded = json.loads(_cfg(home).read_text(encoding="utf-8"))
        for key in ("$schema", "model", "provider"):
            assert loaded[key] == user[key]
        assert loaded["mcp"]["context7"] == user["mcp"]["context7"]
        assert loaded["mcp"]["mind-mem"]["type"] == "local"
        backups = _backups(_cfg(home))
        assert backups == [Path(result["backup"])]
        assert backups[0].read_text(encoding="utf-8") == original

    def test_idempotent_second_run_no_write_no_backup(self, home: Path) -> None:
        _write(_cfg(home), {"model": "a/b"})
        first = install_mcp_config("opencode", str(home / "ws"))
        after_first = _cfg(home).read_text(encoding="utf-8")
        second = install_mcp_config("opencode", str(home / "ws"))
        assert first["written"] is True
        assert second["written"] is False
        assert second["skipped"] is True
        assert _cfg(home).read_text(encoding="utf-8") == after_first
        assert len(_backups(_cfg(home))) == 1

    def test_v2_native_file_gets_native_entry(self, home: Path) -> None:
        _write(
            _cfg(home),
            {
                "mcp": {
                    "timeout": {"startup": 45000},
                    "servers": {"playwright": {"type": "local", "command": ["bunx", "@playwright/mcp"]}},
                }
            },
        )
        ws = str(home / "ws")
        result = install_mcp_config("opencode", ws)
        assert result["dialect"] == "v2-native"
        mcp = json.loads(_cfg(home).read_text(encoding="utf-8"))["mcp"]
        assert "mind-mem" not in mcp  # not duplicated at the V1 level
        assert mcp["timeout"] == {"startup": 45000}
        assert mcp["servers"]["playwright"]["command"] == ["bunx", "@playwright/mcp"]
        entry = mcp["servers"]["mind-mem"]
        assert entry["type"] == "local"
        assert entry["environment"] == {"MIND_MEM_WORKSPACE": ws}
        assert "enabled" not in entry  # V2 uses `disabled`, default false

    def test_v2_native_removes_stale_v1_duplicate(self) -> None:
        srv = {"command": "py", "args": ["s.py"], "env": {"MIND_MEM_WORKSPACE": "/w"}}
        existing = {"mcp": {"mind-mem": {"type": "local", "command": ["old"]}, "servers": {}}}
        out, changed, dialect = merge_opencode_mcp(existing, srv)
        assert dialect == "v2-native"
        assert changed is True
        assert "mind-mem" not in out["mcp"]
        assert out["mcp"]["servers"]["mind-mem"]["command"] == ["py", "s.py"]
        # input not mutated
        assert existing["mcp"]["mind-mem"] == {"type": "local", "command": ["old"]}

    def test_v1_server_named_servers_is_not_mistaken_for_v2(self) -> None:
        srv = {"command": "py", "args": [], "env": {}}
        existing = {"mcp": {"servers": {"type": "local", "command": ["x"]}}}
        out, _changed, dialect = merge_opencode_mcp(existing, srv)
        assert dialect == "v1-compatible"
        assert out["mcp"]["servers"] == {"type": "local", "command": ["x"]}
        assert "mind-mem" in out["mcp"]

    def test_updates_stale_entry_in_place(self, home: Path) -> None:
        _write(_cfg(home), {"mcp": {"mind-mem": {"type": "local", "command": ["/old/launcher"], "enabled": False}}})
        result = install_mcp_config("opencode", str(home / "ws"))
        assert result["written"] is True
        entry = json.loads(_cfg(home).read_text(encoding="utf-8"))["mcp"]["mind-mem"]
        assert entry["enabled"] is True
        assert entry["command"][0] != "/old/launcher"

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_symlinked_config_is_written_through(self, home: Path, tmp_path: Path) -> None:
        real = tmp_path / "dotfiles" / "opencode.json"
        _write(real, {"model": "a/b"})
        link = _cfg(home)
        link.parent.mkdir(parents=True)
        link.symlink_to(real)
        result = install_mcp_config("opencode", str(home / "ws"))
        assert result["written"] is True
        assert link.is_symlink()
        loaded = json.loads(real.read_text(encoding="utf-8"))
        assert loaded["model"] == "a/b"
        assert "mind-mem" in loaded["mcp"]

    def test_force_rewrites_but_keeps_siblings(self, home: Path) -> None:
        _write(_cfg(home), {"model": "a/b"})
        install_mcp_config("opencode", str(home / "ws"))
        forced = install_mcp_config("opencode", str(home / "ws"), force=True)
        assert forced["written"] is True
        assert json.loads(_cfg(home).read_text(encoding="utf-8"))["model"] == "a/b"


class TestMcpRefusesUnsafeFiles:
    @pytest.mark.parametrize(
        "body",
        [
            '{\n  // my comment\n  "model": "a/b",\n}\n',  # JSONC
            '{"model": ',  # truncated / invalid
            "[1, 2]",  # non-object root
            '{"mcp": ["not", "an", "object"]}',
        ],
    )
    def test_file_left_byte_identical(self, home: Path, body: str) -> None:
        path = _cfg(home)
        path.parent.mkdir(parents=True)
        path.write_text(body, encoding="utf-8")
        result = install_mcp_config("opencode", str(home / "ws"))
        assert result["written"] is False
        assert result["skipped"] is True
        assert "left untouched" in result["reason"]
        assert path.read_text(encoding="utf-8") == body
        assert _backups(path) == []
        # The user gets a snippet to paste by hand.
        assert json.loads(result["content"])["mcp"]["mind-mem"]["type"] == "local"

    def test_force_does_not_override_refusal(self, home: Path) -> None:
        path = _cfg(home)
        path.parent.mkdir(parents=True)
        body = '{\n  // keep me\n  "model": "a/b"\n}\n'
        path.write_text(body, encoding="utf-8")
        result = install_mcp_config("opencode", str(home / "ws"), force=True)
        assert result["written"] is False
        assert path.read_text(encoding="utf-8") == body


class TestInstructions:
    def test_agents_md_written_to_global_opencode_dir(self, home: Path) -> None:
        result = install_config("opencode", str(home / "ws"))
        path = home / ".config" / "opencode" / "AGENTS.md"
        assert Path(result["path"]) == path
        body = path.read_text(encoding="utf-8")
        assert "# mind-mem" in body
        assert "Memory Protocol" in body

    def test_agents_md_appends_once_and_keeps_user_text(self, home: Path) -> None:
        path = home / ".config" / "opencode" / "AGENTS.md"
        path.parent.mkdir(parents=True)
        path.write_text("# My rules\n- be terse\n", encoding="utf-8")
        install_config("opencode", str(home / "ws"))
        once = path.read_text(encoding="utf-8")
        second = install_config("opencode", str(home / "ws"))
        assert second["skipped"] is True
        assert path.read_text(encoding="utf-8") == once
        assert once.startswith("# My rules\n- be terse")
        assert once.count("Memory Protocol (mind-mem MCP") == 1


class TestInstallAll:
    def test_install_all_wires_hook_and_mcp(self, home: Path) -> None:
        results = install_all(str(home / "ws"), agents=["opencode"])
        phases = {r["phase"]: r for r in results}
        assert phases["hook"]["written"] is True
        assert phases["mcp"]["written"] is True
        assert os.path.isfile(phases["mcp"]["path"])
