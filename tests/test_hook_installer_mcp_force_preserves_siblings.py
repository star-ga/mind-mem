"""Regression test: install_mcp_config(force=True) must merge, not clobber.

The bug: the MCP writer path gated the load of the existing config on
``os.path.isfile(path) and not force``, so ``force=True`` handed the
writer an empty dict (JSON clients) or an empty string (TOML clients)
and the result was written unconditionally.  Every unrelated key in the
user's config file — editor settings, other MCP servers — was silently
destroyed.  The sibling ``install_config`` already carries the fix; this
pins the same guarantee for ``install_mcp_config``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

from mind_mem.hook_installer import install_mcp_config


def _fake_home(tmp_path: Path):
    # ``os.path.expanduser("~")`` reads USERPROFILE on Windows and HOME
    # on POSIX — patch both so nothing touches the real user home.
    return mock.patch.dict(
        os.environ,
        {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)},
    )


def test_install_mcp_config_force_preserves_json_siblings(tmp_path: Path) -> None:
    """Zed settings.json keeps its editor keys and other context servers."""
    cfg = tmp_path / ".config" / "zed" / "settings.json"
    cfg.parent.mkdir(parents=True)
    prior = {
        "theme": "sentinel-theme",
        "vim_mode": True,
        "languages": {"Python": {"tab_size": 4}},
        "context_servers": {"other-server": {"source": "custom", "command": "other"}},
    }
    cfg.write_text(json.dumps(prior, indent=2), encoding="utf-8")

    with _fake_home(tmp_path):
        res = install_mcp_config("zed", str(tmp_path), force=True)

    assert res["written"] is True
    after = json.loads(cfg.read_text(encoding="utf-8"))
    assert after["theme"] == "sentinel-theme"
    assert after["vim_mode"] is True
    assert after["languages"] == {"Python": {"tab_size": 4}}
    assert "other-server" in after["context_servers"], "sibling MCP server dropped"
    assert after["context_servers"]["other-server"]["command"] == "other"
    assert "mind-mem" in after["context_servers"]


def test_install_mcp_config_force_preserves_generic_json_siblings(tmp_path: Path) -> None:
    """The generic ``mcpServers`` writer (Gemini/Continue/Cursor/Cline)."""
    cfg = tmp_path / ".gemini" / "settings.json"
    cfg.parent.mkdir(parents=True)
    prior = {
        "selectedAuthType": "sentinel-auth",
        "mcpServers": {"other-server": {"command": "other", "args": []}},
    }
    cfg.write_text(json.dumps(prior, indent=2), encoding="utf-8")

    with _fake_home(tmp_path):
        install_mcp_config("gemini", str(tmp_path), force=True)

    after = json.loads(cfg.read_text(encoding="utf-8"))
    assert after["selectedAuthType"] == "sentinel-auth"
    assert "other-server" in after["mcpServers"], "sibling MCP server dropped"
    assert "mind-mem" in after["mcpServers"]


def test_install_mcp_config_force_preserves_toml_siblings(tmp_path: Path) -> None:
    """Codex config.toml keeps its top-level keys and other server sections."""
    cfg = tmp_path / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        'model = "sentinel-model"\napproval_policy = "on-request"\n\n[mcp_servers.other]\ncommand = "other"\nargs = []\n',
        encoding="utf-8",
    )

    with _fake_home(tmp_path):
        install_mcp_config("codex", str(tmp_path), force=True)

    after = cfg.read_text(encoding="utf-8")
    assert 'model = "sentinel-model"' in after
    assert 'approval_policy = "on-request"' in after
    assert "[mcp_servers.other]" in after, "sibling MCP server section dropped"
    assert "[mcp_servers.mind-mem]" in after


def test_install_mcp_config_force_preserves_vibe_toml_entries(tmp_path: Path) -> None:
    """Vibe's flat ``mcp_servers`` array keeps its other entries."""
    cfg = tmp_path / ".vibe" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        'model = "sentinel-model"\nmcp_servers = [\n  { name = "other-server", command = "other", args = [] }\n]\n',
        encoding="utf-8",
    )

    with _fake_home(tmp_path):
        install_mcp_config("vibe", str(tmp_path), force=True)

    after = cfg.read_text(encoding="utf-8")
    assert 'model = "sentinel-model"' in after
    assert 'name = "other-server"' in after, "sibling MCP server entry dropped"
    assert 'name = "mind-mem"' in after


def test_install_mcp_config_force_still_rewrites_identical_config(tmp_path: Path) -> None:
    """``force`` keeps its meaning: write even when nothing changed.

    Guards against a "fix" that merely makes force behave like the
    non-force path (which would skip the write on an unchanged file).
    """
    cfg = tmp_path / ".gemini" / "settings.json"
    cfg.parent.mkdir(parents=True)
    with _fake_home(tmp_path):
        first = install_mcp_config("gemini", str(tmp_path))
        assert first["written"] is True
        second = install_mcp_config("gemini", str(tmp_path))
        assert second["skipped"] is True and second["written"] is False
        forced = install_mcp_config("gemini", str(tmp_path), force=True)

    assert forced["written"] is True
    assert forced["skipped"] is False


def test_install_mcp_config_force_dry_run_does_not_clobber(tmp_path: Path) -> None:
    """dry_run + force must preview the *merged* content, not a stub."""
    cfg = tmp_path / ".config" / "zed" / "settings.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"theme": "sentinel-theme"}, indent=2), encoding="utf-8")

    with _fake_home(tmp_path):
        res = install_mcp_config("zed", str(tmp_path), dry_run=True, force=True)

    assert res["written"] is False
    preview = json.loads(res["content"])
    assert preview["theme"] == "sentinel-theme"
    assert "mind-mem" in preview["context_servers"]
    # The on-disk file is untouched by a dry run.
    assert json.loads(cfg.read_text(encoding="utf-8")) == {"theme": "sentinel-theme"}
