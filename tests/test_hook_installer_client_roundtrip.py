# Copyright 2026 STARGA, Inc.
"""Install -> uninstall round trip for every automatic-MCP client, on a throwaway HOME.

For each client with automatic MCP registration:

* install writes a file that parses in the client's own format (JSON, TOML
  or YAML) and holds the ``mind-mem`` server entry;
* uninstall then gives the user back exactly what they had: the original
  bytes when the file existed before (whatever its formatting), or no file
  at all when mind-mem created it.

Detection is checked to be read-only: probing for every client creates no
file and no directory.

Every negative assertion is paired with a positive control (the entry is
present before it is shown to be gone; the file exists before it is shown
to be removed). The real HOME is never touched: the ``home`` fixture points
``HOME``/``APPDATA``/``PATH`` at a temp directory and asserts it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import test_client_integrations as _ci
from test_client_integrations import GOLDEN, SEEDS, WS, _entry, _load, _nest, _platform

from mind_mem.client_config_io import BACKUP_INFIX
from mind_mem.hook_installer import AGENT_REGISTRY, detect_installed_agents, install_mcp_config, uninstall_mcp_config

# The throwaway-HOME fixture of the golden tests (HOME, APPDATA, PATH -> temp).
home = _ci.home

AUTO = sorted(GOLDEN)
# Continue merges only into a config.yaml it created itself, so there is no
# "file did not exist" case to round-trip for it (covered by
# test_continue_needs_its_config_yaml).
CREATES_FILE = [a for a in AUTO if a != "continue"]


def _seed_text(kind: str, keys: tuple[str, ...]) -> str:
    """A pre-existing config whose formatting differs from what the installer writes."""
    if kind == "json":
        doc = _nest(keys, {"other": {"command": "other-server", "args": []}}, {"theme": "dark", "nested": {"keep": [1, 2]}})
        return json.dumps(doc, indent=4)  # 4-space indent, no trailing newline
    return SEEDS[kind](keys)


def _setup(agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, tuple[str, ...]]:
    rel, (kind, keys, _expected), plat, guarded = GOLDEN[agent]
    _platform(monkeypatch, plat)
    path = home / rel
    if guarded:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path, kind, keys


def test_round_trip_selects_every_auto_mcp_client() -> None:
    auto = {n for n, s in AGENT_REGISTRY.items() if s.mcp_fmt}
    assert set(AUTO) == auto
    assert len(CREATES_FILE) == len(auto) - 1 >= 30, "positive control: the parametrisation is not empty"


@pytest.mark.parametrize("agent", CREATES_FILE)
def test_install_then_uninstall_removes_a_file_mind_mem_created(agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, kind, keys = _setup(agent, home, monkeypatch)
    assert not path.exists()
    result = install_mcp_config(agent, WS)
    assert result["written"] is True, result
    assert path.is_file(), "positive control: install created the file"
    assert _entry(kind, _load(kind, path), keys), "the written file parses and holds mind-mem"
    removed = uninstall_mcp_config(agent, WS)
    assert removed["removed"] is True, removed
    assert not path.exists(), f"{agent}: a file mind-mem created is removed on uninstall"
    assert not list(path.parent.glob(path.name + BACKUP_INFIX + "*")), "no backup litter for a file that never existed"


@pytest.mark.parametrize("agent", AUTO)
def test_install_then_uninstall_restores_existing_file_byte_identical(agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, kind, keys = _setup(agent, home, monkeypatch)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = _seed_text(kind, keys).encode("utf-8")
    path.write_bytes(original)
    result = install_mcp_config(agent, WS)
    assert result["written"] is True, result
    assert path.read_bytes() != original, "positive control: install changed the file"
    assert _entry(kind, _load(kind, path), keys), "the merged file parses and holds mind-mem"
    removed = uninstall_mcp_config(agent, WS)
    assert removed["removed"] is True, removed
    assert path.read_bytes() == original, f"{agent}: uninstall restores the user's bytes"
    assert result.get("backup"), "positive control: install took a backup"
    assert not list(path.parent.glob(path.name + BACKUP_INFIX + "*")), "the restored file leaves no backup litter"


# A pre-existing config with no MCP-server container yet (the common real
# case: a ~/.gemini/settings.json or Zed settings.json holding only settings).
_BARE_SEEDS = {
    "json": '{"theme": "dark"}\n',
    "toml": '# my settings\nmodel = "sentinel"\n',
    "toml-list": '# my settings\nmodel = "sentinel"\n',
    "yaml": "# my settings\nmodel: sentinel\n",
    "yaml-list": "name: Local\nversion: 1.0.0\nschema: v1\n",
}


@pytest.mark.parametrize("agent", AUTO)
def test_install_then_uninstall_restores_file_without_server_container(agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, kind, keys = _setup(agent, home, monkeypatch)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = _BARE_SEEDS[kind].encode("utf-8")
    path.write_bytes(original)
    assert keys[0] not in _load(kind, path) or {}, "seed holds no server container"
    result = install_mcp_config(agent, WS)
    assert result["written"] is True, result
    assert _entry(kind, _load(kind, path), keys), "positive control: the merged file holds mind-mem"
    removed = uninstall_mcp_config(agent, WS)
    assert removed["removed"] is True, removed
    assert removed.get("restored") is True, removed
    assert path.read_bytes() == original, f"{agent}: uninstall restores the user's bytes"
    assert result.get("backup"), "positive control: install took a backup"
    assert not list(path.parent.glob(path.name + BACKUP_INFIX + "*")), "the restored file leaves no backup litter"


@pytest.mark.parametrize("agent", ["cursor", "codex", "goose"])
def test_uninstall_keeps_edits_made_after_install(agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The backup is used only when it equals the post-removal content: a user
    edit made after install must survive (no stale backup is restored)."""
    path, kind, keys = _setup(agent, home, monkeypatch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_seed_text(kind, keys), encoding="utf-8")
    install_mcp_config(agent, WS)
    marker = {"json": ('"theme": "dark"', '"theme": "light"'), "toml": ('"sentinel"', '"edited"'), "yaml": ("sentinel", "edited")}
    old, new = marker[kind]
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    assert uninstall_mcp_config(agent, WS)["removed"] is True
    after = path.read_text(encoding="utf-8")
    assert new in after and "mind-mem" not in after


def test_uninstall_keeps_a_file_that_holds_other_content(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file created by install but later given other servers is not deleted."""
    path, kind, keys = _setup("cursor", home, monkeypatch)
    install_mcp_config("cursor", WS)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["mcpServers"]["mine"] = {"command": "mine"}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    assert uninstall_mcp_config("cursor", WS)["removed"] is True
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == {"mcpServers": {"mine": {"command": "mine"}}}


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


@pytest.mark.parametrize("plat", ["linux", "darwin", "win32"])
def test_detection_creates_no_files(plat: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, plat)
    before = _tree(home)
    detect_installed_agents(WS)
    assert _tree(home) == before, "detection on an empty HOME is read-only"
    # Positive control: with every non-glob detect dir present, detection
    # finds clients and still adds nothing.
    from mind_mem import client_paths

    for spec in AGENT_REGISTRY.values():
        for p in spec.detect_paths:
            if "*" not in p and "{ws}" not in p:
                Path(client_paths.expand(p, WS)).mkdir(parents=True, exist_ok=True)
    populated = _tree(home)
    found = detect_installed_agents(WS)
    assert len(found) > 30, found
    assert _tree(home) == populated
    assert not os.path.exists(WS), "the workspace path is not created by detection"


@pytest.mark.parametrize("agent", AUTO)
def test_uninstall_restores_crlf_file_byte_identical(agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A config saved with Windows line endings comes back with them, on every OS."""
    path, kind, keys = _setup(agent, home, monkeypatch)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = _seed_text(kind, keys).replace("\n", "\r\n").encode("utf-8")
    assert b"\r\n" in original, "positive control: the seed really has CRLF line endings"
    path.write_bytes(original)
    assert install_mcp_config(agent, WS)["written"] is True
    assert uninstall_mcp_config(agent, WS)["removed"] is True
    assert path.read_bytes() == original, f"{agent}: uninstall restores CRLF bytes"
