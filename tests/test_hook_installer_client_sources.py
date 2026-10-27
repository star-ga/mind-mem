# Copyright 2026 STARGA, Inc.
"""Registry entries match each client's current documentation.

Every client's config path, format and MCP key were checked against the
vendor's own documentation; the source URL per client is recorded in the
"Verified sources" table of ``docs/client-integrations.md``. These tests pin
the corrections that check produced and require a source row for every
registry client, so a client cannot be added without one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import test_client_integrations as _ci
from test_client_integrations import WS, _platform

from mind_mem.hook_installer import AGENT_REGISTRY, install_mcp_config

# The throwaway-HOME fixture of the golden tests (HOME, APPDATA, PATH -> temp).
home = _ci.home

ROOT = Path(__file__).resolve().parent.parent
CLIENT_DOC = ROOT / "docs" / "client-integrations.md"
SOURCE_HEADER = "| Client | `mm install` id | Install mode | Source |"
_URL = re.compile(r"https://[^\s|)>]+")


def _source_rows() -> dict[str, list[str]]:
    lines = CLIENT_DOC.read_text(encoding="utf-8").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(SOURCE_HEADER)]
    assert len(starts) == 1, f"expected one table headed {SOURCE_HEADER!r}"
    rows: dict[str, list[str]] = {}
    for ln in lines[starts[0] + 2 :]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        ident = re.findall(r"`([a-z0-9][a-z0-9-]*)`", cells[1])
        assert ident, cells
        rows[ident[-1]] = cells
    return rows


def _mode(name: str) -> str:
    spec = AGENT_REGISTRY[name]
    return "auto MCP" if spec.mcp_fmt else ("manual MCP" if spec.manual else "none")


def test_every_registry_client_has_a_source_row() -> None:
    rows = _source_rows()
    assert set(rows) == set(AGENT_REGISTRY), (
        f"missing {sorted(set(AGENT_REGISTRY) - set(rows))}, extra {sorted(set(rows) - set(AGENT_REGISTRY))}"
    )


def test_every_source_row_cites_a_url_and_the_registry_mode() -> None:
    for name, cells in _source_rows().items():
        assert _URL.search(cells[3]), f"{name}: no https source URL in {cells[3]!r}"
        assert cells[2] == _mode(name), f"{name}: doc says {cells[2]!r}, registry is {_mode(name)!r}"


def test_positive_control_a_row_without_url_is_detected() -> None:
    assert not _URL.search("vendor docs (unverified)")
    assert _URL.search("[docs](https://example.com/mcp)")


# ------------------------------------------------- corrections from the check


def test_anythingllm_writes_the_desktop_storage_plugins_file(home: Path) -> None:
    # docs.anythingllm.com: <storage>/plugins/anythingllm_mcp_servers.json, and
    # the desktop storage dir is <appdata>/anythingllm-desktop/storage.
    storage = home / ".config" / "anythingllm-desktop" / "storage" / "plugins"
    storage.mkdir(parents=True)
    result = install_mcp_config("anythingllm", WS)
    assert result["written"] is True, result
    assert Path(result["path"]) == storage / "anythingllm_mcp_servers.json"
    assert not (home / ".config" / "AnythingLLM").exists()


def test_amp_settings_live_under_dot_config_on_windows_too(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ampcode.com: user settings are ~/.config/amp/settings.json; on Windows
    # %USERPROFILE%\.config\amp, not %APPDATA%.
    _platform(monkeypatch, "win32")
    result = install_mcp_config("amp", WS)
    assert Path(result["path"]) == home / ".config" / "amp" / "settings.json"
    assert not (home / "AppData").exists()


@pytest.mark.parametrize("agent", ["void", "fast-agent"])
def test_unverifiable_user_level_paths_are_manual(agent: str, home: Path) -> None:
    # Void: no maintained documentation of a user-level MCP file (sources
    # disagree on ~/.void vs ~/.void-editor). fast-agent: fast-agent.yaml is
    # discovered from the working directory; no per-user file is documented.
    spec = AGENT_REGISTRY[agent]
    assert spec.mcp_fmt == "" and spec.manual, spec
    before = sorted(p.name for p in home.rglob("*"))
    result = install_mcp_config(agent, WS)
    assert result["written"] is False and result["reason"] == "manual setup required"
    assert "mind-mem" in result["content"]
    assert sorted(p.name for p in home.rglob("*")) == before
