# Copyright 2026 STARGA, Inc.
"""Golden tests: one per MCP-capable client, against a throwaway HOME.

For every client with automatic MCP registration this checks, on a temp
home directory only:

* install writes exactly the expected entry at the expected path;
* a second install is byte-identical and reports ``skipped``;
* other servers and settings already in the file survive;
* uninstall removes only the mind-mem entry;
* detection finds the client by its config dir or its binary.

Manual-setup clients are checked to write nothing and print their steps.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Callable

import pytest

from mind_mem import client_mcp, client_paths, client_registry
from mind_mem.client_config_io import BACKUP_INFIX
from mind_mem.hook_installer import (
    AGENT_REGISTRY,
    detect_installed_agents,
    install_all,
    install_config,
    install_mcp_config,
    uninstall_all,
    uninstall_mcp_config,
)

try:
    import tomllib as _toml
except ImportError:  # pragma: no cover - Python 3.10
    _toml = pytest.importorskip("tomli")

CMD = "/opt/mind-mem/bin/python"
ARGS = ["/opt/mind-mem/mcp_server.py"]
WS = "/srv/mm-ws"
ENV = {"MIND_MEM_WORKSPACE": WS}
STD = {"command": CMD, "args": ARGS, "env": ENV}
OPENCODE = {"type": "local", "command": [CMD, *ARGS], "environment": ENV, "enabled": True}
CLINE = {**STD, "disabled": False, "autoApprove": []}
STDIO = {"type": "stdio", **STD}
GOOSE = {"enabled": True, "type": "stdio", "name": "mind-mem", "cmd": CMD, "args": ARGS, "envs": ENV, "timeout": 300}
_VSU = ".config/Code/User/globalStorage/"


def _json(keys: tuple[str, ...], entry: dict[str, Any]) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    return ("json", keys, entry)


# agent -> (path under HOME on its platform, (kind, key path, expected entry),
#           platform, guarded: directory must pre-exist)
G = dict[str, tuple[str, tuple[str, tuple[str, ...], Any], str, bool]]
GOLDEN: G = {
    "claude-code": (".claude.json", _json(("mcpServers",), STDIO), "linux", False),
    "codex": (".codex/config.toml", ("toml", ("mcp_servers",), STD), "linux", False),
    "grok-build": (".grok/config.toml", ("toml", ("mcp_servers",), STD), "linux", False),
    "open-interpreter": (".openinterpreter/config.toml", ("toml", ("mcp_servers",), STD), "linux", True),
    "vibe": (".vibe/config.toml", ("toml-list", ("mcp_servers",), {"name": "mind-mem", "transport": "stdio", **STD}), "linux", False),
    "opencode": (".config/opencode/opencode.json", _json(("mcp",), OPENCODE), "linux", False),
    "kilo": (".config/kilo/kilo.jsonc", _json(("mcp",), OPENCODE), "linux", False),
    "mimo-code": (".config/mimocode/mimocode.jsonc", _json(("mcp",), OPENCODE), "linux", False),
    "gemini": (".gemini/settings.json", _json(("mcpServers",), STD), "linux", False),
    "cursor": (".cursor/mcp.json", _json(("mcpServers",), STDIO), "linux", False),
    "windsurf": (".codeium/windsurf/mcp_config.json", _json(("mcpServers",), STD), "linux", False),
    "openclaw": (".openclaw/openclaw.json", _json(("mcp", "servers"), STD), "linux", False),
    "continue": (".continue/config.yaml", ("yaml-list", ("mcpServers",), {"name": "mind-mem", **STD}), "linux", True),
    "cline": (_VSU + "saoudrizwan.claude-dev/settings/cline_mcp_settings.json", _json(("mcpServers",), CLINE), "linux", False),
    "roo": (_VSU + "rooveterinaryinc.roo-cline/settings/mcp_settings.json", _json(("mcpServers",), STD), "linux", False),
    "zed": (".config/zed/settings.json", _json(("context_servers",), {"source": "custom", **STD}), "linux", False),
    "copilot-cli": (".copilot/mcp-config.json", _json(("mcpServers",), {"type": "local", **STD, "tools": ["*"]}), "linux", False),
    "claude-desktop": (".config/Claude/claude_desktop_config.json", _json(("mcpServers",), STD), "linux", True),
    "vscode": (".config/Code/User/mcp.json", _json(("servers",), STDIO), "linux", True),
    "visual-studio": (".mcp.json", _json(("servers",), STD), "win32", False),
    "qwen-code": (".qwen/settings.json", _json(("mcpServers",), STD), "linux", False),
    "kimi-code": (".kimi-code/mcp.json", _json(("mcpServers",), STD), "linux", False),
    "amp": (".config/amp/settings.json", _json(("amp.mcpServers",), STD), "linux", False),
    "goose": (
        ".config/goose/config.yaml",
        ("yaml", ("extensions",), GOOSE),
        "linux",
        False,
    ),
    "crush": (".config/crush/crush.json", _json(("mcp",), STDIO), "linux", False),
    "factory-droid": (".factory/mcp.json", _json(("mcpServers",), {"type": "stdio", **STD, "disabled": False}), "linux", False),
    "kiro": (".kiro/settings/mcp.json", _json(("mcpServers",), CLINE), "linux", False),
    "auggie": (".augment/settings.json", _json(("mcpServers",), STD), "linux", False),
    "openhands": (".openhands/mcp.json", _json(("mcpServers",), {**STD, "transport": "stdio"}), "linux", False),
    "junie": (".junie/mcp/mcp.json", _json(("mcpServers",), STD), "linux", False),
    "antigravity": (".gemini/config/mcp_config.json", _json(("mcpServers",), STD), "linux", False),
    "tabnine": (".tabnine/mcp_servers.json", _json(("mcpServers",), STD), "linux", False),
    "hermes-agent": (".hermes/config.yaml", ("yaml", ("mcp_servers",), STD), "linux", False),
    "anythingllm": (
        ".config/anythingllm-desktop/storage/plugins/anythingllm_mcp_servers.json",
        _json(("mcpServers",), {**STD, "anythingllm": {"autoStart": True}}),
        "linux",
        True,
    ),
    "warp": (".warp/.mcp.json", _json(("mcpServers",), STD), "linux", True),
    "amazon-q": (".aws/amazonq/mcp.json", _json(("mcpServers",), STD), "linux", True),
    "lm-studio": (".lmstudio/mcp.json", _json(("mcpServers",), STD), "linux", True),
    "jan": (".local/share/Jan/data/mcp_config.json", _json(("mcpServers",), {**STD, "active": True}), "linux", True),
    "boltai": (".boltai/mcp.json", _json(("mcpServers",), STD), "darwin", True),
    "iflow": (".iflow/settings.json", _json(("mcpServers",), STD), "linux", True),
    "neovim-mcphub": (".config/mcphub/servers.json", _json(("mcpServers",), STD), "linux", True),
}

# A pre-existing config per dialect: another server plus an unrelated setting.
_SEED_JSON = {"theme": "dark", "nested": {"keep": [1, 2]}}
SEEDS: dict[str, Callable[[tuple[str, ...]], str]] = {
    "json": lambda keys: json.dumps(_nest(keys, {"other": {"command": "other-server", "args": []}}, dict(_SEED_JSON)), indent=2),
    "toml": lambda keys: '# my settings\nmodel = "sentinel"\n\n[mcp_servers.other]\ncommand = "other-server"\nargs = []\n',
    "toml-list": lambda keys: '# my settings\nmodel = "sentinel"\n\n[[mcp_servers]]\nname = "other"\ncommand = "other-server"\n',
    "yaml": lambda keys: "# my settings\nmodel: sentinel\n" + _yaml_nest(keys, "other:\n  command: other-server\n"),
    "yaml-list": lambda keys: "name: Local\nversion: 1.0.0\nschema: v1\n# my servers\nmcpServers:\n  - name: other\n    command: x\n",
}


def _nest(keys: tuple[str, ...], leaf: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    node = root
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = leaf
    return root


def _yaml_nest(keys: tuple[str, ...], body: str) -> str:
    out = ""
    for depth, key in enumerate(keys):
        out += "  " * depth + f"{key}:\n"
    pad = "  " * len(keys)
    return out + "".join(pad + line + "\n" for line in body.splitlines())


def _load(kind: str, path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if kind == "json":
        return json.loads(text)
    if kind.startswith("toml"):
        return _toml.loads(text)
    # Gate per test, not per module: only the YAML dialects need PyYAML.
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(text)


def _at(doc: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        doc = doc[key]
    return doc


def _entry(kind: str, doc: Any, keys: tuple[str, ...]) -> Any:
    container = _at(doc, keys)
    if kind in ("toml-list", "yaml-list"):
        found = [x for x in container if x.get("name") == "mind-mem"]
        assert len(found) == 1, container
        return found[0]
    return container["mind-mem"]


def _others(kind: str, doc: Any, keys: tuple[str, ...]) -> Any:
    container = _at(doc, keys)
    if kind in ("toml-list", "yaml-list"):
        return [x for x in container if x.get("name") != "mind-mem"]
    return {k: v for k, v in container.items() if k != "mind-mem"}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    monkeypatch.setenv("APPDATA", str(h / "AppData" / "Roaming"))
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setattr(client_mcp, "mcp_server_spec", lambda ws: {"command": CMD, "args": list(ARGS), "env": {"MIND_MEM_WORKSPACE": ws}})
    _platform(monkeypatch, "linux")
    assert os.path.expanduser("~") == str(h)
    return h


def _platform(monkeypatch: pytest.MonkeyPatch, plat: str) -> None:
    monkeypatch.setattr(client_paths, "current_platform", lambda: plat)
    monkeypatch.setattr(client_registry, "current_platform", lambda: plat)


def _backups(p: Path) -> list[Path]:
    return sorted(p.parent.glob(p.name + BACKUP_INFIX + "*"))


def _prepare(agent: str, home: Path, monkeypatch: pytest.MonkeyPatch, seed: bool) -> tuple[Path, str, tuple[str, ...], Any]:
    rel, (kind, keys, expected), plat, guarded = GOLDEN[agent]
    _platform(monkeypatch, plat)
    path = home / rel
    if guarded or seed:
        path.parent.mkdir(parents=True, exist_ok=True)
    if seed or agent == "continue":
        path.write_text(SEEDS[kind](keys), encoding="utf-8")
    return path, kind, keys, expected


def test_golden_table_covers_every_mcp_client() -> None:
    mcp = {n for n, s in AGENT_REGISTRY.items() if s.mcp_fmt}
    assert set(GOLDEN) == mcp
    assert len(mcp) >= 40, "positive control: the table must not select nothing"


@pytest.mark.parametrize("agent", sorted(GOLDEN))
class TestGoldenPerClient:
    def test_install_writes_expected_entry(self, agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path, kind, keys, expected = _prepare(agent, home, monkeypatch, seed=False)
        result = install_mcp_config(agent, WS)
        assert result["written"] is True, result
        assert Path(result["path"]) == path
        assert _entry(kind, _load(kind, path), keys) == expected

    def test_second_install_is_byte_identical(self, agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path, *_ = _prepare(agent, home, monkeypatch, seed=False)
        install_mcp_config(agent, WS)
        first = path.read_bytes()
        backups = _backups(path)
        again = install_mcp_config(agent, WS)
        assert again["skipped"] is True and again["written"] is False
        assert path.read_bytes() == first
        assert _backups(path) == backups

    def test_keeps_other_servers_and_settings(self, agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path, kind, keys, expected = _prepare(agent, home, monkeypatch, seed=True)
        before = _load(kind, path)
        before_text = path.read_text(encoding="utf-8")
        result = install_mcp_config(agent, WS)
        assert result["written"] is True and result["merged"] is True
        after = _load(kind, path)
        assert _entry(kind, after, keys) == expected
        assert _others(kind, after, keys) == _others(kind, before, keys)
        if kind == "json":
            assert after["theme"] == "dark" and after["nested"] == {"keep": [1, 2]}
        else:
            assert "# my settings" in path.read_text(encoding="utf-8") or "# my servers" in before_text
        assert _backups(path), "a changed file is backed up first"

    def test_uninstall_removes_only_mind_mem(self, agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path, kind, keys, _ = _prepare(agent, home, monkeypatch, seed=True)
        before = _load(kind, path)
        install_mcp_config(agent, WS)
        assert _entry(kind, _load(kind, path), keys)  # positive control
        result = uninstall_mcp_config(agent, WS)
        assert result["removed"] is True, result
        after = _load(kind, path)
        assert _others(kind, after, keys) == _others(kind, before, keys)
        container = _at(after, keys)
        assert "mind-mem" not in (container if isinstance(container, dict) else [x.get("name") for x in container])
        assert uninstall_mcp_config(agent, WS)["removed"] is False, "second uninstall is a no-op"

    def test_detected_by_dir_or_binary(self, agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _platform(monkeypatch, GOLDEN[agent][2])
        spec = AGENT_REGISTRY[agent]
        assert agent not in detect_installed_agents(WS), "negative control: clean HOME, empty PATH"
        if spec.detect_binaries:
            # Windows `which` only finds names with a PATHEXT suffix; ".exe" is always in PATHEXT.
            name = spec.detect_binaries[0]
            suffix = ".exe" if os.name == "nt" and not name.lower().endswith(".exe") else ""
            exe = Path(os.environ["PATH"]) / (name + suffix)
            exe.write_text("#!/bin/sh\n", encoding="utf-8")
            exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
            assert agent in detect_installed_agents(WS)
            exe.unlink()
        dirs = [p for p in spec.detect_paths if "*" not in p]
        assert dirs or spec.detect_binaries, "every client is detectable"
        if dirs:
            Path(client_paths.expand(dirs[0], WS)).mkdir(parents=True, exist_ok=True)
            assert agent in detect_installed_agents(WS)


@pytest.mark.parametrize("agent", sorted(a for a, g in GOLDEN.items() if g[3] and a != "continue"))
def test_guarded_client_writes_nothing_without_its_dir(agent: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, GOLDEN[agent][2])
    result = install_mcp_config(agent, WS)
    assert result["written"] is False and result["skipped"] is True
    assert "not found" in result["reason"] and "mind-mem" in result["content"]
    assert not (home / GOLDEN[agent][0]).parent.exists()


def test_continue_needs_its_config_yaml(home: Path) -> None:
    (home / ".continue").mkdir()
    (home / ".continue" / "config.json").write_text('{"models": []}\n', encoding="utf-8")
    result = install_mcp_config("continue", WS)
    assert result["written"] is False and "config.yaml" in result["reason"]
    assert not (home / ".continue" / "config.yaml").exists()


@pytest.mark.parametrize("agent", sorted(n for n, s in AGENT_REGISTRY.items() if s.manual))
def test_manual_clients_print_steps_and_write_nothing(agent: str, home: Path) -> None:
    result = install_mcp_config(agent, WS)
    assert result["skipped"] is True and result["reason"] == "manual setup required"
    assert "mind-mem" in result["content"] and AGENT_REGISTRY[agent].manual in result["content"]
    hook = install_config(agent, WS)
    assert hook["written"] is False and hook["content"]
    assert list(home.iterdir()) == []


def test_platform_specific_paths(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "win32")
    assert AGENT_REGISTRY["zed"].expand_mcp_path(WS) == str(home / "AppData" / "Roaming" / "Zed" / "settings.json")
    assert AGENT_REGISTRY["goose"].expand_mcp_path(WS) == str(home / "AppData" / "Roaming" / "Block" / "goose" / "config" / "config.yaml")
    assert AGENT_REGISTRY["cline"].expand_mcp_path(WS).startswith(str(home / "AppData" / "Roaming" / "Code" / "User"))
    _platform(monkeypatch, "darwin")
    mac = home / "Library" / "Application Support"
    assert AGENT_REGISTRY["claude-desktop"].expand_mcp_path(WS) == str(mac / "Claude" / "claude_desktop_config.json")
    assert AGENT_REGISTRY["vscode"].expand_mcp_path(WS) == str(home / "Library" / "Application Support" / "Code" / "User" / "mcp.json")
    # Claude Desktop is guarded on Linux only: macOS creates the file.
    assert install_mcp_config("claude-desktop", WS)["written"] is True


def test_platform_only_client_is_skipped_elsewhere(home: Path) -> None:
    result = install_mcp_config("visual-studio", WS)
    assert result["written"] is False and "platform" in result["reason"]
    assert not (home / ".mcp.json").exists()


@pytest.mark.parametrize(
    ("agent", "var", "value", "target"),
    [
        ("vibe", "VIBE_HOME", "vh", "vh/config.toml"),
        ("copilot-cli", "COPILOT_HOME", "ch", "ch/mcp-config.json"),
        ("cline", "CLINE_MCP_SETTINGS_PATH", "cl/settings.json", "cl/settings.json"),
        ("amp", "AMP_SETTINGS_FILE", "amp.json", "amp.json"),
    ],
)
def test_client_relocation_env_vars(agent: str, var: str, value: str, target: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(var, str(home / value))
    result = install_mcp_config(agent, WS)
    assert Path(result["path"]) == home / target and (home / target).is_file()


def test_extra_locations_written_only_when_present(home: Path) -> None:
    devin = home / ".config" / "devin"
    assert "also" not in install_mcp_config("windsurf", WS)
    assert not devin.exists()
    devin.mkdir(parents=True)
    result = install_mcp_config("windsurf", WS)
    assert [Path(r["path"]) for r in result["also"]] == [devin / "mcp_config.json"]
    assert json.loads((devin / "mcp_config.json").read_text(encoding="utf-8"))["mcpServers"]["mind-mem"] == STD
    (home / ".cline" / "data" / "settings").mkdir(parents=True)
    cline = install_mcp_config("cline", WS)
    assert any(Path(r["path"]).as_posix().endswith(".cline/data/settings/cline_mcp_settings.json") and r["written"] for r in cline["also"])
    gone = uninstall_mcp_config("windsurf", WS)
    assert gone["removed"] and gone["also"][0]["removed"]


def test_existing_alternate_file_is_preferred(home: Path) -> None:
    alt = home / ".config" / "kilo" / "kilo.json"
    alt.parent.mkdir(parents=True)
    alt.write_text('{"model": "x"}\n', encoding="utf-8")
    result = install_mcp_config("kilo", WS)
    assert Path(result["path"]) == alt
    assert not (alt.parent / "kilo.jsonc").exists()
    assert json.loads(alt.read_text(encoding="utf-8"))["mcp"]["mind-mem"] == OPENCODE


def test_yaml_flow_style_is_refused_byte_identical(home: Path) -> None:
    cfg = home / ".config" / "goose" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    body = b"extensions: {developer: {enabled: true, type: builtin}}\n"
    cfg.write_bytes(body)
    result = install_mcp_config("goose", WS)
    assert result["written"] is False and result["skipped"] is True and "flow style" in result["reason"]
    assert cfg.read_bytes() == body and _backups(cfg) == []


def test_jsonc_amp_settings_is_refused(home: Path) -> None:
    cfg = home / ".config" / "amp" / "settings.json"
    cfg.parent.mkdir(parents=True)
    body = b'{\n  // comment\n  "amp.mcpServers": {}\n}\n'
    cfg.write_bytes(body)
    result = install_mcp_config("amp", WS)
    assert result["skipped"] is True and cfg.read_bytes() == body
    assert json.loads(result["content"])["amp.mcpServers"]["mind-mem"] == STD


def test_vibe_inline_array_form_is_kept(home: Path) -> None:
    cfg = home / ".vibe" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text('mcp_servers = [\n  { name = "other", command = "x" }\n]\n', encoding="utf-8")
    install_mcp_config("vibe", WS)
    doc = _toml.loads(cfg.read_text(encoding="utf-8"))
    assert [s["name"] for s in doc["mcp_servers"]] == ["other", "mind-mem"]
    assert "[[mcp_servers]]" not in cfg.read_text(encoding="utf-8")
    uninstall_mcp_config("vibe", WS)
    assert [s["name"] for s in _toml.loads(cfg.read_text(encoding="utf-8"))["mcp_servers"]] == ["other"]


def test_toml_escapes_windows_paths(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    win = {"command": "C:\\Python\\python.exe", "args": ["C:\\mm\\mcp_server.py"], "env": {"MIND_MEM_WORKSPACE": "C:\\ws"}}
    monkeypatch.setattr(client_mcp, "mcp_server_spec", lambda ws: win)
    install_mcp_config("codex", WS)
    assert _toml.loads((home / ".codex" / "config.toml").read_text(encoding="utf-8"))["mcp_servers"]["mind-mem"] == win


def test_openclaw_hooks_and_mcp_share_one_file(home: Path) -> None:
    cfg = home / ".openclaw" / "openclaw.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"channels": {"telegram": {"enabled": True}}}), encoding="utf-8")
    install_all(WS, agents=["openclaw"])
    doc = json.loads(cfg.read_text(encoding="utf-8"))
    assert doc["channels"] == {"telegram": {"enabled": True}}
    assert doc["hooks"]["internal"]["entries"]["mind-mem"]["enabled"] is True
    assert doc["mcp"]["servers"]["mind-mem"] == STD


def test_editor_extension_detection_by_glob(home: Path) -> None:
    assert "continue" not in detect_installed_agents(WS)
    (home / ".vscode" / "extensions" / "continue.continue-1.2.3").mkdir(parents=True)
    assert "continue" in detect_installed_agents(WS)


def test_uninstall_all_reports_every_agent(home: Path) -> None:
    install_all(WS, agents=["cursor", "gemini"])
    results = uninstall_all(WS, agents=["cursor", "gemini", "aider", "trae"])
    by = {r["agent"]: r for r in results}
    assert by["cursor"]["removed"] and by["gemini"]["removed"]
    assert not by["aider"]["removed"] and not by["trae"]["removed"]
    # Both files were created by install, so uninstall removes them outright.
    assert by["cursor"].get("deleted") is True
    assert not (home / ".cursor" / "mcp.json").exists()


def test_uninstall_dry_run_changes_nothing(home: Path) -> None:
    install_mcp_config("cursor", WS)
    cfg = home / ".cursor" / "mcp.json"
    before = cfg.read_bytes()
    result = uninstall_mcp_config("cursor", WS, dry_run=True)
    assert result["removed"] is True and result["written"] is False
    assert cfg.read_bytes() == before


def test_uninstall_refuses_jsonc(home: Path) -> None:
    cfg = home / ".cursor" / "mcp.json"
    cfg.parent.mkdir()
    body = b'{ // c\n "mcpServers": {"mind-mem": {}} }\n'
    cfg.write_bytes(body)
    result = uninstall_mcp_config("cursor", WS)
    assert result["removed"] is False and "by hand" in result["reason"]
    assert cfg.read_bytes() == body
