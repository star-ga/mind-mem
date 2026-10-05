# Copyright 2026 STARGA, Inc.
"""``--help`` must print usage and change nothing.

Regressions:
* ``uninstall.sh`` ignored every flag but ``--purge``, so ``uninstall.sh --help``
  ran the full uninstall and removed the mind-mem entry from real client
  configs. Its JSON cleanup also had a Python SyntaxError, so JSON clients were
  never actually cleaned.
* ``mind-mem-migrate`` / ``mind-mem-validate`` / ``mind-mem-capture`` took
  ``sys.argv[1]`` as the workspace, so ``--help`` was treated as a directory
  (``mind-mem-migrate --help`` would migrate a workspace named ``--help``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UNINSTALL_SH = REPO_ROOT / "uninstall.sh"

posix_only = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="POSIX shell script")

CODEX_TOML = (
    '[mcp_servers.other]\ncommand = "other"\n\n'
    '[mcp_servers.mind-mem]\ncommand = "mind-mem-mcp"\n\n'
    '[mcp_servers.mind-mem.env]\nMIND_MEM_WORKSPACE = "/ws"\n'
)
CURSOR_JSON = {"mcpServers": {"other": {"command": "x"}, "mind-mem": {"command": "mind-mem-mcp"}}}
ZED_JSON = {"theme": "dark", "context_servers": {"mind-mem": {"command": {"path": "mm"}}}}
OPENCLAW_JSON = {"gateway": {"port": 1}, "hooks": {"internal": {"entries": {"mind-mem": {"enabled": True}, "other": {}}}}}


def _seed_home(home: Path) -> dict[Path, bytes]:
    files = {
        home / ".codex" / "config.toml": CODEX_TOML.encode(),
        home / ".cursor" / "mcp.json": json.dumps(CURSOR_JSON).encode(),
        home / ".config" / "zed" / "settings.json": json.dumps(ZED_JSON).encode(),
        home / ".openclaw" / "openclaw.json": json.dumps(OPENCLAW_JSON).encode(),
        home / ".gemini" / "settings.json": b'// comment\n{"mcpServers": {"mind-mem": {}}}\n',
        home / ".mind-mem" / "keep.txt": b"workspace data",
    }
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return files


def _snapshot(home: Path) -> dict[str, bytes]:
    return {str(p.relative_to(home)): p.read_bytes() for p in sorted(home.rglob("*")) if p.is_file()}


def _uninstall(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(["bash", str(UNINSTALL_SH), *args], env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    _seed_home(h)
    return h


@posix_only
class TestUninstallScript:
    @pytest.mark.parametrize("flag", ["--help", "-h"])
    def test_help_prints_usage_and_touches_nothing(self, home: Path, flag: str) -> None:
        before = _snapshot(home)
        result = _uninstall(home, flag)
        assert result.returncode == 0, result.stderr
        assert "Usage:" in result.stdout
        assert _snapshot(home) == before

    @pytest.mark.parametrize("args", [["--bogus"], ["--purge", "--hlep"], ["workspace"]])
    def test_unknown_argument_is_refused_before_any_change(self, home: Path, args: list[str]) -> None:
        before = _snapshot(home)
        result = _uninstall(home, *args)
        assert result.returncode == 2
        assert "unknown argument" in result.stderr
        assert _snapshot(home) == before

    def test_dry_run_changes_nothing(self, home: Path) -> None:
        before = _snapshot(home)
        result = _uninstall(home, "--dry-run", "--purge")
        assert result.returncode == 0, result.stderr
        assert "would remove" in result.stdout
        assert _snapshot(home) == before

    def test_uninstall_removes_only_the_mind_mem_entries(self, home: Path) -> None:
        """Positive control: the script still does its job, and the JSON cleanup runs."""
        result = _uninstall(home)
        assert result.returncode == 0, result.stderr
        assert "Traceback" not in result.stderr and "SyntaxError" not in result.stderr

        codex = (home / ".codex" / "config.toml").read_text(encoding="utf-8")
        assert "mind-mem" not in codex and "[mcp_servers.other]" in codex
        cursor = json.loads((home / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
        assert cursor == {"mcpServers": {"other": {"command": "x"}}}
        zed = json.loads((home / ".config" / "zed" / "settings.json").read_text(encoding="utf-8"))
        assert zed == {"theme": "dark", "context_servers": {}}
        claw = json.loads((home / ".openclaw" / "openclaw.json").read_text(encoding="utf-8"))
        assert claw == {"gateway": {"port": 1}, "hooks": {"internal": {"entries": {"other": {}}}}}
        # Changed files were backed up first.
        assert list((home / ".cursor").glob("mcp.json.bak-mind-mem-uninstall-*"))
        # A JSONC config is left byte-identical, not rewritten.
        assert (home / ".gemini" / "settings.json").read_bytes().startswith(b"// comment")
        # Workspace data is only removed with --purge.
        assert (home / ".mind-mem" / "keep.txt").is_file()

    def test_client_flag_limits_the_scope(self, home: Path) -> None:
        result = _uninstall(home, "--cursor")
        assert result.returncode == 0, result.stderr
        assert "mind-mem" not in json.loads((home / ".cursor" / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]
        assert "mind-mem" in (home / ".codex" / "config.toml").read_text(encoding="utf-8")


ENTRY_POINTS = [
    ("mind_mem.schema_version", "mind-mem-migrate"),
    ("mind_mem.validate_py", "mind-mem-validate"),
    ("mind_mem.capture", "mind-mem-capture"),
]


@pytest.mark.parametrize(("module", "prog"), ENTRY_POINTS)
def test_console_script_help_is_help(module: str, prog: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert f"usage: {prog}" in result.stdout
    # Nothing was created: in particular no workspace directory named --help.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("module", "prog"), ENTRY_POINTS)
def test_console_script_rejects_unknown_flags(module: str, prog: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--definitely-not-a-flag"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_capture_scan_all_flag_is_not_taken_as_the_workspace(tmp_path: Path) -> None:
    from mind_mem.capture import _parse_args

    args = _parse_args(["--scan-all"])
    assert args.workspace == "." and args.scan_all is True
    args = _parse_args([str(tmp_path), "--scan-all"])
    assert args.workspace == str(tmp_path) and args.scan_all is True
