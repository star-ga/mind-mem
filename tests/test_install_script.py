from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
INSTALL_SH = os.path.join(REPO_ROOT, "install.sh")
MCP_SERVER = os.path.join(REPO_ROOT, "mcp_server.py")


def test_install_sh_bootstraps_clean_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()

    env = os.environ.copy()
    env["HOME"] = str(home).replace("\\", "/") if os.name == "nt" else str(home)

    if os.name == "nt":
        bash = shutil.which("bash")
        assert bash is not None, "bash is required to run install.sh on Windows"
        cmd = [bash, INSTALL_SH.replace("\\", "/"), "--codex"]
    else:
        cmd = [INSTALL_SH, "--codex"]

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    config_path = home / ".codex" / "config.toml"
    assert config_path.is_file()
    assert "mind-mem" in config_path.read_text()


def test_mcp_server_help_runs_from_source_checkout(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, MCP_SERVER, "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Mind-Mem MCP Server" in result.stdout
