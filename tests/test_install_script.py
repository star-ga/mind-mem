from __future__ import annotations

import os
import re
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
    home_str = str(home).replace("\\", "/") if os.name == "nt" else str(home)
    env["HOME"] = home_str
    # pip --user honours PYTHONUSERBASE on every platform; without it a
    # Windows --user install lands in %APPDATA%, outside the isolated HOME,
    # and the isolation this test claims would not exist.
    env["PYTHONUSERBASE"] = home_str

    # Force the pip installer. The pipx path is exercised independently
    # by the ``install.sh smoke (pipx)`` CI job; this test focuses on
    # client-config wiring under an isolated HOME, where pipx's
    # ``%LOCALAPPDATA%\\pipx`` cache (Windows) and
    # ``~/Library/Application Support/pipx`` (macOS) defy isolation
    # and time out the runner.
    install_args = ["--codex", "--installer", "pip"]

    if os.name == "nt":
        bash = shutil.which("bash")
        assert bash is not None, "bash is required to run install.sh on Windows"
        cmd = [bash, INSTALL_SH.replace("\\", "/"), *install_args]
    else:
        cmd = [INSTALL_SH, *install_args]

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    config_path = home / ".codex" / "config.toml"
    assert config_path.is_file()
    config_text = config_path.read_text()
    assert "mind-mem" in config_text

    # The wired command must be the console script this run just installed
    # under the isolated HOME — never an older ``mind-mem-mcp`` that happens
    # to sit earlier on PATH. Resolving by PATH first smoke-tests one binary
    # and writes a different, stale one into the client config.
    wired = re.search(r'^command = "(.+)"$', config_text, re.MULTILINE)
    assert wired is not None, config_text
    # compare against the same path form install.sh was actually given:
    # on Windows HOME is passed with forward slashes, so str(home) (backslashes)
    # would never match the wired path.
    assert wired.group(1).startswith(home_str), f"install.sh wired {wired.group(1)}, expected a copy under {home_str}"


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
