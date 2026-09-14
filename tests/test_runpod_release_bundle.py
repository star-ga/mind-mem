"""Offline closure tests for the RunPod training/evaluation release bundle."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from train import runpod_deploy as deploy

REPO = Path(__file__).resolve().parents[1]
SECRET = "hf_fixture_secret_must_not_escape"


def _bash_executable() -> str:
    """Resolve bash before a test changes any child environment.

    On Windows the fixture drives path conversion through ``cygpath``, which
    ships with Git Bash / MSYS2 but NOT with WSL's bash. Trusting PATH order can
    resolve ``bash`` to WSL, where ``cygpath`` does not exist and the fixture
    would later abort with an opaque cygpath error that reads like a portability
    bug. So on Windows we require a bash whose ``cygpath`` is reachable and
    otherwise ``skip`` WITH A REASON (never a silent skip, never a misleading
    hard failure). On the GitHub ``windows-latest`` runner ``bash`` is Git Bash,
    so this passes. On POSIX any ``bash`` is fine and no cygpath is needed.
    """
    bash = shutil.which("bash")
    if not bash:
        if os.name == "nt":
            pytest.skip("Windows release fixture requires Git Bash (no bash on PATH)")
        raise AssertionError("the release fixture requires bash")
    resolved = str(Path(bash).resolve())
    if os.name == "nt":
        probe = subprocess.run(
            [resolved, "-lc", "command -v cygpath"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            pytest.skip(f"Windows release fixture requires Git Bash with cygpath; resolved bash ({resolved}) has none (likely WSL bash)")
    return resolved


def _bash_path(path: Path) -> str:
    """Convert a fixture path to the POSIX spelling understood by Git Bash."""
    if os.name != "nt":
        return path.as_posix()
    bash = _bash_executable()
    command = f"cygpath -u -- {shlex.quote(path.as_posix())}"
    completed = subprocess.run(
        [bash, "-lc", command],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    if completed.returncode != 0:
        raise AssertionError(f"cygpath failed for fixture path: {completed.stderr.strip()}")
    result = completed.stdout.strip()
    if not result:
        raise AssertionError("cygpath returned an empty fixture path")
    return result


def _native_path(remote: str) -> Path:
    """Map a POSIX remote fixture path back to the host filesystem for SCP."""
    if os.name != "nt":
        return Path(remote)
    bash = _bash_executable()
    command = f"cygpath -w -- {shlex.quote(remote)}"
    completed = subprocess.run(
        [bash, "-lc", command],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    if completed.returncode != 0:
        raise AssertionError(f"cygpath failed for remote fixture path: {completed.stderr.strip()}")
    result = completed.stdout.strip()
    if not result:
        raise AssertionError("cygpath returned an empty native fixture path")
    return Path(result)


def test_release_bundle_preserves_train_src_and_eval_dependencies(monkeypatch):
    ssh_commands: list[str] = []
    scp_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(deploy, "_ssh_cmd", lambda ip, port, command: ssh_commands.append(command) or "")
    monkeypatch.setattr(
        deploy,
        "_scp_to",
        lambda ip, port, local, remote: scp_calls.append((local, remote)),
    )

    deploy._stage_release_bundle("203.0.113.10", 2200, REPO)

    assert len(ssh_commands) == 2
    assert (
        "mkdir -p /workspace/mind-mem-release/scripts /workspace/mind-mem-release/src/mind_mem /workspace/mind-mem-release/train"
        in ssh_commands[0]
    )
    assert "git init -q" in ssh_commands[1]
    assert "git add --" in ssh_commands[1]
    assert "stage release bundle" in ssh_commands[1]
    staged = {remote: Path(local) for local, remote in scp_calls}
    assert set(staged) == {f"{deploy.REMOTE_SOURCE_ROOT}/{relative}" for relative in deploy.RELEASE_FILES}
    for relative in deploy.RELEASE_FILES:
        assert staged[f"{deploy.REMOTE_SOURCE_ROOT}/{relative}"] == REPO / relative
    assert f"{deploy.REMOTE_SOURCE_ROOT}/train/eval_harness.py" in staged
    assert f"{deploy.REMOTE_SOURCE_ROOT}/train/eval_holdout.py" in staged
    assert f"{deploy.REMOTE_SOURCE_ROOT}/train/eval_receipt.py" in staged
    assert f"{deploy.REMOTE_SOURCE_ROOT}/train/build_corpus.py" in staged
    assert f"{deploy.REMOTE_SOURCE_ROOT}/src/mind_mem/causal_lm_loader.py" in staged
    assert "/workspace/eval_harness.py" not in staged


def test_all_remote_eval_and_release_commands_share_explicit_paths(monkeypatch):
    commands: list[str] = []
    monkeypatch.setattr(deploy, "_ssh_cmd", lambda ip, port, command: commands.append(command) or "")

    deploy._run_release_commands("203.0.113.10", 2200, "v4.0.0", skip_upload=False)

    assert [command.rsplit("/", 1)[-1] for command in commands[:2]] == [
        "eval_harness.py",
        "eval_holdout.py",
    ]
    assert commands[2].endswith("python3 -u train/build_model_card.py")
    assert "train/upload_to_hf.py" in commands[3]
    for command in commands:
        assert f"MM_TRAIN_ROOT={deploy.REMOTE_TRAIN_ROOT}" in command
        assert f"MM_FULLFT_DIR={deploy.REMOTE_FULLFT_DIR}" in command
        assert f"MM_WEIGHTS_DIR={deploy.REMOTE_FULLFT_DIR}" in command
        assert f"MM_HOLDOUT_REPORT={deploy.REMOTE_HOLDOUT_REPORT}" in command
        assert f"MM_CORPUS={deploy.REMOTE_CORPUS}" in command
        assert SECRET not in command


def test_failed_main_eval_stops_card_and_upload(monkeypatch):
    calls: list[str] = []

    def fail_eval(ip, port, command):
        calls.append(command)
        raise RuntimeError("ssh failed: eval failed")

    monkeypatch.setattr(deploy, "_ssh_cmd", fail_eval)
    with pytest.raises(RuntimeError, match="main evaluation failed; refusing release"):
        deploy._run_release_commands("203.0.113.10", 2200, "v4.0.0", skip_upload=False)
    assert len(calls) == 1
    assert "build_model_card.py" not in calls[0]
    assert "upload_to_hf.py" not in calls[0]


def test_skip_upload_still_runs_both_evals_and_builds_card(monkeypatch, capsys):
    commands: list[str] = []
    monkeypatch.setattr(deploy, "_ssh_cmd", lambda ip, port, command: commands.append(command) or "")

    deploy._run_release_commands("203.0.113.10", 2200, "v4.0.0", skip_upload=True)

    assert len(commands) == 3
    assert "eval_harness.py" in commands[0]
    assert "eval_holdout.py" in commands[1]
    assert "build_model_card.py" in commands[2]
    assert all("upload_to_hf.py" not in command for command in commands)
    assert "NOT pushing to HF" in capsys.readouterr().out


def test_training_command_uses_same_paths_without_token_value(monkeypatch):
    monkeypatch.setenv("MM_BASE_MODEL", "Qwen/Qwen3.5-4B")
    command = deploy._training_launch_command()

    assert "train/runpod_full_ft.py" in command
    assert 'export HF_TOKEN="$(cat' in command
    assert "exec python3 -u train/runpod_full_ft.py" in command
    assert "nohup bash -lc" in command
    assert " env " not in command
    assert SECRET not in command
    for name, value in (
        ("MM_TRAIN_ROOT", deploy.REMOTE_TRAIN_ROOT),
        ("MM_FULLFT_DIR", deploy.REMOTE_FULLFT_DIR),
        ("MM_WEIGHTS_DIR", deploy.REMOTE_FULLFT_DIR),
        ("MM_HOLDOUT_REPORT", deploy.REMOTE_HOLDOUT_REPORT),
        ("MM_CORPUS", deploy.REMOTE_CORPUS),
    ):
        assert f"{name}={value}" in command


def test_training_shell_exports_token_without_env_argv_secret(monkeypatch, tmp_path):
    """Execute the generated inner shell with a fake Python worker.

    The worker records its argv and environment. The fixture token must arrive
    only through the environment; an ``env TOKEN=...`` wrapper would expose it
    in the worker launcher argv on the way to Python.
    """
    token_file = tmp_path / "hf-token"
    token_file.write_text(SECRET, encoding="utf-8")
    capture = tmp_path / "capture.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        '#!/bin/sh\nprintf \'argv=%s\\n\' "$*" > "$MM_CAPTURE"\nprintf \'HF_TOKEN=%s\\n\' "$HF_TOKEN" >> "$MM_CAPTURE"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    # The generated command is for a POSIX shell on the Linux pod even when
    # this test itself runs on Windows.  Use slash-form paths at that boundary.
    monkeypatch.setattr(deploy, "REMOTE_SOURCE_ROOT", _bash_path(tmp_path / "source"))
    monkeypatch.setattr(deploy, "REMOTE_TOKEN_FILE", _bash_path(token_file))
    monkeypatch.setattr(deploy, "REMOTE_TRAIN_ROOT", _bash_path(tmp_path / "output"))
    monkeypatch.setattr(deploy, "REMOTE_FULLFT_DIR", _bash_path(tmp_path / "output" / "full-ft"))
    monkeypatch.setattr(deploy, "REMOTE_HOLDOUT_REPORT", _bash_path(tmp_path / "output" / "holdout.json"))
    monkeypatch.setattr(deploy, "REMOTE_CORPUS", _bash_path(tmp_path / "output" / "corpus.jsonl"))
    monkeypatch.setenv("MM_CAPTURE", _bash_path(capture))
    bash = _bash_executable()
    run_env = os.environ.copy()
    fake_bin_posix = _bash_path(fake_bin)
    wrapped_command = f"PATH={shlex.quote(fake_bin_posix)}:$PATH; export PATH; " + deploy._training_command_body()

    completed = subprocess.run(
        [bash, "-c", wrapped_command],
        check=True,
        capture_output=True,
        text=True,
        env=run_env,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.stdout == ""
    assert completed.stderr == ""
    recorded = capture.read_text(encoding="utf-8")
    assert "argv=-u train/runpod_full_ft.py" in recorded
    assert f"HF_TOKEN={SECRET}" in recorded
    assert SECRET not in recorded.splitlines()[0]
    assert SECRET not in deploy._training_launch_command()


def test_staging_smoke_bare_import_and_receipt_sources(monkeypatch, tmp_path):
    """Copy the bundle like SCP, then import the staged evaluators in a subprocess."""
    remote_source = tmp_path / "workspace" / "mind-mem-release"
    remote_output = tmp_path / "workspace" / "train-output"
    # Remote commands always target the POSIX pod filesystem.  The local
    # Windows runner still maps these slash-form paths to its fixture tree.
    monkeypatch.setattr(deploy, "REMOTE_ROOT", _bash_path(tmp_path / "workspace"))
    monkeypatch.setattr(deploy, "REMOTE_SOURCE_ROOT", _bash_path(remote_source))
    monkeypatch.setattr(deploy, "REMOTE_TRAIN_ROOT", _bash_path(remote_output))
    monkeypatch.setattr(deploy, "REMOTE_FULLFT_DIR", _bash_path(remote_output / "full-ft"))
    monkeypatch.setattr(deploy, "REMOTE_CORPUS", _bash_path(remote_output / "corpus.jsonl"))
    monkeypatch.setattr(deploy, "REMOTE_EVAL_REPORT", _bash_path(remote_output / "eval_report.json"))
    monkeypatch.setattr(deploy, "REMOTE_HOLDOUT_REPORT", _bash_path(remote_output / "holdout.json"))

    bash = _bash_executable()

    def fake_ssh(ip, port, command):
        subprocess.run([bash, "-c", command], check=True, timeout=30)
        return ""

    def fake_scp(ip, port, local, remote):
        destination = _native_path(remote)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, destination)

    monkeypatch.setattr(deploy, "_ssh_cmd", fake_ssh)
    monkeypatch.setattr(deploy, "_scp_to", fake_scp)
    deploy._stage_release_bundle("203.0.113.10", 2200, REPO)

    probe = (
        "from pathlib import Path\n"
        "import subprocess\n"
        "import train.eval_harness\n"
        "import train.eval_holdout\n"
        "from eval_receipt import eval_source_paths\n"
        "root = Path.cwd()\n"
        "paths = eval_source_paths(root)\n"
        "assert all(path.is_file() for path in paths), paths\n"
        "assert subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True).returncode == 0\n"
        "print('staged_eval_imports=ok')\n"
        "print('staged_eval_sources=' + str(len(paths)))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(remote_source)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=remote_source,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert "staged_eval_imports=ok" in completed.stdout
    assert "staged_eval_sources=6" in completed.stdout
    assert (remote_source / "src/mind_mem/causal_lm_loader.py").is_file()
    assert (remote_source / "train/build_corpus.py").is_file()


def test_token_is_copied_as_a_file_and_never_embedded_in_command(monkeypatch, tmp_path, capsys):
    token_file = tmp_path / "hf-token"
    token_file.write_text(SECRET, encoding="utf-8")
    scp_calls: list[tuple[str, str]] = []
    ssh_commands: list[str] = []
    monkeypatch.setattr(deploy, "HF_TOKEN_FILE", token_file)
    monkeypatch.setattr(
        deploy,
        "_scp_to",
        lambda ip, port, local, remote: scp_calls.append((local, remote)),
    )
    monkeypatch.setattr(deploy, "_ssh_cmd", lambda ip, port, command: ssh_commands.append(command) or "")

    deploy._stage_hf_token("203.0.113.10", 2200)

    assert scp_calls == [(str(token_file), deploy.REMOTE_TOKEN_FILE)]
    assert ssh_commands == [f"chmod 600 {deploy.REMOTE_TOKEN_FILE}"]
    assert SECRET not in " ".join(ssh_commands)
    assert SECRET not in capsys.readouterr().out
