"""Offline closure tests for the RunPod training/evaluation release bundle."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from train import runpod_deploy as deploy

REPO = Path(__file__).resolve().parents[1]
SECRET = "hf_fixture_secret_must_not_escape"


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
    monkeypatch.setattr(deploy, "REMOTE_SOURCE_ROOT", str(tmp_path / "source"))
    monkeypatch.setattr(deploy, "REMOTE_TOKEN_FILE", str(token_file))
    monkeypatch.setattr(deploy, "REMOTE_TRAIN_ROOT", str(tmp_path / "output"))
    monkeypatch.setattr(deploy, "REMOTE_FULLFT_DIR", str(tmp_path / "output" / "full-ft"))
    monkeypatch.setattr(deploy, "REMOTE_HOLDOUT_REPORT", str(tmp_path / "output" / "holdout.json"))
    monkeypatch.setattr(deploy, "REMOTE_CORPUS", str(tmp_path / "output" / "corpus.jsonl"))
    monkeypatch.setenv("MM_CAPTURE", str(capture))
    run_env = os.environ.copy()
    run_env["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    completed = subprocess.run(
        ["bash", "-c", deploy._training_command_body()],
        check=True,
        capture_output=True,
        text=True,
        env=run_env,
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
    monkeypatch.setattr(deploy, "REMOTE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setattr(deploy, "REMOTE_SOURCE_ROOT", str(remote_source))
    monkeypatch.setattr(deploy, "REMOTE_TRAIN_ROOT", str(remote_output))
    monkeypatch.setattr(deploy, "REMOTE_FULLFT_DIR", str(remote_output / "full-ft"))
    monkeypatch.setattr(deploy, "REMOTE_CORPUS", str(remote_output / "corpus.jsonl"))
    monkeypatch.setattr(deploy, "REMOTE_EVAL_REPORT", str(remote_output / "eval_report.json"))
    monkeypatch.setattr(deploy, "REMOTE_HOLDOUT_REPORT", str(remote_output / "holdout.json"))

    def fake_ssh(ip, port, command):
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
        return ""

    def fake_scp(ip, port, local, remote):
        destination = Path(remote)
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
