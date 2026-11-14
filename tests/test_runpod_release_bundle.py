"""Offline closure tests for the RunPod training/evaluation release bundle."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


class _ResumePlan:
    def __init__(self, pod_id: str = "abc123") -> None:
        self.pod_id = pod_id
        self.gets: list[tuple[int, dict]] = []
        self.start: tuple[int, dict] = (500, {"title": "unset", "status": 500, "detail": "start was not scripted"})
        self.stop: tuple[int, dict] = (500, {"title": "unset", "status": 500, "detail": "stop was not scripted"})
        self.after_stop_get: tuple[int, dict] | None = None
        self.graphql: dict | None = None
        self.stop_seen = False
        self.calls: list[tuple[str, str, str, str | None]] = []
        self.lock = threading.Lock()


def _resume_pod(
    pod_id: str,
    status: str,
    *,
    host: str = "",
    port: int | None = None,
    actions: list[str] | None = None,
    locked: bool = False,
    runtime: tuple[str, int] | None = None,
) -> dict:
    direct = None
    if host and port is not None:
        direct = {"host": host, "port": port, "username": "root"}
    runtime_body = None
    if runtime is not None:
        runtime_body = {
            "uptime": 1,
            "ports": [{"private": 22, "public": runtime[1], "type": "tcp", "ip": runtime[0]}],
        }
    return {
        "id": pod_id,
        "status": status,
        "actions": ["start", "stop", "restart", "terminate"] if actions is None else actions,
        "locked": locked,
        "ssh": {"proxy": {"host": "ssh.runpod.io", "port": 22}, "direct": direct},
        "runtime": runtime_body,
        "env": {"PUBLIC_KEY": "should-not-leak"},
        "cost": 3.99,
    }


def _pop_script(items: list[tuple[int, dict]]) -> tuple[int, dict]:
    if not items:
        raise AssertionError("resume fixture has no scripted response")
    if len(items) == 1:
        return items[0]
    return items.pop(0)


@contextmanager
def _resume_server(plan: _ResumePlan):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            return

        def _read(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0") or "0")
            return self.rfile.read(length) if length else b""

        def _send(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            with plan.lock:
                plan.calls.append(("GET", self.path, "", self.headers.get("Authorization")))
                if self.path != f"/v2/pods/{plan.pod_id}":
                    code, payload = (404, {"title": "Not Found", "status": 404, "detail": "resource not found"})
                elif plan.stop_seen and plan.after_stop_get is not None:
                    code, payload = plan.after_stop_get
                else:
                    code, payload = _pop_script(plan.gets)
            self._send(code, payload)

        def do_POST(self) -> None:
            text = self._read().decode()
            with plan.lock:
                plan.calls.append(("POST", self.path, text, self.headers.get("Authorization")))
                if self.path == "/graphql":
                    code, payload = 200, (plan.graphql or {})
                elif self.path == f"/v2/pods/{plan.pod_id}/action":
                    try:
                        action = json.loads(text or "{}").get("action")
                    except json.JSONDecodeError:
                        action = None
                    if action == "terminate":
                        code, payload = (400, {"title": "Bad Request", "status": 400, "detail": "terminate refused"})
                    elif action == "stop":
                        plan.stop_seen = True
                        code, payload = plan.stop
                    elif action == "start":
                        code, payload = plan.start
                    else:
                        code, payload = (
                            422,
                            {"title": "Unprocessable Entity", "status": 422, "detail": "Request validation failed."},
                        )
                else:
                    code, payload = (404, {"title": "Not Found", "status": 404, "detail": "resource not found"})
            self._send(code, payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v2"
    try:
        yield base
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run_resume(tmp_path: Path, base: str, plan: _ResumePlan, inject: str = "", polls: str = "4"):
    if os.name == "nt":
        pytest.skip("resume script tests run on POSIX runners")
    token = tmp_path / "hf"
    token.write_text("fixture-token\n", encoding="utf-8")
    token.chmod(0o600)
    config = tmp_path / "runpod.toml"
    config.write_text('api_key = "fixture-api-key"\n', encoding="utf-8")
    env = os.environ.copy()
    env["HF_TOKEN_FILE"] = str(token)
    env["RUNPOD_CONFIG"] = str(config)
    env["MM_POD_ID"] = plan.pod_id
    env["MM_RUNPOD_API_BASE"] = base
    env["MM_RUNPOD_GRAPHQL_URL"] = base[: -len("/v2")] + "/graphql"
    env["MM_RESUME_MAX_POLLS"] = polls
    env["MM_RESUME_POLL_SECONDS"] = "0"
    env["MM_RESUME_STOP_POLLS"] = "3"
    env["MM_RESUME_INJECT"] = inject
    completed = subprocess.run(
        [_bash_executable(), str(REPO / "train" / "resume_pod_train.sh")],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    blob = completed.stdout + completed.stderr
    assert "should-not-leak" not in blob
    assert "fixture-api-key" not in blob
    assert "fixture-token" not in blob
    assert "PUBLIC_KEY" not in blob
    for _method, path, body, auth in plan.calls:
        assert auth == "Bearer fixture-api-key"
        assert "/v1/" not in path
        assert not path.endswith("/start")
        assert "terminate" not in body
    return completed


def _bodies(plan: _ResumePlan, suffix: str) -> list[str]:
    return [body for _method, path, body, _auth in plan.calls if path.endswith(suffix)]


def test_resume_script_does_not_call_rest_v1():
    text = (REPO / "train" / "resume_pod_train.sh").read_text(encoding="utf-8")
    assert "rest.runpod.io/v1" not in text
    assert "stat -c" not in text
    assert 'POD_ID="${MM_POD_ID:-}"' in text
    assert "MM_POD_ID:-" not in text.replace('POD_ID="${MM_POD_ID:-}"', "", 1)
    assert '"/pods/${POD_ID}/action"' in text
    assert '\'{"action":"start"}\'' in text
    assert '\'{"action":"stop"}\'' in text
    assert '{"action":"terminate"}' not in text


def test_resume_requires_pod_id_from_the_environment(tmp_path: Path):
    if os.name == "nt":
        pytest.skip("resume script tests run on POSIX runners")
    env = os.environ.copy()
    env.pop("MM_POD_ID", None)
    env["HF_TOKEN_FILE"] = str(tmp_path / "missing")
    completed = subprocess.run(
        [_bash_executable(), str(REPO / "train" / "resume_pod_train.sh")],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert completed.returncode != 0
    assert "MM_POD_ID is required" in completed.stderr


def test_resume_rejects_pod_id_before_any_request(tmp_path: Path):
    if os.name == "nt":
        pytest.skip("resume script tests run on POSIX runners")
    env = os.environ.copy()
    env["MM_POD_ID"] = "../x"
    env["HF_TOKEN_FILE"] = str(tmp_path / "missing")
    completed = subprocess.run(
        [_bash_executable(), str(REPO / "train" / "resume_pod_train.sh")],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert completed.returncode != 0
    assert "invalid pod id" in completed.stderr


def test_resume_parses_v2_problem_and_does_not_stop_an_exited_pod(tmp_path: Path):
    plan = _ResumePlan()
    exited = _resume_pod(plan.pod_id, "EXITED")
    plan.gets = [(200, exited)]
    plan.start = (
        422,
        {
            "title": "Unprocessable Entity",
            "status": 422,
            "detail": "Request validation failed.",
            "errors": ["action: rejected"],
        },
    )
    with _resume_server(plan) as base:
        completed = _run_resume(tmp_path, base, plan)
    assert completed.returncode != 0
    assert "Unprocessable Entity (422): Request validation failed.; action: rejected" in completed.stderr
    assert _bodies(plan, "/action") == ['{"action":"start"}']


def test_resume_stops_a_pod_it_started_when_launch_is_not_confirmed(tmp_path: Path):
    plan = _ResumePlan()
    plan.gets = [
        (200, _resume_pod(plan.pod_id, "EXITED")),
        (200, _resume_pod(plan.pod_id, "RUNNING", host="203.0.113.10", port=4022)),
    ]
    plan.start = (200, _resume_pod(plan.pod_id, "STARTING"))
    plan.stop = (200, _resume_pod(plan.pod_id, "EXITED"))
    plan.after_stop_get = (200, _resume_pod(plan.pod_id, "EXITED"))
    with _resume_server(plan) as base:
        completed = _run_resume(tmp_path, base, plan, inject="fail-after-ready")
    assert completed.returncode != 0
    assert "injected failure after SSH endpoint was ready" in completed.stderr
    assert f"pod {plan.pod_id} status EXITED. Compute is not billed." in completed.stdout
    assert _bodies(plan, "/action") == ['{"action":"start"}', '{"action":"stop"}']


def test_resume_does_not_stop_a_pod_that_was_already_running(tmp_path: Path):
    plan = _ResumePlan()
    plan.gets = [(200, _resume_pod(plan.pod_id, "RUNNING", host="203.0.113.10", port=4022))]
    with _resume_server(plan) as base:
        completed = _run_resume(tmp_path, base, plan, inject="fail-after-ready")
    assert completed.returncode != 0
    assert "pod is already RUNNING" in completed.stdout
    assert "Compute may still be billed" not in completed.stderr
    assert _bodies(plan, "/action") == []


def test_resume_leaves_a_started_pod_running_when_launch_is_confirmed(tmp_path: Path):
    plan = _ResumePlan()
    plan.gets = [
        (200, _resume_pod(plan.pod_id, "EXITED")),
        (200, _resume_pod(plan.pod_id, "RUNNING", host="203.0.113.10", port=4022)),
    ]
    plan.start = (200, _resume_pod(plan.pod_id, "STARTING"))
    with _resume_server(plan) as base:
        completed = _run_resume(tmp_path, base, plan, inject="confirm-launch")
    assert completed.returncode == 0
    assert "injected launch confirmation" in completed.stdout
    assert _bodies(plan, "/action") == ['{"action":"start"}']


def test_resume_reports_billing_when_stop_does_not_reach_exited(tmp_path: Path):
    plan = _ResumePlan()
    running = _resume_pod(plan.pod_id, "RUNNING", host="203.0.113.10", port=4022)
    plan.gets = [(200, _resume_pod(plan.pod_id, "EXITED")), (200, running)]
    plan.start = (200, _resume_pod(plan.pod_id, "STARTING"))
    plan.stop = (200, running)
    with _resume_server(plan) as base:
        completed = _run_resume(tmp_path, base, plan, inject="fail-after-ready")
    assert completed.returncode != 0
    assert f"pod {plan.pod_id} status RUNNING. Compute may still be billed." in completed.stderr
    assert '{"action":"stop"}' in _bodies(plan, "/action")
    assert '{"action":"start"}' in _bodies(plan, "/action")


def test_resume_uses_graphql_ports_when_v2_has_no_ssh_endpoint(tmp_path: Path):
    plan = _ResumePlan()
    plan.gets = [
        (200, _resume_pod(plan.pod_id, "EXITED")),
        (200, _resume_pod(plan.pod_id, "RUNNING")),
    ]
    plan.start = (200, _resume_pod(plan.pod_id, "STARTING"))
    plan.stop = (200, _resume_pod(plan.pod_id, "EXITED"))
    plan.after_stop_get = (200, _resume_pod(plan.pod_id, "EXITED"))
    plan.graphql = {
        "data": {
            "myself": {
                "pods": [
                    {
                        "id": plan.pod_id,
                        "runtime": {
                            "ports": [
                                {
                                    "ip": "203.0.113.10",
                                    "privatePort": 22,
                                    "publicPort": 4022,
                                    "isIpPublic": True,
                                }
                            ]
                        },
                    }
                ]
            }
        }
    }
    with _resume_server(plan) as base:
        completed = _run_resume(tmp_path, base, plan, inject="fail-after-ready")
    assert completed.returncode != 0
    assert "ssh ready: 203.0.113.10:4022" in completed.stdout
    assert any(path == "/graphql" for _method, path, _body, _auth in plan.calls)
    assert _bodies(plan, "/action") == ['{"action":"start"}', '{"action":"stop"}']


def test_resume_reads_ssh_from_v2_runtime_ports(tmp_path: Path):
    plan = _ResumePlan()
    plan.gets = [
        (200, _resume_pod(plan.pod_id, "EXITED")),
        (200, _resume_pod(plan.pod_id, "RUNNING", runtime=("203.0.113.10", 4022))),
    ]
    plan.start = (200, _resume_pod(plan.pod_id, "STARTING"))
    plan.stop = (200, _resume_pod(plan.pod_id, "EXITED"))
    plan.after_stop_get = (200, _resume_pod(plan.pod_id, "EXITED"))
    with _resume_server(plan) as base:
        completed = _run_resume(tmp_path, base, plan, inject="fail-after-ready")
    assert completed.returncode != 0
    assert "ssh ready: 203.0.113.10:4022" in completed.stdout
    assert not any(path == "/graphql" for _method, path, _body, _auth in plan.calls)


def test_resume_stops_an_error_pod_instead_of_starting_it(tmp_path: Path):
    plan = _ResumePlan()
    plan.gets = [(200, _resume_pod(plan.pod_id, "ERROR", actions=["stop", "terminate"]))]
    plan.stop = (200, _resume_pod(plan.pod_id, "EXITED"))
    plan.after_stop_get = (200, _resume_pod(plan.pod_id, "EXITED"))
    with _resume_server(plan) as base:
        completed = _run_resume(tmp_path, base, plan)
    assert completed.returncode != 0
    assert "Not sending start." in completed.stderr
    assert f"pod {plan.pod_id} status EXITED. Compute is not billed." in completed.stdout
    assert _bodies(plan, "/action") == ['{"action":"stop"}']


def test_resume_does_not_stop_a_locked_error_pod(tmp_path: Path):
    plan = _ResumePlan()
    plan.gets = [(200, _resume_pod(plan.pod_id, "ERROR", actions=["stop"], locked=True))]
    with _resume_server(plan) as base:
        completed = _run_resume(tmp_path, base, plan)
    assert completed.returncode != 0
    assert f"pod {plan.pod_id} is locked. Compute may still be billed." in completed.stderr
    assert _bodies(plan, "/action") == []


def test_resume_stops_a_pod_already_starting_when_ssh_never_appears(tmp_path: Path):
    plan = _ResumePlan()
    plan.gets = [(200, _resume_pod(plan.pod_id, "STARTING"))]
    plan.stop = (200, _resume_pod(plan.pod_id, "EXITED"))
    plan.after_stop_get = (200, _resume_pod(plan.pod_id, "EXITED"))
    with _resume_server(plan) as base:
        completed = _run_resume(tmp_path, base, plan, polls="2")
    assert completed.returncode != 0
    assert "pod failed to expose SSH" in completed.stderr
    assert _bodies(plan, "/action") == ['{"action":"stop"}']
    assert f"pod {plan.pod_id} status EXITED. Compute is not billed." in completed.stdout
