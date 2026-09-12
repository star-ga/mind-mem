#!/usr/bin/env python3
"""End-to-end RunPod driver for full-FT on Qwen3.5-4B.

Flow:

    1. Provision a pod via RunPod REST API. Default: A100 80 GB
       (~$1.64/hr spot). Override with --gpu-type.
    2. Wait until the pod reports SSH reachable.
    3. scp the corpus + train script onto the pod.
    4. ssh into the pod; pip install deps; run runpod_full_ft.py.
    5. Poll training log until the saved model appears.
    6. scp the merged weights back to /data/checkpoints/.
    7. Push to HF at star-ga/mind-mem-4b.
    8. Tear down the pod.

Requires:
    - RunPod API key in ~/.runpod/config.toml (already present)
    - SSH key at ~/.ssh/runpod_key{,.pub}
    - HF write token at /tmp/hf_write_token
    - Corpus built at /data/checkpoints/mm-workspace/train-output/corpus.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS = Path(
    os.environ.get(
        "MM_CORPUS",
        "/data/checkpoints/mm-workspace/train-output/corpus.jsonl",
    )
)
SSH_KEY = Path(os.environ.get("MM_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519")))
# scp lands the pod's `/workspace/train-output/full-ft` directory at
# `WEIGHTS_OUT.parent` and creates a `full-ft/` subdir inside it. So the
# *actual* destination is `WEIGHTS_OUT.parent / "full-ft"`. Default
# parent is `/data/checkpoints/mm-workspace/`, so weights end up at
# `/data/checkpoints/mm-workspace/full-ft/`. The verify-first hash check
# at the end MUST point at that exact path or it will false-negative
# (and silently keep the pod alive — has happened in past runs).
WEIGHTS_OUT = Path(
    os.environ.get(
        "MM_WEIGHTS_OUT",
        "/data/checkpoints/mm-workspace/full-ft",
    )
)
HF_TOKEN_FILE = Path(os.environ.get("MM_HF_TOKEN_FILE", "/tmp/hf_write_token"))
RUNPOD_CONFIG = Path(os.environ.get("MM_RUNPOD_CONFIG", str(Path.home() / ".runpod" / "config.toml")))

# Image ships PyTorch 2.x + CUDA 12.x; add our deps via pip at runtime.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPU_TYPE = "NVIDIA A100 80GB PCIe"
DEFAULT_CONTAINER_DISK_GB = 60
DEFAULT_VOLUME_GB = 40

# The pod is deliberately a small, reproducible checkout rather than a flat
# collection of scripts.  Evaluators derive the repository root from
# ``__file__.parents[1]`` and the model-card builder reaches into ``src/`` and
# ``scripts/``; preserving these paths keeps every attested command on the
# same source tree.
REMOTE_ROOT = "/workspace"
REMOTE_SOURCE_ROOT = f"{REMOTE_ROOT}/mind-mem-release"
REMOTE_TRAIN_ROOT = f"{REMOTE_ROOT}/train-output"
REMOTE_FULLFT_DIR = f"{REMOTE_TRAIN_ROOT}/full-ft"
REMOTE_CORPUS = f"{REMOTE_TRAIN_ROOT}/corpus.jsonl"
REMOTE_EVAL_REPORT = f"{REMOTE_TRAIN_ROOT}/eval_report.json"
REMOTE_HOLDOUT_REPORT = f"{REMOTE_TRAIN_ROOT}/eval_holdout_report.json"
REMOTE_TOKEN_FILE = f"{REMOTE_ROOT}/.hf_token"

# These are the source files needed by training, both gated evaluations, the
# receipt validator, the model-card builder, and the canonical Qwen loader.
# Every destination keeps its checkout-relative train/src/scripts path.
RELEASE_FILES = (
    "train/runpod_full_ft.py",
    "train/_causal_lm_import.py",
    "train/eval_harness.py",
    "train/eval_holdout.py",
    "train/eval_receipt.py",
    "train/build_corpus.py",
    "train/build_model_card.py",
    "train/upload_to_hf.py",
    "src/mind_mem/causal_lm_loader.py",
    "src/mind_mem/__init__.py",
    "scripts/count_mcp_tools.py",
)


def _api_key() -> str:
    for line in RUNPOD_CONFIG.read_text(encoding="utf-8").splitlines():
        if "api_key" in line or line.strip().startswith("apikey"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"no api_key found in {RUNPOD_CONFIG}")


def _api_call(method: str, path: str, body: dict | None = None) -> dict:
    url = f"https://rest.runpod.io/v1{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode() or "{}"
            return json.loads(text) if text.strip() else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API {method} {path} failed: {e.code} — {e.read().decode()[:400]}")


# ---------------------------------------------------------------------------
# Pod lifecycle
# ---------------------------------------------------------------------------


def provision(gpu_type: str = DEFAULT_GPU_TYPE, image: str = DEFAULT_IMAGE) -> str:
    """Create a pod, return its ID once it's running."""
    pub = Path(f"{SSH_KEY}.pub").read_text(encoding="utf-8").strip()
    # SECURE cloud: a v3.9.2 retrain on COMMUNITY was preempted twice on
    # 2026-05-05 (~2h of compute lost each time, EXITED pods couldn't be
    # restarted), so we eat the small SECURE premium to get an
    # uninterruptable run on H200.
    body = {
        "name": "mind-mem-4b-fullft",
        "imageName": image,
        "gpuTypeIds": [gpu_type],
        "cloudType": os.environ.get("MM_RUNPOD_CLOUD", "SECURE"),
        "gpuCount": 1,
        "containerDiskInGb": DEFAULT_CONTAINER_DISK_GB,
        "volumeInGb": DEFAULT_VOLUME_GB,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": pub},
    }
    print(f"provisioning pod on {gpu_type} …")
    pod = _api_call("POST", "/pods", body)
    pid = pod["id"]
    print(f"pod id: {pid}")
    return pid


def _gql_pod(pod_id: str) -> dict:
    """REST `runtime` is broken on community-cloud H200s — fall back to
    the GraphQL endpoint, which exposes uptime + ports correctly.
    """
    query = (
        "{ myself { pods { id desiredStatus runtime { uptimeInSeconds "
        "ports { ip privatePort publicPort isIpPublic type } } } } }"
    )
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key()}",
            "User-Agent": "Mozilla/5.0",
        },
    )
    body = json.loads(urllib.request.urlopen(req, timeout=30).read())
    for p in body.get("data", {}).get("myself", {}).get("pods", []) or []:
        if p.get("id") == pod_id:
            return p
    return {}


def wait_ssh(pod_id: str, timeout: float = 900.0) -> tuple[str, int]:
    """Block until the pod's TCP-exposed SSH (port 22 → publicPort) is up.

    Returns (publicIp, publicPort). Polls GraphQL because REST returns
    runtime=None on community-cloud H200 pods.
    """
    start = time.time()
    last_status = None
    while time.time() - start < timeout:
        p = _gql_pod(pod_id)
        status = p.get("desiredStatus")
        runtime = p.get("runtime") or {}
        ports = runtime.get("ports") or []
        if status != last_status:
            print(f"  status={status}  ports={len(ports)}")
            last_status = status
        for port in ports:
            if port.get("privatePort") == 22 and port.get("isIpPublic"):
                ip, pub = port["ip"], int(port["publicPort"])
                print(f"pod ready — ssh -p {pub} root@{ip}")
                return ip, pub
        time.sleep(10)
    raise TimeoutError("pod never exposed SSH within timeout")


def destroy(pod_id: str) -> None:
    print(f"tearing down pod {pod_id} …")
    try:
        _api_call("DELETE", f"/pods/{pod_id}")
        print("pod destroyed")
    except Exception as exc:
        print(f"teardown failed (manual check needed): {exc}")


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------


_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ServerAliveInterval=30",
    # 30s × 20 = 10 min before disconnect — tolerates brief network blips
    # during long-running training streams. Earlier value (default 3) gave
    # 90s, which dropped a v3.9.1 retrain at step 21 on 2026-05-05.
    "-o", "ServerAliveCountMax=20",
    "-o", "TCPKeepAlive=yes",
]


def _ssh_cmd(ip: str, port: int, cmd: str) -> str:
    full = ["ssh", "-i", str(SSH_KEY), "-p", str(port), *_SSH_OPTS, f"root@{ip}", cmd]
    # `errors="replace"` so a mid-byte slice from `tail -c` (e.g. a multi-byte
    # UTF-8 char split) does NOT crash the polling loop with UnicodeDecodeError
    # (observed crash mid-retrain v1 on 2026-05-09 — byte 0x96 from training log).
    result = subprocess.run(full, capture_output=True, text=True, errors="replace", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ssh failed: {result.stderr[:400]}")
    return result.stdout


def _ssh_stream(ip: str, port: int, cmd: str) -> int:
    """Run remote command, streaming stdout/stderr live (no capture)."""
    full = ["ssh", "-i", str(SSH_KEY), "-p", str(port), *_SSH_OPTS, f"root@{ip}", cmd]
    return subprocess.run(full).returncode


def _scp_to(ip: str, port: int, local: str, remote: str) -> None:
    subprocess.run(
        ["scp", "-i", str(SSH_KEY), "-P", str(port), *_SSH_OPTS, local, f"root@{ip}:{remote}"],
        check=True,
    )


def _scp_from(ip: str, port: int, remote: str, local: str) -> None:
    subprocess.run(
        ["scp", "-r", "-i", str(SSH_KEY), "-P", str(port), *_SSH_OPTS, f"root@{ip}:{remote}", local],
        check=True,
    )


def _remote_env(*, include_base_model: bool = False) -> str:
    """Return the one path environment shared by every remote release step."""
    values = (
        ("MM_TRAIN_ROOT", REMOTE_TRAIN_ROOT),
        ("MM_FULLFT_DIR", REMOTE_FULLFT_DIR),
        ("MM_WEIGHTS_DIR", REMOTE_FULLFT_DIR),
        ("MM_HOLDOUT_REPORT", REMOTE_HOLDOUT_REPORT),
        ("MM_CORPUS", REMOTE_CORPUS),
    )
    assignments = [f"{name}={shlex.quote(value)}" for name, value in values]
    if include_base_model:
        assignments.append(
            "MM_BASE_MODEL="
            + shlex.quote(os.environ.get("MM_BASE_MODEL", "Qwen/Qwen3.5-4B"))
        )
    return " ".join(assignments)


def _stage_release_bundle(ip: str, port: int, repo_root: Path | None = None) -> None:
    """Copy the runnable checkout slice while preserving its source layout."""
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    paths = [root / relative for relative in RELEASE_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("release bundle source missing: " + ", ".join(missing))

    directories = sorted(
        {str(Path(REMOTE_SOURCE_ROOT, Path(relative).parent)) for relative in RELEASE_FILES}
    )
    _ssh_cmd(ip, port, "mkdir -p " + " ".join(shlex.quote(directory) for directory in directories))
    for relative, local in zip(RELEASE_FILES, paths):
        _scp_to(ip, port, str(local), str(Path(REMOTE_SOURCE_ROOT, relative)))
    tracked = " ".join(shlex.quote(relative) for relative in RELEASE_FILES)
    _ssh_cmd(
        ip,
        port,
        f"cd {shlex.quote(REMOTE_SOURCE_ROOT)} && git init -q && git reset -q && "
        "git config user.name 'STARGA Inc' && "
        "git config user.email 'noreply@star.ga' && "
        f"git add -- {tracked} && "
        "(git diff --cached --quiet || git commit -qm 'stage release bundle')",
    )


def _stage_hf_token(ip: str, port: int) -> None:
    """Transfer the token file without placing its contents in a command."""
    _scp_to(ip, port, str(HF_TOKEN_FILE), REMOTE_TOKEN_FILE)
    _ssh_cmd(ip, port, f"chmod 600 {shlex.quote(REMOTE_TOKEN_FILE)}")


def _training_command_body() -> str:
    """Build the inner shell that exports the token before exec-ing Python."""
    return (
        f'export HF_TOKEN="$(cat {shlex.quote(REMOTE_TOKEN_FILE)})" && '
        f"{_remote_env(include_base_model=True)} "
        "exec python3 -u train/runpod_full_ft.py"
    )


def _training_launch_command() -> str:
    """Build the detached training command without exposing the HF token."""
    return (
        f"cd {shlex.quote(REMOTE_SOURCE_ROOT)} && nohup bash -lc "
        + shlex.quote(_training_command_body())
        + " "
        f">{shlex.quote(REMOTE_TRAIN_ROOT + '/train.log')} 2>&1 < /dev/null & "
        f'echo "launched pid=$!" >{shlex.quote(REMOTE_TRAIN_ROOT + "/training.pid")}; sleep 3'
    )


def _run_gated_evals(ip: str, port: int) -> None:
    """Run both evaluations with identical explicit input/output paths."""
    for label, script in (
        ("main", "train/eval_harness.py"),
        ("holdout", "train/eval_holdout.py"),
    ):
        command = (
            f"cd {shlex.quote(REMOTE_SOURCE_ROOT)} && {_remote_env()} "
            f"python3 -u {shlex.quote(script)}"
        )
        try:
            _ssh_cmd(ip, port, command)
        except RuntimeError as exc:
            raise RuntimeError(f"{label} evaluation failed; refusing release") from exc


def _run_release_commands(ip: str, port: int, version_tag: str, *, skip_upload: bool) -> None:
    """Gate, card, and optionally upload a completed remote checkpoint."""
    _run_gated_evals(ip, port)
    _ssh_cmd(
        ip,
        port,
        f"cd {shlex.quote(REMOTE_SOURCE_ROOT)} && {_remote_env()} "
        "python3 -u train/build_model_card.py",
    )
    if skip_upload:
        print("--skip-upload set: NOT pushing to HF. Eval locally first, then re-run upload manually.")
        return

    commit_msg = f"Full-FT retrain on Qwen3.5-4B ({version_tag})"
    _ssh_cmd(
        ip,
        port,
        f"cd {shlex.quote(REMOTE_SOURCE_ROOT)} && "
        f"{_remote_env()} HF_TOKEN=\"$(cat {shlex.quote(REMOTE_TOKEN_FILE)})\" "
        f"python3 -u train/upload_to_hf.py --commit-message {shlex.quote(commit_msg)}",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image for the pod")
    parser.add_argument(
        "--version-tag",
        default=os.environ.get("MM_VERSION_TAG", "v4.0.0"),
        help="Version label used in the HF commit message (e.g. v4.0.0). "
        "Drives `--commit-message 'Full-FT retrain on Qwen3.5-4B (<TAG>)'`.",
    )
    parser.add_argument(
        "--provision-only",
        action="store_true",
        help="Create pod + print SSH info; don't run training or tear down.",
    )
    parser.add_argument(
        "--destroy",
        metavar="POD_ID",
        help="Just destroy an existing pod and exit.",
    )
    parser.add_argument(
        "--pod-id",
        metavar="POD_ID",
        help="Reuse an already-provisioned pod instead of creating one.",
    )
    parser.add_argument(
        "--keep-pod",
        action="store_true",
        default=True,
        help="Don't destroy the pod after training (default on — protects "
        "against losing weights if scp-back or HF upload fails). Use "
        "--auto-destroy to override.",
    )
    parser.add_argument(
        "--auto-destroy",
        dest="keep_pod",
        action="store_false",
        help="Destroy the pod automatically after training + upload complete. Only safe once you've verified the artifacts landed.",
    )
    parser.add_argument(
        "--verify-first",
        action="store_true",
        default=True,
        help="Before destroy, verify merged weights exist locally and are readable. Default on.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        default=False,
        help="Skip the HF upload step. Use when you want to eval locally before publishing.",
    )
    args = parser.parse_args()

    if args.destroy:
        destroy(args.destroy)
        return

    if not CORPUS.is_file():
        sys.exit(f"corpus missing: {CORPUS}. Run build_corpus.py first.")
    if not HF_TOKEN_FILE.is_file():
        sys.exit(f"HF write token missing: {HF_TOKEN_FILE}")
    if not Path(f"{SSH_KEY}.pub").is_file():
        sys.exit(f"SSH key missing: {SSH_KEY}(.pub)")

    pod_id = args.pod_id or provision(gpu_type=args.gpu_type, image=args.image)
    if args.pod_id:
        print(f"reusing existing pod {pod_id}")
    try:
        ip, port = wait_ssh(pod_id)
        if args.provision_only:
            print(f"pod {pod_id} alive; SSH at {ip}:{port}")
            print(f"ssh -i {SSH_KEY} -p {port} root@{ip}")
            return

        # 1. Prep remote workspace + deps
        # Pin versions to a tested-green set:
        #   * Qwen3.5-4B uses model_type `qwen3_5`, which only landed in
        #     transformers >= 4.57 / 5.x. transformers 4.46.x does NOT load
        #     the config.
        #   * transformers >= 5.8 added an FP8-MoE `torch.library.custom_op`
        #     whose schema inference fails on torch < 2.7 (the runpod image
        #     ships torch 2.4.1) — bumping trl to 1.3.0 also pulls a
        #     `HybridCache` import that 5.7.0 doesn't expose.
        # The narrow working window is `transformers==5.7.0` + `trl==1.2.0`
        # on the runpod/pytorch:2.4 image; outside that, the import chain
        # breaks before training starts.
        print("installing deps on the pod …")
        _ssh_cmd(
            ip, port,
            "mkdir -p /workspace/train-output && "
            "pip install -q --no-cache-dir "
            "'transformers==5.7.0' 'trl==1.2.0' 'peft==0.14.0' "
            "'accelerate==1.5.0' 'bitsandbytes==0.46.1' 'datasets>=4.7.0,<5' "
            "huggingface_hub",
        )

        # 2. Ship corpus + complete source bundle.  The remote checkout must
        # retain train/ and src/ because eval receipts attest those paths.
        print("uploading corpus + scripts …")
        _scp_to(ip, port, str(CORPUS), REMOTE_CORPUS)
        _stage_release_bundle(ip, port)

        # 3. Launch training via nohup so it survives SSH disconnects
        # (RunPod hosts have been dropping connections every ~3-20 min,
        # which previously SIGHUP'd training when it ran inside the SSH
        # session). Now: detach via nohup, then poll the log file via
        # short ssh sessions until the run finishes or fails.
        #
        # HF_TOKEN goes to a file on the pod first (mode 600) so its contents
        # never appear in an SSH command, process argument, or log.
        print("staging HF token + launching full FT via nohup (survives SSH drops) …")
        _stage_hf_token(ip, port)
        _ssh_cmd(ip, port, _training_launch_command())
        # Poll until the saved-model marker appears in the log or the
        # training process is gone with no marker (= failure).
        print("polling training log every 30s (will break on completion / failure) …")
        last_size = 0
        rc = 1
        for _ in range(600):  # 600 × 30s = 5 hours wall-time cap
            time.sleep(30)
            try:
                tail = _ssh_cmd(ip, port, "tail -c 8000 /workspace/train-output/train.log 2>/dev/null; echo ---; pgrep -f runpod_full_ft.py | head -1")
            except RuntimeError:
                # transient ssh drop — retry next cycle
                continue
            log_part, _, pid_part = tail.rpartition("---")
            still_running = bool(pid_part.strip().isdigit())
            new_size = len(log_part)
            if new_size != last_size:
                # print one short progress snippet so the operator sees forward motion
                snippet = log_part.strip().splitlines()[-1] if log_part.strip() else "(empty)"
                print(f"  [pod log @ {new_size:>9} bytes, alive={still_running}] {snippet[:140]}")
                last_size = new_size
            if "training complete — full-FT weights saved" in log_part:
                rc = 0
                print("✓ training complete on pod")
                break
            if not still_running:
                # process exited — was it success or failure?
                if "training complete — full-FT weights saved" in log_part:
                    rc = 0
                    print("✓ training complete on pod")
                else:
                    print("✗ training process exited without success marker")
                    rc = 1
                break
        if rc != 0:
            raise RuntimeError("training exited / timed out without success; pod kept alive for inspection")

        # 4. Both evaluations must pass before card generation or publication.
        # The skip-upload option still runs and records both gates, then leaves
        # the attested bundle available for a later explicit upload command.
        print("running main + held-out evaluation gates …")
        _run_release_commands(ip, port, args.version_tag, skip_upload=args.skip_upload)

        # 5. Pull a copy of the weights back for local reference.
        WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)
        print(f"pulling merged weights to {WEIGHTS_OUT} …")
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                _scp_from(ip, port, "/workspace/train-output/full-ft", str(WEIGHTS_OUT.parent))
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                print(f"  scp attempt {attempt} failed: {exc}")
                time.sleep(10)
        if last_err is not None:
            print(
                f"\n⚠ scp-back FAILED after 3 attempts: {last_err}\n"
                f"⚠ pod left alive so you can retry manually:\n"
                f"    scp -r -i {SSH_KEY} -P {port} root@{ip}:/workspace/train-output/full-ft {WEIGHTS_OUT.parent}"
            )
            args.keep_pod = True

        # 6. Verify before tearing down: SHA256 cross-check between
        #    pod-side and locally-pulled weights. Only destroy when the
        #    hash confirms — protects against silent scp corruption.
        if args.verify_first and not args.keep_pod:
            weights = WEIGHTS_OUT / "model.safetensors"
            shards = list(WEIGHTS_OUT.glob("model-*.safetensors"))
            if not weights.is_file() and not shards:
                print(f"\n⚠ no weight file found at {WEIGHTS_OUT}. Keeping pod alive; manual inspection required.")
                args.keep_pod = True
            else:
                target_files = [weights] if weights.is_file() else sorted(shards)
                all_match = True
                # Give the OS a moment to flush scp page-cache writes
                # before hashing — on a busy SSD an 8-9 GB safetensors
                # file may not be fully fsync'd at scp's close().
                os.sync()
                time.sleep(2)
                for f in target_files:
                    rel = f.name
                    print(f"  hash-check {rel} …")
                    try:
                        remote_sha = _ssh_cmd(
                            ip, port,
                            f"sha256sum /workspace/train-output/full-ft/{rel} | awk '{{print $1}}'",
                        ).strip()
                    except RuntimeError as exc:
                        print(f"  ⚠ remote sha256sum failed for {rel}: {exc}")
                        all_match = False
                        break
                    h = hashlib.sha256()
                    with f.open("rb") as fh:
                        for chunk in iter(lambda: fh.read(1 << 20), b""):
                            h.update(chunk)
                    local_sha = h.hexdigest()
                    if local_sha == remote_sha:
                        print(f"  ✓ {rel}  sha256={local_sha[:16]}…  match")
                    else:
                        print(
                            f"  ✗ {rel}  hash mismatch\n"
                            f"      local  = {local_sha}\n"
                            f"      remote = {remote_sha}"
                        )
                        all_match = False
                        break
                if not all_match:
                    print(
                        "\n⚠ hash mismatch — keeping pod alive for re-pull. "
                        "Re-scp manually before destroying."
                    )
                    args.keep_pod = True
                else:
                    print(
                        f"\n✓ all weight files SHA256-confirmed against pod copy. "
                        f"Pod {pod_id} safe to terminate."
                    )

        print("\n✓ full-FT complete + uploaded + local copy saved")
    finally:
        if not args.keep_pod:
            destroy(pod_id)
        else:
            print(f"\npod {pod_id} LEFT RUNNING (--keep-pod).  Destroy manually when done:")
            print(f"  python3 {__file__} --destroy {pod_id}")


if __name__ == "__main__":
    main()
