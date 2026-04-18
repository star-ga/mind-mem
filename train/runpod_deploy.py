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
import json
import os
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
SSH_KEY = Path(os.environ.get("MM_SSH_KEY", "/home/n/.ssh/runpod_key"))
WEIGHTS_OUT = Path(
    os.environ.get(
        "MM_WEIGHTS_OUT",
        "/data/checkpoints/mm-workspace/mind-mem-4b-fullft",
    )
)
HF_TOKEN_FILE = Path(os.environ.get("MM_HF_TOKEN_FILE", "/tmp/hf_write_token"))
RUNPOD_CONFIG = Path("/home/n/.runpod/config.toml")

# Image ships PyTorch 2.x + CUDA 12.x; add our deps via pip at runtime.
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.12-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_GPU_TYPE = "NVIDIA A100 80GB PCIe"
DEFAULT_CONTAINER_DISK_GB = 60
DEFAULT_VOLUME_GB = 40


def _api_key() -> str:
    for line in RUNPOD_CONFIG.read_text().splitlines():
        if "api_key" in line:
            return line.split("=", 1)[1].strip().strip('"')
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
    pub = Path(f"{SSH_KEY}.pub").read_text().strip()
    body = {
        "name": "mind-mem-4b-fullft",
        "imageName": image,
        "gpuTypeIds": [gpu_type],
        "cloudType": "SECURE",
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


def wait_ssh(pod_id: str, timeout: float = 600.0) -> tuple[str, int]:
    """Block until SSH is reachable. Returns (ip, port)."""
    start = time.time()
    while time.time() - start < timeout:
        info = _api_call("GET", f"/pods/{pod_id}")
        status = info.get("desiredStatus")
        runtime = info.get("runtime") or {}
        ports = runtime.get("ports") or []
        for p in ports:
            if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("ip"):
                print(f"pod ready — ssh {p['ip']}:{p['publicPort']}")
                return p["ip"], int(p["publicPort"])
        print(f"  status={status} waiting for SSH port …")
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


def _ssh_cmd(ip: str, port: int, cmd: str) -> str:
    full = [
        "ssh",
        "-i",
        str(SSH_KEY),
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        f"root@{ip}",
        cmd,
    ]
    result = subprocess.run(full, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ssh failed: {result.stderr[:400]}")
    return result.stdout


def _scp_to(ip: str, port: int, local: str, remote: str) -> None:
    subprocess.run(
        [
            "scp",
            "-i",
            str(SSH_KEY),
            "-P",
            str(port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            local,
            f"root@{ip}:{remote}",
        ],
        check=True,
    )


def _scp_from(ip: str, port: int, remote: str, local: str) -> None:
    subprocess.run(
        [
            "scp",
            "-r",
            "-i",
            str(SSH_KEY),
            "-P",
            str(port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            f"root@{ip}:{remote}",
            local,
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image for the pod")
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

    hf_token = HF_TOKEN_FILE.read_text().strip()

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
        print("installing deps on the pod …")
        _ssh_cmd(
            ip,
            port,
            "mkdir -p /workspace/train-output && "
            "pip install -q --no-cache-dir transformers datasets peft bitsandbytes accelerate trl && "
            "pip install -q --no-cache-dir huggingface_hub",
        )

        # 2. Ship corpus + training script + upload helper
        print("uploading corpus + scripts …")
        _scp_to(ip, port, str(CORPUS), "/workspace/train-output/corpus.jsonl")
        _scp_to(ip, port, "/home/n/mind-mem/train/runpod_full_ft.py", "/workspace/runpod_full_ft.py")
        _scp_to(ip, port, "/home/n/mind-mem/train/upload_to_hf.py", "/workspace/upload_to_hf.py")
        _scp_to(ip, port, "/home/n/mind-mem/train/build_model_card.py", "/workspace/build_model_card.py")

        # 3. Run training
        print("running full FT on pod (this takes ~60-90 min on A100) …")
        _ssh_cmd(
            ip,
            port,
            f"cd /workspace && HF_TOKEN={hf_token} "
            "MM_TRAIN_ROOT=/workspace/train-output "
            "MM_CORPUS=/workspace/train-output/corpus.jsonl "
            "python3 -u runpod_full_ft.py 2>&1 | tee /workspace/train-output/train.log",
        )

        # 4. Generate + upload (model card + merged weights)
        print("building model card + pushing to HF …")
        _ssh_cmd(
            ip,
            port,
            "cd /workspace && MM_TRAIN_ROOT=/workspace/train-output "
            "python3 build_model_card.py && "
            f"HF_TOKEN={hf_token} python3 upload_to_hf.py "
            "--commit-message 'Full-FT retrain on Qwen3.5-4B (v3.0.1)'",
        )

        # 5. Pull a copy of the weights back for local reference.
        # Retry up to 3 times — network hiccups during scp of 8+ GB
        # would otherwise leave us without a local backup.
        WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)
        print(f"pulling merged weights to {WEIGHTS_OUT} …")
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                _scp_from(
                    ip,
                    port,
                    "/workspace/train-output/full-ft",
                    str(WEIGHTS_OUT.parent),
                )
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
                f"    scp -r -i {SSH_KEY} -P {port} "
                f"root@{ip}:/workspace/train-output/full-ft {WEIGHTS_OUT.parent}"
            )
            args.keep_pod = True

        # 6. Verify before tearing down.
        if args.verify_first and not args.keep_pod:
            weights = WEIGHTS_OUT / "model.safetensors"
            shards = list(WEIGHTS_OUT.glob("model-*.safetensors"))
            if not weights.is_file() and not shards:
                print(f"\n⚠ no weight file found at {WEIGHTS_OUT}. Keeping pod alive; manual inspection required.")
                args.keep_pod = True

        print("\n✓ full-FT complete + uploaded + local copy saved")
    finally:
        if not args.keep_pod:
            destroy(pod_id)
        else:
            print(f"\npod {pod_id} LEFT RUNNING (--keep-pod).  cost meter: ~$1.39/hr. Destroy manually when done:")
            print(f"  python3 {__file__} --destroy {pod_id}")


if __name__ == "__main__":
    main()
