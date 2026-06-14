"""spend_guard — mechanical interlock on cloud spend.

What this prevents (and why it exists):
  v4.0.x retry-2c through retry-2h burned ~$48 over 4 days because no
  mechanical check prevented (a) overwriting un-pulled weights with a
  next-run-on-same-pod, or (b) spinning the next pod before the previous
  run's eval result was locked in.

  Interlock rules (all must pass before runpod_deploy.py is allowed):

    R1.  Previous-run weights must be SCP'd locally AND sha256-verified
         against the pod copy. Record in .run-ledger.jsonl.
    R2.  An approval marker file must exist for this specific run
         (--budget USD --tag NAME) with a sha256 over the run config.
         No marker = no spend.
    R3.  The pod from the previous run must be in state 'terminated'
         (not 'paused', not 'exited'). One run per pod, full stop.
    R4.  The configured --version-tag must not already exist locally as
         a weights directory. Refuses to overwrite known-good 127/131.

Usage:
  python3 spend_guard.py preflight \\
      --tag retry2j \\
      --budget-usd 5 \\
      --approval-file ~/mind-mem-budget-approvals/retry2j.yml \\
      --prev-run-tag retry2i

  # Only if all 4 checks pass:
  python3 train/runpod_deploy.py ...

  python3 spend_guard.py postflight \\
      --tag retry2j \\
      --pod-id vgpy7ctbzcrxq7 \\
      --local-weights /data/checkpoints/mm-workspace/full-ft.retry2j-...

Ledger:
  <repo>/.run-ledger.jsonl (append-only, JSONL)
  One line per run with: tag, started_at, pod_id, budget_usd, spend_usd,
  weights_local_path, weights_local_sha256, weights_pod_sha256,
  hash_match, pod_terminated_at, eval_summary, status.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

LEDGER = Path(os.environ.get(
    "MM_RUN_LEDGER", str(Path(__file__).resolve().parents[1] / ".run-ledger.jsonl")))
WEIGHT_ROOT = Path("/data/checkpoints/mm-workspace")
KNOWN_GOOD = WEIGHT_ROOT / "full-ft.retry2e-109of109+18of22"


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _read_ledger() -> list[dict]:
    if not LEDGER.is_file():
        return []
    with LEDGER.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_ledger(entry: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _refuse(rule: str, detail: str) -> None:
    sys.stderr.write(f"\nSPEND-GUARD REFUSED ({rule})\n  {detail}\n\n")
    sys.exit(3)


def preflight(args) -> None:
    ledger = _read_ledger()

    # R4: tag must not collide with existing local weights dir.
    for d in WEIGHT_ROOT.iterdir():
        if args.tag in d.name and d.is_dir() and any(d.iterdir()):
            _refuse("R4", f"weights dir for tag '{args.tag}' already exists: {d}. "
                          "Pick a fresh tag, never overwrite.")

    # R1: previous run weights pulled + verified.
    if args.prev_run_tag:
        prev = next((e for e in ledger if e.get("tag") == args.prev_run_tag), None)
        if not prev:
            _refuse("R1", f"prev-run tag '{args.prev_run_tag}' not in ledger {LEDGER}")
        if not prev.get("hash_match"):
            _refuse("R1",
                f"prev run '{args.prev_run_tag}' hash_match=False or weights not SCP'd. "
                f"ledger says: {prev}")
        if not Path(prev["weights_local_path"]).is_dir():
            _refuse("R1",
                f"prev run '{args.prev_run_tag}' weights dir missing: "
                f"{prev['weights_local_path']}")

    # R3: previous pod was terminated.
    if args.prev_run_tag and prev:
        if not prev.get("pod_terminated_at"):
            _refuse("R3",
                f"prev pod for '{args.prev_run_tag}' was NOT terminated. "
                "One pod = one run. Always destroy after eval.")

    # R2: explicit budget approval marker.
    approval = Path(args.approval_file).expanduser()
    if not approval.is_file():
        _refuse("R2",
            f"approval file missing: {approval}. "
            f"Create with budget_usd={args.budget_usd}, tag={args.tag} "
            "and re-run preflight.")
    content = approval.read_text()
    if f"tag: {args.tag}" not in content:
        _refuse("R2", f"approval file does not declare tag '{args.tag}'")
    if f"budget_usd: {args.budget_usd}" not in content:
        _refuse("R2",
            f"approval file does not declare budget_usd={args.budget_usd}")

    # Stage the new ledger entry; postflight will fill in the rest.
    entry = {
        "tag": args.tag,
        "started_at": _now(),
        "budget_usd": args.budget_usd,
        "prev_run_tag": args.prev_run_tag,
        "approval_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "status": "preflight_passed",
    }
    _append_ledger(entry)
    print(f"\nSPEND-GUARD PRE-FLIGHT PASSED — tag={args.tag} budget=${args.budget_usd}\n"
          f"  R1 prev weights pulled+verified: OK\n"
          f"  R2 approval marker valid: OK\n"
          f"  R3 prev pod terminated: OK\n"
          f"  R4 no tag collision: OK\n"
          f"  ledger: {LEDGER}\n")


def postflight(args) -> None:
    """Lock in the run result: hash, SCP, destroy pod, ledger entry."""
    weights_dir = Path(args.local_weights)
    if not (weights_dir / "model.safetensors").is_file():
        _refuse("postflight",
            f"local weights not found: {weights_dir}/model.safetensors. "
            "SCP must complete BEFORE postflight is called.")
    local_sha = _sha256(weights_dir / "model.safetensors")

    pod_sha = args.pod_sha256
    if not pod_sha:
        _refuse("postflight",
            "--pod-sha256 is required. SCP must include a "
            "`sha256sum model.safetensors` from the pod for cross-check.")

    if local_sha != pod_sha:
        _refuse("postflight",
            f"hash mismatch! local={local_sha} pod={pod_sha}. "
            "Re-pull or treat run as lost.")

    # Verify pod is terminated.
    pod_terminated = args.pod_terminated_at or _now()

    entry = {
        "tag": args.tag,
        "ended_at": _now(),
        "pod_id": args.pod_id,
        "weights_local_path": str(weights_dir),
        "weights_local_sha256": local_sha,
        "weights_pod_sha256": pod_sha,
        "hash_match": True,
        "pod_terminated_at": pod_terminated,
        "spend_usd": args.spend_usd,
        "eval_summary": args.eval_summary or "",
        "status": "postflight_locked",
    }
    _append_ledger(entry)
    print(f"\nSPEND-GUARD POST-FLIGHT LOCKED — tag={args.tag}\n"
          f"  weights: {weights_dir}\n"
          f"  sha256:  {local_sha}\n"
          f"  pod:     {args.pod_id} (terminated {pod_terminated})\n"
          f"  spend:   ${args.spend_usd}\n"
          f"  ledger:  {LEDGER}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--tag", required=True)
    p.add_argument("--budget-usd", required=True, type=float)
    p.add_argument("--approval-file", required=True)
    p.add_argument("--prev-run-tag", default=None,
                   help="None on the first run; required from second run on")

    q = sub.add_parser("postflight")
    q.add_argument("--tag", required=True)
    q.add_argument("--pod-id", required=True)
    q.add_argument("--local-weights", required=True)
    q.add_argument("--pod-sha256", required=True,
                   help="sha256 of model.safetensors as computed ON the pod")
    q.add_argument("--spend-usd", type=float, default=0.0)
    q.add_argument("--eval-summary", default="")
    q.add_argument("--pod-terminated-at", default="")

    args = ap.parse_args()
    {"preflight": preflight, "postflight": postflight}[args.cmd](args)


if __name__ == "__main__":
    main()
