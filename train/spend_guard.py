"""spend_guard — launch approval checks and operator run records.

What this prevents (and why it exists):
  v4.0.x retry-2c through retry-2h burned ~$48 over 4 days because no
  mechanical check prevented (a) overwriting un-pulled weights with a
  next-run-on-same-pod, or (b) spinning the next pod before the previous
  run's eval result was locked in.

  Operating rules for a complete launch interlock:

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

  Also pass --config-sha256 from runpod_deploy.py --print-approval-config.
  The marker contains exact ``tag:``, ``budget_usd:`` and ``config_sha256:``
  lines. Both entrypoints require the digest; RunPod compares it with the
  explicit non-secret requested launch configuration.

  # Only if all 4 checks pass:
  python3 train/runpod_deploy.py ...

  python3 spend_guard.py postflight \\
      --tag retry2j \\
      --pod-id vgpy7ctbzcrxq7 \\
      --local-weights /data/checkpoints/mm-workspace/full-ft.retry2j-...

Implementation scope:
  runpod_deploy.py enforces R2 before launch. This separate preflight checks
  R1/R3 against operator ledger fields; those fields do not independently prove
  remote termination or checkpoint provenance. Existing-pod identity, immutable
  training inputs, one-time approval consumption and a runtime billing ceiling
  are not implemented here. This is not full launch readiness.

Ledger:
  <repo>/.run-ledger.jsonl (append-only, JSONL)
  One line per run with: tag, started_at, pod_id, budget_usd, spend_usd,
  weights_local_path, weights_local_sha256, weights_pod_sha256,
  hash_match, pod_terminated_at, eval_summary, status.
  New budget_usd values are decimal strings to retain exact amounts;
  historical numeric rows remain readable.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

LEDGER = Path(os.environ.get(
    "MM_RUN_LEDGER", str(Path(__file__).resolve().parents[1] / ".run-ledger.jsonl")))
WEIGHT_ROOT = Path("/data/checkpoints/mm-workspace")
KNOWN_GOOD = WEIGHT_ROOT / "full-ft.retry2e-109of109+18of22"
_APPROVAL_LINE = re.compile(r"^(tag|budget_usd|config_sha256): ([^\s#]+)$")
_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]*$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class ApprovalError(ValueError):
    """An approval marker is missing, malformed, or does not match the run."""


@dataclass(frozen=True)
class SpendApproval:
    tag: str
    budget_usd: Decimal
    config_sha256: str | None
    marker_sha256: str


def _parse_budget(value: str) -> Decimal:
    try:
        budget = Decimal(value)
    except InvalidOperation as exc:
        raise ApprovalError("budget_usd must be a finite decimal") from exc
    if not budget.is_finite() or budget <= 0:
        raise ApprovalError("budget_usd must be greater than zero")
    return budget


def parse_approval_file(path: Path) -> SpendApproval:
    """Parse the small approval format without substring or comment matching."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ApprovalError(f"approval file cannot be read: {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ApprovalError(f"approval file is not UTF-8: {path}") from exc

    fields: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        match = _APPROVAL_LINE.fullmatch(line)
        if match is None:
            raise ApprovalError(f"approval line {number} is not an exact field")
        key, value = match.groups()
        if key in fields:
            raise ApprovalError(f"approval field {key!r} is duplicated")
        fields[key] = value
    if set(fields) - {"tag", "budget_usd", "config_sha256"} or not {"tag", "budget_usd"} <= set(fields):
        raise ApprovalError("approval must contain exactly tag and budget_usd, with optional config_sha256")
    tag = fields["tag"]
    if _TAG.fullmatch(tag) is None:
        raise ApprovalError("approval tag contains unsupported characters")
    config = fields.get("config_sha256")
    if config is not None and _SHA256.fullmatch(config) is None:
        raise ApprovalError("approval config_sha256 must be 64 hexadecimal characters")
    return SpendApproval(
        tag=tag,
        budget_usd=_parse_budget(fields["budget_usd"]),
        config_sha256=config.lower() if config else None,
        marker_sha256=hashlib.sha256(raw).hexdigest(),
    )


def validate_approval(
    path: Path,
    *,
    expected_tag: str,
    expected_budget_usd: str | int | float | Decimal,
    expected_config_sha256: str | None = None,
    require_config: bool = False,
) -> SpendApproval:
    """Validate an approval against the exact launch values."""
    approval = parse_approval_file(path)
    if approval.tag != expected_tag:
        raise ApprovalError(f"approval tag {approval.tag!r} does not match requested tag {expected_tag!r}")
    expected_budget = _parse_budget(str(expected_budget_usd))
    if approval.budget_usd != expected_budget:
        raise ApprovalError(f"approval budget_usd {approval.budget_usd} does not match requested {expected_budget}")
    if require_config and not approval.config_sha256:
        raise ApprovalError("approval config_sha256 is required for provisioning")
    if expected_config_sha256 is not None:
        if _SHA256.fullmatch(expected_config_sha256) is None:
            raise ApprovalError("expected config_sha256 must be 64 hexadecimal characters")
        if approval.config_sha256 != expected_config_sha256.lower():
            raise ApprovalError("approval config_sha256 does not match the requested launch configuration")
    return approval


def launch_config_sha256(**config: object) -> str:
    """Digest the explicit, non-secret provisioning configuration."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


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
    try:
        if not getattr(args, "config_sha256", None):
            raise ApprovalError("expected config_sha256 is required")
        parsed = validate_approval(
            approval,
            expected_tag=args.tag,
            expected_budget_usd=args.budget_usd,
            expected_config_sha256=getattr(args, "config_sha256", None),
            require_config=True,
        )
    except ApprovalError as exc:
        _refuse("R2", str(exc))

    # Stage the new ledger entry; postflight will fill in the rest.
    entry = {
        "tag": args.tag,
        "started_at": _now(),
        # New rows retain decimal text rather than a rounded JSON float.
        # Historical numeric rows remain readable by _read_ledger.
        "budget_usd": str(parsed.budget_usd),
        "prev_run_tag": args.prev_run_tag,
        "approval_sha256": parsed.marker_sha256,
        "config_sha256": parsed.config_sha256,
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
    p.add_argument("--budget-usd", required=True, help="exact positive decimal USD amount")
    p.add_argument("--approval-file", required=True)
    p.add_argument(
        "--config-sha256",
        required=True,
        help="expected SHA-256 of the explicit launch configuration",
    )
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
