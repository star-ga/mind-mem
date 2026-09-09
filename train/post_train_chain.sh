#!/usr/bin/env bash
# Post-training chain: wait for deploy → verify scp + SHA256 + pod-destroy markers → run eval.
#
# Triggered after `runpod_deploy.py --auto-destroy --skip-upload --version-tag v4.0.0`
# launches in background. Deploy already does scp + SHA256 + auto-destroy on success;
# this wrapper waits, verifies, and runs the un-softened eval harness + held-out paraphrase eval.
set -uo pipefail

LOG=/data/checkpoints/mm-workspace/train-output/h200_retrain.log
EVAL_LOG=/data/checkpoints/mm-workspace/train-output/eval.retry2f.log
HOLDOUT_LOG=/data/checkpoints/mm-workspace/train-output/eval_holdout.retry2f.log
WEIGHTS_DIR=/data/checkpoints/mm-workspace/full-ft
DEPLOY_PID="${1:-}"

if [[ -z "$DEPLOY_PID" ]]; then
    DEPLOY_PID=$(pgrep -f "runpod_deploy.py" | head -1)
fi
echo "[chain] waiting on deploy PID $DEPLOY_PID …"

# Wait for deploy process to exit (no polling burn; tail follows the log instead)
while kill -0 "$DEPLOY_PID" 2>/dev/null; do
    sleep 30
done
echo "[chain] deploy PID $DEPLOY_PID exited at $(date -Iseconds)"

# Verify success markers in log
if ! grep -q "all weight files SHA256-confirmed" "$LOG"; then
    echo "[chain] FAIL: SHA256 confirmation marker not found — NOT running eval"
    grep -E "✗|FAIL|Traceback|Error|Exited|Killed|hash mismatch" "$LOG" | tail -20
    exit 1
fi
if ! grep -q "pod destroyed" "$LOG"; then
    echo "[chain] WARN: pod-destroy marker not found — pod may still be alive"
    grep -E "pod " "$LOG" | tail -10
fi

# Verify weights landed locally
if [[ ! -f "$WEIGHTS_DIR/model.safetensors" ]] && [[ -z "$(ls $WEIGHTS_DIR/model-*.safetensors 2>/dev/null)" ]]; then
    echo "[chain] FAIL: no weights at $WEIGHTS_DIR"
    ls -la "$WEIGHTS_DIR" || true
    exit 1
fi
echo "[chain] weights present at $WEIGHTS_DIR; running un-softened eval …"

cd "$(git rev-parse --show-toplevel)"
export MM_FULLFT_DIR="$WEIGHTS_DIR"

# 1) un-softened eval_harness (109 probes, target 109/109)
python3 -u train/eval_harness.py 2>&1 | tee "$EVAL_LOG"
EVAL_RC=$?
echo "[chain] eval_harness exit=$EVAL_RC"

# 2) held-out paraphrase eval (22 probes, target ≥90%)
python3 -u train/eval_holdout.py 2>&1 | tee "$HOLDOUT_LOG"
HOLDOUT_RC=$?
echo "[chain] eval_holdout exit=$HOLDOUT_RC"

RECEIPT_RC=1
if [[ "$EVAL_RC" -eq 0 && "$HOLDOUT_RC" -eq 0 ]]; then
    EVAL_REPORT=/data/checkpoints/mm-workspace/train-output/eval_report.json \
    HOLDOUT_REPORT=/data/checkpoints/mm-workspace/full-ft/eval_holdout_report.json \
    python3 - <<'PY'
import json
import os
import sys

from train.eval_receipt import receipt_is_valid

for name in ("EVAL_REPORT", "HOLDOUT_REPORT"):
    path = os.environ[name]
    try:
        report = json.loads(open(path, encoding="utf-8").read())
        receipt = report["receipt"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[chain] FAIL: {name} has no usable receipt: {exc}")
        sys.exit(1)
    if not receipt.get("complete") or not receipt_is_valid(receipt):
        print(f"[chain] FAIL: {name} receipt is incomplete or tampered")
        sys.exit(1)
print("[chain] evaluation receipts complete and self-consistent")
PY
    RECEIPT_RC=$?
fi
if [[ "$RECEIPT_RC" -ne 0 ]]; then
    echo "[chain] FAIL: refusing to report a shippable eval without complete receipts"
    exit 2
fi

echo "[chain] DONE — eval_harness rc=$EVAL_RC, eval_holdout rc=$HOLDOUT_RC"
echo "[chain] reports:"
echo "  $EVAL_LOG"
echo "  $HOLDOUT_LOG"
echo "  /data/checkpoints/mm-workspace/full-ft/eval_holdout_report.json"
echo "  /data/checkpoints/mm-workspace/train-output/eval_report.json"
