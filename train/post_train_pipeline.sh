#!/usr/bin/env bash
# Post-training pipeline for mind-mem-4b v3.9.2 (augmented-corpus retrain).
#
# Pre-conditions:
#   * H200 pod uz2uajluzskmm2 finished training and dumped weights to
#     /workspace/train-output/full-ft on the pod.
#   * SSH key at ~/.ssh/id_ed25519, pod reachable via MM_POD_IP / MM_POD_PORT.
#
# Steps:
#   1. scp weights pod → /data/checkpoints/mm-workspace/full-ft-v3.9.2/
#   2. Preserve previous v3.9.1 weights as full-ft-v3.9.1-failed-eval/
#   3. Symlink full-ft → full-ft-v3.9.2 at canonical location
#   4. Run eval gate (eval_harness.py)
#   5. Print decision: PASS → push HF + GGUF; FAIL → diagnose
set -euo pipefail

POD_IP="${MM_POD_IP:?set MM_POD_IP to the training pod IP}"
POD_PORT="${MM_POD_PORT:-22}"
SSH_KEY="${MM_SSH_KEY:-$HOME/.ssh/id_ed25519}"
WORKSPACE="/data/checkpoints/mm-workspace"
NEW_DIR="${WORKSPACE}/full-ft-v3.9.2"
OLD_DIR="${WORKSPACE}/mind-mem-4b-fullft/full-ft"
CANONICAL="${WORKSPACE}/mind-mem-4b-fullft/full-ft"

echo "=== Step 1: scp weights from pod ==="
mkdir -p "${NEW_DIR}"
scp -r -i "${SSH_KEY}" -P "${POD_PORT}" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    "root@${POD_IP}:/workspace/train-output/full-ft/." "${NEW_DIR}/"
echo "weights at ${NEW_DIR}:"
ls -lh "${NEW_DIR}" | head -10

echo ""
echo "=== Step 2: preserve previous v3.9.1 weights ==="
if [[ -d "${OLD_DIR}" && ! -L "${OLD_DIR}" ]]; then
    PRESERVE="${WORKSPACE}/mind-mem-4b-fullft/full-ft-v3.9.1-failed-eval-2026-05-05"
    if [[ ! -d "${PRESERVE}" ]]; then
        mv "${OLD_DIR}" "${PRESERVE}"
        echo "preserved old weights → ${PRESERVE}"
    else
        echo "preserve target exists already: ${PRESERVE}"
        rm -rf "${OLD_DIR}"
    fi
fi

echo ""
echo "=== Step 3: install new weights at canonical location ==="
ln -sf "${NEW_DIR}" "${CANONICAL}"
ls -la "${CANONICAL}"

echo ""
echo "=== Step 4: run eval gate ==="
cd "$(git rev-parse --show-toplevel)"
MM_FULLFT_DIR="${CANONICAL}" python3 train/eval_harness.py 2>&1 | tee "${WORKSPACE}/train-output/eval_v3.9.2.log"
RC=$?
echo ""
if [[ "${RC}" -eq 0 ]]; then
    echo "=== ✓ EVAL GATE PASSED ==="
    echo "Next: push to HF + build GGUF + import to Ollama"
else
    echo "=== ✗ EVAL GATE FAILED (exit ${RC}) ==="
    echo "Inspect ${WORKSPACE}/train-output/eval_v3.9.2.log + eval_report.json"
fi
exit "${RC}"
