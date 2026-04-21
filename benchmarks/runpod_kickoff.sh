#!/usr/bin/env bash
# mind-mem-4b v2 — Runpod one-shot kickoff.
#
# CRITICAL: all artifacts go to /runpod-volume (persists across pod
# terminations). Container disk (/workspace) is ephemeral — only the
# repo clone and pip cache live there. A pod termination mid-training
# will keep every completed save_steps checkpoint on the volume.
#
# Runs on the pod once the operator has:
#   1. Spun up a Runpod A100 80GB (or H200) pod with template
#      runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#      Must have a volume attached — the default mount is /runpod-volume.
#   2. Uploaded the training corpus:
#         scp /tmp/mm-train/*.jsonl root@<pod>:/runpod-volume/mm-train/
#      or via ``runpodctl send`` / web uploader.
#   3. SSH'd into the pod.
#
# Then:
#   curl -sSL https://raw.githubusercontent.com/star-ga/mind-mem/main/benchmarks/runpod_kickoff.sh \
#        | GPU=a100 bash
#
# Env:
#   GPU=a100|h200   (default: auto-detected)
#   HF_TOKEN=<your HF token>   — required for pulling star-ga/mind-mem-4b
#   WANDB_API_KEY=<optional>   — set to enable W&B logging
#   VOLUME=/runpod-volume      — override if your mount is elsewhere
set -euo pipefail

# ---------------------------------------------------------------------------
# 0a. Volume sanity — everything expensive lives here. Fail fast if it's
#     not mounted so we don't write 30GB of checkpoints to container disk
#     that evaporates on pod terminate.
# ---------------------------------------------------------------------------
VOLUME="${VOLUME:-/runpod-volume}"
if [[ ! -d "${VOLUME}" ]]; then
  echo "ERROR: volume not mounted at ${VOLUME}."
  echo "       Attach a Runpod volume to this pod (Settings → Volume) and re-run."
  echo "       Or override with VOLUME=/some/other/mount bash ..."
  exit 2
fi
mkdir -p "${VOLUME}/mm-train" "${VOLUME}/hf-cache" "${VOLUME}/logs"
export HF_HOME="${VOLUME}/hf-cache"           # model + dataset caches persist
export TRANSFORMERS_CACHE="${VOLUME}/hf-cache"
echo "[info] volume: ${VOLUME} (hf-cache, checkpoints, logs all persist here)"

# ---------------------------------------------------------------------------
# 0b. Detect GPU (unless forced via env)
# ---------------------------------------------------------------------------
GPU="${GPU:-}"
if [[ -z "${GPU}" ]]; then
  if nvidia-smi | grep -qi h200; then
    GPU=h200
  elif nvidia-smi | grep -qi "a100.*80"; then
    GPU=a100
  elif nvidia-smi | grep -qi "a100.*40"; then
    echo "WARN: A100 40GB detected — marginal for 4B full FT. Consider QLoRA."
    GPU=a100
  else
    echo "Could not auto-detect GPU. Set GPU=a100|h200 and re-run."
    nvidia-smi
    exit 2
  fi
fi
echo "[info] GPU=${GPU}"

# ---------------------------------------------------------------------------
# 1. Clone repo (or reuse existing)
# ---------------------------------------------------------------------------
cd /workspace
if [[ ! -d mind-mem ]]; then
  git clone --depth 1 https://github.com/star-ga/mind-mem.git
fi
cd /workspace/mind-mem

# ---------------------------------------------------------------------------
# 2. Install training stack
# ---------------------------------------------------------------------------
pip install --upgrade pip
pip install -e ".[train]"
pip install "huggingface_hub[cli]" "datasets" "accelerate" "trl" "peft"

# ---------------------------------------------------------------------------
# 3. Auth — HuggingFace (required)
# ---------------------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: set HF_TOKEN env var (HuggingFace token with read access to star-ga/mind-mem-4b)"
  exit 1
fi
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential

# Optional: wandb
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  pip install wandb
  wandb login "${WANDB_API_KEY}"
  REPORT_TO=wandb
else
  REPORT_TO=none
fi

# ---------------------------------------------------------------------------
# 4. Validate corpus is present (on the volume)
# ---------------------------------------------------------------------------
CORPUS_DIR="${VOLUME}/mm-train"
# Back-compat: if the operator uploaded to /workspace/mm-train instead,
# move it to the volume once so re-runs don't re-upload.
if [[ ! -f "${CORPUS_DIR}/dispatchers.jsonl" ]] && [[ -f /workspace/mm-train/dispatchers.jsonl ]]; then
  echo "[info] migrating /workspace/mm-train → ${CORPUS_DIR} (volume)"
  cp /workspace/mm-train/*.jsonl "${CORPUS_DIR}/"
fi
for f in dispatchers.jsonl retrieval.jsonl; do
  if [[ ! -f "${CORPUS_DIR}/${f}" ]]; then
    echo "ERROR: ${CORPUS_DIR}/${f} missing — upload /tmp/mm-train/*.jsonl to ${CORPUS_DIR} first."
    exit 1
  fi
done

# Concatenate + shuffle once so the trainer gets a single file.
MIXED="${CORPUS_DIR}/mixed.jsonl"
cat "${CORPUS_DIR}"/dispatchers.jsonl "${CORPUS_DIR}"/retrieval.jsonl \
  | shuf --random-source=/dev/urandom > "${MIXED}"
echo "[info] mixed corpus: $(wc -l < "${MIXED}") examples"

# ---------------------------------------------------------------------------
# 5. Select config per GPU
# ---------------------------------------------------------------------------
case "${GPU}" in
  h200) CONFIG=benchmarks/train_config.yaml ;;
  a100) CONFIG=benchmarks/train_config_a100.yaml ;;
  *)    echo "ERROR: unsupported GPU=${GPU}"; exit 2 ;;
esac

# Override report_to if wandb is live (avoid editing committed yaml).
RUN_ENV=(
  "MIND_MEM_REPORT_TO=${REPORT_TO}"
  "TRANSFORMERS_NO_ADVISORY_WARNINGS=1"
  "TOKENIZERS_PARALLELISM=true"
)

OUTPUT_DIR="${VOLUME}/mind-mem-4b-v2"
LOG_FILE="${VOLUME}/logs/training-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "${OUTPUT_DIR}"

echo "[info] output_dir: ${OUTPUT_DIR} (PERSISTS across pod terminations)"
echo "[info] log: ${LOG_FILE}"
echo "[info] launching training — ~6-10h on H200, ~10-16h on A100 80GB"
env "${RUN_ENV[@]}" python3 benchmarks/train_mind_mem_4b.py \
  --base-model star-ga/mind-mem-4b \
  --data "${MIXED}" \
  --output-dir "${OUTPUT_DIR}" \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG_FILE}"

# ---------------------------------------------------------------------------
# 6. Post-run: tarball the checkpoint to the volume so the operator can
#    download even after pod termination via runpodctl.
# ---------------------------------------------------------------------------
cd "${VOLUME}"
OUTPUT_TAR="mind-mem-4b-v2-$(date +%Y%m%d).tar.zst"
tar --use-compress-program='zstd -T0 -10' \
    -cf "${OUTPUT_TAR}" "$(basename "${OUTPUT_DIR}")"
echo "[done] training complete."
echo "[done] checkpoint dir: ${OUTPUT_DIR}"
echo "[done] checkpoint tar: ${VOLUME}/${OUTPUT_TAR} ($(du -h "${OUTPUT_TAR}" | cut -f1))"
echo "[done] log:            ${LOG_FILE}"
echo ""
echo "Next steps (operator, locally):"
echo "  1. Download: runpodctl receive ${VOLUME}/${OUTPUT_TAR}"
echo "  2. huggingface-cli upload star-ga/mind-mem-4b ./mind-mem-4b-v2 . --commit-message 'v2 full fine-tune'"
echo "  3. ollama create mind-mem:4b -f Modelfile && ollama push mind-mem:4b"
echo ""
echo "SAFE TO TERMINATE POD — checkpoint is on the volume, survives terminate/restart."
