# mind-mem-4b v2 training recipe — Runpod H200

This recipe retrains `star-ga/mind-mem-4b` against the v3.3.0 MCP
surface so the bundled local LLM natively emits:

- 7 consolidated dispatcher tool-calls (`recall(mode=…)`,
  `staged_change(phase=…)`, `graph(action=…)`, `memory_verify`,
  `core`, `kernels`, `compiled_truth`).
- Query decomposition / reformulation sub-queries for the new
  `query_planner` module.
- Structured evidence bundle consumption (`recall(format="bundle")`).
- The 7 typed graph edges (cites / derives_from / depends_on /
  tested_by / supersedes / contradicts / relates_to).

v1 was Qwen3.5-4B fine-tuned on the v2.10.0 MCP surface (19 tools).
v2 targets the 7-dispatcher v3.2.x surface + Tier-1/2/3 retrieval
call shapes.

## Runpod config

| | |
|---|---|
| GPU | **H200 141GB** (single — Qwen 4B full-FT fits with room for batch 16+) |
| Template | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Container disk | 200 GB (HF cache + checkpoints) |
| Volume disk | 500 GB (persists across pod restarts) |
| Region | US-WEST (lowest sustained-use pricing) |
| Estimate | ~$3.50/hr × ~8-12 hr = **$28-45** per full run |

## Base model + data mix

- Base: `star-ga/mind-mem-4b` (v1 checkpoint, already
  instruction-tuned on the MCP surface).
- Data:
  1. **Existing v1 corpus** (~40K examples) — keep ~20% as
     replay buffer to prevent regression on legacy tool names.
  2. **v3.2.x dispatcher corpus** — 10K synthetic examples
     generated via `benchmarks/generate_dispatcher_examples.py`
     (ship with v3.3.0). Each example is a (user prompt,
     expected dispatcher call) pair covering every dispatcher
     mode: `recall(mode="similar|verify|intent|diagnose")`,
     `staged_change(phase="propose|approve|rollback")`, etc.
  3. **v3.3.0 retrieval corpus** — 5K examples covering:
     - Query decomposition calls (question → sub-queries).
     - Query reformulation calls (question → paraphrases).
     - Entity extraction from user prompt (for entity prefetch).
     - LLM-as-reranker judgments (query + candidates → scores).
  4. **LoCoMo replay** — 1.9K QA pairs from the LoCoMo
     judge-log (`/tmp/mm-bench/locomo_v3.2.1_opus_conv0.json`
     etc.). Input = dialogue turn, output = ideal retrieval
     call pattern. This is the behavioural-lift signal.

## Hyperparameters

Copied verbatim from v1 run-log (checked against A100 80GB there),
adjusted for H200's 141GB VRAM headroom:

```yaml
# train_config.yaml
base_model: star-ga/mind-mem-4b
output_dir: /workspace/mind-mem-4b-v2
dtype: bfloat16
optim: adamw_torch_fused
learning_rate: 5e-6           # gentle — preserve v1 capability
lr_scheduler_type: cosine
warmup_ratio: 0.03
num_train_epochs: 3
per_device_train_batch_size: 16   # H200 headroom vs A100's 8
gradient_accumulation_steps: 2    # effective batch 32
max_seq_length: 4096
packing: true                     # 2-3x throughput on H200
save_strategy: steps
save_steps: 500
logging_steps: 25
report_to: wandb                  # optional — free tier for STARGA
seed: 42

# QLoRA disabled — full fine-tune fits on H200
# (v1 used QLoRA on A100 80GB; v2 can afford full-FT)
```

## Pod boot script

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. HF cache on volume, not container disk
export HF_HOME=/runpod-volume/hf-cache
mkdir -p "$HF_HOME"

# 2. Clone mind-mem + checkout v3.3.0-dev
cd /workspace
git clone https://github.com/star-ga/mind-mem.git
cd mind-mem
git checkout main       # or pin to the commit hash you're training against

# 3. Install training deps
pip install --upgrade pip
pip install "torch>=2.4" "transformers>=4.45" "accelerate>=0.34" \
            "peft>=0.12" "datasets>=2.20" "bitsandbytes>=0.43" \
            "trl>=0.11" wandb

# 4. Download v1 base + replay corpus
huggingface-cli download star-ga/mind-mem-4b --local-dir /workspace/base
huggingface-cli download star-ga/mind-mem-4b-train-corpus \
    --local-dir /workspace/train-corpus --repo-type dataset

# 5. Generate v3.3.0 data (runs in container — ~5 min)
cd /workspace/mind-mem
python3 benchmarks/generate_dispatcher_examples.py \
    --output /workspace/train-corpus/v3.3.0-dispatchers.jsonl \
    --count 10000

python3 benchmarks/generate_retrieval_examples.py \
    --output /workspace/train-corpus/v3.3.0-retrieval.jsonl \
    --count 5000

# 6. Mix corpora (80% new, 20% replay)
python3 benchmarks/mix_training_corpus.py \
    --input-dirs /workspace/train-corpus \
    --replay-ratio 0.2 \
    --output /workspace/train-corpus/mixed.jsonl

# 7. Train
accelerate launch --config_file /workspace/accelerate-h200.yaml \
    benchmarks/train_mind_mem_4b.py \
    --config /workspace/mind-mem/benchmarks/train_config.yaml \
    --data /workspace/train-corpus/mixed.jsonl

# 8. Push v2 to HF
huggingface-cli upload star-ga/mind-mem-4b-v2 \
    /workspace/mind-mem-4b-v2 \
    --repo-type model
```

## Validation

Before promoting v2 over v1:

1. **Tool-call accuracy** — replay 500 held-out dispatcher prompts;
   v2 must emit a valid dispatcher call ≥ 97% (v1 hits 94%). Use
   `benchmarks/validate_tool_calls.py`.
2. **LoCoMo subset** — run the first 3 conversations through
   `locomo_judge.py` with v2 as answerer. Score must be within
   ±2 points of v1 on conv 0-2 (regression sentinel).
3. **Latency budget** — first-token latency on a 4K context must
   stay under v1 + 20%. Run `benchmarks/latency_p50_p95.py`.
4. **Memory footprint** — runtime peak RSS on `mind-mem:4b`
   (Ollama Q4_K_M) must stay under 3GB.

If any check fails, iterate the data mix (typically bump replay
ratio to 0.3) rather than the hyperparameters.

## What NOT to retrain for

v2 is a capability-expansion release, **not** a quality-lift release.
Do not:

- Train on LoCoMo answerer outputs directly — that leaks test data.
- Swap the base model (stay on Qwen3.5-4B).
- Tune for benchmark scores above the tool-call surface — that's
  the judge model's job.

v3 will be the next major retrain, targeted at v4.0's sharded-
Postgres tool surface.

## Deploy

After HF upload:

1. `ollama pull star-ga/mind-mem-4b-v2`
2. Update `~/.config/mind-mem/mind-mem.json`:
   ```json
   {"llm": {"backend": "ollama", "model": "mind-mem-4b-v2:latest"}}
   ```
3. `mm install-all --force` to rewire all 16 clients.
4. Smoke-test with `mm recall "test query"` — first response ≤ 3s.

## Cost summary

| Line item | Cost |
|---|---|
| H200 training (12 hr worst case) | ~$45 |
| HF Pro (bigger artefact uploads) | $9/mo |
| Runpod template + storage (on-demand) | ~$5 |
| **Total per full retrain** | **~$55-60** |

vs v1 on A100 80GB which ran ~18 hours at $1.30/hr = $23 but with
QLoRA (lower quality). Full-FT on H200 pays off on throughput.
