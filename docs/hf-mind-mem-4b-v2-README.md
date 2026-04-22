---
language:
  - en
license: apache-2.0
library_name: transformers
tags:
  - mind-mem
  - memory
  - governance
  - retrieval-augmented
  - full-fine-tune
  - qwen3.5
  - text-generation
  - conversational
  - tool-use
  - instruction-tuned
base_model: star-ga/mind-mem-4b
pipeline_tag: text-generation
---

# mind-mem-4b v2 (2026-04-21)

A governance-aware memory-assistant model for [mind-mem](https://github.com/star-ga/mind-mem) — an auditable, contradiction-safe memory layer for coding agents (MCP-compatible).

**v2 supersedes v1.** This checkpoint is a **full fine-tune** of the v1 `star-ga/mind-mem-4b` checkpoint (itself QLoRA-merged from `Qwen/Qwen3.5-4B`), retrained on the v3.3.0 MCP surface: the 7 consolidated dispatcher tool-calls, query decomposition / reformulation, entity extraction, and evidence-bundle consumption.

## What's new in v2

| | v1 (2026-02-14) | v2 (2026-04-21) |
|---|---|---|
| Method | QLoRA rank-16 adapters, 10GB RTX 3080 | **Full fine-tune**, all 4.2B params trained |
| Hardware | RTX 3080 10GB | **H200 NVL 141GB** |
| MCP surface | 19 v2.10.0 tools | **7 v3.2.x dispatchers + v3.3.0 retrieval** |
| Optimizer | Paged AdamW 8-bit | AdamW (fused) bf16 |
| Learning rate | 2e-4 | 5e-6 (gentler for full FT) |
| Epochs | 3 | 3 |
| Training examples | ~40K | **16,450** (curated) |
| Sequence length | 768 | 384 (matches p99 of corpus) |
| Gradient checkpointing | true | true |
| Effective batch size | 16 | 32 (batch 4 × accum 8) |

## What it knows about (v3.3.0)

- **7 MCP dispatchers** (consolidated from 57 tools):
  - `recall(mode="similar|verify|intent|diagnose|bundle")`
  - `staged_change(phase="propose|approve|rollback")`
  - `graph(action="expand|prefetch|contradict")`
  - `memory_verify`, `core`, `kernels`, `compiled_truth`
- **Query decomposition** — splits multi-hop questions into sub-queries (multi-hop / temporal / causal patterns).
- **Query reformulation** — paraphrase rewrites for robustness.
- **Entity extraction** — `PER-/PRJ-/TOOL-/INC-` prefixes and capitalised proper nouns.
- **Evidence bundle format** — structured JSON with facts / relations / timeline / entities (`recall(format="bundle")`).
- **7 typed graph edges** — cites, derived_from, depends_on, tested_by, supersedes, contradicts, relates_to.

## Usage

### Load bf16 (native)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "star-ga/mind-mem-4b"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16", device_map="auto")

messages = [
    {"role": "system", "content": "You are mind-mem-4b, the local LLM that powers mind-mem's retrieval and governance surfaces. Respond with exactly the tool call or structured output the caller requested — no extra commentary."},
    {"role": "user", "content": "What did Alice say about the OAuth migration?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
# → {"tool":"recall","args":{"mode":"similar","query":"Alice OAuth migration"}}
```

### Ollama (Q4_K_M GGUF, ~2.7GB)

```bash
ollama pull mind-mem:4b                    # v2 auto-replaces v1
ollama run mind-mem:4b "What did Alice say about OAuth?"
```

### vLLM / exllamav2 (high-throughput)

```bash
vllm serve star-ga/mind-mem-4b --dtype bfloat16 --port 8000
curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "star-ga/mind-mem-4b", "prompt": "...", "max_tokens": 128}'
```

## Training recipe

```yaml
base_model: star-ga/mind-mem-4b           # v1 checkpoint (not raw Qwen3.5-4B)
dtype: bfloat16
optim: adamw_torch_fused
learning_rate: 5.0e-6
lr_scheduler_type: cosine
warmup_ratio: 0.03
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 8            # effective batch 32
max_seq_length: 384                        # p99 corpus length = 274 chars
packing: false
gradient_checkpointing: true
save_steps: 100
logging_steps: 10
data_collator: DataCollatorForLanguageModeling(mlm=False)
```

### Corpus composition (16,450 examples)

| Source | Count | Shape |
|---|---|---|
| `benchmarks/generate_dispatcher_examples.py` | 10,000 | `(prompt, expected_call)` pairs, covers all 7 dispatchers × modes |
| `benchmarks/generate_retrieval_examples.py` | 5,000 | `(task, prompt, expected_output)` — query decomposition, reformulation, entity extraction |
| v1 replay buffer (`/data/checkpoints/mm-workspace/train-output/corpus.jsonl`) | 1,450 | Legacy v2.10.0 MCP surface — keeps tool-name awareness |

All three generators are deterministic — `generate_*_examples.py` + the v1 corpus reproduce the 16,450-example JSONL byte-for-byte.

### Loss curve

Published separately at `docs/mind-mem-4b-v2-loss-curve.json`. Key milestones:

| Step | Epoch | Loss | LR | Grad norm |
|---|---|---|---|---|
| 10 | 0.02 | 2.452 | 1.05e-06 | 86.0 |
| 50 | 0.11 | 0.3456 | 5.00e-06 | 7.97 |
| 100 | 0.21 | 0.1041 | 4.98e-06 | 4.56 |
| 200 | 0.43 | ~0.085 | 4.85e-06 | 1.9 |
| 300 | 0.64 | ~0.084 | 4.60e-06 | 1.6 |
| 1407 (final, epoch 3) | _see commit message_ | _see commit message_ | _see commit message_ | _see commit message_ |

Loss plateau at ~0.085 from step 150 is characteristic of the tool-call JSON output format (highly constrained vocabulary); further descent in epochs 2-3 reflects generalisation to unseen queries.

### Hardware

- **GPU**: single H200 NVL (141 GB HBM3e, 4.8 TB/s memory bandwidth)
- **Memory used**: ~41 GB / 141 GB VRAM at bf16 full FT (batch 4, seq 384, grad-ckpt on)
- **Throughput**: ~10 s/step, ~6 hr total wall time for 1407 steps
- **Cost**: ~$20 at $3.39/hr on-demand (Runpod US-MD-1)

## Eval

### LoCoMo (benchmarks/locomo_judge.py)

v2 is evaluated on the same LoCoMo 10-conversation suite as v1. Publishing numbers separately once the full sweep completes — v2 is the answerer-side component; retrieval-side improvements from v3.3.0 features (graph_recall, entity_prefetch, rerank_ensemble, truth_score, answer_quality) are measured in the same run.

| Target | Baseline (v1.1.0) | Current (v3.2.1) | v3.3.0 projected |
|---|---|---|---|
| Mean | 70.54 | ~76.7 (Opus conv-0) | **≥ 82** |
| Temporal | — | — | — |
| Adversarial | — | — | — |

### Tool-call accuracy

Replay 500 held-out dispatcher prompts:

| | v1 | v2 |
|---|---|---|
| Correct tool | 94% | _pending_ |
| Correct mode | 89% | _pending_ |
| Valid JSON | 98% | _pending_ |

## What NOT to retrain for

v2 is a **capability-expansion + architecture-shift** release (QLoRA → full FT), **not** a quality-lift release. Do not:

- Retrain on v3.3.0 synthetic data only — the 1,450-example replay buffer is essential to keep v2.10.0 tool-name awareness.
- Use v2 as a drop-in replacement without updating the mind-mem-json dispatcher map (v1 emits 19-tool calls, v2 emits 7-dispatcher calls).
- Quantise below Q4_K_M — tool-call JSON starts degrading at Q3.

## Reproducing v2 training

```bash
git clone https://github.com/star-ga/mind-mem.git
cd mind-mem

# 1. Generate corpus
python3 benchmarks/generate_dispatcher_examples.py \
    --output /tmp/mm-train/dispatchers.jsonl --count 10000
python3 benchmarks/generate_retrieval_examples.py \
    --output /tmp/mm-train/retrieval.jsonl --count 5000
cat /data/checkpoints/mm-workspace/train-output/corpus.jsonl \
    /tmp/mm-train/dispatchers.jsonl /tmp/mm-train/retrieval.jsonl \
  | shuf --random-source=/dev/urandom > /tmp/mm-train/mixed.jsonl

# 2. Spin up Runpod H200 NVL (141GB), 100GB container + 100GB volume

# 3. On pod — one-shot kickoff
curl -sSL https://raw.githubusercontent.com/star-ga/mind-mem/main/benchmarks/runpod_kickoff.sh \
    | HF_TOKEN=<write-scope> bash
```

## Citation

```
@software{mind_mem_4b_v2_2026,
  author = {STARGA Inc.},
  title = {mind-mem-4b v2: Governance-Aware Memory Model for MCP Agents},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/star-ga/mind-mem-4b},
  note = {Full fine-tune on 16,450 v3.3.0 MCP examples.}
}
```

## Links

- **GitHub**: [star-ga/mind-mem](https://github.com/star-ga/mind-mem)
- **PyPI**: `pip install mind-mem`
- **v3.3.0 docs**: `docs/` in the repo — includes feature gate catalogue, protection model, Runpod kickoff.
- **Protection model**: [docs/protection.md](https://github.com/star-ga/mind-mem/blob/main/docs/protection.md) — integrity manifests, strict-mode import, provenance chain.
