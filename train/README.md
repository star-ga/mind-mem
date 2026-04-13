# mind-mem-7b training pipeline

Scripts to retrain the [`star-ga/mind-mem-7b`](https://huggingface.co/star-ga/mind-mem-7b) governance-aware memory-assistant model on a fresh checkout of the mind-mem repo.

Training artifacts land in `/home/n/mm-train-output/`, which is gitignored — the repo only carries the scripts, not the weights.

## Pipeline

```
build_corpus.py  →  train_qlora.py  →  eval_harness.py  →  export_gguf.py  →  upload_to_hf.py
     │                   │                    │                    │                    │
 corpus.jsonl      adapter/*.safetensors   eval_report.json    *-Q4_K_M.gguf     HF commit
```

## Steps

### 1. Harvest corpus (deterministic)

```bash
cd /home/n/mind-mem
python3 train/build_corpus.py
```

Produces `/home/n/mm-train-output/corpus.jsonl`. Five sources: MCP tool docstrings, block schemas, CHANGELOG, docs/, governance workflows. No LLM calls; running twice yields byte-identical output.

### 2. Fine-tune

```bash
python3 train/train_qlora.py
```

QLoRA on Qwen2.5-7B-Instruct. Fits in 10 GB VRAM on a 3080. Wall time ≈ 2-4 hours on a 393-example corpus with 3 epochs. Override base model via `MM_BASE_MODEL`.

### 3. Evaluate

```bash
python3 train/eval_harness.py
```

Runs three benchmarks. Exits non-zero if any target fails (tool-call ≥ 95%, block-schema ≥ 98%, workflow ≥ 90%) so you can gate upload on green eval.

### 4. Export GGUF (optional — needed for llama.cpp / Ollama users)

```bash
python3 train/export_gguf.py
```

Merges the LoRA back into the base, converts to GGUF, then quantizes to Q4_K_M. Prerequisites: `llama.cpp` cloned + built at `/home/n/llama.cpp`.

### 5. Upload to HuggingFace

```bash
HF_TOKEN=hf_... python3 train/upload_to_hf.py
```

Pushes adapter + README + GGUF to `star-ga/mind-mem-7b`. **Requires a write-scope token** — the default read-only token cached in `~/.cache/huggingface/token` will be rejected by the upload script before bytes are sent.

## Constraints

- GPU: ≥ 10 GB VRAM (RTX 3080 target)
- Disk: ~20 GB free in `~/.cache/huggingface/hub` + `/home/n/mm-train-output`
- Python: 3.10+ with `transformers`, `peft`, `bitsandbytes`, `accelerate`, `trl`, `datasets`, `torch`
- HF token with **write** scope for upload (read-only token is enough for training base-model pulls)
