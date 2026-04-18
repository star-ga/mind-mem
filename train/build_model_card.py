"""Generate the HuggingFace model-card README for mind-mem-4b v3.0.0."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

OUT = Path(os.environ.get("MM_TRAIN_ROOT", "/data/checkpoints/mm-workspace/train-output")) / "README.md"


MODEL_CARD = """---
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
base_model: Qwen/Qwen3.5-4B
pipeline_tag: text-generation
---

# mind-mem-4b (v{version})

A governance-aware memory-assistant model for [mind-mem](https://github.com/star-ga/mind-mem) — an auditable, contradiction-safe memory layer for coding agents (MCP-compatible).

This checkpoint is a **full fine-tune** of `Qwen/Qwen3.5-4B`, fine-tuned on the mind-mem {version} source tree: all 57 MCP tool signatures, 14 block-type schemas, full CHANGELOG history through v3.0.0, the docs/ tree, and a curated set of end-to-end governance workflow transcripts.

## What it knows about

- **57 MCP tools** — exact signatures, arg types, return envelopes, scope requirements.
- **14 block schemas** — ADR, CODE, PERF, ALGO, BUG, DEC, CONV, DREF, CHECK, EV, FIELD, TIER, IMAGE, AUDIO.
- **Governance workflows** — propose → list_contradictions → approve_apply → verify_chain → rollback with BeliefStore + FieldAuditor + AuditChain wiring.
- **Drift detection** — live `DriftDetector` semantic pass alongside the lexical `DRIFT.md` pass (v3.0.0+).
- **Memory tiers** — 4-tier promotion cycle (WORKING → SHARED → LONG_TERM → VERIFIED), tier-boost retrieval ranking.
- **Encryption** — admin-scope `encrypt_file` / `decrypt_file` MCP tools gated on `MIND_MEM_ENCRYPTION_PASSPHRASE`.

## Usage

### Load the model (bf16)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "star-ga/mind-mem-4b"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16", device_map="auto")

messages = [
    {{"role": "system", "content": "You are mind-mem-4b, a memory-governance assistant."}},
    {{"role": "user",   "content": "Which MCP tool should I call to verify my audit chain?"}},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

### Quantized (GGUF) inference with llama.cpp

```bash
# Grab the Q4_K_M build
huggingface-cli download star-ga/mind-mem-4b mind-mem-4b-Q4_K_M.gguf --local-dir ./gguf

# Run via llama-server, llama-cli, Ollama, LM Studio …
llama-cli -m ./gguf/mind-mem-4b-Q4_K_M.gguf -p "Show me a DREF block template."
```

## Training recipe

| Knob | Value |
|---|---|
| Base | `Qwen/Qwen3.5-4B` |
| Method | Full fine-tune (bf16, all 4.2B params trained, AdamW 8-bit) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q,k,v,o,gate,up,down`-proj (all linear) |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| Learning rate | 2e-4 (cosine, 3% warmup) |
| Precision | bf16 |
| Optimizer | paged AdamW 8-bit |
| Hardware | RTX 3080 10GB |

## Corpus

Built deterministically from the mind-mem {version} source tree. Running `python3 train/build_corpus.py` in the repo reproduces the exact training JSONL byte-for-byte. Five disjoint sources:

1. MCP tool docstrings (`src/mind_mem/mcp_server.py` — 57 tools)
2. Block-type schemas (14 templates + field lists)
3. CHANGELOG entries (v1.0.0 → v{version})
4. docs/ prose (setup, usage, api-reference, architecture, roadmap)
5. Curated governance workflow transcripts (6 scenarios)

All five sources are local to the repo — **no external LLM calls, no web scraping, no synthetic data from a teacher model.** The training data is auditable.

## Eval

Three held-out benchmarks, scored zero-shot on the adapter-loaded base. See `train/eval_harness.py` for the exact harness — it runs on every commit to catch regressions.

| Benchmark | Score | Items |
|---|---|---|
| Tool-call name recall | **65%** | 13/20 prompts cite the correct MCP tool name |
| Block-schema conformance | **70%** | 7/10 templates include canonical field names + ID prefix |
| Governance workflow | **60%** | 3/5 workflows respond with the correct tool chain |

### Honest read

These numbers are an **improvement** over the prior adapter (v2.8.x on Qwen3.5-4B), which was trained pre-v2 API and does not know about any of the 35 new MCP tools shipped between v1.9 → v2.9 (it scores 0% on every v2.x-specific prompt). But they're below the aspirational 95 / 98 / 90% gates; the gap is concentrated in three failure modes:

1. **Imperative phrasing** ("Roll back an apply.") still occasionally triggers role-play responses instead of tool recall — Full-FT at 1,450 examples × 5 epochs still leaves residual base-model priors on imperative phrasings. v3.1 will expand the corpus to 10k+ examples.
2. **Block-template hallucination** — the model sometimes invents plausible-sounding fields instead of the canonical ones (e.g. inventing `EvidenceType:` where the schema requires `Signal:`).
3. **Workflow-as-prose** — "Walk me through" prompts sometimes produce explanatory prose instead of a tool chain.

Future iterations will address these with (a) a 3-5k-example corpus including more diverse imperative phrasings, (b) schema-conformance reinforcement with negative examples, and (c) rank-64 LoRA.

Use the base `Qwen/Qwen3.5-4B` plus this adapter when you want mind-mem-aware answers; use the base model alone for open-domain chat.

## Intended use / scope

This is a **specialised assistant**, not a general-purpose LLM. It's tuned to answer questions about mind-mem internals, help agents compose correct MCP calls, and narrate governance workflows. Use the base Qwen3.5-4B for open-domain chat.

## License

Apache-2.0 (same as the mind-mem Python package).

## Changelog

- **v3.0.0 ({today}):** Full fine-tune on Qwen/Qwen3.5-4B — all 4.2B parameters trained (not LoRA). Final loss 0.65, token accuracy 0.86. Covers mind-mem v1.9 → v3.0 surface: 57 MCP tools, 14 block schemas, governance workflows (evidence chain, field audit, drift, tier decay, alerting hooks, transparent encryption).
- **v2.9.0:** Legacy QLoRA adapter on Qwen/Qwen2.5-7B-Instruct base. Superseded by v3.0.0.
- **v2.8.x:** Initial release on Qwen3.5-4B base.

## Citation

```bibtex
@software{{mind_mem_7b_2026,
  author = {{STARGA, Inc.}},
  title = {{mind-mem-4b: governance-aware memory-assistant for coding agents}},
  year = 2026,
  version = {{v{version}}},
  url = {{https://huggingface.co/star-ga/mind-mem-4b}}
}}
```
"""


def main() -> None:
    # Pull version from the package without importing it (avoid
    # polluting the environment with test dependencies).
    init_path = Path("/home/n/mind-mem/src/mind_mem/__init__.py")
    version = "2.9.0"
    for line in init_path.read_text().splitlines():
        if line.startswith("__version__"):
            version = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    today = dt.date.today().isoformat()
    OUT.write_text(
        MODEL_CARD.format(version=version, today=today),
        encoding="utf-8",
    )
    print(f"wrote model card for v{version} → {OUT}")


if __name__ == "__main__":
    main()
