---
language:
  - en
license: apache-2.0
library_name: transformers
tags:
  - MIND-Mem
  - memory
  - governance
  - retrieval-augmented
  - full-fine-tune
  - qwen3.5
  - text-generation
  - conversational
  - tool-use
  - instruction-tuned
  - cognitive-kernel
  - knowledge-graph
  - v4
base_model: Qwen/Qwen3.5-4B
pipeline_tag: text-generation
---

# mind-mem-4b v4.0.0

A governance-aware memory-assistant model for
[MIND-Mem](https://github.com/star-ga/mind-mem) — an auditable,
contradiction-safe memory layer for coding agents (MCP-compatible).

v4.0.0 is a full fine-tune of Qwen3.5-4B trained on the v4.0.0 corpus.
It supersedes the v3.12.0-fullft weights (pinned at the `v3.12.0` HF
revision).

---

## What v4 knows

v4 knows all 84 MCP tools from v3.x, plus the following v4 surfaces:

**Cognition**
- `tier_memory` — block tier promotion, `StaleVersionError`, CAS
  semantics via `block_version`
- `cognitive_kernel` — `KernelKind` enum values and dispatch semantics
  (`SURPRISE_WEIGHTED`, `LINEAGE_FIRST`, `RECENT_FIRST`,
  `CONTRADICTS_FIRST`, `GRAPH_WALK`, `DEFAULT`), `mind_recall` call
  signature, `register_kernel` / `is_kernel_registered`
- `surprise_retrieval` — `compute_surprise` return range `[0.0, 1.0]`,
  `FallbackPolicy` variants (`NEUTRAL`/`PROMOTE`/`DEMOTE`/`RAISE`),
  `EmbeddingFailureError`

**Knowledge graph**
- `block_kinds` — `block_kind_tags` junction table, `add_kind_tag` /
  `get_kind_tags`, multi-label semantics
- `block_metadata` — `set_block_metadata` / `get_block_metadata` /
  `list_blocks_by_tag`, TTL via `ttl_seconds` key, schema validator
  registration, `SchemaValidationResult` fields
- `kind_summaries` — `refresh_summary(workspace, kind)` / `get_summary`
- `embedding_pipeline` — `register_embedder` / `embed`, backend
  parameter
- `consolidation_worker` — `plan_consolidation` is a pure function,
  `ConsolidationPlan.apply()`

**Resilience**
- `eviction` — four policies (`LRU`/`LOW_SURPRISE`/`AGE`/`COMPOSITE`),
  `set_active_policy` / `active_policy`, `EvictionPlan.debug_plan()`,
  `is_policy_registered`
- `federation` — `block_tier_vclock`, `tier_conflict_log`,
  `MergeStrategy` variants
- `self_editing` — `propose_edit` / `approve_edit` / `reject_edit`,
  `block_edits` table; no direct mutation path
- `pq` — `PQCodec.train` / `.encode` / `.decode` / `.save` / `.load`,
  `M=32` `K=256` defaults
- `hnsw_kind_index` — `build_kind_index` / `query_kind_index`,
  `sqlite-vec` detection with brute-force fallback
- `circuit_breaker` — `CircuitBreaker` constructor args
  (`failure_threshold`, `recovery_timeout`, `half_open_probes`),
  `CircuitState` values (`CLOSED`/`OPEN`/`HALF_OPEN`),
  `@circuit_breaker` decorator, `default_breaker` singleton
- `backpressure` — `BackpressureController`, `recommended_pause` vs
  `current_pause`, `controller` singleton
- `health` — `health_check(workspace)` return shape, 7 built-in probe
  names, `register_health_probe`, `BaseException`-safe contract,
  `disabled_count`

**Observability**
- `observability` — `counter` / `gauge` / `histogram`, `@timed`,
  `set_exporter`, `MAX_CARDINALITY=10000`, overflow sentinel
  `"__overflow__"`
- `logging_context` — `with_context` / `with_correlation_id`,
  async-safety, `StructuredLogFilter`

**Foundation**
- `feature_flags` — `is_enabled` / `require_enabled` / `flag_config`,
  `FeatureDisabledError`, 35-flag inventory, startup rejection of
  unknown flags

**Corrected from v3.12.1**
- `KIND_DECAY['cites']` is `0.8` — not `0.4`. The v3.12.1 model
  confabulated the `refines` value (0.4) when asked about `cites`.
  The v4 corpus applies the per-kind reinforcement block from
  `train/V4_RETRAIN_TODO.md` to fix this.
- `quality_gate` escape hatch is `force=True` on `validate_block` —
  not `quality_gate.mode = "off"` (which is not a legal value).

---

## Eval results

Harness: `train/eval_harness.py` — 109 probes (95 v3.x + 14
`V4_SURFACES`), un-softened. **Primary ship gate hit: 109/109 = 100%.**

| Category | Pass / Total | % |
|----------|--------------|---|
| tool_call | 20 / 20 | 100% |
| block_schema | 10 / 10 | 100% |
| workflow | 5 / 5 | 100% |
| v3.9 new tools | 13 / 13 | 100% |
| v3.9 transform-hash | 3 / 3 | 100% |
| v3.9 transport-guard | 4 / 4 | 100% |
| v3.11 new tools | 10 / 10 | 100% |
| v3.11 explain field | 10 / 10 | 100% |
| v3.12 quality-gate strict-mode | 10 / 10 | 100% |
| v3.12 lineage-staleness | 10 / 10 | 100% |
| v4 surfaces | 14 / 14 | 100% |
| **Total (un-softened eval_harness)** | **109 / 109** | **100%** |

Held-out paraphrase eval (`train/eval_holdout.py` — 22 probes that
do **not** appear verbatim in the training corpus):

| Group | Pass / Total | % |
|-------|--------------|---|
| v4 holdout paraphrases | 13 / 14 | 92.86% |
| v3.12 holdout paraphrases | 6 / 8 | 75.00% |
| **Total holdout** | **19 / 22** | **86.36%** |

The 3 held-out misses are documented in **Known limitations** below.

---

## Usage

### Transformers (bf16)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "star-ga/mind-mem-4b"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype="bfloat16", device_map="auto"
)

messages = [
    {
        "role": "system",
        "content": (
            "You are mind-mem-4b, the local LLM that powers mind-mem's "
            "retrieval and governance surfaces. Respond with exactly the "
            "tool call or structured output the caller requested — no "
            "extra commentary."
        ),
    },
    {"role": "user", "content": "What did Alice say about the OAuth migration?"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
out = model.generate(inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
# → {"tool":"recall","args":{"mode":"similar","query":"Alice OAuth migration"}}
```

### Ollama (Q4_K_M GGUF, ~2.7 GB)

```bash
ollama pull mind-mem:4b
ollama run mind-mem:4b "What is KIND_DECAY['cites']?"
# → 0.8
```

### vLLM

```bash
vllm serve star-ga/mind-mem-4b --dtype bfloat16 --port 8000
```

---

## Training recipe

```yaml
base_model: Qwen/Qwen3.5-4B
dtype: bfloat16
optim: paged_adamw_8bit
learning_rate: 2.0e-5
lr_scheduler_type: cosine
warmup_ratio: 0.03
num_train_epochs: 4
per_device_train_batch_size: 2
gradient_accumulation_steps: 16      # effective batch 32
packing: false                       # each example gets its own seq
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
max_length: 3072                     # accommodates longest changelog dumps
max_grad_norm: 1.0
save_strategy: "no"                  # 40 GB volume cannot hold intermediate ckpts
logging_steps: 5
seed: 42
```

### Corpus

<!-- CORPUS_STATS -->

The v4 corpus extends the v3.12.0-fullft corpus with:

- v4 surface probes for all 19 new modules
- Per-kind reinforcement block for all five `KIND_DECAY` values
  (≥10 probes each) — corrects the v3.12.1 `cites=0.4` confabulation
- Denial / negation probes for all five edge-kind decay values
- Corrected canonical escape-hatch probes (removes the `"off"` answer
  entirely; canonical is `force=True` on `validate_block`)
- 22 held-out paraphrase probes (not in training set)
- v4 retry-2c diversity block: ≥9 canonical-token answers per probe
  (audit-verified by `train/audit_canonical_coverage.py`), so the
  model gets ≥144 gradient passes per canonical token across 4 epochs

### Hardware

Trained on a single H200 SXM (NVIDIA, 141 GB HBM3e). Wall-clock
~30 min for 4 epochs at effective batch 32 on the 4793-example
corpus. Peak GPU memory ≈ 60 GB.

---

## Known limitations

- **`FallbackPolicy.RAISE` propagation** — `EmbeddingFailureError`
  propagates through `mind_recall`; callers using the cognitive kernel
  must handle it explicitly when `RAISE` is the active policy.
- **`federation.py` transport** — the VClock and conflict-log data model
  ships in v4.0.0; active sync transport across hosts is not included.
  `MergeStrategy.MANUAL` is the safe default for multi-host deployments
  until a transport layer is available.
- **`consolidation_worker.py` is advisory** — `plan_consolidation` is a
  pure function and never writes. Callers must call `.apply()` explicitly
  after reviewing the plan.
- **Held-out paraphrase eval — 3 misses (19/22 = 86.4%):**
  1. `register_schema_validator` paraphrase: when asked "How does a
     caller plug a per-kind validator into v4 block_metadata before
     `validate_block` is called?", the model may answer with a
     plausible-sounding but incorrect function name. Workaround: call
     the canonical `register_schema_validator(kind, fn)` from
     `src/mind_mem/block_metadata.py`.
  2. `validate_block` default mode: in an under-specified pre-flight
     question, the model returns the function name but may omit that
     `advisory` is the default `quality_gate.mode`. The default IS
     `advisory`; strict mode requires explicit opt-in via
     `mind-mem.json`.
  3. `propagate_lineage_staleness` file location: the model may report
     `src/mind_mem/block_lineage.py` when asked which file ships the
     function. The actual file is `src/mind_mem/lineage_staleness.py`;
     `block_lineage.py` is a sibling module that exports the typed-edge
     model. Both modules co-operate but the propagator lives in
     `lineage_staleness.py`.

---

## Version history

| Revision | Weights | Eval | Notes |
|----------|---------|------|-------|
| `v4.0.0` | v4.0.0 fullft | 109/109 un-softened | This card |
| `v3.12.0` | v3.12.0-fullft (v5) | 95/95 patched | `cites=0.4` known error |
| `v3.0.0` | v3.0.0 QLoRA | — | Legacy |
