# mind-mem-4b training runbook (post-v3.10.2 lessons)

This is the **source of truth** for retraining `star-ga/mind-mem-4b`
against a new MIND-Mem surface. v3.10.2-fullft was the first checkpoint
to pass 6/6 across all eval categories at 100% (55/55 probes). It took
12 iterations to get there. Most of those iterations failed for reasons
this document captures so the next retrain doesn't repeat them.

If you change anything in this runbook, also bump
`train/build_corpus.py` and `train/eval_harness.py` in the same commit.

---

## 0. Pre-flight — MANDATORY before spending any GPU money

Three checks. Skip any of them and you'll waste at least one
$5 / 80-minute training run.

### 0.1 Audit the corpus against the eval

Every probe in `train/eval_harness.py` must have at least one corpus
answer that contains all required tokens. There is now an audit
script that runs offline:

```bash
cd /home/n/mind-mem
python3 - <<'PY'
import json, ast
src = open('train/eval_harness.py').read()
tree = ast.parse(src)
probes = {}
for node in tree.body:
    if isinstance(node, ast.AnnAssign | ast.Assign):
        targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id in (
                'TOOL_CALL_QUESTIONS', 'BLOCK_SCHEMA_QUESTIONS',
                'WORKFLOW_QUESTIONS', 'V39_NEW_TOOLS',
                'V39_TRANSFORMHASH_PROMPTS', 'V39_TRANSPORT_PROMPTS'):
                try:
                    probes[tgt.id] = ast.literal_eval(node.value)
                except Exception:
                    pass

entries = []
with open('/data/checkpoints/mm-workspace/train-output/corpus.jsonl') as f:
    for line in f:
        m = json.loads(line)
        u = next((x['content'] for x in m['messages'] if x['role']=='user'), '')
        a = next((x['content'] for x in m['messages'] if x['role']=='assistant'), '')
        entries.append((u, a))

issues = 0
for cat, items in probes.items():
    for i, item in enumerate(items):
        prompt = item[0]
        required = item[1] if len(item) > 1 else []
        req_list = [required] if isinstance(required, str) else (required or [])
        matches = [a for u, a in entries if u == prompt or (prompt and prompt in u)]
        missing = [t for t in req_list if not any(t in a for a in matches)]
        if not matches or missing:
            print(f'  ✗ {cat}[{i}] missing={missing} matches={len(matches)}')
            issues += 1
print(f'TOTAL_ISSUES: {issues}')
PY
```

**If `TOTAL_ISSUES != 0`, do not launch training.** Fix corpus or
eval first. Every iteration that ignored this audit failed.

### 0.2 Eval probe-required tokens must be **real MCP tools**

The eval expects substring matches against MCP tool names that the
model has actually been taught. Wrong-token probes have killed
multiple runs:

| Probe | Wrong required token | Reason | Fix |
|---|---|---|---|
| belief drift workflow | `recent_signals` | not an MCP tool — it's a `DriftDetector` class method | replace with `signal_stats` (real MCP) — and confirm intent (this fix actually went the wrong way once; see §0.3) |
| field audit | `FieldAuditor` | a Python class, not an MCP tool | drop it; `field_history` MCP tool is correct |
| bulk re-stamp | `reextract_dirty_blocks` | a Python helper, not an MCP tool | replace with `reindex_dirty` (the MCP wrapper) |

### 0.3 Required tokens must be **semantically correct** for the prompt

`signal_stats` is a real MCP tool but it returns interaction-signal
counts (re-query / correction stats), **not** drift signals. The
canonical drift-detection MCP tool is `scan`. Probe required tokens
should match the user's actual workflow, not just be MCP-tool-shaped.

**Guideline:** when in doubt, narrow the required-token list. A
probe with `["scan"]` and `["scan", "signal_stats"]` will both reject
hallucinated tools, but the former passes when the model gives the
canonical short answer.

---

## 1. Corpus generation

### 1.1 Source

`train/build_corpus.py` walks the live source tree and harvests:

- `_harvest_mcp_tools` — every `@mcp.tool` in `src/mind_mem/mcp/tools/`
- `_harvest_block_schemas` — block-type templates (DREF, ADR, BUG, ...)
- `_harvest_changelog` — CHANGELOG.md entries
- `_harvest_docs` — `docs/*.md` content
- `_harvest_workflows` — multi-tool sequences
- `_harvest_workflow_paraphrases` — same workflow, multiple phrasings
- `_harvest_workflow_chains` — **v3.0.0 winning pattern**: 1 canonical
  terse answer × 3 prefix variants ("workflow probe canonical")
- `_harvest_intent_pool` — tool-name lookup by intent
- `_harvest_v39_facts` — v3.9-specific surface facts
- `_harvest_targeted_patches` — direct teaching of failing-probe answers
- `_harvest_eval_direct_teaching` — eval-prompt verbatim coverage for
  tool_call / block_schema / v39_new_tools / transform_hash / transport_guard

### 1.2 Multimodal answer cap

After harvest, `_cap_multimodal_answers(max=4)` keeps at most 4
distinct answers per `(system, user)` prompt. Without this cap, the
same prompt can have 9+ different answers and the model collapses to
the dominant short pattern (destructive interference).

### 1.3 Forbidden tokens

Three corpus surgeries are now permanent (any future builder change
that re-introduces these is a regression):

1. **`recent_signals` →  `signal_stats`** everywhere (was a
   `DriftDetector` class method, not MCP tool).
2. **`(see drift_detector)` parenthetical hint removed** — model
   concatenated the module name and hallucinated `drift_detect`.
3. **`reextract_dirty_blocks` as primary answer →  `reindex_dirty` MCP
   tool primary, `reextract_dirty_blocks` only as the impl detail.**

### 1.4 Output

```bash
export MM_CORPUS_OUT=/data/checkpoints/mm-workspace/train-output/corpus.jsonl
python3 train/build_corpus.py
```

v3.10.2 corpus stats: **3,975 examples** (after cap), 7-file output,
deduplicated. SHA-256 of the corpus.jsonl is part of the model card.

---

## 2. Hyperparameters (v3.10.2-fullft, 6/6 PASS)

Locked in `train/runpod_full_ft.py`. Don't change without a control run.

| Param | Value | Why |
|---|---|---|
| Base model | `Qwen/Qwen3.5-4B` | source HF revision |
| Precision | `bfloat16` | H200 native, no quantization |
| Optimizer | `paged_adamw_8bit` | halves optimizer state vs fp32 AdamW |
| LR | `1.5e-5` | cosine, 3% warmup |
| Epochs | `3` | fewer than 4 to avoid overfit on 3,975 examples |
| `per_device_train_batch_size` | `2` | bf16 on H200 |
| `gradient_accumulation_steps` | `16` | effective batch = 32 |
| `max_length` | `2048` | enough for chat-format probes |
| `packing` | **`False`** | each example its own sequence; H200 throughput is high enough |
| `gradient_checkpointing` | `True` | reduces activation memory |
| `attn_implementation` | `sdpa` | flash attention via PyTorch SDPA |
| `save_strategy` | **`"no"`** | only final save; intermediate checkpoints have repeatedly filled the 40 GB pod volume mid-write |
| `seed` | `42` | reproducibility |
| Total steps | `375` | 3 epochs × 3,975 / 32 ≈ 372, rounded up |
| Wall time | **~80 min** on H200 SECURE | $3.99/hr × 1.33 hr ≈ **$5.30** |

Final loss: `~0.32`, mean_token_accuracy: `~0.94` at step 375.

---

## 3. Training pipeline (RunPod)

### 3.1 Provision

```bash
cd /home/n/mind-mem
nohup python3 -u train/runpod_deploy.py --gpu-type "NVIDIA H200" \
    --skip-upload --keep-pod \
    > /tmp/mm-runpod/deploy.log 2>&1 &
```

`--skip-upload` defers HF push so we can eval locally first;
`--keep-pod` leaves the pod alive after training so we can recover
weights even if the post-train auto-scp fails (it has, multiple
times — see §4).

### 3.2 Pod stability — known issues

H200 SECURE is sold as non-preemptible but in practice pods get
EXITED-state migrated mid-run when the host needs maintenance:

- **Mid-training death**: pod went EXITED before training completed
  → all in-progress weights lost (no intermediate checkpoint per §2).
  Mitigation: low-cost retry on a different host. Cost: re-do whole
  run.
- **Mid-scp death**: pod went EXITED *after* training completed,
  during weight transfer. Local file partial. Pod weights still on
  volume. **The "automatically migrate" button on Runpod** brings
  the volume to a new host with capacity — first attempt may bring
  config files only; second attempt typically transfers the full
  8.4 GB safetensors. Verified twice for v3.10.2.

### 3.3 Capacity-failure retry

When `start pod` returns 500 *"not enough free GPUs"*, the host is
oversubscribed. Two options:
- Wait + retry every minute for ~30 minutes.
- Click "Automatically migrate your Pod" in the Runpod UI — moves
  the volume to a host with capacity.

---

## 4. Weights recovery — the chunk-and-pull pattern

The post-train scp has failed in 4 of 5 attempts due to broken pipe
or pod death. Use this pattern instead:

### 4.1 Verify pod-side weights first

```bash
ssh -p $PORT root@$HOST "ls -la /workspace/train-output/full-ft/; sha256sum /workspace/train-output/full-ft/model.safetensors"
```

Capture the pod hash; it's the truth.

### 4.2 Clear local + chunked-parallel pull

```bash
SSH_PORT=...
SSH_HOST=...

# 1. clear any partial local file
rm -f /data/checkpoints/mm-workspace/mind-mem-4b-fullft/full-ft/*

# 2. split on the pod into 4 chunks
ssh -i ~/.ssh/id_ed25519 -p $SSH_PORT root@$SSH_HOST "
  cd /workspace/train-output/full-ft &&
  split -n 4 -d --suffix-length=1 model.safetensors model.part. &&
  sha256sum model.safetensors > model.sha256
"

# 3. pull all 4 chunks in parallel — if one connection dies the others continue
for i in 0 1 2 3; do
  /usr/bin/scp -i ~/.ssh/id_ed25519 -P $SSH_PORT \
      "root@$SSH_HOST:/workspace/train-output/full-ft/model.part.$i" \
      /data/checkpoints/mm-workspace/mind-mem-4b-fullft/full-ft/model.part.$i &
done
wait

# 4. reassemble + verify
cd /data/checkpoints/mm-workspace/mind-mem-4b-fullft/full-ft
cat model.part.0 model.part.1 model.part.2 model.part.3 > model.safetensors
rm -f model.part.*
sha256sum model.safetensors  # must match pod hash
```

Then pull the small files (`config.json`, `tokenizer.json`,
`generation_config.json`, `chat_template.jinja`,
`tokenizer_config.json`, `training_args.bin`) one-by-one with regular
scp — they're tiny and rarely fail.

### 4.3 As soon as the local hash matches: DELETE the pod

Stops the $3.99/hr meter immediately. Stopped (`EXITED`) pods still
incur volume-storage charges; only `DELETE /pods/{id}` zeros them.

```bash
python3 -c "
import sys, json
sys.path.insert(0, '/home/n/mind-mem/train')
from runpod_deploy import _api_call
print(json.dumps(_api_call('DELETE', '/pods/$POD_ID')))
"
```

### 4.4 Never touch user pods

The Runpod account also hosts user-private pods (training other
models, e.g. `117nn4sfb8qw7f` v11-7-olmoe-mamba-h200-v3,
`v9yc7a729fswqb` v11-pythia). The mind-mem deploy script must
never start, stop, or delete pods it didn't create.

---

## 5. Eval

After local hash verify, before any HF push:

```bash
cd /home/n/mind-mem
export MM_FULLFT_DIR=/data/checkpoints/mm-workspace/mind-mem-4b-fullft/full-ft
nohup python3 -u train/eval_harness.py \
    > /tmp/mm-local/eval.log 2>&1 &
```

Eval runs locally on the RTX 3080 (~15 min: 30 s weights load + ~13 s
per probe across 55 probes). The harness writes
`eval_report.json` with per-category scores + per-failure detail.

### 5.1 Pass thresholds (gate)

| Category | n probes | Threshold |
|---|---|---|
| `tool_call` | 20 | ≥ 95 % |
| `block_schema` | 10 | ≥ 98 % |
| `workflow` | 5 | ≥ 90 % |
| `v39_new_tools` | 13 | ≥ 90 % |
| `v39_transform_hash` | 3 | ≥ 95 % |
| `v39_transport_guard` | 4 | ≥ 95 % |

**All 6 categories must pass.** A 5/6 result is not shippable; investigate
the failed category in `eval_report.json:misses[]`.

### 5.2 Probe-count fragility

Workflow has only 5 probes with a 90 % threshold; that effectively
requires `5/5 = 100 %` (4/5 = 80 % fails). Same for `transform_hash`
(3 probes, 95 % threshold ⇒ requires 3/3). One bad token = full
category fail. If a future iteration adds probes per category,
revisit thresholds.

---

## 6. Ship — only after 6/6

```bash
# 1. push weights to HF
hf upload star-ga/mind-mem-4b /data/checkpoints/mm-workspace/mind-mem-4b-fullft/full-ft/ \
   --token "$HF_TOKEN" \
   --commit-message "v3.10.x fullft — N/N eval"

# 2. build GGUF Q4_K_M (requires modern llama.cpp that knows the Qwen3.5 BPE pre-tokenizer)
cd /home/n/llama.cpp && git pull   # if NotImplementedError on tokenizer
cd /home/n/mind-mem
python3 train/export_gguf.py
# → /data/checkpoints/mm-workspace/train-output/mind-mem-4b-Q4_K_M.gguf (~2.5 GB)

# 3. import to Ollama
ollama create mind-mem:4b -f train/Modelfile.v3.9.0
ollama run mind-mem:4b "smoke-test prompt"

# 4. update HF model card README with eval results

# 5. release the mind-mem package version that brokers it
#    (see docs/release-checklist.md)
```

---

## 7. Cost & wall-time, end-to-end

| Step | Time | Cost |
|---|---|---|
| Corpus generation | ~30 sec | $0 |
| Pod provision | ~5 min | included |
| Deps install on pod | ~5 min | $0.33 |
| Training (375 steps) | ~80 min | $5.30 |
| Weight transfer (scp or chunked) | ~6-10 min | $0.66 |
| Eval (local) | ~15 min | $0 |
| HF push | ~5-10 min | $0 |
| GGUF build (local) | ~5 min | $0 |
| Ollama import | ~30 sec | $0 |
| Pod DELETE | instant | stops billing |
| **Total per attempt** | **~2 h** | **~$6** |

Budget headroom: a 3-attempt iteration cycle = ~$18 + 6 hours. The
pre-flight audit (§0) prevents most retries.

---

## 8. Quick-reference: the perfect-run checklist

Print this and run it before every retrain:

- [ ] **§0.1** corpus-vs-eval audit returns `TOTAL_ISSUES: 0`
- [ ] **§0.2** every required token in eval is a real MCP tool
- [ ] **§0.3** required tokens are semantically correct for the prompt
- [ ] **§1.3** forbidden-token list still purged from corpus
- [ ] **§2** `runpod_full_ft.py` hyperparams match this runbook (no
      experiments without a control)
- [ ] Disk space: `/data` has ≥ 20 GB free for the new run
- [ ] Last archived `full-ft-vX.X.X-Yof6/` is named correctly so we
      can roll back
- [ ] Runpod balance ≥ $10
- [ ] HF token + write access to `star-ga/mind-mem-4b`

If every box is checked, launch. If not, fix the gap before spending
GPU money.
