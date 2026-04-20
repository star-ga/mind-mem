# Full-capability benchmark mode (v3.3.0)

For LoCoMo / NIAH / LongMemEval to measure the ceiling of what v3.3.0
can do, every optional retrieval feature has to be active. This doc
is the single checklist.

## 1. Start every local service

```bash
# Redis (L2 cache). Skip if `redis-cli ping` already returns PONG.
redis-server --daemonize yes

# Ollama + mind-mem:4b (local extractor LLM). Skip if already loaded.
ollama serve &>/dev/null &
ollama pull mind-mem:4b    # first run only

# claude-proxy (Opus OAuth routing for the LoCoMo judge answerer).
# Skip if the service is already running.
systemctl --user status claude-proxy || \
    nohup node /home/n/claude-proxy.js > /tmp/claude-proxy.log 2>&1 &
```

Verify:

```bash
redis-cli ping                                     # PONG
curl -s http://localhost:11434/api/tags | jq       # list of loaded models
curl -s http://127.0.0.1:8766/health               # {"status": "ok", ...}
```

## 2. Warm HuggingFace caches

```bash
# Cross-encoder (CE) — already cached on most dev boxes.
python3 -c "from sentence_transformers import CrossEncoder; \
           CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# BGE reranker (ensemble Tier 4 #9).
python3 -c "from sentence_transformers import CrossEncoder; \
           CrossEncoder('BAAI/bge-reranker-v2-m3')"

# Sentence embedder for sqlite_vec backend.
python3 -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('all-MiniLM-L6-v2')"
```

## 3. Optional: per-tier learned weights

If you've run LoCoMo once already and want to learn the best tier
multipliers:

```bash
python3 benchmarks/tier_weight_search.py \
    --input 'benchmarks/locomo_judge_results_*.json' \
    --output benchmarks/tier_weights_best.json
```

Paste the suggested `retrieval.tier_boost_weights` block into
[`benchmarks/locomo_v3.3.0_benchmark_config.json`](locomo_v3.3.0_benchmark_config.json)
before step 4. Otherwise the baseline 0.7 / 1.0 / 1.5 / 2.0 applies.

## 4. Run the benchmark

The LoCoMo harness copies a workspace config per conversation; the
benchmark-mode snippet lives at
[`benchmarks/locomo_v3.3.0_benchmark_config.json`](locomo_v3.3.0_benchmark_config.json).
Point the harness at it by editing `benchmarks/locomo_judge.py`'s
`_build_workspace` helper, or by running a wrapper that overrides the
per-conv config after the harness writes it.

```bash
# Option A — Mistral answerer + Mistral judge (v1.1.0 apples-to-apples baseline):
python3 benchmarks/locomo_judge.py \
    --answerer-model mistral-large-latest \
    --judge-model mistral-large-latest \
    --hybrid \
    --output /tmp/mm-bench/locomo_v3.3.0_mistral.json

# Option B — Opus 4.7 answerer via claude-proxy (OAuth, free) + Mistral judge
#  (same judge as v1.1.0 → comparable score ceiling).
python3 benchmarks/locomo_judge.py \
    --answerer-model claude-proxy/claude-opus-4-7 \
    --judge-model mistral-large-latest \
    --hybrid \
    --output /tmp/mm-bench/locomo_v3.3.0_opus.json
```

Expected runtime:

| Variant | Wall time | Cost |
|---|---|---|
| Mistral answerer + Mistral judge | ~2-3 h | ~$3-5 |
| Opus answerer (CLI) + Mistral judge | ~8-14 h | ~$1 (judge only) |
| Opus answerer (API) + Mistral judge | ~3-4 h | ~$220 |

## 5. Sanity-check the run

Before committing any result:

```bash
# Verify the config that actually ran:
jq '.config.retrieval' /tmp/mm-bench/locomo_v3.3.0_opus.json | head

# Check that v3.3.0 features fired:
grep -c "query_decomposition_auto_enabled\|graph_expand_applied\|entity_prefetch_merged\|cross_encoder_auto_enabled\|session_boost_applied" \
    /tmp/mm-bench/locomo_v3.3.0_opus.log

# Per-category breakdown:
jq '.metrics.by_category' /tmp/mm-bench/locomo_v3.3.0_opus.json
```

Fail conditions:

* Overall mean below v1.1.0 (70.54) → **don't publish**, re-check.
* Any feature's log-count is 0 across the whole run → that feature
  didn't fire; debug before publishing.
* `acc_50` below 73.77 → regression vs v1.1.0 baseline.

## 6. Publish

If the sanity check passes, commit:
* `benchmarks/locomo_v3.3.0_opus.json` (raw judge output)
* `benchmarks/LOCOMO.md` update with the new overall + per-category
  table
* README + PyPI badge bumps if the number moves the headline

Don't delete `/tmp/mm-bench/*.log` — the log stream is the only
evidence that every feature actually fired.
