# Mem-OS v1.0.1 Benchmark Report

**Tag:** `v1.0.1`
**Date:** 2026-02-17

## Environment

| Component | Value |
|---|---|
| Python | 3.12.3 |
| OS | Linux 6.17.0-14-generic (x86_64) |
| SQLite | system (FTS5 enabled) |
| Answerer model | gpt-4o-mini |
| Judge model | gpt-4o-mini |
| Temperature | 0.0 |
| Top-k retrieval | 10 |
| Dataset | LoCoMo (10 conversations, 1986 QA pairs) |
| Dataset cache | `benchmarks/.cache/locomo10.json` |

## Reproduction

```bash
git checkout v1.0.1
pip install -e .
python benchmarks/locomo_judge.py \
  --answerer gpt-4o-mini \
  --judge gpt-4o-mini \
  --top-k 10 \
  --output benchmarks/locomo_judge_results_v9_1.json
```

Requires `OPENAI_API_KEY` in environment.

## Overall Results

| Metric | v3 (baseline) | v1.0.1 | Delta |
|---|---|---|---|
| **Acc>=50** | 58.2% | **67.3%** | **+9.1pp** |
| Mean Score | 54.3 | **61.4** | +7.1 |
| Acc>=75 | 36.5% | **48.8%** | +12.3pp |

## Per-Category (Acc>=50)

| Category | N | v3 | v1.0.1 | Delta |
|---|---|---|---|---|
| open-domain | 841 | 75.7% | **86.6%** | +10.8pp |
| temporal | 96 | 70.8% | **78.1%** | +7.3pp |
| single-hop | 282 | 56.0% | **68.8%** | +12.8pp |
| multi-hop | 321 | 48.6% | **55.5%** | +6.9pp |
| adversarial | 446 | 30.7% | **36.3%** | +5.6pp |

## Per-Category (Mean Score)

| Category | N | v3 | v1.0.1 | Delta |
|---|---|---|---|---|
| open-domain | 841 | 68.3 | **78.3** | +10.1 |
| temporal | 96 | 61.5 | **65.7** | +4.3 |
| single-hop | 282 | 50.2 | **59.1** | +8.9 |
| multi-hop | 321 | 44.4 | **48.4** | +3.9 |
| adversarial | 446 | 36.3 | **39.5** | +3.3 |

## Per-Conversation Breakdown

| Conv | Sample | N | Mean | Acc>=50 |
|---|---|---|---|---|
| 0 | conv-40 | 199 | 64.2 | 74.9% |
| 1 | conv-41 | 105 | 61.8 | 67.6% |
| 2 | conv-42 | 193 | 64.4 | 71.0% |
| 3 | conv-26 | 260 | 58.2 | 60.8% |
| 4 | conv-43 | 242 | 60.9 | 66.5% |
| 5 | conv-44 | 158 | 63.8 | 70.9% |
| 6 | conv-47 | 190 | 55.0 | 54.7% |
| 7 | conv-48 | 239 | 63.1 | 71.1% |
| 8 | conv-49 | 196 | 62.6 | 69.9% |
| 9 | conv-50 | 204 | 61.7 | 68.1% |
| **ALL** | | **1986** | **61.4** | **67.3%** |

## Architecture

Pure deterministic retrieval pipeline — no vector DB, no embeddings.

1. **Ingestion:** Session-aware chunking with speaker labels and timestamps
2. **Index:** SQLite FTS5 (Porter stemmer) + in-memory BM25 scan fallback
3. **Query processing:**
   - Query type detection (open-domain, single-hop, multi-hop, temporal, adversarial)
   - Morphological normalization (irregular verbs, month names)
   - Controlled synonym expansion (full or morph-only based on query type)
4. **Retrieval:** Wide candidate pool (top-200) with deterministic rerank
   - Speaker-match boost
   - Time-proximity signal
   - Entity overlap scoring
   - Bigram coherence
   - Recency decay
5. **Context packing** (append-only post-retrieval):
   - Rule 1: Dialog adjacency (question-answer pairs)
   - Rule 2: Multi-entity diversity enforcement
   - Rule 3: Pronoun rescue (antecedent recovery)
6. **Evidence packing:** Adversarial-aware formatting with misclassification guard

## Result Files

| File | Description |
|---|---|
| `benchmarks/locomo_judge_results_v9_1.json` | Full 10-conv results (1986 QA pairs) |
| `benchmarks/locomo_judge_results_v9_1.json.conv{0-9}.jsonl` | Per-conversation detail |
| `benchmarks/locomo_judge_results_v3.json` | v3 baseline for comparison |

## Version Lineage

| Tag | Commit | Description |
|---|---|---|
| `v7_fts5_snapshot_minimal` | — | FTS5 backend + minimal snapshot |
| `v8_phaseD_wide_speaker_extractor` | — | Wide retrieval + speaker-aware rerank |
| `v9_phaseE_recall_hardening` | `527636c` | Month norm, irregular verbs, synonyms, context pack |
| `v9_1_adv_gate_expansion` | `111bfe7` | Adversarial synonym gating |
| `v1.0.1` | — | Full 10-conv validated release (this report) |
