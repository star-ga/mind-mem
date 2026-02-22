# mind-mem Roadmap

## v1.0.6 (current) — Hybrid Retrieval Pipeline

- [x] Date field passthrough in all retrieval paths
- [x] Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) in hybrid path
- [x] Module shadowing fix (filelock.py rename)
- [x] llama.cpp embedding provider (Qwen3-Embedding-8B, 4096d)
- [x] sqlite-vec local vector backend
- [x] Pinecone integrated inference
- [x] fastembed ONNX support
- [ ] Full 10-conversation LoCoMo benchmark with cross-encoder (in progress)

## v1.1.0 — Retrieval Quality Push

Target: **top-3 on LoCoMo** (currently 67.2%, need ~72%+)

- [x] **top_k 10 → 18** — 80% more context blocks from RRF fusion pool (e86b59e)
- [x] **Temporal extra_limit_factor 1.5 → 2.0** — Wider candidate retrieval for date-bearing queries
- [x] **Temporal-multi-hop cross-boost** — "When did X do Y?" gets multi-hop signal boost
- [x] **Cross-encoder A/B tested** — ms-marco-MiniLM-L-6-v2 tested, net -0.8 on conv-0, reverted
- [x] **Detection test suite** — 32 tests for _recall_detection.py (5c2a27a)
- [x] **Benchmark comparison tool** — compare_runs.py for side-by-side A/B analysis (154a04c)
- [ ] **Answerer prompt tuning** — Rules 2+3 force hallucination ("Always give your best answer", "INFER from partial evidence"). Replace with evidence-grounded instructions
- [ ] **Judge prompt calibration** — Remove "core facts = 70+" anchor that inflates scores
- [ ] **Abstention for multi-hop** — Currently only fires on adversarial; extend to low-confidence multi-hop
- [ ] **BM25F weight grid search** — Optimize field weights for title/excerpt/tags

## v1.2.0 — Trajectory Memory

**Goal:** Add case-based reasoning so agents learn from past task executions.

### Trajectory Block Type

New block type `[TRAJECTORY]` that stores full task execution traces:

```markdown
[TRAJECTORY]
Task: Deploy v1.0.5 to production
Date: 2026-02-19
Duration: 45min
Tools: git, pytest, docker, mindc
Outcome: SUCCESS
Reward: 1.0
Lessons:
  - Always run pytest before tagging
  - FORTRESS build needs --mindc flag
  - Never skip smoke tests on staging
Steps:
  1. git checkout main && git pull
  2. pytest tests/ -x
  3. ./build.sh --release
  4. docker build -t mind-mem:v1.0.5 .
  5. smoke_test.sh staging
  6. git tag v1.0.5 && git push --tags
```

### How It Works

1. **Capture**: Auto-extract trajectories from session transcripts (tools used, outcomes, duration)
2. **Store**: New `[TRAJECTORY]` blocks in the existing Markdown + SQLite system
3. **Recall**: "I'm about to deploy" → retrieve similar trajectories (successes AND failures)
4. **Replay**: Present relevant lessons before the agent starts a similar task
5. **Consolidate**: Periodic reflective pass merges overlapping trajectories into patterns

### Why This Matters

| Current Recall | + Trajectory Memory |
|---|---|
| "What did we discuss about X?" | "Last time we did X, step 3 failed because..." |
| Text block retrieval | Full task execution replay |
| No outcome awareness | Success/failure signals inform planning |
| Each session starts fresh | Agents learn from accumulated experience |

### Design Constraints

- **Local-first**: No cloud services, no fine-tuning, no external APIs in the retrieval loop
- **Lightweight**: Fits existing Markdown + SQLite + BM25 architecture
- **Governed**: Trajectories go through the same propose/approve/apply pipeline
- **Backward compatible**: Existing blocks and recall work unchanged

## v1.3.0 — Multi-Hop Query Decomposition

- [ ] Decompose complex queries into sub-queries (e.g., "When did X do Y?" → "Find conversations about Y" + "Extract dates from those conversations")
- [ ] Parallel sub-query execution with result merging
- [ ] Chain-of-retrieval with iterative refinement

## v2.0.0 — Reflective Consolidation

- [ ] Sleep-time memory consolidation (periodic background pass)
- [ ] Pattern extraction from trajectory clusters
- [ ] Automatic contradiction detection across trajectories
- [ ] Memory importance scoring with decay
