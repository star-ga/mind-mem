# mind-mem — Post-v4.4.0 Roadmap Plan (reference)

> **Status: REFERENCE ONLY.** Authored 2026-07-31 as the sequenced plan for the
> capability work above v4.4.0. Execution is **deferred until after the MIND
> compiler reaches Rust-independence**, at which point mind-mem's core is ported
> to pure MIND (mic@3 + MAP evidence chain + federation) and *these* phases land
> on that substrate. Do not treat any checkbox here as scheduled work; do not
> cite any held/unpublished benchmark number from this doc. Companion brief:
> `docs/AGENTIC-MEMORY-SOTA.md` (P1–P7 + refused-trap list).

## Concerns to clear before any implementation session

1. **Stale ROADMAP checkboxes — verify against the tree first.** Several "open"
   items already ship: `token rotation` + `N-08`/`T-007` (claimed shipped
   v4.0.15 per `CLAUDE.md`); `granularity_align.py` + `block_maturity.py` exist
   in `src/mind_mem/` despite Group H listing them open; H.1 "fuse
   `KnowledgeGraph.neighbors()` into recall" is substantially `kg_fusion.py`
   (shipped K.0, flag-gated OFF). **Phase 0 of every session = confirm the
   checkbox against the live module before writing code.** Do not re-implement
   shipped modules.
2. **Real distinct open set ≈ 72 items** (ROADMAP.md has ~117 `- [ ]` lines but
   ~45 are historical-section duplicates of the canonical top list, lines 17–543).
3. **Measurement debt (honesty gate, not dev gate):** full-corpus LoCoMo run +
   the LongMemEval provenance hold (benchmarks/STATUS.md). No held number may be
   cited until resolved.

**Moat every item must preserve:** cross-substrate determinism + governed write
(`propose_update → approve_apply`, HITL) + per-run recall attestation.
**Refused categories (never build, however phrased):** online RL /
outcome-weighted reranking; autonomous self-edit of source of truth (incl. any
"auto-approve low-risk" fast path); LLM-judged decay/importance in the retrieval
core.

## Standing gates (release criteria on every phase)

- **G1 Determinism** — byte-identical replay test per new recall-path feature;
  any ranking-affecting feature ships flag-gated default-OFF with an A/B
  determinism proof before default-ON.
- **G2 HITL** — test proving the new surface can only write source of truth via
  `propose_update`/`approve_apply` (the v4.4.0 `propose_edge`/`approve_edge`
  split pattern: proposer touches staging only; approver is sole committer).
- **G3 Preimage** — no new field enters the sealed Q16.16 audit-hash preimage;
  new scores/tiers/credits are sidecar; test asserts preimage stability across
  the toggle.
- **G4 Attestation** — any new retrieval leg registers in
  `recall_attestation.py` leg reporting; test asserts it.
- **G5 LLM-free CI** — every model call is an injected `Callable`
  (`compressors.py` discipline); full suite green with zero API calls.
- **G6 Bookkeeping** — ROADMAP checkbox flip + CHANGELOG + ACL scope +
  `mm doctor` awareness per new tool; honest MCP tool-count in README.

## Phases (bucket-A, pure-Python, sequenced)

- **Phase 1 — v4.5.0 "Trust Tiers + The Graph Goes Live"** (M, no deps) —
  `trust_tier.py` (VERIFIED/ATTESTED/REPORTED/INFERRED, pure function of
  provenance, sidecar); populate the typed KG (backfill over the corpus, HITL
  approve, flip `kg_fusion` ON after A/B); schema-versioning; predicate vocab v2;
  per-hit feedback credit; `recall(as_of=…)` plumb-through; stale-checkbox
  reconciliation.
- **Phase 2 — v4.6.0 "Credit Ledger + Honest Measurement"** (M, dep 1) —
  `credit_ledger.py` (append-only, attestation×outcome join; per-actor
  reputation falls out); `record_outcome`/`credit_report`; recall-sufficiency
  score; `bench/repro_harness.py` (pinned dataset+commit+seed, raw per-query
  artifacts) — the biggest external-credibility lever; before/after recompaction
  bench that gates the 4b-retrain *decision* (recorded, not executed).
- **Phase 3 — v4.7.0 "Task Frames + Dead-End Registry"** (M, dep 1) —
  `[TASK-FRAME]` kind + `resume_brief` (governed, batch-review staging from day
  one); `[DEAD-END]` kind + deterministic matcher (warns, never vetoes;
  registration HITL). Negative memory nobody else has.
- **Phase 4 — v4.8.0 "World-State Validity + Validity-Gated Fusion + Poisoning
  Defense"** (M, dep 1) — `world_anchor.py` (at-write git_sha/file-digest/
  config-hash capture, read-only compare, demote-and-flag never auto-archive);
  validity component in fusion (default OFF→A/B→ON); `poison_defense.py`
  (per-actor write-anomaly + canary blocks); public/private workspaces.
- **Phase 5 — v4.9.0 "Reflexion Repair + Consolidation Completion"** (M, dep
  2+4) — `reflexion.py` (post-FAILURE drafts supersede/repair proposals with
  evidence; DRAFT-only, never auto-approve — trap-B pressure lives here);
  `lint_autofix`; write-path edge extraction into the same proposal; `mm
  recompact` verb + dream-cycle pass 6; surface `block_maturity` +
  `granularity_align` (verify-before-build).
- **Phase 6 — v4.10.0 "Edge-Grounded Answers + Chat"** (M/L, dep 1) —
  `entity_resolution.py` (deterministic blocking + injected-arbiter clustering,
  singleton never dropped); `edge_grounded_answer.py` (k-hop subgraph → per-claim
  {edge_id, source_block_id} citations + explicit "not in graph" gaps); edge
  confidence from cross-source corroboration (sidecar); `chat_with_memory`;
  degree-gated hub-node profiles.
- **Phase 7 — v4.11.0 "Skill Distillation + Advanced Primitives"** (M/L, dep
  2+5) — `[SKILL]`/`[CAUSAL]` kinds; `skill_distill.py` (cluster VERIFIED
  trajectories → recompaction fixed-point → `[SKILL]` proposal; success-rate =
  pure fold over VERIFIED ledger rows, never an LLM vibe); cross-domain adapter;
  `mm export-episode` evidence-chain submission bundle.
- **Phase 8 — v4.12.0 "Local Hot Path + Adoption + Hardening"** (M, schedule
  last) — `novel_term_gate.py` + `anticipation_cache.py` (Redis push transport;
  cache hits are an attestation leg); `mind-mem connect`; OpenAPI/AsyncAPI specs;
  local-file importers (letta/mem0/chroma/qdrant → bulk-staged REPORTED-tier
  proposals); `mm usage`; plugin SDK; v3.2.0 security batch; per-peer signed-write
  envelopes; TLS 1.3 floor + pinning.

## Honest deferrals (bucket C / infra-gated bucket B) — do not pad phase scope

Pure-MIND core port (out of scope until compiler done) · `mm view` web UI /
managed console · PBFT · WASM + Rust/Java/Ruby SDK stubs · ActivityPub ·
`[VISUAL]` kind · T-008 SQLCipher · `mind-mem:4b` retrain (a *decision* gated on
the Phase-2 bench + external GPU spend) · all bucket-B publishing/infra (npm/Go
publish, SLSA L3, C2PA, KMS row-encryption, mTLS×2, WORM, K8s/Helm, PyOxidizer,
Kafka/NATS, cloud importers, chaos harness) — behind an explicit "infra unlock"
checklist · LoCoMo full run / LongMemEval hold (measurement debt).

## SOTA success criteria (capability, not benchmark rank)

"The only agent memory that can prove what it served, verify what it stored, and
show its repairs." Demonstrable, deterministically, on the live corpus with the
audit chain intact: provable serving (v4.4.0+P2) · evidence-graded memory (P1) ·
exact credit assignment (P2) · populated governed KG (P1+5) · long-horizon
continuity (P3) · negative memory (P3) · repo-speed validity (P4) · auditable
self-repair (P5) · grounded cited answers (P6) · procedural memory w/ verified
success rates (P7) · frictionless + fast (P8). Recall-benchmark *rank* is
explicitly NOT a phase goal — the repro harness (P2) makes numbers honest; chasing
rank waits for the pure-MIND + mind-lab stage.
