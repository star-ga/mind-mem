# mind-mem → SOTA Agentic Memory (v4.5.0+ design brief)

> Forward-looking plan (2026-07-30). Strategic frame: competitors (Mem0, Zep,
> Letta) fight on **recall quality**, where the honest LoCoMo/LongMemEval bench
> puts mind-mem at ~BM25-level. The winning move is a different axis —
> **memory an agent can act on without re-verifying**: trust, outcome-closure,
> task continuity. These are unreachable for competitors without rebuilding
> their storage layer, because they lack the attestation + governed-write
> substrate. **Keystone asset: `recall_attestation.py` (v4.4.0)** — the only
> memory that can prove *exactly which blocks were served into which run*,
> enabling **exact** credit assignment (everyone else correlates fuzzily).

## Ranked proposals (build order: 1 → 2 → (3 ∥ 4 ∥ 6) → 7 → 5)

| # | Name | What | Agent-SOTA differentiator | Wedge guardrail | Effort |
|---|------|------|---------------------------|-----------------|--------|
| **1** | **Trust-Tier Evidence Grading** (keystone) | Every block carries a deterministically-derived tier: VERIFIED (machine-checkable evidence handle / exit code / CI run) / ATTESTED (agent claim) / REPORTED (human) / INFERRED (consolidation). Recall surfaces it; agents gate risky actions on it. | Agents *act* on memory — today they must re-verify (token waste) or trust-all (drift). None distinguish "agent believes tests pass" from "here's the stored test-run handle proving it." | Tier is a pure versioned function of existing provenance fields — never an LLM judgment; lives in sidecar metadata, **never in the sealed audit-hash preimage**. | S |
| **2** | **Attestation→Outcome Credit Ledger** | Join each run's attestation (blocks served) with its terminal outcome (exit code / eval / TRAJ SUCCESS-FAILURE) → append-only per-block credit ledger (served / co-success / co-failure). | The credit-assignment problem for memory, solved **exactly** not statistically. Turns "recall ~BM25" into offense: stop optimizing rank, measure "did the served set cause success." | Ledger is evidence, never actuation — ranking changes only via a versioned config diff through propose_update/HITL. No online reweighting. | M |
| **3** | **Governed Task Frames** | `[TASK-FRAME]` block: goal, plan steps, per-step status, open loops, blockers, decisions — checkpointed at session end; deterministic `resume_brief(frame_id)` emits a bounded citation-bearing resume context. | Multi-session long-horizon work is THE agentic gap (proven by the operator's own hand-written RESUME notes). Letta has state but no governance/audit/determinism — a mind-mem frame is *replayable* + *auditable*. | Frame mutations are governed writes (propose→approve); documented low-friction batch-review for step ticks, **never** silent mutation of goal/plan/decisions. | M |
| **4** | **World-State Anchored Validity** | Bind blocks to external substrate at write time (git SHA, file digests, config hash); recall deterministically flags "recorded at commit X; referenced file since changed." | Coding-agent memory rots at repo speed, not calendar. Time-decay/lineage can't see `auth.ts` was rewritten. Evidence-based staleness = pure hash compare, un-expressible for fuzzy-embedding rivals. | Anchors captured at write, compared read-only at recall; verdict is a logged hash compare — demotes + flags, never auto-archives (repair = scan finding → proposal). | S/M |
| **5** | **Trajectory→Skill Distillation** | Cluster successful `[TRAJECTORY]` blocks → distill (via the existing recompaction fixed-point + injected compressor) into `[SKILL]` blocks (preconditions/steps/failure-modes); success-rate maintained deterministically from VERIFIED outcomes (P1) only. | Procedural memory is what separates agent memory from RAG. Success-rate = count of machine-verified outcomes, not an LLM's vibe (anti-Voyager). Feeds the skills hub + skill_opt critique loop. | Output lands as propose_update with `source_ids`+`input_digest` (recompaction discipline); fixed-point loop + injected `Callable` keeps CI LLM-free; source-of-truth never self-modifies. | M/L |
| **6** | **Dead-End Registry** | `[DEAD-END]` block (approach, context, why-failed, evidence handle) + deterministic match (entity/tool/predicate overlap) → **warning channel** in recall when a plan/task-frame matches a known dead-end. | Autonomous agents re-run known-failed approaches — costliest long-horizon failure (proven by the operator's "do NOT re-run AVX2 campaign" note). Action-space negative memory; zero competitors have it. | Deterministic overlap match; a dead-end can only *warn* (flagged result + failure evidence), never gate/veto autonomously; registration is HITL. | S |
| **7** | **Reflexion Repair Loop** | After a FAILURE, a post-run pass joins the credit ledger (P2) + contradiction/staleness detectors and **auto-drafts** governance proposals ("block D-x served into this failed run, anchored-stale/contradicted → proposed supersede, evidence attached"); operator batch-approves. | Reflection made auditable — competitors write LLM "insights" straight into memory (silent drift). Here reflection produces *proposals with machine evidence*; the human approves the diff. Closing actuator of 1→2→7. | May only *draft* — never auto-approve, however confident; drafts carry the full evidence chain (attestation id, outcome, detector finding). | S/M (after 1+2) |

**The SOTA claim that falls out:** *"the only agent memory that can prove what it served, verify what it stored, and show its repairs."*

## Wedge-incompatible — DO NOT BUILD
- **A. Online outcome-weighted reranking (RL on recall weights).** Learned nondeterministic rewiring; breaks byte-identical replay + bypasses HITL. Compliant form = P2's versioned-config route. Any "memory learns to retrieve from rewards, automatically" is this in disguise.
- **B. Autonomous agent self-edit of core memory (Letta/MemGPT).** Breaks governed-write. `self_editing.py` already does it correctly (propose/approve/reject) — the trap is an "auto-approve low-risk edits" fast path under UX pressure.
- **C. LLM-judged importance/decay per run.** Nondeterministic at the retrieval core; deterministic A-MEM decay + P4 evidence-staleness cover the need.

## Open concern (fold in up front)
P3's step-tick friction (HITL on every plan-step update) needs the **batch-review mode designed up front**, or agents route around the frame with scratch files and the feature dies. Permissible relaxation: stage-for-batch-review, never silent apply, never for goal/plan/decision fields.

---
*Source: Fable design pass, 2026-07-30. Opus reviews + implements the fit ones after v4.4.0 ships.*
