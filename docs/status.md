# MIND-Mem — implementation status (alignment companion)

Three-column ledger of the current source checkout, experimental work and roadmap items. The 5.0.4 source is a release candidate; its changelog entry does not establish publication. Companion to `docs/audit_response.md`, `docs/governance.md`, and `ROADMAP.md` / `CHANGELOG.md` (which remain product-authoritative).

## Implemented now (operational, tested)

### Core retrieval

Package is a flat `src/mind_mem/` — there is no `core/` subpackage.

| Component | Source | Notes |
|---|---|---|
| BM25F retrieval (Porter stemming + RM3 expansion) | `src/mind_mem/recall.py` (+ `_recall_core.py`, `_recall_scoring.py`, `_recall_tokenization.py`) | Per-field weighting; English stemmer baseline. |
| Hybrid search (BM25 + sqlite-vec + RRF fusion) | `src/mind_mem/hybrid_recall.py` | Reciprocal Rank Fusion across lexical and dense scores. |
| Cross-encoder reranking (opt-in) | `src/mind_mem/cross_encoder_reranker.py` | Config-gated; off by default. |
| 9-type intent router | `src/mind_mem/intent_router.py` | Adaptive confidence weights. |
| ConnectionManager (WAL read/write split) | `src/mind_mem/connection_manager.py` | Thread-safe SQLite pool. |
| BlockStore | `src/mind_mem/block_store.py` | A-MEM blocks with metadata evolution. |

### Governance + audit

Package is a flat `src/mind_mem/` — there is no `governance/` subpackage.

| Component | Source | Notes |
|---|---|---|
| Contradiction detection | `src/mind_mem/contradiction_detector.py` | Surfaces conflicting memories on read. |
| Drift detection | `src/mind_mem/drift_detector.py` | Long-window memory-shape monitoring. |
| Proposal queue (write gate) | `src/mind_mem/governance_gate.py` (`GovernanceGate`) | Human-approval pathway. |
| Audit chain (TAG_v1 NUL-separated preimages) | `src/mind_mem/audit_chain.py` | Q16.16 fixed-point scoring in hash preimages. |
| Alerting hooks (webhook / Slack) | `src/mind_mem/alerting.py` | Webhook-first; Slack template included. |

### Storage + tier decay

| Component | Source | Notes |
|---|---|---|
| At-rest encryption (HMAC-SHA256 keystream + encrypt-then-MAC; not AES/SQLCipher) | `src/mind_mem/encryption.py` | v3.0.0+. |
| Tier decay (TTL + idle demotion) | `src/mind_mem/memory_tiers.py` (`TierPolicy`, `run_promotion_cycle`) | RA.0 collapsed three tier ladders to this one and deleted the other two. |
| Delta-based snapshot rollback | `src/mind_mem/block_store.py` (`MANIFEST.json` write/read) | MANIFEST.json for O(manifest) restore. |

### MCP server (107 tools, 8 resources)

| Component | Source | Notes |
|---|---|---|
| MCP server entry | `src/mind_mem/mcp_server.py` | 107 tools across recall / write / governance / observability / audit. |
| Native MCP integration (19 clients) | `src/mind_mem/hook_installer.py` | `mm install-all` wires Claude Code, Claude Desktop, Codex CLI, Gemini CLI, GitHub Copilot CLI, Cursor, Windsurf, Zed, OpenClaw + 10 more. |
| Multi-backend LLM extractor | `src/mind_mem/llm_extractor.py` | ollama / openai-compatible / vLLM / exllamav2 backends selected by `backend="auto"`. |
| Model provenance (audit / sign / verify) | `src/mind_mem/model_provenance.py`, `src/mind_mem/mcp/tools/model.py` | `audit_model_tool`, `sign_model_tool`, `verify_model_tool` are in the shipped tool surface — no feature flag — with 28 tests and their own `Audit Pinned Models` workflow. This row said "not yet shipped" while all three were counted in the tool badge. |

### MIND scoring kernels

Kernel sources live in `mind/` (repo root), not `kernels/`.

| Component | Source | Notes |
|---|---|---|
| Score fusion kernel | `mind/ranking.mind` (`weighted_rank`) | Deterministic weighted sum of BM25/recency/graph/importance signals. |
| Audit-chain integrity verify | `src/mind_mem/mind_kernels.py` (`sha3_512_chain_verify`) | Python-side kernel wrapper; no standalone `.mind` source file yet for this step. |
| Tier decay adjustment | `src/mind_mem/memory_tiers.py` (`TierPolicy`) | Pure Python today — not yet ported to a `.mind` kernel. |

### Local model

| Component | Source | Notes |
|---|---|---|
| `star-ga/mind-mem-4b` (v4.1.1) | `docs/mind-mem-4b-setup.md` / [HF model card](https://huggingface.co/star-ga/mind-mem-4b) | Q4_K_M @ 2.7GB via Ollama; reported 133/133 eval (111 main + 22 held-out, two using inference-time anchors). Trained on 83 MCP tools; the current server exposes 107 MCP tools. This is not a new independent evaluation. |
| Backend dispatcher | `src/mind_mem/llm_extractor.py` | `mind-mem.json` → `{"backend": "ollama", "model": "mind-mem:4b"}`. |

## Experimental (in-tree, behind feature flags)

| Component | Source | Status |
|---|---|---|
| Adversarial-memory + Jepsen stress tests | _(planned — no dedicated suite yet)_ | v3.0.0+; gated on long-haul CI runner. |

## Release and roadmap status — 2026-09-15

The 5.0.4 source is a release candidate with publication pending. The verified
publication baseline is [MIND-Mem 5.0.3](https://pypi.org/project/mind-mem/5.0.3/)
(checked 2026-09-15 before candidate preparation). The 5.0.4 changes require
their own release checks against the exact commit. The historical 5.1.0 release remains yanked;
its published tag and archives are not rewritten.
Statuses below are implementation and evidence states, not a checkbox-derived
completion percentage. The full roadmap and release history remain in
`ROADMAP.md` and `CHANGELOG.md`.

| Area | Current status | Evidence and boundary |
|---|---|---|
| Current release work | 5.0.3 published; 5.0.4 candidate pending | Published archives and later source changes have separate identities. The current source exposes 107 MCP tools; see the 5.0.4 candidate changelog for the later retrieval, graph, training-preparation and release checks. |
| Pure-MIND core port | Planned; not started | `ROADMAP.md` describes the compiler and Rust-independence gate, missing `std.tensor`, and unresolved reductions before migration. Existing Python/native prototypes are not a completed Pure-MIND port. |
| 4B retraining | Deferred until after the Pure-MIND port | Published weights remain trained on 83 tools and report 133/133 evaluation; the current server exposes 107. A source-only MCP contract inventory supports preparation but is not independent evaluation or training approval. A newer base, including Qwen3.8-4B, is not asserted as available. Any post-port run requires a reviewed corpus, independent evaluation, rented GPU approval/funding, and new receipts. |
| Ground-truth evaluation | Open and blocked | `docs/design/eval-set-ground-truth.md` requires 60+ labelled cases against a hash-pinned snapshot. The current 36-query pool is an unlabelled development pool, not an independent evaluation or a basis for tuning and release claims. |
| JavaScript/TypeScript SDK | Source exists; registry publication pending | `sdk/js/` is tested in-tree. `sdk/release/README.md` requires a built staged artifact and npm authentication for `@star-ga/mind-mem-client`; no registry publication is claimed. |
| Go SDK | Source retrievable; tagged SDK release pending | `github.com/star-ga/mind-mem/sdk/go/v5` resolves through the Go proxy as a pseudo-version. The 2026-09-15 check found no `sdk/go/v5.x.y` release tag; source availability is distinct from a tagged SDK release. |
| Independent CVS / MIND Witness | Open design and implementation | `ROADMAP.md` RE.3 and `docs/specs/retrieval-receipt-contract.md` define the split and acceptance boundary. Local receipts remain local evidence until an independent producer/verifier interoperates. |
| AsyncAPI | Published in 5.0.3 for the opt-in outbound Redis Streams publisher | `sdk/spec/asyncapi.json` and `tests/test_sdk_asyncapi_drift.py` bind the `XADD` `data` field and observed source emitters. This is a best-effort outbound notification contract; no consumer, retry, ordering or at-least-once guarantee is claimed. |

## What this file is not

- It is not a release-claim. The README + CHANGELOG are authoritative.
- It is not auto-generated yet. Future workstream will derive it from the MCP tool catalogue + the `[invariant]` table.
- It does not cover consumers (MindLLM uses MIND-Mem for L4 retrieval). Each carries its own `docs/status.md`.
