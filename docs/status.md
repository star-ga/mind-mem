# MIND-Mem — implementation status (alignment companion)

Three-column ledger of what is operational on the current `main`, what is in-tree as experimental, and what is on the roadmap. Companion to `docs/audit_response.md`, `docs/governance.md`, and `ROADMAP.md` / `CHANGELOG.md` (which remain product-authoritative).

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

### MCP server (96 tools, 8 resources)

| Component | Source | Notes |
|---|---|---|
| MCP server entry | `src/mind_mem/mcp_server.py` | 96 tools across recall / write / governance / observability / audit. |
| Native MCP integration (18 clients) | `src/mind_mem/hook_installer.py` | `mm install-all` wires Claude Code, Claude Desktop, Codex CLI, Gemini CLI, GitHub Copilot CLI, Cursor, Windsurf, Zed, OpenClaw + 9 more. |
| Multi-backend LLM extractor | `src/mind_mem/llm_extractor.py` | ollama / openai-compatible / vLLM / exllamav2 backends selected by `backend="auto"`. |

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
| `star-ga/mind-mem-4b` (fully trained) | `docs/mind-mem-4b-setup.md` | Q4_K_M @ 2.7GB via Ollama. |
| Backend dispatcher | `src/mind_mem/llm_extractor.py` | `mind-mem.json` → `{"backend": "ollama", "model": "mind-mem:4b"}`. |

## Experimental (in-tree, behind feature flags)

| Component | Source | Status |
|---|---|---|
| Model provenance v1.0 (in-flight) | `src/mind_mem/model_provenance.py`, `src/mind_mem/mcp/tools/model.py`, `tests/test_model_provenance.py` | Active development on local main; not yet shipped. |
| Adversarial-memory + Jepsen stress tests | _(planned — no dedicated suite yet)_ | v3.0.0+; gated on long-haul CI runner. |

## Future roadmap (alignment-driven)

For product roadmap, see `ROADMAP.md` and `CHANGELOG.md`. Cross-repo alignment items:

| ID | Item | Target |
|---|---|---|
| `EVD-MIND-1` | Native-MIND scoring kernels emit evidence calls from MIND modules | v3.2.0 |
| `MIC-1` | MIC-B serialization for cross-repo evidence payloads | v3.2.0 |
| `CI-AM-1` | Nightly arch-mind regression with audit-trail event emission | v3.2.0 |
| `STATUS-AUTO-1` | Auto-generate this file from the MCP tool catalogue | v3.2.0 |

## What this file is not

- It is not a release-claim. The README + CHANGELOG are authoritative.
- It is not auto-generated yet. Future workstream will derive it from the MCP tool catalogue + the `[invariant]` table.
- It does not cover consumers (MindLLM uses MIND-Mem for L4 retrieval). Each carries its own `docs/status.md`.
