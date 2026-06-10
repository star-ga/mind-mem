# MIND-Mem — implementation status (alignment companion)

Three-column ledger of what is operational on the current `main`, what is in-tree as experimental, and what is on the roadmap. Companion to `docs/audit_response.md`, `docs/governance.md`, and `ROADMAP.md` / `CHANGELOG.md` (which remain product-authoritative).

## Implemented now (operational, tested)

### Core retrieval

| Component | Source | Notes |
|---|---|---|
| BM25F retrieval (Porter stemming + RM3 expansion) | `src/mind_mem/core/retrieval.py` | Per-field weighting; English stemmer baseline. |
| Hybrid search (BM25 + sqlite-vec + RRF fusion) | `src/mind_mem/core/hybrid.py` | Reciprocal Rank Fusion across lexical and dense scores. |
| Cross-encoder reranking (opt-in) | `src/mind_mem/core/rerank.py` | Config-gated; off by default. |
| 9-type intent router | `src/mind_mem/core/intent.py` | Adaptive confidence weights. |
| ConnectionManager (WAL read/write split) | `src/mind_mem/core/connection.py` | Thread-safe SQLite pool. |
| BlockStore | `src/mind_mem/core/store.py` | A-MEM blocks with metadata evolution. |

### Governance + audit

| Component | Source | Notes |
|---|---|---|
| Contradiction detection | `src/mind_mem/governance/contradiction.py` | Surfaces conflicting memories on read. |
| Drift detection | `src/mind_mem/governance/drift.py` | Long-window memory-shape monitoring. |
| Proposal queue (write gate) | `src/mind_mem/governance/proposal.py` | Human-approval pathway. |
| Audit chain (TAG_v1 NUL-separated preimages) | `src/mind_mem/audit.py` | Q16.16 fixed-point scoring in hash preimages. |
| Alerting hooks (webhook / Slack) | `src/mind_mem/governance/alerts.py` | Webhook-first; Slack template included. |

### Storage + tier decay

| Component | Source | Notes |
|---|---|---|
| At-rest encryption (HMAC-SHA256 keystream + encrypt-then-MAC; not AES/SQLCipher) | `src/mind_mem/encryption.py` | v3.0.0+. |
| Tier decay (TTL + LRU aging) | `src/mind_mem/core/decay.py` | v3.0.0+. |
| Delta-based snapshot rollback | `src/mind_mem/core/snapshot.py` | MANIFEST.json for O(manifest) restore. |

### MCP server (83 tools, 8 resources)

| Component | Source | Notes |
|---|---|---|
| MCP server entry | `src/mind_mem/mcp_server.py` | 83 tools across recall / write / governance / observability / audit. |
| Native MCP integration (18 clients) | `src/mind_mem/hook_installer/` | `mm install-all` wires Claude Code, Claude Desktop, Codex CLI, Gemini CLI, GitHub Copilot CLI, Cursor, Windsurf, Zed, OpenClaw + 9 more. |
| Multi-backend LLM extractor | `src/mind_mem/extractors/` | ollama / openai-compatible / vLLM / exllamav2. |

### MIND scoring kernels

| Component | Source | Notes |
|---|---|---|
| Score fusion kernel | `kernels/score_fusion.mind` | Q16.16 deterministic weighted sum. |
| Tier decay kernel | `kernels/tier_decay.mind` | Q16.16 TTL/LRU adjustment. |
| Audit-integrity preimage encoder | `kernels/audit_preimage.mind` | TAG_v1 NUL-separated hash preimages. |

### Local model

| Component | Source | Notes |
|---|---|---|
| `star-ga/mind-mem-4b` (fully trained) | `docs/mind-mem-4b-setup.md` | Q4_K_M @ 2.7GB via Ollama. |
| Backend dispatcher | `src/mind_mem/extractors/dispatcher.py` | `mind-mem.json` → `{"backend": "ollama", "model": "mind-mem:4b"}`. |

## Experimental (in-tree, behind feature flags)

| Component | Source | Status |
|---|---|---|
| Model provenance v1.0 (in-flight) | `src/mind_mem/model_provenance.py`, `src/mind_mem/mcp/tools/model.py`, `tests/test_model_provenance.py` | Active development on local main; not yet shipped. |
| Adversarial-memory + Jepsen stress tests | `tests/jepsen/` | v3.0.0+; gated on long-haul CI runner. |

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
