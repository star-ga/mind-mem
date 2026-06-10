# MIND-Mem — governance design (5 layers)

MIND-Mem is the L4 retrieval-time layer of the MIND ecosystem: persistent, auditable, contradiction-safe memory for AI agents. This document is the local five-layer mapping; the canonical description lives in [`512-mind/docs/governance.md`](https://github.com/star-ga/512-mind/blob/main/docs/governance.md).

## Layer summary

| Layer | What it enforces | Primary source | Verified by |
|---|---|---|---|
| **L1 architectural** | Acyclicity, governance-kernel coverage, depth/equality/redundancy/purity floors of MIND-Mem | `.arch-mind/rules.mind` (this repo) | `arch-mind check-rules` (CI per-repo) |
| **L2 training-time** | Local-model fine-tune integrity (`mind-mem-4b` checkpoint provenance) | `docs/mind-mem-4b-setup.md` + planned `model_provenance.py` | Checkpoint manifest + reproducibility runbook |
| **L3 inference-time** | Per-recall request shape; rate limiting; auth-token validation | `src/mind_mem/mcp_server.py` + `src/mind_mem/auth.py` | MCP server tests; auth-failure tests |
| **L4 retrieval-time** | **The load-bearing layer.** Contradiction detection, drift, proposal queue, audit chain (TAG_v1), tier decay, at-rest encryption | `src/mind_mem/governance/*.py` + `src/mind_mem/core/store.py` | `tests/test_governance.py`, `tests/test_audit.py`, `tests/jepsen/` |
| **L5 continuous** | CI on every PR; planned nightly arch-mind regression; PyPI release on tag | `.github/workflows/ci.yml`, `.github/workflows/release.yml` | OIDC trusted-publishing via PyPI; LoCoMo benchmark snapshot per release |

## L1 — Architectural

`.arch-mind/rules.mind` declares nine `[arch_rule]` constraints. Floors recalibrated 2026-05-01 (see `audit_response.md` F1) for the Python+MIND hybrid reality. Per-rule current values are in `audit_response.md`.

The cross-repo contract: a regression on any of the nine rules halts MIND-Mem's nightly regression and surfaces in the ecosystem health dashboard.

## L2 — Training-time

MIND-Mem ships a local fine-tuned model (`star-ga/mind-mem-4b`, Qwen3.5-4B base). L2 governance covers:

- **Checkpoint provenance.** Every release of `mind-mem-4b` is reproducible from the data + base + recipe documented in `docs/mind-mem-4b-setup.md`.
- **Bundle integrity.** The Q4_K_M GGUF format ships with the standard llama.cpp checksum.
- **Planned: model provenance v1.0.** `src/mind_mem/model_provenance.py` (currently in-flight on local main) will record the source-tree SHA, training-data hash, and base-model hash into a manifest that travels with every fine-tuned weights bundle.

## L3 — Inference-time

MIND-Mem's request-time surface is the MCP server. Every MCP tool call flows through:

1. **Auth check.** `X-MindMem-Token` header validated against the configured token list.
2. **Rate limit.** Sliding-window per-token + global; the limiter primitive is shared with 512-mind.
3. **Tool dispatch.** 83 MCP tools, each with a typed input schema. Schema mismatch rejects with a structured error.
4. **Audit chain entry.** The request is recorded with the calling `auth_hash` (when supplied) so a downstream auditor can replay.

## L4 — Retrieval-time

The load-bearing layer. Every `recall` (read) or `propose_update` (write) flows through:

1. **Contradiction detection.** Conflicting memories surface as a structured contradiction event on the witness chain.
2. **Drift detection.** Long-window memory-shape monitoring; an alert fires if the corpus shape diverges from the recent baseline.
3. **Proposal queue.** Writes go to the proposal queue; approval requires either explicit operator sign-off or automated approval under a declared policy (rate-limited, contradiction-free).
4. **Audit chain.** TAG_v1 NUL-separated hash preimages over Q16.16-scored entries. The chain is replayable: feed the same inputs to the same source-tree SHA and get the same hash.
5. **At-rest encryption.** Optional authenticated encryption of on-disk block files (HMAC-SHA256 keystream + encrypt-then-MAC with a PBKDF2-derived key — *not* AES/SQLCipher; the FTS5/sqlite-vec recall index is not encrypted). Decryption is per-process; the running server is the only entity that can read the plaintext.

## L5 — Continuous

The drift-detection layer:

- **CI** on every push and PR — full pytest matrix (5,465+ tests across the suite).
- **PyPI release** on tag push via OIDC trusted publishing (no long-lived tokens).
- **LoCoMo benchmark snapshot** per release; regression on any axis (mean / adversarial / temporal) is documented in CHANGELOG.

Planned:

- **Nightly arch-mind regression** — `.github/workflows/arch-mind.yml`.
- **Adversarial-memory long-haul** — Jepsen-style stress tests on a long-running runner.

## Cross-repo discipline

MIND-Mem is one consumer of the 512 kernel; the same kernel runs in:

- `512-mind` (kernel itself)
- `mind-inference` (transformer inference)
- `rfn-mind` (RFN classifier inference)
- `MindLLM` (request-time HTTP surface; uses MIND-Mem for L4)
- `mind-mem` (this repo — L4 retrieval-time)
- `arch-mind` (L1 ecosystem-wide)

A change to the kernel surface in 512-mind propagates here through the MIND-Mem MCP server's auth-hash binding; the consumer's audit-chain replay catches any mismatch.

---

*Memory governance design v3.1.x, 2026-05-02. Canonical kernel description: [`512-mind/docs/governance.md`](https://github.com/star-ga/512-mind/blob/main/docs/governance.md).*
