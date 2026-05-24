# mind-mem: agent instructions (auto-written)

Before every response, run `mm context "$QUERY"` and prepend the output.


<claude-mem-context>
# Memory Context

# [mind-mem] recent context, 2026-05-09 2:53am PDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 4 obs (1,646t read) | 85,780t work | 98% savings

### Apr 21, 2026
**6661** 9:25p 🟣 **mind-mem v3.4.0 Pre-Release Security Hardening**
The mind-mem v3.4.0 release introduces four new recall modules for enhanced AI memory retrieval. A two-agent security audit identified and fixed six issues before final review: prompt injection via evidence wrapping, missing iteration limits on iterative_retrieve (now capped at 5 rounds, 3 followups, 20 total queries), unsafe follow-up query patterns (now regex-validated), out-of-bounds temporal dates, unsafe object deduplication using id() instead of content fingerprints, and a NameError when sub_queries was empty. The LoCoMo benchmark on conversation-0 jumped from 77.06 to 95.0 with the new features enabled, indicating significant retrieval quality improvements.
~378t 🛠️ 19,507

### May 2, 2026
**6702** 11:17p 🔵 **mind-mem v3.8.x Ship Train Complete — 11 Releases Audited**
The mind-mem v3.8.x release train spanning 10 versions completed the Model Safety Audit theme end-to-end: Ed25519 signing of model manifests, HuggingFace base_model allowlists, MCP tool wrappers for audit/sign/verify, a load-gate registry enforced inside the transformers extractor, CI hooks for auditing pinned models, and a full MIC/MAP Python toolchain for MIND IR serialization. The v3.8.8 release added Hypothesis fuzzing and 26 adversarial DoS tests which caught a real UnicodeDecodeError leak. v3.8.9 added a streaming MICB parser with bounded memory. v3.8.10 added an optional Cython accelerator for hot-path serialization primitives. Social Ingestion was architecturally descoped from mind-mem and moved to the agent layer, recognizing that fetching is an agent concern not a memory concern. Next priorities include v3.9.0 (hash-of-code invalidation, per-byte lineage) and v4.0.0 (sharded Postgres, Kubernetes operator, gRPC, event fan-out).
~519t 🔍 22,157

**6705** 11:42p 🟣 **mind-mem v3.8.11 MIC/MAP Discoverability Layer**
mind-mem v3.8.11 shipped the discoverability layer for MIC/MAP codec functionality. The core codec shipped pure-Python in v3.8.5; this release adds the user-facing surface: two MCP tools (mic_convert for format conversion between mic2 text and micb binary with auto-detection and 8 MiB input guard, mic_inspect for structural analysis), CLI commands (mm mic convert/inspect), a runnable quickstart example demonstrating residual block building and round-trip with streaming parser, comprehensive user documentation, and README feature callout. The implementation honors critical constraints: zero new dependencies preserving mind-mem's zero-dep status, no exposure of private STARGA Rust backends, and separation from the MIC/MAP v15 patent filing track.
~439t 🛠️ 22,021

### May 6, 2026
**6709** 6:32p ⚖️ **mind-mem 4B Full Fine-Tune GO/NO-GO Checkpoint**
GO/NO-GO checkpoint for mind-mem 4B full fine-tune training run. Configuration shows strong alignment between training corpus and eval spec: 55 production queries define success criteria, corpus includes verbatim prompts plus 2700+ paraphrases for robustness, keyword coverage is saturated (23-670 mentions per eval keyword), and system prompts are matched. Learning rate 1.5e-5 with cosine over 3 epochs is conservative for 4B scale. Zero contamination claim valid if paraphrases are semantically distinct. This setup optimizes for the specific 55-query production spec rather than general capability.
~310t ⚖️ 22,095


Access 86k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>