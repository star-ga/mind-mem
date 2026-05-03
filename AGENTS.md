<claude-mem-context>
# Memory Context

# [mind-mem] recent context, 2026-05-02 11:42pm PDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 2 obs (897t read) | 41,664t work | 98% savings

### Apr 21, 2026
**6661** 9:25p 🟣 **mind-mem v3.4.0 Pre-Release Security Hardening**
The mind-mem v3.4.0 release introduces four new recall modules for enhanced AI memory retrieval. A two-agent security audit identified and fixed six issues before final review: prompt injection via evidence wrapping, missing iteration limits on iterative_retrieve (now capped at 5 rounds, 3 followups, 20 total queries), unsafe follow-up query patterns (now regex-validated), out-of-bounds temporal dates, unsafe object deduplication using id() instead of content fingerprints, and a NameError when sub_queries was empty. The LoCoMo benchmark on conversation-0 jumped from 77.06 to 95.0 with the new features enabled, indicating significant retrieval quality improvements.
~378t 🛠️ 19,507

### May 2, 2026
**6702** 11:17p 🔵 **mind-mem v3.8.x Ship Train Complete — 11 Releases Audited**
The mind-mem v3.8.x release train spanning 10 versions completed the Model Safety Audit theme end-to-end: Ed25519 signing of model manifests, HuggingFace base_model allowlists, MCP tool wrappers for audit/sign/verify, a load-gate registry enforced inside the transformers extractor, CI hooks for auditing pinned models, and a full MIC/MAP Python toolchain for MIND IR serialization. The v3.8.8 release added Hypothesis fuzzing and 26 adversarial DoS tests which caught a real UnicodeDecodeError leak. v3.8.9 added a streaming MICB parser with bounded memory. v3.8.10 added an optional Cython accelerator for hot-path serialization primitives. Social Ingestion was architecturally descoped from mind-mem and moved to naestro-bot, recognizing that fetching is an agent concern not a memory concern. Next priorities include v3.9.0 (hash-of-code invalidation, per-byte lineage) and v4.0.0 (sharded Postgres, Kubernetes operator, gRPC, event fan-out).
~519t 🔍 22,157


Access 42k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>