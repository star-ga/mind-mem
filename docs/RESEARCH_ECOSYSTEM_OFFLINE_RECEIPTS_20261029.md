# Governed memory consumption of offline research receipts

Status: proposed / review required / default OFF. No implementation or deployment is claimed.

The memory product is the governed store of record, not an experiment engine. The offline research service owns proposals and result generation. The user-facing layer owns orchestration.

- Define opt-in, provenance-linked storage of externally verified experiment receipts, with bounded source, corpus, and model revision identifiers.
- Keep speculative hypotheses and unverified research leads separate from approved factual memory.
- Require propose, review, and approve for durable promotion. Preserve contradiction, lifecycle, privacy, and access checks, and keep the evidence replayable.
- Prefer existing hybrid retrieval, atomic fact indexing, and graph associations. Evaluate a new concept-composition feature only against a measured incremental benefit.
- Validate revocation, correction, stale studies, conflicting evidence, and source retraction without silent overwrites.

No external scorer can silently mutate memory or bypass approval.
