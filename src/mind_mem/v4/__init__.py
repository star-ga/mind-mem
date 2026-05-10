"""mind-mem v4.0 surface — side-by-side scaffolding, default OFF.

This package holds the v4.0 implementation. Every surface in here is
gated by an explicit feature flag in ``mind-mem.json`` under the ``v4``
key; without the flag set, the surface raises ``FeatureDisabledError``
and v3.x behaviour is preserved unchanged.

Canonical task list: ``ROADMAP.md`` ``## v4.0.0`` (Groups A–F).
Design rationale: ``docs/roadmap-v4.md``.

Surfaces (mapped to ROADMAP groups):

A. cognition / model layer
   - tier-aware block schema additions   →  v4.tier_memory
   - Cognitive Mind Kernel API           →  v4.cognitive_kernel
   - surprise-weighted retrieval term    →  v4.surprise_retrieval

B. knowledge graph (multi-page entity / concept extraction)
   - block kinds (entity/concept/...)    →  v4.block_kinds
   - long-context recall mode            →  v4.long_context_recall
   - LLM-driven knowledge fusion         →  v4.fusion
   - streaming recall                    →  v4.streaming_recall
   - conversational chat layer           →  v4.chat
   - schema layer for prompts            →  v4.prompt_schema

C. knowledge graph governance / UX
   - idle-only background ingest         →  v4.idle_ingest
   - AI lint with auto-fix               →  v4.lint
   - contradiction state machine         →  v4.contradiction_states
   - self-healing index                  →  v4.self_heal
   - local visual viewer                 →  v4.viewer
   - real-time contradiction stream      →  v4.contradiction_stream

D. platform scale
   - Rust hot path                       →  v4.rust_hot_path (optional dep)
   - pluggable embedding fallback        →  v4.embedding_fallback
   - (sharded Postgres / K8s / gRPC are tracked separately in
     existing platform paths)

E. compliance-sensitive opt-in extensions
   - pluggable redaction layer           →  v4.redaction
   - time-bounded recall                 →  v4.time_bounded_recall
   - vocabulary-bound fields             →  v4.vocabulary
   - provenance fields                   →  v4.provenance
   - evidence/confidence as first-class  →  v4.evidence
   - tenant KMS                          →  v4.tenant_kms
   - per-tenant audit chains             →  v4.tenant_chains
   - compliance export                   →  v4.compliance_export
   - contraindicates / supersedes edges  →  v4.contraindicates_edges

F. anti-patterns explicitly forbidden — see docs/roadmap-v4.md.

The ``v4`` package adds NO behaviour to v3.x code paths until a flag
flips. Every public symbol is additive; nothing in v3.x imports from
``mind_mem.v4``. This is the contract.

Copyright STARGA, Inc.
"""

from __future__ import annotations

from .feature_flags import (
    ALL_V4_FLAGS,
    FeatureDisabledError,
    is_enabled,
    require_enabled,
)

__all__ = [
    "FeatureDisabledError",
    "is_enabled",
    "require_enabled",
    "ALL_V4_FLAGS",
]
