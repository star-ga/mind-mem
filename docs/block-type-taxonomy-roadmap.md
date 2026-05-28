# Block-Type Taxonomy Enhancement — Roadmap Note

**Status:** Proposed, not implemented. Action on next mind-mem touch — not now.
**Date filed:** 2026-05-27
**Source pattern:** External improvement-harness 3-tier knowledge layout
(durable playbook + ephemeral hints + generated tools). Conceptual borrow,
clean-room — pattern only, no upstream code referenced.

## Why this exists

mind-mem currently models blocks flat — `decision`, `signal`, `task`,
`entity`, `fact` — with metadata tags. The flat shape is fine for the
current ~300-block scale per deployment, but as deployments grow, two
axes of meaning are getting conflated inside the same block-type:

1. **Persistence axis** — is this knowledge durable across sessions, or
   ephemeral within a single run? Currently decided implicitly by tag
   choice and curator discipline.
2. **Provenance axis** — was this block human-authored, agent-proposed,
   or auto-generated (e.g., a tool spec emitted by an architect role)?
   Currently inferred from block history, not first-class.

Adding the persistence axis as a first-class tag (without restructuring
block types) lets recall and ranking weight blocks differently, and
lets the apply-engine ignore ephemeral blocks during long-term audit
queries.

## Proposed change

Add an optional `lifecycle` field to block frontmatter, with three
values:

| Value | Meaning | Default behavior |
|---|---|---|
| `durable` | Persists across runs, indexed in long-term recall | Default for `decision`, `entity`, `fact` |
| `ephemeral` | Session-scoped hint, expired after N days | Default for transient `signal` blocks |
| `generated` | Auto-emitted by an agent role (e.g., architect-emitted tool spec) | Default for blocks proposed by sub-agents |

Recall and hybrid_search gain an optional `lifecycle` filter; existing
queries default to `durable` to preserve current behavior.

## Why this is small

- No schema break — `lifecycle` is optional, defaults preserve current
  behavior.
- No new block type — existing decision/signal/task/entity/fact types
  stay unchanged.
- No retroactive backfill — older blocks default to `durable`
  semantically, which matches current implicit usage.
- Apply-engine + propose_update unchanged in surface, just gain the
  filter parameter.

## When to do this

Not now. Next time mind-mem ships a non-trivial change (block schema,
search-rank tuning, retrieval primitive) — bundle this enhancement into
that work so the schema change isn't a standalone release.

Forward triggers:

- A deployment crosses ~1,000 blocks and recall starts surfacing stale
  ephemeral signals as top hits.
- An agent role (proposer, architect — see Naestro R11 acceptance
  criteria) starts emitting structured proposals that need a distinct
  retrieval lane.
- A buyer's audit requires distinguishing human-authored from
  agent-emitted knowledge in the evidence chain.

## Cross-references

- Naestro ROADMAP R11 5-role acceptance criteria (proposer / analyst /
  coach / architect / curator).
- Naestro ROADMAP R25 Production-Trace Instrumentation (some R25
  traces may promote into mind-mem as `generated` lifecycle blocks).
- Source pattern: external multi-generation improvement harness
  observed 2026-05-27 (see STARGA-internal autocontext review note).

## Out of scope

- Cross-tenant retrieval differentiation — separate concern, owned by
  Naestro Track D D1.
- Automatic ephemeral → durable promotion based on access patterns —
  too clever for v1; humans still curate via apply-engine.
- Block-type renames or restructures — explicitly avoided to keep this
  enhancement reversible.
