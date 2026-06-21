# Hyperedge + temporal-anchor design (Hyper-Extract steal)

Date: 2026-06-17
Status: SPEC — not implemented. Review the shape before any code lands.
Trigger: github.com/yifanfeng97/Hyper-Extract (Apache-2.0) evaluated as a
"steal" candidate for the mind-mem knowledge graph.

## TL;DR

There is **no algorithm worth porting** from Hyper-Extract. Their hyperedge is a
Pydantic `list` field + 3 user lambdas; their "extraction" is two LLM prompts +
a dangling-edge prune (`participants ⊆ known_nodes`); their temporal model is a
string-concatenated dedup key. **Our store is already ahead on temporal** (real
`valid_from`/`valid_until` columns w/ validation, see `knowledge_graph.py:307`).

The only genuine gap our store has is **N-ary hyperedges** (one edge over 3+
entities). That maps to a standard relational incidence table — our own code, no
dependency. Plus one prompt-engineering idea worth adopting (`observation_time`).

**Does NOT require retraining the 4B mind-mem model.** The 4B model embeds/ranks
text blocks (the vector leg of RRF). The graph layer is a separate SQLite store
the model never reads or writes. This work is orthogonal to the U1 QLoRA retrain.

## Current state (verified live, knowledge_graph.py)

- `edges` table is a **dyadic S-P-O triple**: `(subject, predicate, object,
  source_block_id, confidence, valid_from, valid_until, metadata)`,
  PK `(subject, predicate, object, source_block_id)` — `knowledge_graph.py:307`.
- Temporal columns + ISO8601 validation already present (`:313`, `:375-381`).
- Typed predicates already present (`Predicate` enum); `test_typed_edges_group_h.py`
  + `causal_graph.py` already exercise typed/decayed edges.
- MCP surface: `graph_add_edge` / `graph_query` / `traverse_graph` /
  `graph_stats` in `mcp/tools/graph.py`. `graph_add_edge` is hard-coded to
  subject/object (`:29-60`).

## Proposed extension (additive — triple table untouched)

### New tables

```sql
CREATE TABLE IF NOT EXISTS hyperedges (
    edge_id          TEXT PRIMARY KEY,   -- stable hash of sorted members+predicate+source
    predicate        TEXT NOT NULL,
    label            TEXT,               -- human label for the joint fact
    source_block_id  TEXT NOT NULL,
    confidence       REAL NOT NULL DEFAULT 1.0,
    valid_from       TEXT,               -- mirror the triple table's temporal model
    valid_until      TEXT,
    metadata         TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS hyperedge_members (
    edge_id    TEXT NOT NULL REFERENCES hyperedges(edge_id),
    entity_id  TEXT NOT NULL,            -- resolved via EntityRegistry, same as triples
    role       TEXT,                     -- optional: agent/instrument/enables/... NULL = unordered set
    PRIMARY KEY (edge_id, entity_id, role)
);
CREATE INDEX IF NOT EXISTS idx_hmembers_entity ON hyperedge_members(entity_id);
CREATE INDEX IF NOT EXISTS idx_hyperedges_predicate ON hyperedges(predicate);
```

`edge_id` = stable hash over `sorted(member_entity_ids) + predicate + source_block_id`
so `{A,B,C}` and `{C,A,B}` collapse (this is the one correctness rule Hyper-Extract
makes the caller responsible for — we bake it in instead).

### Tool changes

- `add_hyperedge(members: list[str], predicate, *, label=None, roles=None,
  source_block_id, confidence=1.0, valid_from=None, valid_until=None, metadata=None)`
  — resolves each member via `EntityRegistry.resolve` (same path as triples),
  inserts one `hyperedges` row + N `hyperedge_members` rows. Reject if <2 members
  (use `add_edge` for binary). Idempotent on `edge_id`.
- `traverse_graph` gains hyperedge incidence: "find all hyperedges containing
  entity X" = `SELECT edge_id FROM hyperedge_members WHERE entity_id=?` then join
  back to co-members. Plain SQL join, our own code — no ANN, no LLM.
- Referential integrity on insert: each member must resolve to a known entity
  (our equivalent of their `_prune_dangling_edges`, enforced at write not read).

### Prompt tweak (the one real idea worth adopting)

In the extractor (`extractor.py`), anchor temporal extraction with a concrete
`observation_time` (today's date) so the LLM resolves "last year" / "currently"
against a fixed point before parsing. ~3 lines in the prompt. Independent of the
hyperedge tables — could ship alone.

## What we deliberately do NOT take from Hyper-Extract

- Their Pydantic-lambda data model (we have a real schema).
- Their two-prompt extraction pipeline (we have a governed extractor + HITL).
- Their string-concat temporal key (we have real temporal columns — better).
- Their ANN-only "traversal" (we want real incidence traversal).
- Their LLM-merge dedup (we have contradiction-safe `propose_update` governance).

Adopting their RAG engines would *compete* with mind-mem's governance thesis
(auditable + contradiction-safe), not extend it. Hard no.

## Open questions for review

1. Roles: ship `role` column now (nullable) or defer until a consumer needs it?
   Cheap to include nullable; YAGNI says defer the *population* but the column is
   one line.
2. Should hyperedges feed recall ranking (a graph-walk signal fused into RRF)?
   That is a *separate* future phase, still not a retrain — a reranking input.
   Out of scope for this spec.
3. Migration: pure additive (CREATE TABLE IF NOT EXISTS) — no data migration of
   existing triples needed. Existing 103 dyadic edges stay as triples.
