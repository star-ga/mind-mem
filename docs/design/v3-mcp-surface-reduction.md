# v3.0 Design: MCP Tool Surface Reduction

Status: **draft**
GH issue: [#501](https://github.com/star-ga/mind-mem/issues/501)

## Problem

mind-mem v2.10 ships **57 MCP tools**. Gemini 3 Pro's arch audit
flagged this as a high-friction surface for LLM clients:

> "57 MCP tools will overwhelm the context window and the agent's
> tool-selection capabilities."

Each MCP tool costs ~200–400 tokens of schema in the agent's
context on every turn. 57 × 300 ≈ 17 000 tokens — a serious chunk of
a 128K window, permanently reserved.

## Proposal

Consolidate granular tools into **task-oriented compounds** with a
`mode` / `op` discriminator. Target: 57 → **~25 tools**.

### Consolidation map

| New compound | Replaces (5 → 1) | Mode param |
|---|---|---|
| `recall(mode=…)` | `recall`, `hybrid_search`, `find_similar`, `intent_classify`, `prefetch` | `bm25`, `hybrid`, `similar`, `intent`, `prefetch` |
| `graph(op=…)` | `graph_query`, `graph_stats`, `graph_add_edge`, `traverse_graph`, `propagate_staleness` | `query`, `stats`, `add_edge`, `traverse`, `propagate` |
| `core(op=…)` | `build_core`, `load_core`, `unload_core`, `list_cores` | `build`, `load`, `unload`, `list` |
| `verify(scope=…)` | `verify_chain`, `verify_merkle`, `mind_mem_verify` | `chain`, `merkle`, `full` |
| `staged_change(phase=…)` | `propose_update`, `approve_apply`, `rollback_proposal` | `propose`, `apply`, `rollback` |
| `block(op=…)` | `get_block`, `delete_memory_item`, `find_similar` | `get`, `delete`, `similar` |
| `observe(op=…)` | `observe_signal`, `signal_stats` | `record`, `stats` |
| `vault(op=…)` | `vault_scan`, `vault_sync` | `scan`, `sync` |

**Keep as-is** (no natural consolidation):
- `scan`, `list_contradictions`, `category_summary`, `index_stats`,
  `reindex`, `export_memory`, `memory_evolution`, `memory_health`,
  `stream_status`, `dream_cycle`, `plan_consolidation`,
  `pack_recall_budget`, `retrieval_diagnostics`, `project_profile`,
  `ontology_load`, `ontology_validate`, `list_evidence`,
  `calibration_feedback`, `calibration_stats`, `stale_blocks`,
  `agent_inject`, `governance_health_bench`, `encrypt_file`,
  `decrypt_file`, `list_mind_kernels`, `get_mind_kernel`,
  `compiled_truth_*` (3 tools)

## Migration

**Two-phase rollout:**

1. **v3.0**: Introduce the compound tools alongside the granular
   ones. Granular tools gain a `deprecated: true` flag in their
   docstring + a `Deprecated` tag in the tool registry. No behavior
   change — old clients still work.

2. **v4.0**: Remove the deprecated granular tools. Hard break.

Between v3.0 and v4.0 every agent integration updates its prompts.
The mind-mem-7b model needs **one retrain** after v3.0 ships so it
learns both surfaces (new compounds + deprecated granulars) and is
biased toward the compounds.

## Schema design

Every compound tool follows the shape:

```python
@mcp.tool
def recall(
    query: str,
    mode: str = "hybrid",  # bm25 | hybrid | similar | intent | prefetch
    limit: int = 10,
    **kwargs: Any,
) -> str:
    """Dispatch to the right recall backend based on mode."""
```

The dispatcher is a thin switch inside `mcp_server.py` that routes
to the existing granular implementations — no logic duplicated.

## Open questions

1. Should `staged_change` also absorb `list_contradictions` (which
   informs the propose phase)? Argument for: it's the natural place
   to ask "what needs resolving?". Argument against: it's a
   read-only query that doesn't mutate state.

2. Do we keep `scan` as a top-level tool or roll it under
   `governance(op=…)` alongside `list_contradictions` +
   `list_evidence`? The governance compound would net 4 tools → 1.

3. Do we preserve the `agent_inject` name or rename to
   `inject_context`? Legacy MCP clients reference the exact name.

## Plan

| Step | Owner | Deliverable |
|---|---|---|
| Design doc review | STARGA | This file, signed off |
| Implement compound tools | mind-mem | 8 new tools, each routing to existing impls |
| Mark granular tools deprecated | mind-mem | docstring + registry flag |
| Update docs | mind-mem | MCP reference page |
| Retrain mind-mem-7b | mind-mem | new adapter on HF |
| Hard deprecation cycle | mind-mem | 3-month notice before v4.0 removes them |

Estimated effort: **1 week implementation + 4 hrs retrain**.
