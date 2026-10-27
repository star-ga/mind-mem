# API Reference

## Python API

### Core Functions

#### `recall(workspace: str, query: str, limit: int = 10, **kwargs) -> list[dict]`
Search blocks using BM25F scoring with graph boost, fact card aggregation, and knee cutoff.

**Parameters:**
- `workspace` — Path to the MIND-Mem workspace
- `query` — Search query (supports stemming and domain-aware expansion)
- `limit` — Maximum results (default: 10)
- `active_only` — Only return blocks with `Status: active` (default: False)
- `include_pending` — Include pending signals in results (default: False)

**Returns:** List of dicts with `_id`, `score`, `Statement`, `Type`, and matched fields.

#### `init_workspace(path: str) -> None`
Initialize a new MIND-Mem workspace with the standard directory structure.

#### `parse_blocks(workspace: str) -> list[dict]`
Parse all memory blocks from `.md` files in a workspace.

### Intent Drift Baselines (`mind_mem.baseline_snapshot`)

Intent-distribution drift detection is a Python API, not an MCP tool.
`freeze_baseline(workspace, ...)` captures a versioned snapshot of the intent
classification distribution, `detect_drift(workspace, ...)` compares the
current state against a frozen baseline with a chi-squared test,
`compare_baselines(workspace, v1, v2)` diffs two snapshots, and
`list_baselines(workspace)` enumerates them.

### Client Installation (`mind_mem.hook_installer`)

v3.1.0 added programmatic access to the hook + MCP installers. Use these when embedding MIND-Mem into bootstrap scripts, package installers, or custom orchestrators.

#### `install_config(agent: str, workspace: str, *, force: bool = False, dry_run: bool = False) -> dict`
Write the text-hook configuration for `agent` into its client-specific
config file (e.g., `~/.cursor/rules/mind-mem.mdc`,
`~/.codex/AGENTS.md`). Idempotent; re-running updates the `# mind-mem`
marker block. Returns `{agent, path, written, skipped, reason}`.

#### `install_mcp_config(agent: str, workspace: str, *, force: bool = False, dry_run: bool = False) -> dict`
**(New in v3.1.0)** Write the native MCP server entry for `agent` into
its MCP config file. Supports 41 MCP-aware clients across TOML
(`[mcp_servers.mind-mem]`, `[[mcp_servers]]`), YAML (Goose `extensions`,
Hermes `mcp_servers`, Continue `mcpServers` list)
and JSON dialects (`mcpServers`, VS Code `servers`, Zed `context_servers`,
OpenClaw `mcp.servers`, Amp `"amp.mcpServers"`, OpenCode-style `mcp`); the
per-client table is in `docs/client-integrations.md`. Only the mind-mem entry
changes; further documented locations that exist are written too and returned
under `also`. Returns `{agent, path, written, merged, skipped, reason}`.
Skipped with `reason="agent does not support native MCP"` for clients
without MCP, and `reason="manual setup required"` (steps in `content`) for
clients whose MCP list lives in their UI.

#### `uninstall_mcp_config(agent: str, workspace: str, *, dry_run: bool = False) -> dict`
Remove only the `mind-mem` entry from `agent`'s MCP config file(s); other
servers and settings are kept and a changed file is backed up first. Returns
`{agent, path, written, removed, reason}`. `uninstall_all(workspace, *,
agents=None, dry_run=False)` applies it to every detected or named client
(CLI: `mm uninstall-all`).

#### `install_all(workspace: str, *, agents: list[str] | None = None, include_mcp: bool = True, force: bool = False, dry_run: bool = False) -> list[dict]`
**(Updated in v3.1.0)** Install configs for every detected agent. When
`include_mcp=True` (default), emits BOTH the hook phase AND the MCP
phase per agent — returning 2 results per MCP-aware agent. Pass
`include_mcp=False` to write only the hook configs.

#### `detect_installed_agents(workspace: str) -> list[str]`
Scan `PATH` plus known config directories to determine which AI
clients are installed on the current machine. Returns a list of
recognised agent names. Used by the `mm detect` CLI.

#### `mcp_server_spec(workspace: str) -> dict`
**(New in v3.1.0)** Return the canonical MCP server command+env dict
used by every `install_mcp_config` writer. Shape:
`{"command": "python3", "args": [...], "env": {"MIND_MEM_WORKSPACE": ws}}`.
Useful if you're writing a custom MCP config writer.

---

## MCP Server (107 tools, 8 resources)

The MCP server exposes 107 distinct tools via JSON-RPC. See [MCP Tool Examples](mcp-tool-examples.md) and [MCP Integration Guide](mcp-integration.md).

### Starting the Server

```bash
MIND_MEM_WORKSPACE=/path/to/workspace mind-mem-mcp
```

The workspace is read from `MIND_MEM_WORKSPACE`; the server has no
`--workspace` flag. From a source checkout, `python3 mcp_server.py` is
equivalent. HTTP transport: `--transport http --port 8765` (token via
`MIND_MEM_TOKEN`).

### Tool Reference

#### Retrieval

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `recall` | BM25/hybrid search with graph boost and knee cutoff; with `explain=True` returns step-by-step reasoning | `query`, `limit`, `active_only`, `backend`, `explain` |
| `hybrid_search` | BM25+Vector RRF fusion (wrapper for `recall(backend="hybrid")`); with `explain=True` includes fusion rationale | `query`, `limit`, `active_only`, `explain` |
| `find_similar` | Vector similarity search from a block ID | `block_id`, `limit` |
| `intent_classify` | Classify query intent (WHY/WHEN/ENTITY/WHAT/HOW/LIST/VERIFY/COMPARE/TRACE) | `query` |
| `prefetch` | Pre-assemble context from conversation signals | `signals`, `limit` |
| `chat_with_memory` | Answer from admitted memory with structural citation checks; `semantic_required=True` abstains while no entailment verifier is available | `question`, `limit`, `generator`, `semantic_required` |
| `retrieval_diagnostics` | Pipeline diagnostics: stage rejection rates, intent distribution | `last_n`, `max_age_days` |

Chat and graph answers expose `semantic_verification: "not_established"`.
Their `grounded` field describes structural citation checks, not proof that
every sentence follows from its evidence. Python chat and graph callers can
require semantic verification with `semantic_required=True`; the graph CLI
offers `mm graph-answer ENTITY --semantic-required --json`. The current runtime
refuses the requested answer before generation when that capability is missing.

#### Memory Management

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `propose_update` | Propose a decision or task; both require a caller-supplied rationale with at least eight non-whitespace characters (writes to SIGNALS.md) | `block_type`, `statement`, `rationale`, `tags`, `confidence` |
| `propose_slot_update` | Stage a value for an explicitly declared closed slot; admin scope, separate approval required | `namespace`, `slot`, `value`, `rationale`, optional provenance fields |
| `approve_apply` | Apply a staged proposal with contradiction check (dry_run default) | `proposal_id`, `dry_run` |
| `rollback_proposal` | Rollback an applied proposal by receipt timestamp | `receipt_ts` |
| `delete_memory_item` | Delete a block by ID from its source file | `block_id` |
| `export_memory` | Export all blocks as JSONL | `format`, `include_metadata` |
| `memory_evolution` | A-MEM metadata: importance, access patterns, keywords | `block_id`, `action` |

`propose_slot_update` requires a workspace `closed_slots` declaration. Unknown
slot names are refused; a changed occupied slot stages a superseding decision,
and reasserting the same value leaves the corpus unchanged. Approval rechecks
the current declaration, occupancy and configured write policies. See the
[closed-slot contract](design/m4-closed-set-slots-implementation.md) for the
configuration, provenance fields and refusal cases. Slot namespaces name sets
of fields; they do not grant access to retrieval namespaces.

#### Workspace Operations

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `scan` | Integrity scan: contradictions, drift, dead decisions | — |
| `list_contradictions` | Detected contradictions with resolution strategies | — |
| `reindex` | Rebuild FTS5 + optional vector index | `include_vectors` |
| `index_stats` | Block counts, staleness, vector coverage, kernel status | — |
| `category_summary` | Category-based topic summaries | `topic`, `limit` |

#### Quality Gates & Lineage (v3.11.0+)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `validate_block` | Deterministic quality gate over candidate block text, honouring the workspace's `quality_gate.mode` (the same verdict `propose_update` would reach) | `text`, `strict`, `force` |
| `block_lineage` | Breadth-first traversal of the typed lineage graph rooted at a block (cites/implements/refines/contradicts/cooccurrence/supersedes) | `block_id`, `max_depth` (1-3), `kind_filter`, `node_cap` |
| `add_block_edge` | Record a typed lineage edge between two blocks; re-adding the same triple bumps its hit count | `src`, `dst`, `kind` (`cites`/`implements`/`refines`/`contradicts`/`cooccurrence`/`supersedes`), `weight` |

#### Corpus Lint (`v4.lint`, default OFF)

Post-storage scan of the governed corpus, and the one governed way to repair
what it finds. Complementary to `validate_block`, not a replacement: the
quality gate filters a *candidate* before it is staged, the lint scans blocks
that are already stored, and the two share no rule. Both tools return a
structured `v4.lint is disabled` payload until the flag is set in
`mind-mem.json`.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `lint` | Report deterministic corpus defects — `stale_date`, `missing_metadata`, `duplicate_block` — each with a stable content-addressed `LF-xxxxxxxx` id. Read-only | `rule` (optional single rule) |
| `lint_autofix` | Stage a repair **proposal** for one finding. Never writes the corpus; `approve_apply` remains the only thing that changes a block (ADMIN) | `finding_id` |

#### MIND Kernels

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `list_mind_kernels` | List available `.mind` kernel configs | — |
| `get_mind_kernel` | Read a specific kernel configuration | `name` |

#### Audit & Evidence

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `verify_merkle` | Verify a block's Merkle inclusion against the live tree | `block_id`, `content_hash` |
| `verify_chain` | Verify every hash-linked ledger as one verdict — `hash_chain`, `evidence_chain`, `audit_sidecar`, `served_ledger` — by delegating to the single verifier (`mind-mem-verify`). `valid` is false when any one of them fails; `workspace_valid` additionally folds in the non-ledger checks (`spec_binding`, `snapshot_anchor`), and each check returns its own row. Reads only: it creates nothing in the workspace | — |
| `mind_mem_verify` | Verify a workspace (optionally against a snapshot) | `snapshot` |
| `list_evidence` | List evidence-chain records | filter parameters |
| `anchor_root` | Record the CURRENT Merkle root in the local external-anchor trail. Computes the root from the live index — it takes no root argument, because anchoring a caller-supplied root would anchor the caller's claim rather than the store's state. Records `status="pending"` when no external poster is wired (ADMIN) | `chain`, `tx_hash`, `block_height` |
| `anchor_history` | List recorded external anchors, newest last, with any integrity `problems` — a damaged trail line is reported, never silently skipped | `limit` |

#### Entity Observations

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `entity_observations` | Return an entity's accreted observation facts (read-only) | `entity` |
| `entity_add_observation` | Append an accreted fact to an entity's observation list (ADMIN) | `entity`, observation fields |

#### Typed-Edge Proposals

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `list_edge_proposals` | List typed-edge proposals, optionally filtered by status | `status` |
| `reject_edge` | Reject a staged typed-edge proposal. No edge is written (ADMIN) | `proposal_id` |

#### Governed Entity Equivalence

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `propose_entity_merge` | Stage a reversible equivalence between existing entity IDs; requires at least eight non-whitespace rationale characters | `winner_id`, `loser_id`, `rationale` |
| `list_entity_merge_proposals` | List staged, applied or reversed merge proposals | `status`, `limit` |
| `approve_entity_merge` | Approve one staged `SAME_AS` assertion (ADMIN) | `proposal_id` |
| `reverse_entity_merge` | Retract the approved equivalence edge (ADMIN) | `proposal_id` |
| `graph_query` | Traverse existing graph relationships; optionally resolve equivalence at each traversal level | `entity`, `depth`, `predicate`, `direction`, `limit`, `resolve_same_as` |

Merging preserves both entity IDs, aliases, observations and original source
edges. `graph_query(resolve_same_as=True)` opts into the derived equivalent
neighborhood; reversal retracts the specific assertion. Generic edge-write
tools cannot create `SAME_AS` relations. The [design note](design/ra4-entity-equivalence.md)
describes the governance and persistence boundaries. Readers of a graph with
this new predicate must support entity equivalence; older readers are not
assumed compatible. Automatic resolution and REST merge routes are separate work.

#### Arch-MIND Governance Wrappers

Thin wrappers over the external `arch-mind` binary, which scores a repository's
structural metrics and chains each session into a tamper-evident log. **All
seven are ADMIN-scope**, including the read-only ones: they shell out against an
absolute path on the host, so they are a directory-listing and file-existence
oracle outside the workspace.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `arch_baseline` | Initialise the Arch-MIND store at a repo with a baseline event | `repo` |
| `arch_session_start` | Open an Arch-MIND session evidence node | `repo` |
| `arch_session_end` | Close the open Arch-MIND session at a repo | `repo` |
| `arch_delta` | Per-metric delta between two Arch-MIND baselines | `repo`, baseline ids |
| `arch_check_rules` | Apply a `rules.mind` to a fresh Arch-MIND scan | `repo`, rules path |
| `arch_history` | List events in the Arch-MIND store at a repo | `repo` |
| `arch_metric_explain` | Per-metric human-readable breakdown for a fixture | `metric` |

### Complete Tool Index

The tables above describe tools by task. This index lists every registered
tool and is regenerated from the server's registry by
`python3 scripts/check_tool_surface.py --write-api-index`;
`--check-doc-names` (run in CI) fails when it is stale, when a registered tool
has no row, or when any row in this section names a tool that does not exist.

<!-- tool-index:begin (generated by scripts/check_tool_surface.py --write-api-index; do not edit by hand) -->
All 107 registered tools, generated from the live registry. `[param]` is optional; scope `admin` needs an admin-scoped token.

| Tool | Scope | Module | Parameters | Summary |
|------|-------|--------|------------|---------|
| `add_block_edge` | admin | `lineage` | `src`, `dst`, `kind`, `[weight]` | Record an explicit typed lineage edge from ``src`` to ``dst``. |
| `agent_inject` | user | `agent` | `query`, `[agent]`, `[limit]`, `[scoring_instant]` | Render a context snippet in the target agent's expected format. |
| `anchor_history` | user | `audit` | `[limit]` | List recorded external anchors, newest last, with integrity problems. |
| `anchor_root` | admin | `audit` | `[chain]`, `[tx_hash]`, `[block_height]` | Record the CURRENT Merkle root in the local external-anchor trail. |
| `approve_apply` | admin | `governance` | `proposal_id`, `[dry_run]` | Apply a staged proposal from intelligence/proposed/. |
| `approve_edge` | admin | `graph` | `proposal_id` | Commit a staged typed-edge proposal to the graph (ADMIN-scope). |
| `approve_entity_merge` | admin | `graph` | `proposal_id` | Approve a staged entity-equivalence proposal (admin-scope). |
| `arch_baseline` | admin | `arch_mind` | `repo`, `fixture` | Initialise the arch-mind store at *repo* with a baseline event. |
| `arch_check_rules` | admin | `arch_mind` | `repo`, `fixture`, `[rules]`, `[mode]` | Apply a ``rules.mind`` to a fresh arch-mind scan. |
| `arch_delta` | admin | `arch_mind` | `repo`, `before`, `after` | Compute the per-metric delta between two arch-mind baselines. |
| `arch_history` | admin | `arch_mind` | `repo`, `[days]`, `[verify]` | List events in the arch-mind store at *repo*. |
| `arch_metric_explain` | admin | `arch_mind` | `metric`, `fixture` | Per-metric human-readable breakdown for a given fixture. |
| `arch_session_end` | admin | `arch_mind` | `repo`, `fixture` | Close the open arch-mind session at *repo*. |
| `arch_session_start` | admin | `arch_mind` | `repo`, `fixture`, `agent_id`, `commit_sha` | Open an arch-mind session evidence node. |
| `audit_model_tool` | user | `model` | `path`, `[allow_publisher]`, `[include_manifest]` | Static security scan of a local model checkpoint. |
| `block_lineage` | user | `lineage` | `block_id`, `[max_depth]`, `[kind_filter]`, `[node_cap]` | BFS-traverse the block-lineage graph rooted at ``block_id``. |
| `build_core` | admin | `core` | `namespace`, `version`, `[filename]` | Build a .mmcore bundle from the active workspace's blocks. |
| `calibration_feedback` | admin | `calibration` | `query_id`, `[block_ids_useful]`, `[block_ids_not_useful]`, `[feedback_type]` | Record retrieval quality feedback for calibration. |
| `calibration_stats` | user | `calibration` | - | Report calibration health — per-block scores, per-query-type accuracy. |
| `category_summary` | user | `benchmark` | `topic`, `[limit]` | Returns category summaries for the ADMITTED blocks matching a topic. |
| `chat_with_memory` | user | `chat` | `question`, `[limit]`, `[generator]`, `[on_invalid]`, `[require_in_evidence]`, `[semantic_required]`, `[graph_seed]` | Answer *question* from workspace memory with verified citations. |
| `check_dead_ends` | user | `frames` | `[tool]`, `[command]`, `[intent]`, `[paths]` | Return the recorded failures that overlap an about-to-happen action. |
| `check_guardrails` | user | `guardrails` | `[tool]`, `[command]`, `[intent]`, `[paths]` | Return the guardrails that fire for an about-to-happen action. |
| `compact` | admin | `memory_ops` | `[dry_run]`, `[archive_days]`, `[signal_days]`, `[snapshot_days]` | Run workspace compaction — archive old blocks, clean snapshots, remove resolved signals. |
| `compile_truth_walkthrough` | user | `walkthrough_persona` | `topic`, `[limit]`, `[active_only]` | Return *topic* memories in dependency order (foundations → current). |
| `compiled_truth` | user | `public` | `action`, `[entity_id]`, `[observation]`, `[source]`, `[confidence]`, `[entity_type]` | Unified compiled-truth page lifecycle. |
| `compiled_truth_add_evidence` | admin | `kernels` | `entity_id`, `observation`, `[source]`, `[confidence]`, `[entity_type]` | Add evidence to a compiled truth page and auto-recompile. |
| `compiled_truth_contradictions` | user | `kernels` | `entity_id` | Detect contradictions in a compiled truth page. |
| `compiled_truth_load` | user | `kernels` | `entity_id` | Load a compiled truth page for an entity. |
| `core` | user | `public` | `action`, `[namespace]`, `[version]`, `[filename]`, `[verify]` | Unified .mmcore bundle lifecycle. |
| `decrypt_file` | admin | `encryption` | `file_path` | Return plaintext bytes (base64-encoded) for an encrypted file. |
| `delete_memory_item` | admin | `memory_ops` | `block_id`, `[rationale]` | Delete a block by ID from the workspace's blocks of record (admin-scope). |
| `dream_cycle` | user | `consolidation` | `[auto_repair]`, `[lookback_days]`, `[stale_days]` | Run the dream cycle — autonomous memory enrichment. |
| `encrypt_file` | admin | `encryption` | `file_path` | Encrypt a single workspace file at rest. |
| `entity_add_observation` | admin | `graph` | `entity`, `fact` | Append an accreted fact to an entity's observation list (ADMIN-scope). |
| `entity_observations` | user | `graph` | `entity` | Return an entity's accreted observation facts (user-scope, read-only). |
| `export_core` | admin | `core` | `name`, `[format]` | Render a .mmcore bundle into a static interchange format. |
| `export_memory` | admin | `memory_ops` | `[format]`, `[include_metadata]`, `[max_blocks]` | Export every ADMITTED workspace block as JSONL. |
| `find_similar` | user | `recall` | `block_id`, `[limit]`, `[kind]` | Find blocks co-retrieved with a given block (co-occurrence, not embeddings). |
| `get_block` | user | `memory_ops` | `block_id`, `[namespace]` | Retrieve a single ADMITTED block by its ID, with full content. |
| `get_mind_kernel` | user | `kernels` | `name` | Read a specific .mind kernel configuration as structured JSON. |
| `governance_health_bench` | admin | `benchmark` | - | Run the governance health benchmark suite. |
| `graph` | user | `public` | `action`, `[subject]`, `[predicate]`, `[object]`, `[source_block_id]`, `[confidence]`, `[entity]`, `[depth]`, `[direction]`, `[limit]`, `[block_id]` | Unified knowledge/causal graph entry point. |
| `graph_add_edge` | admin | `graph` | `subject`, `predicate`, `object`, `source_block_id`, `[confidence]` | Record a typed relationship in the knowledge graph — DIRECT admin write. |
| `graph_query` | user | `graph` | `entity`, `[depth]`, `[predicate]`, `[direction]`, `[limit]`, `[resolve_same_as]` | N-hop traversal from *entity*. |
| `graph_stats` | user | `graph` | - | Aggregated knowledge-graph stats for the active workspace. |
| `hybrid_search` | user | `recall` | `query`, `[limit]`, `[active_only]`, `[explain]`, `[scoring_instant]`, `[intended_use]` | Hybrid BM25+Vector recall with RRF fusion. |
| `index_stats` | user | `memory_ops` | - | Block counts, index staleness, vector coverage, and MIND kernel status. |
| `intent_classify` | user | `recall` | `query` | Show the routing strategy for a query. |
| `kernels` | user | `public` | `[action]`, `[name]` | Unified .mind kernel read surface. |
| `lint` | user | `lint` | `[rule]` | Report deterministic corpus defects (read-only, USER-scope). |
| `lint_autofix` | admin | `lint` | `finding_id` | Stage a repair PROPOSAL for one lint finding (ADMIN-scope). |
| `list_contradictions` | user | `governance` | - | List detected contradictions with resolution analysis. |
| `list_cores` | user | `core` | - | List every currently-loaded .mmcore bundle (namespace + stats). |
| `list_edge_proposals` | user | `graph` | `[status]`, `[limit]` | List typed-edge proposals, optionally filtered by status (user-scope). |
| `list_entity_merge_proposals` | user | `graph` | `[status]`, `[limit]` | List governed entity-equivalence proposals (user-scope). |
| `list_evidence` | user | `audit` | `[block_id]`, `[action]`, `[limit]` | List governance evidence objects, optionally filtered by block_id or action. |
| `list_mind_kernels` | user | `kernels` | - | List available .mind kernel configuration files. |
| `load_core` | admin | `core` | `filename`, `[verify]` | Load a .mmcore bundle from the workspace's cores/ directory. |
| `memory_evolution` | user | `governance` | `block_id`, `[action]` | A-MEM metadata for a block — importance, access patterns, keywords. |
| `memory_health` | user | `memory_ops` | - | Deep health dashboard for the memory workspace. |
| `memory_verify` | admin | `public` | `[mode]`, `[block_id]`, `[content_hash]`, `[snapshot]` | Unified memory-integrity verification. |
| `mic_convert_tool` | user | `mic_map` | `input`, `[input_format]`, `[output_format]` | Convert a MIND IR graph between mic@2 (text) and mic-b (binary). |
| `mic_inspect_tool` | user | `mic_map` | `input`, `[input_format]` | Return a structural summary of a MIC/MAP payload without re-emitting it. |
| `mind_mem_verify` | user | `audit` | `[snapshot]` | Run the standalone `mind-mem-verify` CLI against the current workspace. |
| `observe_signal` | user | `signal` | `session_id`, `previous_query`, `new_query`, `[previous_results]` | Capture a re-query / refinement / correction feedback signal. |
| `ontology_load` | admin | `ontology` | `spec`, `[make_active]` | Load an ontology from an inline JSON spec. |
| `ontology_validate` | user | `ontology` | `block`, `type_name`, `[strict]` | Validate *block* (JSON object) against the active ontology. |
| `outcome_stats` | user | `calibration` | `[block_id]`, `[task_id]`, `[top_n]` | Query recorded outcomes — which memories actually earned their keep. |
| `pack_recall_budget` | user | `recall` | `query`, `[max_tokens]`, `[limit]`, `[scoring_instant]`, `[model]` | Run a recall, then pack the result list under a token budget. |
| `pipeline_status` | user | `pipeline` | - | Return current pipeline hash + dirty-block summary. |
| `plan_consolidation` | user | `consolidation` | `[importance_threshold]`, `[stale_days]`, `[archive_after_days]`, `[grace_days]`, `[maturity_gate]`, `[min_maturity]` | Dry-run the cognitive forgetting cycle. |
| `prefetch` | user | `recall` | `signals`, `[limit]` | Pre-assembles likely-needed context from recent conversation signals. |
| `project_profile` | user | `consolidation` | `[name]`, `[top_k]` | Auto-generate a structured project intelligence profile. |
| `propagate_staleness` | user | `consolidation` | `seed_block_ids`, `[max_hops]` | Diffuse staleness outward from seed blocks over the xref graph. |
| `propose_edge` | user | `graph` | `subject`, `predicate`, `object`, `source_block_id`, `[confidence]` | Stage a typed knowledge-graph edge as an HITL proposal (user-scope). |
| `propose_entity_merge` | user | `graph` | `winner_id`, `loser_id`, `rationale` | Stage an explicit winner/loser ``SAME_AS`` proposal (user-scope). |
| `propose_slot_update` | admin | `governance` | `namespace`, `slot`, `value`, `rationale`, `[actor_id]`, `[actor_role]`, `[session_id]`, `[tool_id]`, `[purpose]` | Stage a closed-slot update for review by ``approve_apply``. |
| `propose_update` | admin | `governance` | `block_type`, `statement`, `[rationale]`, `[tags]`, `[confidence]`, `[actor_id]`, `[actor_role]`, `[session_id]`, `[tool_id]`, `[purpose]`, `[content_source]`, `[supersedes]` | Propose a new decision or task. |
| `recall` | user | `public` | `query`, `[mode]`, `[limit]`, `[active_only]`, `[backend]`, `[block_id]`, `[axes]`, `[weights]`, `[max_tokens]`, `[signals]`, `[scoring_instant]`, `[explain]`, `[since]`, `[until]`, `[lifecycle]`, `[event_id]`, `[min_maturity]`, `[intended_use]` | Unified retrieval entry point. |
| `recall_with_axis` | user | `recall` | `query`, `[axes]`, `[weights]`, `[limit]`, `[active_only]`, `[adversarial]`, `[allow_rotation]`, `[scoring_instant]` | Axis-aware recall under the Observer-Dependent Cognition model. |
| `recall_with_guardrails` | user | `guardrails` | `query`, `[tool]`, `[command]`, `[intent]`, `[paths]`, `[limit]`, `[active_only]` | Recall *query*, with the constraints for this context surfaced first. |
| `recall_with_persona` | user | `walkthrough_persona` | `query`, `[persona]`, `[limit]`, `[active_only]`, `[scoring_instant]` | Recall memories and project the result list through *persona*. |
| `reindex` | admin | `memory_ops` | `[include_vectors]` | Trigger FTS index rebuild, optionally with vector indexing. |
| `reindex_dirty` | user | `pipeline` | `[limit]`, `[dry_run]` | Re-stamp blocks whose pipeline hash is stale. |
| `reject_edge` | admin | `graph` | `proposal_id` | Reject a staged typed-edge proposal (ADMIN-scope). |
| `reject_proposal` | admin | `governance` | `proposal_id`, `reason` | Reject a staged proposal explicitly, preserving the rationale. |
| `report_outcome` | admin | `calibration` | `[block_ids]`, `[outcome]`, `[query_id]`, `[run_id]`, `[task_id]`, `[actor_id]`, `[session_id]`, `[tool_id]`, `[evidence]`, `[project_to_calibration]` | Report whether recalled blocks actually *helped* — utility, not relevance. |
| `resume_brief` | user | `frames` | `[frame_id]` | Return the resume brief for the active task frame. |
| `retrieval_diagnostics` | user | `recall` | `[last_n]`, `[max_age_days]` | Pipeline diagnostics: per-stage rejection rates, intent distribution, and hard negative summary. |
| `reverse_entity_merge` | admin | `graph` | `proposal_id` | Retract an approved entity-equivalence edge (admin-scope). |
| `rollback_proposal` | admin | `governance` | `receipt_ts`, `[reason]` | Rollback an applied proposal using its receipt timestamp. |
| `scan` | user | `governance` | - | Run integrity scan — contradictions, drift, dead decisions, impact graph. |
| `sign_model_tool` | admin | `model` | `path`, `[key_file]`, `[generate_key_prefix]`, `[write_sidecars]` | Sign every file in a local model checkpoint with Ed25519. |
| `signal_stats` | user | `signal` | - | Return aggregated interaction-signal counts for the workspace. |
| `similar_trajectories` | user | `trajectory` | `task`, `[tools]`, `[outcome]`, `[limit]`, `[scoring_instant]` | Recall past task executions similar to the one you are about to attempt. |
| `staged_change` | admin | `public` | `phase`, `[block_type]`, `[statement]`, `[rationale]`, `[tags]`, `[confidence]`, `[proposal_id]`, `[dry_run]`, `[receipt_ts]` | Single entry point for the propose / apply / rollback flow. |
| `stale_blocks` | user | `memory_ops` | `[limit]`, `[clear_block_id]` | List blocks flagged as stale, or clear a staleness flag. |
| `stream_status` | user | `agent` | - | Current change-stream publish / delivery / drop counters. |
| `traverse_graph` | user | `graph` | `block_id`, `[depth]`, `[direction]` | Navigate the causal dependency graph from a block. |
| `unload_core` | admin | `core` | `namespace` | Unload a previously-loaded core by namespace. |
| `validate_block` | user | `quality` | `text`, `[strict]`, `[force]` | Run the deterministic quality gate against ``text``. |
| `vault_scan` | user | `agent` | `vault_root`, `[sync_dirs]` | Walk an Obsidian-style vault and return parsed VaultBlocks (JSON). |
| `vault_sync` | admin | `agent` | `vault_root`, `block_id`, `relative_path`, `body`, `[block_type]`, `[title]`, `[overwrite]`, `[include_links]` | Write a single block back into a vault at a relative path. |
| `verify_chain` | admin | `audit` | - | Verify every hash-linked ledger in the workspace, as one verdict. |
| `verify_merkle` | user | `audit` | `block_id`, `content_hash` | Verify a block's Merkle inclusion against the live tree. |
| `verify_model_tool` | user | `model` | `path`, `[pubkey_path]` | Verify a previously-signed checkpoint. |
<!-- tool-index:end -->

### Configuration

The server reads `mind-mem.json` from the workspace root:

```json
{
  "recall": {
    "top_k": 18,
    "min_score": 0.1,
    "knee_cutoff": true
  },
  "auto_ingest": {
    "enabled": true,
    "transcript_scan": true,
    "entity_ingest": true
  }
}
```
