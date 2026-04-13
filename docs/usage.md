# Usage

How to actually drive mind-mem from every surface the package
provides. See [setup.md](setup.md) first.

## Surfaces at a glance

| Surface | When to use |
|---|---|
| **MCP server** | Claude Code, OpenClaw, any MCP agent. 54 tools. |
| **`mm` CLI** | Non-MCP agents: codex, gemini CLI, Cursor, Windsurf, Aider, plain shell. |
| **`mind-mem-verify` CLI** | Third-party integrity audits. Standalone; no network. |
| **Python library** | Direct import from your own code. |

## MCP tool index (54 tools)

### Retrieval
- `recall(query, limit, active_only, backend)` — BM25 / hybrid / auto
- `recall_with_axis(query, axes, weights, …)` — Observer-Dependent Cognition
- `hybrid_search(query)` — deprecated; use `recall(backend="hybrid")`
- `find_similar(block_id)` — nearest neighbours
- `prefetch(query)` — warm blocks for a likely next hop

### Writing / governance
- `propose_update(type, statement, rationale, tags)` — draft a proposal
- `approve_apply(proposal_id)` — admin scope
- `rollback_proposal(proposal_id)` — admin scope
- `delete_memory_item(block_id)` — admin scope
- `reindex()` — rebuild FTS + vector indexes

### Integrity / audit
- `verify_chain()` — walk the SHA3-512 hash chain
- `list_evidence(block_id, action)` — governance evidence
- `verify_merkle(block_id, content_hash)` — Merkle inclusion proof
- `mind_mem_verify(snapshot)` — run the standalone verifier

### Observability
- `index_stats()` — block counts + cache stats + prefetch stats + MRS
- `memory_health()` — workspace health
- `retrieval_diagnostics(query)` — per-axis score breakdown
- `category_summary(category)` — per-category digest

### Knowledge graph (v2.2)
- `graph_add_edge(subject, predicate, object, source_block_id)`
- `graph_query(entity, depth, predicate, direction, limit)`
- `graph_stats()`

### Context cores (v2.3)
- `build_core(namespace, version)` — snapshot current workspace
- `load_core(filename)` — mount a core
- `unload_core(namespace)` — unmount
- `list_cores()` — active cores + stats

### Cognitive memory (v2.4)
- `plan_consolidation(importance_threshold, stale_days, archive_after_days, grace_days)` — dry-run the forget cycle
- `pack_recall_budget(query, max_tokens, limit)` — recall + token-budget packing

### Ontology + streaming (v2.5)
- `ontology_load(spec, make_active)` — register an ontology
- `ontology_validate(block, type_name, strict)` — validate a block

### Competitive intel (v2.6)
- `propagate_staleness(seed_block_ids, max_hops)`
- `project_profile(name, top_k)`
- `stream_status()` — change stream stats

### Agent bridge (v2.7)
- `agent_inject(query, agent, limit)` — render context for any CLI
- `vault_scan(vault_root, sync_dirs)`
- `vault_sync(vault_root, block_id, relative_path, body, …)`

### Interaction signals (v2.1)
- `observe_signal(session_id, previous_query, new_query, previous_results)`
- `signal_stats()`

---

## `mm` CLI

```bash
# Search memory
mm recall "jwt auth"

# Token-budgeted context snippet (for prompt injection)
mm context "jwt auth" --max-tokens 2000

# Pre-rendered snippet for a specific agent
mm inject --agent codex "jwt auth"
mm inject --agent gemini "jwt auth" > /tmp/gem.md
mm inject --agent claude-code "jwt auth"

# Status of current workspace
mm status

# Vault sync
mm vault scan /path/to/obsidian/vault
mm vault scan /path/to/vault --sync-dirs decisions entities
mm vault write /path/to/vault entities/Alice.md \
  --id E-ALICE --type entity --title Alice --body "Engineer."
```

### Shell integration

```bash
# .bashrc / .zshrc
export MIND_MEM_WORKSPACE="$HOME/.openclaw/workspace"

# Alias example: inject memory before invoking codex
codex-mem() {
  mm inject --agent codex --quiet "$*" > /tmp/codex-ctx.md
  codex --context /tmp/codex-ctx.md "$@"
}
```

### Pre-session vs post-session

```bash
# Before a session: generate context
mm context "the feature I'm about to work on" > /tmp/ctx.md

# During a session: your agent posts tool outputs back into mind-mem
# via the hook auto-capture (see §Hook installer below)

# After a session: summarise and store
cat session-transcript.txt | mm capture --stdin
```

## `mind-mem-verify` CLI

Standalone integrity audit — reads snapshots + evidence only, no
network, no writes.

```bash
# Verify the workspace in CWD
mind-mem-verify .

# Verify a snapshot against the live chain
mind-mem-verify . --snapshot snapshots/snap-2026-04-13

# JSON output for CI integration
mind-mem-verify . --json
```

Exit codes: 0 ok / 1 generic / 2 chain / 3 spec / 4 evidence /
5 merkle / 6 snapshot. Use these to gate CI promotions.

## Hook installer

```python
from mind_mem.hook_installer import install_config

# Preview what would be written
install_config("claude-code", "/home/n/ws", dry_run=True)

# Actually write the config
install_config("codex", "/home/n/ws")
install_config("gemini", "/home/n/ws")
install_config("cursor", "/home/n/ws")
install_config("windsurf", "/home/n/ws")
install_config("aider", "/home/n/ws")
```

Each agent gets the right config file name (`~/.claude/settings.json`,
`AGENTS.md`, `GEMINI.md`, `.cursorrules`, `.windsurfrules`,
`.aider.conf.yml`).

## Python library

```python
from mind_mem.recall import recall
from mind_mem.axis_recall import recall_with_axis
from mind_mem.observation_axis import AxisWeights, ObservationAxis
from mind_mem.cognitive_forget import pack_to_budget, plan_consolidation
from mind_mem.knowledge_graph import KnowledgeGraph, Predicate
from mind_mem.context_core import build_core, load_core
from mind_mem.verify_cli import verify_workspace
from mind_mem.interaction_signals import SignalStore

# Plain recall
results = recall("/ws", "jwt", limit=5)

# Axis-aware recall
weights = AxisWeights.uniform([ObservationAxis.TEMPORAL, ObservationAxis.ENTITY_GRAPH])
packed = recall_with_axis("/ws", "who decided jwt", weights=weights)

# Token budget
pb = pack_to_budget(results, max_tokens=2000)

# Knowledge graph
with KnowledgeGraph("/ws/memory/knowledge_graph.db") as kg:
    kg.add_edge("Alice", Predicate.AUTHORED_BY, "Project X", source_block_id="D-001")
    neighbours = kg.neighbors("Alice", depth=2)

# Context core
manifest = build_core("/tmp/foo.mmcore", namespace="foo", version="1.0")
loaded = load_core("/tmp/foo.mmcore")

# Integrity
report = verify_workspace("/ws")
print(report.as_dict())

# Interaction signals
store = SignalStore("/ws/memory/interaction_signals.jsonl")
store.observe_pair(
    session_id="s1",
    previous_query="jwt",
    new_query="jwt rotation",
    previous_results=["D-001"],
)
```

## Proprietary code protection

mind-mem is a **public Apache-2.0 package**. It has **no proprietary
code**. Modules that CAN use proprietary accelerators (the
MIND-compiled `.so` for hot paths) do so via the
`MIND_MEM_KERNELS_SO` env var — when the library is present it's
loaded via `ctypes`; when absent the pure-Python fallback in
`mind_mem.mind_kernels` is used.

See `mind_mem.mind_kernels.load_kernels()` for the exact contract.
STARGA's proprietary `libmindmem_kernels.so` (built from
the separately-distributed accelerator source) implements the same symbol set with all
mind-runtime protections (optional native accelerator; opt-in
with pure-Python fallback when
the accelerator library is distributed separately and is not part
of this public repo.

## Worked example

```bash
# 1. init
mind-mem-init /tmp/demo
cd /tmp/demo
export MIND_MEM_WORKSPACE=/tmp/demo

# 2. add a couple of blocks
cat > decisions/auth.md <<'EOF'
[D-AUTH-001]
Statement: Use OAuth2 with PKCE
Status: active
Date: 2026-04-13
EOF

# 3. build the index
python3 -m mind_mem.intel_scan

# 4. recall
mm recall "oauth"

# 5. verify
mind-mem-verify .

# 6. snapshot + share
python3 -c "
from mind_mem.context_core import build_core
from mind_mem.sqlite_index import merkle_leaves
leaves = merkle_leaves('.')
blocks = [{'_id': bid, 'content_hash': h} for bid, h in leaves]
build_core('/tmp/demo.mmcore', namespace='demo', version='1.0', blocks=blocks)
"

# 7. load elsewhere
python3 -c "
from mind_mem.context_core import load_core
c = load_core('/tmp/demo.mmcore')
print(c.manifest.as_dict())
"
```
