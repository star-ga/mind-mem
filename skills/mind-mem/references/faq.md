# FAQ

### What is mind-mem?

Persistent, auditable memory for coding agents. Decisions, tasks, entities
and daily notes are stored as structured Markdown blocks in a workspace, an
index makes them searchable, and every change to active memory goes through a
governed proposal-and-review path with an evidence trail. Agents reach it
through the `mm` CLI or the MCP server. It is Apache-2.0 licensed.

### Where is my memory stored?

In the workspace directory: `decisions/DECISIONS.md`, `tasks/TASKS.md`,
`entities/*.md`, daily logs in `memory/<YYYY-MM-DD>.md`, scan output and
proposals in `intelligence/`, and the search index in `.mind-mem-index/`.
`mm status` prints the workspace path. The `postgres` block-store backend
keeps blocks in Postgres instead; `encrypted` encrypts the Markdown at rest.

### Does it need a database, a GPU or a network connection?

No. The default setup is plain files plus an SQLite index from the Python
standard library, and recall runs locally. Postgres, vector embeddings, a
cross-encoder and the local `mind-mem:4b` model (via Ollama) are optional.
`mm usage` reports model-call token counts from a local ledger.

### How does recall rank results?

BM25 full-text scoring with stemming and query expansion by default, plus a
deterministic rerank. Hybrid retrieval adds a vector leg fused with
Reciprocal Rank Fusion when embeddings are configured. `mm explain "<query>"`
shows the per-stage scores for a real query.

### Why can't the CLI just write a decision?

Because automated writes are the main way memory gets poisoned. Capture and
scans produce signals and proposals; a reviewer approves them with
`mm review`. The governance mode (`detect_only`, `propose`, `enforce`)
decides how far automation may go. See [../SKILL.md](../SKILL.md#writing-memory-governed).

### Can several agents share one memory?

Yes. Point every agent at the same `MIND_MEM_WORKSPACE`; `mm install-all`
wires every detected client to one workspace. Agents can leave each other
notes with `mm send` and read them with `mm inbox`.

### How do I bring in memory from somewhere else?

`mm import --from markdown ~/notes --dry-run` previews an import from a
note directory; other `--from` formats read other note trees or JSON dumps.
Run it again without `--dry-run` to stage it.
Imported blocks are quarantined until released through review. The
supported formats are listed under `mm import` in [cli.md](cli.md).

### How do I back up or move a workspace?

```bash
mind-mem-backup backup "$MIND_MEM_WORKSPACE" -o backup.tar.gz
mind-mem-backup restore /path/to/new-workspace -i backup.tar.gz
mind-mem-backup export "$MIND_MEM_WORKSPACE" -o blocks.jsonl
```

For an auditable export of admitted memory, `mm export` (requires
`v4.compliance_export`) produces a byte-identical bundle for an unchanged
corpus.

### How do I know the memory has not been tampered with?

`mind-mem-verify "$MIND_MEM_WORKSPACE"` checks the hash chains, the evidence
chain and the config binding offline. `mm chain survey "$MIND_MEM_WORKSPACE"`
reports any break in the evidence chain.

### How do I keep the workspace small?

`mind-mem-compact "$MIND_MEM_WORKSPACE" --dry-run` shows what would be
archived (completed blocks, old snapshots, daily logs, resolved signals);
drop `--dry-run` to do it. Archived blocks move to `*_ARCHIVE.md` files.

### Where is the full documentation?

The `docs/` directory of the source repository: `getting-started.md`,
`configuration.md`, `cli-reference.md`, `mcp-integration.md` and
`troubleshooting.md`.
