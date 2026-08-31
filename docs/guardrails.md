# GUARDRAIL blocks

A guardrail is a **prohibition with a trigger**. It is retrieved by matching
what the agent is *about to do* — not by similarity to the query.

Similarity ranking is the wrong model for a rule like *"never run
`git reset --hard` without checking `git status` first"*. That rule has to
fire when the agent reaches for `git reset`, and the query at that moment is
usually about something else entirely — so a ranked memory returns it exactly
when it does not matter and drops it exactly when it does. A `[GR-...]` block
carries declarative trigger conditions instead, and matching guardrails bypass
the ranker.

## Authoring

Guardrails live in `guardrails/GUARDRAILS.md` by default. Nothing in the
guardrail code writes to the store — the file is **operator-authored** and only
ever read back.

`propose_update` mints `decision` and `task` blocks into `SIGNALS.md`; it has
no `guardrail` block type and cannot author a `[GR-...]` block. So guardrail
authoring is an operator write to a workspace file today, not an agent
proposal. That is deliberate for this release: a guardrail bypasses the ranker
and is surfaced unconditionally, so the smallest trustworthy minting surface is
a human editing the file. Extending `propose_update` → HITL to guardrails would
add an agent-reachable minting path and needs its own review; until it exists,
do not describe guardrails as agent-proposable.

```markdown
[GR-20260827-001]
Type: Guardrail
Statement: Never run `git reset --hard` without checking `git status` first.
Severity: critical
TriggerTools: Bash
TriggerCommands: git reset --hard, git clean -fd
Status: active

[GR-20260827-002]
Type: Guardrail
Statement: Migrations under db/ require a reviewed rollback script.
Severity: high
TriggerPaths:
- db/migrations/**/*.sql
- db/*.sql
Status: active
```

| Field | Meaning |
|-------|---------|
| `Statement` | The constraint text. Required. |
| `Severity` | `critical` / `high` / `medium` / `low`. Orders surfacing. Default `medium`. |
| `TriggerTools` | Tool names the action would invoke (exact match, case-insensitive). |
| `TriggerCommands` | Command patterns (substring match on the normalised command). |
| `TriggerIntents` | Intent classes (exact match). |
| `TriggerPaths` | Path globs. |
| `Status` | Only `active` / `wip` (or absent) fire. `deprecated` is inert. |

Every trigger field accepts either a comma-separated scalar
(`TriggerTools: Bash, Shell`) or a markdown list.

### Provenance restriction

**A block that arrived from outside the governed store can never mint a
guardrail**, whatever its content or metadata declares. Recognition is refused
before a single trigger field is read when the block carries:

| Signal | Example |
|--------|---------|
| An external `ActorRole` | `ActorRole: importer` / `ingest` / `crawler` / `scraper` / `feed` / `sync` |
| An ingest token on `ToolId` or `Source` | `imported:slack`, `import:notion`, `ingest:…`, `external:…` |
| An ingest-authored block type | `Type: ImportedMemory` |
| An `external-ingest` provenance class | anything `mind_mem.provenance_class` classifies as external |
| A declared external content source (T-001) | `ContentSource: external` |

`ContentSource` is the T-001 *declared* content tag, vocabulary-bound to
`{agent, user, external}`. It is read fail-closed: anything outside that
vocabulary — including a hand-edited `ContentSource: operator` — yields no
signal at all rather than a class of the writer's choosing, so the tag can
only ever demote a block, never promote one.

The markers are read straight off the block *before* any role-based promotion,
so a crafted `ActorRole: operator` sitting next to `Source: imported:slack`
(or next to `ContentSource: external`) is still refused — an imported corpus cannot launder itself into a constraint by
claiming a trusted role. The refusal is logged as
`guardrail_provenance_refused` and the rest of the file still loads: one
poisoned block cannot take the constraint set down.

Blocks with **no** provenance fields stay eligible (a corpus predating those
fields is not demoted, matching how absence is treated everywhere else), and
agent-authored guardrails still work — the threat model is untrusted *content*,
not an authenticated agent.

Why this is load-bearing: guardrails bypass the ranker and are surfaced
unconditionally, so a trigger-bearing block is an injection primitive. Content
an attacker can get imported must not be able to declare one.

### Matching rules

* **AND across declared dimensions, OR within one.** The first block above
  fires only for a `Bash` call whose command contains one of the two command
  patterns. A guardrail declaring *no* trigger is refused at load time
  (fail-closed) — an always-on guardrail is just noise.
* **Deterministic.** Literal and glob matching only: no model call, no
  embedding, no clock, no randomness. Same context + same corpus ⇒ same
  guardrails, in the same order, on every machine.
* **Glob grammar.** `*` matches within one path segment, `**` crosses
  segments, `**/` matches zero or more leading segments, `?` matches one
  non-separator character. Everything else is literal.

## Retrieval

```python
from mind_mem.recall import recall

results = recall(
    workspace,
    "how do we roll back a bad deploy",
    guardrail_context={"tool": "Bash", "command": "git reset --hard HEAD~3"},
)
```

Firing guardrails are returned **first**, ahead of every ranked hit, whatever
their similarity score — including guardrails the ranker never retrieved.
They are marked so a client can render them as constraints rather than as
evidence:

```json
{
  "_id": "GR-20260827-001",
  "type": "guardrail",
  "guardrail": true,
  "guardrail_severity": "critical",
  "guardrail_triggers": ["tool", "command"],
  "guardrail_constraint": "Never run `git reset --hard` without checking `git status` first.",
  "surfaced_by": "guardrail_trigger"
}
```

Surfacing runs *after* every post-retrieval filter (date, lifecycle,
event id, maturity): a filter aimed at evidence must not be able to drop a
constraint. A guardrail the ranker already returned is promoted in place and
keeps its score.

### Bounds

At most `max_surfaced` guardrails are injected (default **3**, hard cap
**10**), and the response keeps its original length — so guardrails displace
at most `max_surfaced` ranked hits, never the whole page.

```json
{
  "recall": {
    "guardrails": {
      "enabled": true,
      "max_surfaced": 3,
      "sources": ["guardrails/GUARDRAILS.md"]
    }
  }
}
```

`enabled: false` is the kill switch. Sources are workspace-relative; a path
that escapes the workspace root is refused.

## MCP tools

| Tool | Use |
|------|-----|
| `check_guardrails(tool, command, intent, paths)` | Pure trigger evaluation — the constraints that apply to an action, with no query and no ranker. Call it before a risky action. |
| `recall_with_guardrails(query, tool, command, intent, paths, limit)` | Ordinary recall with the constraints for this context surfaced first. |

Both are read-only, `USER_TOOLS` scope — they evaluate and surface guardrails,
they never mint one. `recall_with_guardrails` deliberately bypasses the recall
cache: the cache key does not include the guardrail context, so a cached
envelope could otherwise answer with the wrong constraints.

## Zero regression

With no guardrail blocks present — or with no `guardrail_context` supplied —
recall output is byte-identical to a build without this feature. The whole
path is skipped, not merely filtered to empty. See
`tests/test_guardrail_blocks.py::TestZeroRegression`, and
`TestProvenanceRestriction` for the minting rules above.
