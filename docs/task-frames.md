# Task Frames & the Dead-End Registry

Two block kinds for multi-session agent work:

- **`[TF-...]` — TASK-FRAME.** What a task *is*, what has been tried, what is
  currently believed, and what remains. Session N+1 opens with a brief instead
  of re-deriving context.
- **`[DE-...]` — DEAD-END.** An approach that was tried and failed, with the
  reason and an evidence handle. Negative action-space memory: the thing an
  autonomous agent most expensively forgets.

Both are read back by `resume_brief`, `mm resume`, `mm dead-ends`, and the MCP
tools `resume_brief` / `check_dead_ends`.

---

## Why not just recall them?

Ranked retrieval is the wrong model for both.

"What am I working on" is a **pointer**, not a similarity query — a pointer
that has to win a relevance contest against the whole corpus is not a pointer.
And "we already tried X and it failed" only helps at the moment the agent is
about to try X again, which is exactly when the query is about something else
and the record loses the ranking contest.

So neither is ranked. A frame is selected by ID; a dead end fires on a
**declarative overlap test** against the frame's declared approach surface.

---

## Authoring

Frames live in `frames/FRAMES.md`, dead ends in `frames/DEAD-ENDS.md` (both
configurable — see [Configuration](#configuration)).

### `[TF-...]` — a task frame

```
[TF-20260829-001]
Type: TaskFrame
Goal: Close the last two AGI3 floors without a net regression.
Status: active
Steps:
- done: rederive the floor count from the live pass
- doing: pin the SAT-BMC encoding for L5
- todo: package the bundle for the cloud run
- blocked: L8 needs more RAM than the box has
Tried:
- explicit BFS over the level graph
- additive per-object assignment bound
Believed: flow bounds survive multi-carrier levels, additive bounds do not
Remaining: fund the cloud run
Blockers: cloud budget
References: docs/floors.json, research/runpod/bundle.md
ApproachTools: Bash
ApproachCommands: rederive_floor_count.py, sat-bmc
ApproachIntents: prove_floor
ApproachPaths: tools/**/*.py
```

| Field | Meaning |
|---|---|
| `Goal` | **Required.** What the task is, in one line. |
| `Steps` | Plan steps as `"<status>: <text>"`. Status is one of `todo` / `doing` / `done` / `blocked`; a bare line is a `todo`. The vocabulary is closed so a brief can state what remains without interpreting free text. |
| `Tried` | What has already been attempted. |
| `Believed` | What is currently believed to be true. |
| `Remaining` | What is left. **Falls back to every step that is not `done`** when omitted, so a frame that ticks steps need not also maintain this field. |
| `Blockers` | What is stopping progress. |
| `References` | Evidence handles carried into the brief as citations. |
| `Approach*` | The task's **approach surface** — see below. |
| `Status` | `active` / `wip` / `doing` keep a frame live; anything else is parsed and ignored. |

`Tried` / `Believed` / `Remaining` / `Blockers` are **prose** and are never
comma-split: *"flow bounds survive, additive bounds do not"* is one belief, not
two. Use a markdown list for multiple entries. `References` and `Evidence` are
**handles** and *are* comma-split.

### `[DE-...]` — a dead end

```
[DE-20260826-001]
Type: DeadEnd
Approach: Additive per-object assignment lower bound for AGI3 floors.
WhyFailed: A co-carrier helper divides the matching by K, so the bound lands
    under the already-proven floor and can never tighten it.
Outcome: refuted
Evidence: docs/AGI3_FLOOR_METHOD_ARSENAL.md#assignment
TriggerTools: Bash
TriggerIntents: prove_floor
Status: active
```

| Field | Meaning |
|---|---|
| `Approach` | **Required.** What was tried. |
| `WhyFailed` | **Required.** Why it did not work. |
| `Outcome` | Closed vocabulary, ranked most-conclusive first: `refuted` → `regressed` → `blocked` → `inconclusive`. Defaults to `blocked`. |
| `Evidence` | Handle(s) to the proof — a log, a results file, a run id. |
| `Trigger*` | **At least one required.** When to raise this — see below. |

A dead end that declares no trigger is **refused**, exactly like a guardrail: a
record that can never fire is noise, not memory.

---

## The overlap test

`Trigger*` on a dead end and `Approach*` on a frame are two halves of one
match, and both are parsed by the **same code a `[GR-...]` guardrail uses**
(`mind_mem.guardrail_patterns`). Same glob grammar, same normalisation, same
per-dimension bounds. There is no second trigger dialect.

| Dimension | Dead end declares | Frame declares | Match |
|---|---|---|---|
| tool | `TriggerTools` | `ApproachTools` | exact or glob |
| command | `TriggerCommands` | `ApproachCommands` | substring or glob |
| intent | `TriggerIntents` | `ApproachIntents` | exact or glob |
| path | `TriggerPaths` | `ApproachPaths` | glob |

**AND across declared dimensions, OR within one.** The dead end above fires
only for a task whose approach involves `Bash` *and* the `prove_floor` intent.
Fail-closed: an empty trigger or an empty approach surface never matches.

`mind_mem.dead_ends.match_dead_ends(frame, dead_ends)` is a **pure function of
its two arguments** — no clock, no model, no learned score, no ranking signal.
Same inputs produce byte-identical output in any process on any machine.
Determinism is the product wedge here; a learned similarity detector would
forfeit it and is out of scope by design.

Warnings are ordered `(outcome, block_id)`. The registry listing is ordered by
`block_id`.

### A dead end warns; it never blocks

A firing dead end produces a warning carrying the reason, the evidence handle,
and `matched` (which declared dimensions were responsible). Nothing refuses an
action, filters a plan, or changes an exit code. Re-running a recorded failure
is sometimes exactly right — the operator decides, and this machinery only
makes sure they decide informed.

---

## Reading them back

### Python

```python
from mind_mem.resume_brief import resume_brief, render_resume_brief

brief = resume_brief(workspace)            # or resume_brief(ws, "TF-20260829-001")
brief.goal, brief.tried, brief.believed, brief.remaining
[w.dead_end.block_id for w in brief.dead_ends]
print(render_resume_brief(brief))
```

An empty workspace yields an empty brief (`frame_id == ""`), never an error, so
a session can ask unconditionally at start-up. An explicit `frame_id` that
names no live frame raises `FrameSpecError` — silently resuming a *different*
task would be worse than failing.

With several live frames, the **active** frame is the one with the highest
block ID. IDs are `TF-YYYYMMDD-NNN`, so that is the most recently minted frame,
read off the ID and never off a clock.

### CLI

```bash
mm resume                                  # rendered brief + dead-end warnings
mm resume --frame TF-20260829-001 --json
mm dead-ends                               # the whole registry, JSON
mm dead-ends --tool Bash --intent prove_floor   # overlap against one action
```

`mm resume` exits `0` even when dead ends fire — a warning must never look like
a failed command.

### MCP

| Tool | Use |
|---|---|
| `resume_brief(frame_id="")` | Session start. Returns the brief plus `dead_ends` / `dead_end_count`. |
| `check_dead_ends(tool, command, intent, paths)` | About to act. Returns the recorded failures overlapping that one action. |

Both are `USER_TOOLS` scope and read-only.

### Nothing is dropped silently

Negative memory that disappears quietly is worse than negative memory that was
never written, because the reader now has *positive evidence of absence*. So
every surface publishes what it could not show:

| Field | On | Meaning |
|---|---|---|
| `rejected` | brief, `resume_brief`, `check_dead_ends`, `mm resume`, `mm dead-ends` | Every `[TF-...]` / `[DE-...]` block the corpus declared that could not be read, with the source file and the parser's reason. |
| `dead_end_total` / `dead_ends_elided` | brief, `resume_brief` | How many dead ends actually fired, versus how many fit under `max_warnings`. |
| `total_matched` / `elided` | `check_dead_ends`, `mm dead-ends` | The same two counts for a single about-to-act check. |

An empty `frame_id` with a non-empty `rejected` is a **different fact** from an
empty `frame_id` with none: *"continuity exists but is unreadable"* against
*"there is no continuity"*. They lead to opposite behaviour, so `mm resume`
renders them differently rather than printing `No active task frame.` for both.

`enabled: false` silences the warning channel but still reports
`dead_end_total`: a kill switch may stop the noise, it may not fake the
absence of the signal.

---

## Configuration

`mind-mem.json`, under `recall.frames` (all keys optional):

```json
{
  "recall": {
    "frames": {
      "enabled": true,
      "max_warnings": 5,
      "frame_sources": ["frames/FRAMES.md"],
      "dead_end_sources": ["frames/DEAD-ENDS.md"]
    }
  }
}
```

`enabled: false` is the kill switch for the **warning channel only** — briefs
still resolve their frame. `max_warnings` is bounded by a hard cap of 20 so a
misconfigured policy can never flood a brief. An invalid value falls back to
the default with a logged warning rather than taking a brief down.

---

## Governance

Frames and dead ends are **blocks**, so they inherit the governance gate:
authoring goes through `propose_update` → `approve_apply` like any other block
kind. Nothing in `task_frames.py`, `dead_ends.py`, `resume_brief.py` or the MCP
tool module writes — they read the corpus back and nothing else.

> **Status (5.0.0): the governed write path is not yet open for these
> prefixes.** `approve_apply` routes `append_block` through
> `BlockStore.write_block`, which resolves a block's canonical file from the
> one corpus table — `CORPUS_TABLE` in `corpus_registry.py`, projected as
> `_BLOCK_PREFIX_MAP` in `block_store.py` — and refuses an unrecognised
> prefix:
>
> ```
> append_block failed: no canonical file mapping for block id
> 'TF-20260829-001'; add a row to corpus_registry.CORPUS_TABLE to enable writes
> ```
>
> Until `TF` and `DE` are added there, frames and dead ends are
> operator-authored in their workspace files and read back read-only — exactly
> the position `[GR-...]` guardrails ship in today (see `docs/guardrails.md`
> §Authoring). `GR`, `TF` and `DE` are all absent from that table. The
> prefixes that do carry a row are whatever `corpus_registry.CORPUS_TABLE`
> lists — read them from there rather than from a copy here, which is the
> same reason the table exists: one definition, no second list to go stale.
>
> **Two map entries are necessary but not sufficient.** `frames/` is also
> outside `corpus_registry.SNAPSHOT_DIRS` (`decisions`, `tasks`, `entities`,
> `summaries`, `memory`, `maintenance/tracked`), so an apply that wrote a frame
> today would land outside the rollback snapshot — applied, receipted, and not
> restorable by `rollback_proposal`. Opening the governed route means opening
> the prefix map *and* the snapshot scope together. Both live in boundary files
> owned elsewhere; no module in this feature needs to change either way.

### Provenance restriction

A frame steers an agent's next session and a dead end steers it away from an
action, so both are injection primitives. A block carrying external-ingest or
imported provenance may **never** mint either, whatever its content or metadata
declares — the check is `guardrails.guardrail_provenance_refusal`, shared
verbatim with guardrails so the two threat models cannot drift apart. One
poisoned block is skipped with a loud warning; it never disables the rest of
the file.

---

## Related

- `docs/guardrails.md` — `[GR-...]` blocks, the trigger grammar these reuse.
- `docs/governance.md` — the `propose_update` → `approve_apply` route.
