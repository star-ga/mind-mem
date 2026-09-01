# Trajectory memory

Case-based reasoning over your own past task executions. Not "which block
is relevant to this query?" but "how did this go the last few times someone
tried it, and how did it end?"

Two halves, both behind one flag:

| Half | Surface | What it does |
|---|---|---|
| Capture | `report_outcome` (MCP tool / library) | mirrors each recorded verdict into a `TRAJ-` block under `<workspace>/trajectories/` |
| Recall | `similar_trajectories` (MCP tool, USER scope) | ranks that store against the task you are about to attempt |

Tuned by `mind/trajectory.mind`; a workspace can override any knob with its
own `<workspace>/mind/trajectory.mind`.

## Enabling it

Off by default. In `mind-mem.json`:

```json
{
  "v4": { "trajectory": { "enabled": true } }
}
```

With the flag off, `report_outcome` behaves exactly as it did before the
feature existed — no `trajectories/` directory is created, its return value
is unchanged, and `similar_trajectories` refuses without reading anything.

## Capture

Every `report_outcome` call that records a *new* outcome writes one block:

```
[TRAJ-20260221-001]
Task: upgrade the runtime
Date: 2026-02-21
Status: active
Tools: cargo
Outcome: SUCCESS
Reward: 1.0
Context: sess-7
Outcome_Id: 9f2c...
Lessons:
  - all 8691 tests green
```

* The `report_outcome` verdict vocabulary is **mapped**, not passed through:
  `success` → `SUCCESS`, `failure` → `FAILURE`, `neutral` → `PARTIAL`. The
  reward comes from the kernel's `[outcome] default_reward_*` knobs.
* `Date` is taken from the outcome's own `recorded_at`, never from the
  clock, so a back-dated report captures at its own date.
* **Replays capture nothing.** The outcome id is the SHA-256 of the
  canonical payload; a replayed report records no new evidence and must not
  mint a second trajectory for the same event.
* Provenance strings are caller-supplied, so every whitespace run in them —
  newlines included — is collapsed before it is written. Text submitted as a
  value can never become a second field or a forged block header.
* Capture failure is contained: a full disk turns into a log line, never
  into a failed outcome report.

### This is a sidecar, not the corpus

`trajectories/` is the same class of artifact as the calibration database
that `report_outcome` already appends to. Nothing in it is served by
`recall()`, and promoting a lesson into the corpus still goes through
`propose_update` → HITL like every other block. The governed write path is
untouched.

Reads are nonetheless admission-filtered: the parsed store goes through
`admit_corpus` before any caller sees it, so a file carrying
`Status: quarantined`, `Status: pending`, or a status nobody has named is
withheld fail-closed. Selecting a status is not filtering on it.

## Recall — `similar_trajectories`

```
similar_trajectories(task, tools="", outcome="", limit=0, scoring_instant="")
```

| Argument | Meaning |
|---|---|
| `task` | what you are about to do |
| `tools` | comma- or space-separated tools you expect to use |
| `outcome` | bias toward trajectories that ended this way |
| `limit` | max results; `0` uses the kernel's `recall_limit` (ceiling 50) |
| `scoring_instant` | UTC `YYYY-MM-DD` the recency decay measures age from |

Scoring is task-word overlap + tool overlap + outcome agreement, discounted
by an exponential recency half-life. A trajectory from 60 days ago scores at
a quarter of the same trajectory recorded today, on the shipped 30-day
half-life — stale procedure is worse than no procedure.

### Determinism

The answer is a pure function of `(store, kernel, scoring_instant)`. The
instant is resolved **once, at the tool boundary**, exactly as `recall()`
does it, and threaded into every comparison; the ranking loop reads no clock
and no randomness. Pass `scoring_instant` and the whole call is clock-free;
omit it and exactly one clock read happens, before any scoring. Every
response carries the instant it scored against, so an earlier run replays by
passing its instant back. Ties break on block id, so the order never depends
on directory iteration.

## Kernel knobs — `mind/trajectory.mind`

| Section | Knob | Default | Effect |
|---|---|---|---|
| `recall` | `recall_limit` | 5 | results when `limit` is 0 |
| `recall` | `recency_halflife` | 30 | days for a score to halve |
| `recall` | `outcome_weight` | 0.3 | weight of outcome agreement |
| `recall` | `tool_overlap_boost` | 1.5 | multiplier on the tool-overlap term |
| `outcome` | `default_reward_success` | 1.0 | reward stamped on a SUCCESS capture |
| `outcome` | `default_reward_partial` | 0.5 | …on a PARTIAL capture |
| `outcome` | `default_reward_failure` | 0.0 | …on a FAILURE capture |
| `outcome` | `default_reward_aborted` | 0.0 | …on an ABORTED capture |

A knob whose value will not coerce keeps its default: a typo in a kernel
must not take the surface down.

> **Historical note.** Until 5.1.0 these knobs were unreachable. The loader
> resolved `dirname(__file__)/../mind/trajectory.mind`, which is `src/mind/`
> — a directory that has never existed — so every knob silently fell back to
> its default. The bug was invisible because the shipped kernel's values
> *are* the defaults; it only became observable once a workspace could
> override one. Resolution now goes through the same `get_mind_dir` resolver
> every other kernel consumer uses.
