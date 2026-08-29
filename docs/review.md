# `mm review` — batch approval for the HITL queue

> Approval throughput is the product metric. Every governance gate stays on;
> the tool only makes approving fast.

## Why

Every write into mind-mem goes through a human. That is the moat, and it is also
the churn engine: an operator who has to approve thirty proposals one at a time,
with no diff and no evidence in front of them, stops approving — and an agent
whose memory never fills stops using it.

Before `mm review`, clearing the queue looked like this:

| Step | Command |
|------|---------|
| discover — no listing verb existed | `cat $MIND_MEM_WORKSPACE/intelligence/proposed/*_PROPOSED.md` |
| read the target — no pre-apply diff existed | `mm inspect D-20260801-001` |
| dry run | `python3 -m mind_mem.apply_engine P-20260829-001 $MIND_MEM_WORKSPACE --dry-run` |
| apply | `python3 -m mind_mem.apply_engine P-20260829-001 $MIND_MEM_WORKSPACE` |
| see what changed — only available *after* the apply | `cat …/intelligence/applied/<TS>/APPLY_RECEIPT.md` |

Five commands and ~316 keystrokes per proposal, and the diff arrives too late to
inform the decision it was needed for.

`mm review` is one command, one keystroke per proposal, with the diff, the
provenance, the chain status and the staleness flag already on screen.

## Commands

```
mm review                       # list the queue with health and blockers
mm review --json                # same, machine-readable
mm review --show P-20260829-001 # full detail: diff, evidence, chain, staleness
mm review -i                    # keyboard session, then commit
mm review --approve P-…-001,P-…-002
mm review --reject P-…-003 --reason "superseded upstream"
```

### Keyboard session

| Key | Action |
|-----|--------|
| `a` | approve this proposal |
| `r` | reject (prompts for a rationale; blank cancels) |
| `s` / space | skip — leaves it staged and undecided |
| `b` | back one proposal |
| `u` | undo the last decision |
| `c` | commit the staged decisions |
| `q` | quit, applying nothing |

**One operator keystroke, one decision.** There is no select-all and no
"approve the rest" — thirty keystrokes clears thirty proposals, which costs
seconds, and it keeps "a human approved this" true for every applied block.

## What it is not

`mm review` is a **front end**. Approvals leave through the same
`approve_apply` the MCP server exposes, and rejections through
`reject_proposal`. There is no second write path, and there is
**no auto-approve fast path at any risk level** — not for `Risk: low`, not for
a trusted source, not unattended. The review modules never branch on risk at
all; `tests/test_review_no_autoapprove.py` fails the build if that changes.

## Atomicity

Atomicity is **per proposal**, not per batch. If proposal 7 of 30 fails, 1-6
stay applied, 8-30 still run, and 7 is reported with its reason. A half-applied
batch that silently rolled six good applies back would leave the audit chain
describing work that no longer exists.

## The metric

Every batch prints what it achieved:

```
applied=30  rejected=0  failed=0
proposals/minute: 106.3    applied/minute: 106.3    (over 16.9s of operator session, 15.6s applying)
median proposal age at approval: 39.0h  (sample 30/30, coverage 100%)
```

`median proposal age at approval` is the churn signal — a queue whose median age
is days old is a queue the agent has already routed around. It is reported only
for proposals that carry a `Created` timestamp, and `coverage` says what fraction
that was. An unknown age renders as `?`, never as `0s`.

**The rate covers the whole operator session, not the applies.** The denominator
is measured from the top of the `mm review` invocation — before the queue is even
drawn — through the last governed apply, so it includes the time the operator
spent reading. The two spans are printed side by side (`over Xs of operator
session, Ys applying`) precisely so the number cannot be quoted out of context.
Measuring only the applies would report the apply engine's throughput and call it
approval throughput, which is the flattering number, not the true one.

## Governance gates you will meet

`mm review` reports these rather than working around them.

| Gate | Where | Effect |
|------|-------|--------|
| `MIND_MEM_SCOPE` | `mcp/infra/acl.py` | `approve_apply` is admin-scope. Export `MIND_MEM_SCOPE=admin`. `mm review` will not elevate it for you. |
| `governance_mode` | `memory/intel-state.json` | `detect_only` blocks every apply. |
| backlog limit | `mind-mem.json` → `proposal_budget.backlog_limit` | At or over the limit, the apply engine refuses. Default 30 — the size of the queue batch review targets. |
| no-touch window | `apply_engine.check_no_touch_window` | A hard-coded 10 minutes since the last successful apply. **This is the throughput ceiling**: 30 serial approvals means 29 × 10 min of enforced waiting. `mm review` reports it per proposal and applies what it can. |

The blockers are printed **before** the operator spends any decisions — at the
top of the listing, at the top of a keyboard session, and ahead of a `--approve`
/ `--reject` batch:

```
BLOCKERS — approvals will fail until these clear:
  * apply rate limit active — No-touch window: 9m 58s remaining
```

Ordering is the point. A gate named after thirty keystrokes is an epitaph, not a
warning, and the operator has to re-make every decision.

## Read-only guarantees

* Listing, `--show` and the keyboard session write nothing to the corpus and add
  no entry to the governance chain.
* A preview that fails **never** ends the review. Containment is by kind of
  failure, not by a list of exception types: the governance gate raises
  `GovernanceBypassError` (a bare `Exception` subclass) on spec drift and on an
  admission that will not resolve in the hash chain, and letting that propagate
  discarded every decision staged before it. The proposal is marked
  `(unavailable: <type>: <reason>)` and the session continues.
* `FilesTouched` is proposal-supplied and the preview *reads* every path it
  names, so containment under the workspace root is checked on both separators —
  `..\..\etc` has no `/` to split on.
* The pre-apply diff is produced by the production op executors replayed against
  a **temporary sandbox copy**, under an admission opened on the *sandbox's* gate
  — never the workspace's. A preview is not an apply, and the chain must not
  record one.
* Reading provenance and verifying the chain may materialise their own empty
  backing stores on first use. Nothing that already existed is modified.
