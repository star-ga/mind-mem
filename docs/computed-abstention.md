# Computed abstention

`recall` returns ranked candidates and leaves one question to the caller's
discipline: *is any of this a clear match?* "Five blocks came back and none of
them is a clear match" and "five blocks came back and the first one is clearly
right" are both a list of five. Computed abstention derives the answer from the
score distribution the engine already computed, so a caller that skips the
judgement still gets a warning.

It is a description of the served set, never an edit of it: the results, their
order and their scores are exactly what they were without it. There is no model,
no network call, no clock and no randomness in it.

## The statistic

One number for the **set**, not a score on each block:

```
separation = (top - runner_up) / top        # 0.0 .. 1.0, four decimal places
```

How far the best candidate stands clear of the second best, relative to its own
size. It is scale-free on purpose — a BM25 gap, an RRF gap and a cosine gap
describe the same shape the same way, and the engine really does mix those scales
from one query to the next.

| Served set | `separation` |
|---|---|
| nothing recalled, or nothing with a positive score | `0.0` |
| one candidate | `0.5` — a lone hit has nothing to be separated *from*, so the distribution is silent (neither a clear win nor ambiguity) |
| two or more | the gap above; an exact tie is `0.0` |

Force-surfaced guardrail hits are constraints, not evidence, and are excluded.
Under RRF fusion the score is rank-derived, so the statistic there measures
*agreement structure* (both legs rank one hit first while the runner-up appears in
a single leg); the verdict records its `basis` so the two regimes are scored
separately.

## Three bands, scaled by use

`act` / `proceed_with_caution` / `do_not_act`. The boundary is not one number:
what you intend to *do* with the answer sets how much separation you should
demand before acting without confirmation.

| `intended_use` | `act` at separation ≥ | `proceed_with_caution` at ≥ |
|---|---|---|
| `summary` — the reader sees the sources and can discount a miss | 0.35 | 0.10 |
| `answer` — answering a question directly from the blocks (default) | 0.55 | 0.20 |
| `governed_write` — feeding a proposal or decision | 0.70 | 0.35 |

A policy that makes a stricter use looser than a less strict one is rejected, so a
recall feeding a governed write can never be easier to act on than one feeding a
summary. Nothing recalled is always `do_not_act`.

These defaults are **provisional**. They were set from a deterministic labelled
fixture and are evidence that the statistic carries signal on a ranker that finds
the right block — nothing more. Measure your own workspace (below) and let the
outcome loop score them.

## The response field

```json
{
  "results": ["…"],
  "abstention": {
    "abstain": false,
    "band": "act",
    "separation": 0.8395,
    "candidates": 5,
    "intended_use": "answer",
    "thresholds": {"act": 0.55, "caution": 0.2},
    "reason": "top_candidate_clearly_separated",
    "basis": "sqlite"
  }
}
```

`abstain` is `true` exactly when `band` is `do_not_act`. `reason` is one of
`no_candidates`, `no_positive_score`, `single_candidate`,
`top_candidate_clearly_separated`, `top_candidate_weakly_separated`,
`no_clear_winner`.

## Asking for it

It is **off by default, and invisible when off**: with neither request below the
response is byte-identical to one from a build that never had the field.

| Door | How to ask |
|---|---|
| MCP `recall` / `hybrid_search`, REST `POST /v1/recall`, gRPC | `intended_use` parameter, **or** `recall.abstention.enabled: true` for every recall |
| `mm recall` | `--intended-use USE` (the output becomes an object with `results`, `abstention`, `attestation`; the default stays a bare list) |
| stdlib `POST /query` | `"intended_use"` in the body |

The workspace-wide switch applies to the MCP / REST / gRPC envelope. `mm recall`
and the stdlib route change shape when they carry a verdict, and honouring the
switch there would cost a config read on every request that never asked, so they
are per-request only. An unknown `intended_use` is refused before any work; on the
`recall` MCP tool it is also refused for the modes that do not rank
(`similar`, `axis`, `pack`, …) rather than accepted and dropped. `format="bundle"`
carries no result list and gets no verdict. gRPC has no field for it in its
message schema, so it honours only the workspace switch.

```json
{
  "recall": {
    "abstention": {
      "enabled": true,
      "default_use": "answer",
      "bands": {"governed_write": {"act": 0.8, "caution": 0.4}}
    }
  }
}
```

Partial overrides are merged over the defaults. An invalid table (out-of-range,
non-numeric, `caution` above `act`, a stricter use made looser) falls back to the
shipped defaults **for every use** and reports why in `config_warnings`.

## Closing the loop: abstention is a prediction

A verdict nobody checks is decoration. When a verdict is returned it is recorded —
in a sibling table of the same `recall.db` that holds `calibration_feedback` and
`recall_outcome` — against the run identity the response already publishes
(`attestation.query_id`, the value you pass to `report_outcome`). When the work
that used the recall succeeds or fails and you report it:

```text
report_outcome(run_id=attestation.query_id, outcome="success" | "failure")
```

the join scores the prediction. Read it in `outcome_stats` (the `abstention`
section appears only once a verdict has been recorded):

```json
"abstention": {
  "predictions": 12,
  "scored_predictions": 9,
  "bands": {"act": {"success": 5, "failure": 0, "votes": 5, "success_rate": 1.0}, "…": "…"},
  "by_use": {"…": "…"},
  "by_basis": {"…": "…"},
  "calibration": {"verdict": "discriminating", "silent": false, "overconfident_failures": 0}
}
```

`calibration.verdict` compares the success rate of recalls called `act` with those
called `do_not_act`, each side needing the outcome loop's own minimum evidence:
`discriminating`, `not_discriminating`, `inverted` (confident recalls fail *more*),
or `insufficient_evidence`. `silent` is the pairing worth watching: failures
accumulating while the abstention never fires. One reporter is one vote per
(run, verdict), exactly as in the outcome projection; the scoring is pure
arithmetic over stored counts (no clock, no window).

The prediction is a record that a read was served, in the same class as the served
ledger. Nothing recall does reads it, and writing it never touches block content.
A (run, use) already recorded is detected by a read and left alone (`recall.db` is
part of the recall cache's corpus stamp). A run recalled under several intended
uses carries one prediction per use and each is scored against the run's outcomes,
since an outcome report does not say which use the work was for.

## Does it carry signal on *your* workspace?

The proposal that introduced this carried its own kill switch: *run recall over a
labelled set where the answer is known to be absent; if the statistic does not
separate those from the present ones, there is no signal and no threshold tuning
will create one.* That is executable:

```bash
mm abstention-eval --labelled queries.jsonl --intended-use answer
# queries.jsonl:  {"query": "which region hosts the datalake", "present": false}
```

It serves each query through `mm recall`'s door (without writing the served-set
ledger), and reports the AUC of the statistic as a present-vs-absent classifier,
the per-class distribution, how often the bands would `act` on an absent answer
(the costly error) and stop on a present one, and a verdict: `separates`
(AUC ≥ 0.80), `weak` (≥ 0.60), `no_signal` (exit status 3 — close this as declined
for the workspace) or `insufficient` (fewer than five labelled queries per class).

### What was measured while building it

On a deterministic fixture (40 present and 40 absent queries over 40 blocks, the
absent queries matching many blocks weakly; `tests/_abstention_fixture.py`), served
through `mm recall`'s door:

| Serving path | AUC | `answer` band: acts on present / absent | `do_not_act` on present / absent |
|---|---|---|---|
| FTS-indexed (`recall.backend: sqlite`) | **0.92** — `separates` | 25 of 40 / **0 of 40** | 5 of 40 / 31 of 40 |
| Unindexed full-scan fallback | **0.48** — `no_signal` | 0 of 40 / 0 of 40 | 29 of 40 / 30 of 40 |

The scan fallback did not put the right block first, so the statistic was
faithfully reporting that no clear winner had been found; a `no_signal` verdict says
"this serving path does not produce separable scores for these queries". Other
bands on the indexed path: `summary` acts on 32 of 40 present and 2 of 40 absent;
`governed_write` acts on 7 of 40 present and 0 of 40 absent.

A single recalled hit cannot be told apart by score: on a separate fixture a lone
strong hit and a lone weak hit were indistinguishable (one hybrid answer scored
`0.033` where a three-hit answer on the same corpus scored `16.4`), which is why a
set of one is the neutral `0.5` and lands at `proceed_with_caution` for answers and
writes.

This is a fixture, not a real workspace. It shows the experiment runs and can come
out negative; it does not discharge the measurement on yours.
