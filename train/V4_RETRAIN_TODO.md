# v4 Retrain — Probe Honesty TODO

> **Owner:** training corpus
> **Created:** 2026-05-10 (during v3.12.1 ship of `mind-mem-4b` v5).
> **Must-fix-before:** the next full retrain (v4 release).
>
> ## Hard verification gate (non-negotiable)
>
> The v4 retrain ships **only if** the model passes the **un-softened**
> eval harness at **95/95 = 100%**. Every `# V4 RETRAIN TODO` marker in
> `train/eval_harness.py` must be reverted before the v4 eval run.
> Soft-passing on the patched harness is **not** ship criteria for v4 —
> it was the one-time emergency exit for v3.12.1.
>
> If the v4 model can't clear the un-softened harness, the corpus work
> in §0 + §1 below didn't take — go back, fix the corpus, retrain.
> Do not soften any further probes. Do not ship at <95/95.

---

## Why this file exists

`mind-mem-4b` v3.12.1 (v5 weights, `model.safetensors` SHA256
`4638ac0b6e9b50ef886cca1fd6d49eee36c12d445c3efaa42b742ba66ae1627a`)
ships at 95/95 against the *patched* eval harness. Two probes were
softened to clear the bar:

| # | Probe ID | Patch type | Honesty status |
|---|----------|------------|----------------|
| 1 | `v312_quality_gate_strict_mode` "Is there an escape hatch …" | weaken required tokens from `["force", "strict"]` → `["mode"]` | **half-honest** — corpus has internal contradictions about the canonical escape hatch, and `quality_gate.py:80-81` only accepts `mode in ("advisory", "strict")` (no `"off"`). True canonical escape hatch in code is `force=True` on `validate_block`. Model emits "set quality_gate.mode = ..." which contains "mode" — the patch keeps the probe in the suite while accepting the model's vague-but-not-wrong answer |
| 2 | `v312_lineage_staleness` "What is the decay multiplier for a `cites` edge …" | drop the numeric requirement, accept just `["cites"]` | **dishonest** — model is factually wrong (emits `0.4`, the refines value). Source of truth is `block_lineage.py:67` (`KIND_DECAY['cites'] = 0.8`). The patch lets the eval pass while the model returns wrong arithmetic |

Patch 2 is the load-bearing one. The eval claims green, but anyone
running the model and asking "what's the decay multiplier for a cites
edge?" will get `0.4` — the **refines** value. This is a real model
error, not a probe-design issue.

---

## What v4 retrain MUST do

### 0. Fix internally-contradictory escape-hatch corpus probes

Today the corpus contains two probes for the same question that
contradict each other:

- One says: *"NOT via `propose_update(force=True)` … set
  `quality_gate.mode = "off"`"*.
- Another says: *"pass `force=True` to `validate_block` … OR set
  `quality_gate.mode = "off"`"*.

Neither is right against the actual code:

- `quality_gate.py:80-81` validates `mode in ("advisory", "strict")` —
  `"off"` is not a legal value and would raise `ValueError`.
- The real library-level escape hatch is `force=True` on
  `validate_block`. There is no `propose_update(force=True)` path.

For v4: collapse to **one canonical answer** per probe variant,
matching the actual code in `quality_gate.py`. Drop the `"off"` answer
entirely. Then revert the eval probe to `["force", "strict"]` (or
`["force", "validate_block"]`).

### 0a. Audit every other corpus probe against actual code

The cites=0.8 failure surfaced because corpus content drifted from
source code without anyone noticing. Before v4 trains, run an
end-to-end consistency pass:

- For every corpus probe that quotes a numeric value, function name,
  config key, or class field — `grep` the answer string against the
  current `src/mind_mem/` tree and confirm it still matches.
- For every CLI command quoted in the corpus — confirm it parses with
  the current `mm` argparse tree (run `mm <subcmd> --help` for each).
- For every file path quoted (`mind-mem.json`, `docs/*.md`,
  `src/mind_mem/*.py`) — confirm the file exists in v4.
- For every two corpus probes that answer the same question —
  diff their answers; if they contradict, pick the one that matches
  the code and delete the other.

Drift is corrosive: every contradictory probe halves the gradient
signal for the right answer. The cites=0.4 failure is what happens
when corpus drift compounds.

### 1. Rebalance per-edge-kind saturation in `train/build_corpus.py`

The current corpus over-trained `refines=0.4` (>10 reinforcement
probes added during v3.10 to fix a prior `0.3` failure) without equal
treatment of the other four kinds. The model now defaults to `0.4`
whenever asked about a single edge kind in isolation, because that's
the most-reinforced numeric anchor.

**Required for v4:**

- Add a parallel reinforcement block (≥10 probes each) for **every**
  edge kind, not just `refines`:
  - `contradicts → 1.0` (already strong, ≥10 probes)
  - `cites → 0.8` (currently ~3, **needs ≥10**)
  - `implements → 0.6` (currently ~3, **needs ≥10**)
  - `cooccurrence → 0.5` (currently ~3, **needs ≥10**)
  - `refines → 0.4` (currently ~12, fine — DO NOT remove)
- Vary surface form for each: "decimal value", "numeric value", "what
  is `KIND_DECAY['<kind>']`", "decay multiplier for `<kind>`",
  "kind-aware multiplier for `<kind>`", contrast probes ("`cites` is X
  not Y"), etc.
- Add **denial / negation** probes: "is the cites multiplier 0.4?
  No — it's 0.8."

**Concrete templates for the cites=0.8 reinforcement block** (drop
into `train/build_corpus.py` next to the existing refines block at
`Failure 8` comment):

```python
# v4 reinforcement: cites=0.8 (mirrors refines=0.4 saturation block)
(
    "Decimal value of the `cites` edge decay multiplier?",
    "`0.8`. Full table: `contradicts=1.0`, `cites=0.8`, `implements=0.6`, `cooccurrence=0.5`, `refines=0.4`.",
),
(
    "What's KIND_DECAY['cites']?",
    "`0.8`. Defined in `src/mind_mem/block_lineage.py:67`. The cites multiplier sits between `contradicts` (1.0) and `implements` (0.6).",
),
(
    "Numeric decay value for cites edges in lineage staleness?",
    "Exactly `0.8`. A `cites` seed edge contributes 80% of the maximum signal (1.0 = `contradicts`). Block at hop 1 from a `cites` seed gets `0.8 * 1.0 = 0.8`; hop 2 gets `0.8 * 0.9 = 0.72`; hop 3 gets `0.8 * 0.5 = 0.4`.",
),
(
    "Cites edge — what's the kind decay coefficient?",
    "`0.8` — second-strongest of the five kinds, behind `contradicts` at 1.0.",
),
(
    "What is the decay multiplier for a `cites` edge in lineage staleness?",
    "The `cites` decay multiplier is `0.8` — applied ONCE at the seed edge.",
),
(
    "Is the cites edge decay multiplier 0.4?",
    "**No.** `0.4` is the `refines` value. The `cites` multiplier is `0.8`. Do not confuse cites (citation) with refines (narrowing).",
),
(
    "Confirm: KIND_DECAY['cites'] = 0.4 — true or false?",
    "**False.** `KIND_DECAY['cites'] = 0.8`. `0.4` is `KIND_DECAY['refines']`. The two are easy to confuse but not interchangeable.",
),
(
    "What numeric value does cites contribute as a seed-edge KIND_DECAY?",
    "`0.8`. Citing block A → block B propagates 80% of A's staleness penalty into the BFS frontier seeded by that edge.",
),
(
    "Smallest of the five edge-kind decay values?",
    "`0.4` — the `refines` kind. NOT cites (which is `0.8`).",
),
(
    "Second-largest edge-kind decay value?",
    "`0.8` — the `cites` kind. Largest is `contradicts` (1.0); `cites` (0.8) is second.",
),
```

Drop the same shape for `implements=0.6` and `cooccurrence=0.5`.

### 2. Revert both eval-harness patches before v4 eval

Open `train/eval_harness.py` and restore the *original* probes.
Search for `# V4 RETRAIN TODO` markers — every one of them is a probe
that was softened for v5 and must be tightened back for v4:

```python
# qg.escape_hatch — original required = ["force", "strict"]
# lin.cites — original required = ["cites", "0.8"]
```

The v4 model has to clear those *original* requirements without any
softening. If it doesn't, the corpus rebalancing in §1 didn't take —
do not ship.

### 3. Verify against the un-softened harness

Run, with the *reverted* harness:

```bash
MM_FULLFT_DIR=<v4 weights dir> python3 train/eval_harness.py
```

Expected: 95/95 = 100% with no probe softening.

If qg.escape_hatch fails again, that is a probe-design call — split
into two probes ("which mode-level escape hatch …" + "which write-time
escape hatch …") rather than softening. Only the cites probe is a
true model-quality gate.

---

## Why we're patching instead of retraining now

The user's call:

> "it does not matter as long as we fix it in v4 release training"

v3.12.1 needs to ship today; another full retrain on the H200 (~$30,
~6h) for one factual error is not the right tradeoff when v4 retrain
is already on the roadmap and will happen against the much larger v4
surface (cognitive kernel, knowledge graph, transport, network
connectivity, governance UX). The corpus fix lands as part of that
retrain, not standalone.

---

## Honesty surface

The model card on Hugging Face must also document the v3.12.1 cites
gap explicitly so external users aren't misled by the eval number.
See `train/upload_to_hf.py` and the model-card template — add a
"Known model errors" section listing:

- `KIND_DECAY['cites']`: model returns `0.4`, correct value is `0.8`
  (per `src/mind_mem/block_lineage.py:67`). Will be fixed in v4.

If you forget to add this to the HF card, the discrepancy between the
eval and reality will be the first thing a careful reader catches.
