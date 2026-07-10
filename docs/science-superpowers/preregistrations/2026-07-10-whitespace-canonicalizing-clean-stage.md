# Pre-registration: whitespace-canonicalizing clean stage in `_clean_response`

**Frozen at commit:** (this file, committed before running the bench harness)
**Target file:** `src/mind_mem/compressors.py` (only)
**Experiment #:** 20 (autoresearch recompaction loop)

## Context / stagnation pivot

The reported metric `recompaction_score = fact_retention × convergence_rate` is
already **saturated at 1.000** by the current champion (`691a190`): the
probe-guard forces `fact_retention → 1.0` and the deterministic clean +
idempotent guard forces `convergence_rate → 1.0`. The last three accepted
iterations were all *probe-guard* tweaks (append/substring-collapse). The loop
is in EXPLORE / stagnation-pivot: a qualitatively different mechanism is
required, not another guard refinement.

The only remaining axis of improvement, since the score cannot exceed 1.0, is
the **champion tiebreaker**: among score-1.0 candidates, the champion is the one
with the lowest `compression_ratio < 1.0` (it actually compressed something).
Lower `compression_ratio` at retention=1.0/convergence=1.0 = a strictly better
champion.

## The change (qualitatively new mechanism)

Add ONE new deterministic sub-step to `_clean_response`: a **whitespace
canonicalizer** that, after the existing `<think>` / fence / preamble strips,
collapses runs of intra-line spaces/tabs to a single space and runs of blank
lines to a single blank line (and strips trailing whitespace per line). This is
NOT a guard tweak — it operates on the model *body*, not the appended probe
trailer, and it targets convergence non-determinism (item #2 "response
cleaning" in program.md), which no prior iteration addressed.

Rationale: a chat-tuned model re-emitting the same facts with drifting internal
whitespace (double spaces, extra blank lines, trailing spaces) never reaches a
byte fixed point even when the *content* has settled. Canonicalizing whitespace
removes that non-determinism source, and it strictly removes bytes, lowering
`compression_ratio`.

## Hypotheses
- **H0:** Adding whitespace canonicalization to `_clean_response` does not lower
  the benchmark's mean `compression_ratio` and/or breaks a safety property
  (echo control ≠ 1.0, retention < 1.0, convergence < 1.0, or a red test).
- **H1 (directional):** With the whitespace canonicalizer, mean
  `compression_ratio` is **≤** the pre-change value (strictly `<` on any cluster
  whose model body contains collapsible whitespace), while
  `recompaction_score` stays `1.000`, the `EchoCompressor` control stays `1.0`,
  and all 82 existing tests + ruff + mypy stay green.

## Primary analysis (exact)
- **Retention-safety proof (deterministic, not statistical):** every bench probe
  (number `\b\d[\d,]*(?:\.\d+)?\b`, quoted `"([^"]{2,80})"`, ISO date
  `\d{4}-\d{2}-\d{2}`, capitalized entity) is a single token with **no internal
  run of whitespace and no newline**. Collapsing multi-space/blank-line runs to
  a single space/newline therefore cannot delete or split any probe: if probe
  `p` was a substring of the pre-canonicalized text, `p` (containing at most one
  internal space, which canonicalization maps to exactly one space) remains a
  substring. Quoted probes may contain a single internal space; a `"a  b"` →
  `"a b"` collapse WOULD change such a probe. **Mitigation, fixed now:** the
  canonicalizer only collapses runs of ≥2 spaces to one space *outside* of the
  probe-preserving concern by operating line-wise on leading/trailing +
  inter-word runs — but to be provably safe I restrict the collapse to
  **trailing-whitespace stripping + blank-line-run collapse + leading-indent
  normalization**, and DO NOT touch inter-word single/multi spaces, so no quoted
  probe with an internal space is ever altered. (See decision rule.)
- **Idempotence proof:** each sub-op (rstrip per line, collapse ≥2 blank lines to
  1, strip leading/trailing blank lines) is a projection onto a normal form;
  applying it twice equals applying it once. So `_clean_response` stays
  idempotent, which the fixed point requires.
- **Bench:** run `recompaction_bench` against the local `mind-mem:4b` model (echo
  control + the model path) on `recall.db`; read `recompaction_score`,
  `fact_retention`, `convergence_rate`, `compression_ratio`.

## Prediction
- Direction: `compression_ratio` decreases or is unchanged; score stays 1.000.
- Magnitude: small (whitespace is a few % of body bytes); the win is
  *convergence robustness* + a strictly-not-worse tiebreaker, plus the finding
  of whether whitespace drift is a real `mind-mem:4b` convergence blocker.

## Decision rule
- **Confirm H1 if:** all 82 tests + ruff + mypy green AND the two idempotence
  tests (existing `test_ollama_compressor_cleaning_is_idempotent` +
  `test_ollama_compressor_strips_preamble_and_fences_deterministically`) stay
  green AND no existing exact-output test regresses (the canonicalizer must be a
  no-op on already-canonical single-line bodies like "plain text response").
- **Disconfirm / revert if:** any test goes red, OR the canonicalizer alters any
  quoted-probe-with-internal-space (proven impossible by restricting to
  line-boundary whitespace only), OR the echo control drops below 1.0 (would be
  a harness/mechanism break, investigate before claiming).

## Safety-rule check (the three inviolable rules)
1. Fixed point, not fixed count: unchanged — still converges on byte-identity,
   still raises `NonConvergenceError` at the bound. Whitespace canonicalization
   only *removes* a non-determinism source, never papers over oscillation.
2. Retention floor: untouched (`recompaction.py` not edited).
3. Purity: unchanged — no clock, no RNG, temperature=0/seed pinned.
4. Not a no-op: the model still compresses; canonicalization removes bytes
   (`compression_ratio` can only drop), so `changed`/ratio gates still fire.

## Multiplicity
- One confirmatory comparison (compression_ratio before vs after), one primary
  decision rule. No forking.

## Exploratory (labeled)
- Iteration histogram before/after, to see if whitespace drift was actually
  costing extra passes on `mind-mem:4b` (a finding for the retrain question).
