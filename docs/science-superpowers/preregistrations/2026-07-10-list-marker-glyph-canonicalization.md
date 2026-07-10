# Pre-registration: list-marker glyph canonicalization for recompaction convergence

**Frozen at commit:** (this file's commit)
**Target file:** `src/mind_mem/compressors.py`
**Experiment:** autoresearch recompaction #24 (EXPLORE / stagnation pivot)

## Background / gap

The metric `recompaction_score = fact_retention × convergence_rate` is at the
1.000 champion ceiling. The real headroom is *convergence robustness*: which
small-model output shapes can reach a byte fixed point at all. Every prior
champion axis (`_dedup_lines`, `_canonicalize_whitespace`,
`_canonicalize_bullet_order`, the probe guard, `num_predict`, the prompt) either
handles whitespace/order/duplicate drift or floors retention.

**Unhandled failure mode (novel):** the sole list-recognition regex in the file
is `_BULLET_RE = re.compile(r"^- ")` — it matches ONLY the `- ` marker. A
`mind-mem:4b`-shaped model asked for a one-fact-per-line list has no stable
prior on the *list glyph* and re-emits the identical fact set under a DIFFERENT
marker pass-to-pass (`* fact`, `+ fact`, `1. fact`, `1) fact`, `• fact`). Two
byte-distinct marker choices for the same facts:
1. are NOT seen as bullets by `_canonicalize_bullet_order` → not sorted →
   ordering drift not collapsed,
2. are NOT equal to their `- `-prefixed twin by `_dedup_lines` → duplicate not
   collapsed,
3. therefore oscillate forever → `NonConvergenceError` → that cluster scores 0.

This is a distinct *convergence* axis (marker GLYPH, not marker ORDER).

## Hypotheses
- H0: normalizing leading list-marker glyphs to a single canonical `- ` marker
  does NOT change whether a marker-glyph-drifting model output reaches a fixed
  point, and/or perturbs retention / echo control / idempotence.
- H1 (directional): adding a deterministic, idempotent `_canonicalize_list_markers`
  sub-step to `_clean_response` (rewrite a leading `*`/`+`/`•`/`1.`/`1)`-style
  ordered/unordered list marker to `- `) rescues a marker-glyph-drifting output
  from non-convergence (score 0) to convergence, while leaving retention=1.0,
  the echo control=1.0, and existing behavior byte-unchanged.

## Primary analysis (exact)
- Add `_canonicalize_list_markers(text)` to `compressors.py`, called inside
  `_clean_response` BEFORE `_dedup_lines` (so glyph-normalized lines dedup and
  bullet-order-canonicalize as `- ` lines).
- Verification, all run BEFORE claiming:
  1. **Convergence rescue (through the REAL loop):** a stub compressor that
     emits the same fact set alternating markers (`* `/`- `) across passes goes
     from `NonConvergenceError` (score 0) to `converged=True` via
     `recompact_cluster`, with `changed` well-defined and `fact_retention=1.0`
     on the source probes.
  2. **Idempotence:** `_clean_response(_clean_response(x)) == _clean_response(x)`
     for marker-normalized text (required for the fixed point).
  3. **Retention-safety:** normalizing only the leading marker glyph (never the
     fact text after it) leaves every bench probe (numbers, quoted IDs, ISO
     dates, capitalized entities) a substring of the output.
  4. **Echo control unchanged:** `EchoCompressor` still returns input verbatim
     (it does not route through `_clean_response`).
  5. **No regression:** full `pytest tests/test_compressors.py` +
     `tests/test_recompaction.py` + `tests/test_recompaction_bench.py` green;
     `ruff` + `mypy` green. In particular the read-only cleaning/idempotence
     tests (`test_ollama_compressor_strips_preamble_and_fences_deterministically`,
     `test_ollama_compressor_cleaning_is_idempotent`) must still pass — the new
     step must be a no-op on their non-list inputs.

## Prediction
- Direction: convergence_rate strictly UP on marker-glyph-drifting outputs
  (0 → converges); `recompaction_score` on the scored path unchanged at 1.000
  (retention stays 1.0, echo control stays 1.0); `compression_ratio` unchanged
  or slightly lower (canonical `- ` marker is ≤ the width of `1. `/`1) `).

## Decision rule
- Confirm H1 if: the marker-drifting stub converges through the real
  `recompact_cluster` loop (was `NonConvergenceError`), retention on its probes
  is 1.0, the two read-only cleaning tests still pass, echo control is verbatim,
  and the full targeted suite + ruff + mypy are green.
- Disconfirm / revert if: any inviolable rule breaks (retention floor lowered,
  purity lost, echo control ≠ 1.0, a no-op-required test fails), OR the step is
  not idempotent, OR it changes any existing non-list cleaning output.

## Safety w.r.t. the three inviolable rules
1. Fixed point: the step is a pure projection onto a canonical marker (leading
   glyph → `- `), so applying it twice equals once → idempotent → the loop's
   byte-equality test is preserved; no `max_iterations` change, no silent
   last-iterate return.
2. Retention floor: untouched (no floor edit); the step only rewrites a leading
   marker glyph and never deletes a fact line, so retention is provably ≥ prior.
3. Purity: no clock, no RNG, temperature=0/seed intact — a regex substitution is
   a pure function of its input.
4. Not a no-op compressor: only normalizes marker glyphs; the model still does
   the consolidation, `changed`/`compression_ratio` still catch echo behavior.

## Exploratory (labeled, not confirmatory)
- Whether real `mind-mem:4b` actually drifts markers on the live corpus (needs
  a GPU run; recorded as a finding if observed, not claimed here).
