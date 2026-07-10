# Pre-registration: strip leading <think></think> block in _clean_response

**Experiment:** autoresearch #9 (recompaction). Editable file: src/mind_mem/compressors.py only.

## Root cause (observed BEFORE registration, exploratory grounding)
Empirically, `mind-mem:4b` via the CURRENT prompt does NOT echo verbatim (contra the
iter-2 dead-end note). It compresses (cluster 0: 4983 -> 2699 bytes on pass 1) but
prepends an empty Qwen3-family reasoning block `<think>\n\n</think>\n\n` that the
current `_clean_response` leaves in place. This is non-summary noise carried through
byte-comparison and inflates every output.

## Hypotheses
- H0: stripping a leading `<think>...</think>` block in `_clean_response` does not
  change recompaction_score vs the current champion (0.0) and/or breaks a gate.
- H1 (directional): it yields recompaction_score > 0.0 with compression_ratio < 0.999,
  all three inviolable rules + GATE-A/B/C intact.

## Primary analysis (exact)
- Change: add an idempotent regex strip of a single leading `<think> ... </think>`
  block (DOTALL, only at string start, after existing strip/fence handling) to
  `_clean_response`. No other change.
- Metric: `recompaction_score` from run_mind_recompaction.sh (RECOMPACT_MODEL=mind-mem:4b,
  live corpus), = fact_retention * convergence_rate.
- Gates that must stay green: ruff+mypy; GATE-A (tests/test_recompaction.py +
  tests/test_compressors.py); GATE-B (echo control == 1.0); GATE-C (ratio < 0.999).

## Prediction
- Direction: recompaction_score strictly increases from 0.0 to > 0.0.
- The 7 existing `_clean_response` parametrized cases contain NO `<think>` tags, so
  they are unaffected (idempotence + prefix-only guarantee). Echo control unaffected
  (EchoCompressor does not call `_clean_response`).

## Decision rule
- Confirm H1 if: full run emits recompaction_score > 0.0 AND compression_ratio < 0.999
  AND GATE-A/B/C all pass (script does not DISCARD).
- Disconfirm / null if: recompaction_score == 0.0, OR any gate fails, OR ratio >= 0.999.

## Falsifiable failure mode (watch-it-fail)
If mind-mem:4b's per-cluster outputs still fail to reach a byte fixed point once the
think-tag is gone (e.g. it paraphrases the remaining body differently each pass), the
bench records NonConvergenceError -> score stays 0.0. That is a DISCONFIRM and a
finding (record trajectory in dead_ends.md), NOT a reason to raise max_iterations.

## Sample size & stopping
- Clusters: the wrapper default (30). No optional stopping; one full run is the result.

## Multiplicity
- One confirmatory metric (recompaction_score). No forking.

## Non-negotiables preserved
- Fixed point, not fixed count: max_iterations untouched; NonConvergenceError raise
  untouched.
- Retention floor untouched.
- Purity: temperature=0/seed pinned untouched; the new strip is a pure, idempotent,
  deterministic prefix removal.
- Not a no-op: strip only removes a leading think block; real body still emitted, so
  ratio < 1.0 remains (observed 0.54 on cluster 0 before strip).
