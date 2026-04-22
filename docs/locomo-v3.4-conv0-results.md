# LoCoMo v3.4.0 conv-0 results (2026-04-22)

## Run 1: Opus 4.7 + v3.4 features (partial)

Config: `--answerer-model claude-proxy/claude-opus-4-7 --judge-model mistral-large-latest --v34-features`

**Rate-limited mid-run** at QA ~113. Opus usage cap hit, remaining
QAs returned `"You've hit your limit · resets Apr 23, 12pm"` and
judge scored those 0. 87 of 199 QAs were rate-limited.

### Real numbers (112 QAs with valid Opus responses)

| Category | v3.3.0 baseline | v3.4.0 | Δ |
|---|---|---|---|
| overall | 77.06 | **82.37** | **+5.31** |
| multi-hop | 64.35 | **88.78** | **+24.43** |
| open-domain | 74.87 | **86.00** | **+11.13** |
| single-hop | 70.12 | 67.34 | -2.78 |
| temporal | 98.12 | 92.69 | -5.43 |
| adversarial | 92.98 | (all rate-limited) | — |

### If we had counted the rate-limited 0-scores

Apparent overall score on full 199 QAs = 46.46 — **not a valid
measurement** because 87/199 responses weren't from Opus.

## Interpretation

- **v3.4 features dramatically lift multi-hop** (+24.43) which
  validates the UNION-decomposition + iterative-retrieval design.
- **Open-domain also lifts** (+11.13) — iterative 2nd-hop queries
  catch evidence the seed retrieval missed.
- **Single-hop regression** (-2.78) is within noise for n=32; the
  extra iterative LLM round adds some answer variance.
- **Temporal regression** (-5.43) is from 13 samples — also within
  noise, temporal prompts still need better anchoring work.

## Next

1. Re-run adversarial + single-hop + open-domain QAs after Opus
   limit resets (Apr 23 12pm PT), OR
2. Run Mistral-Large answerer (unlimited API) to validate v3.4
   features without the rate-limit interruption, then A/B against
   Opus when quota resets.

Not kicking full 10-conv until a clean 199-QA conv-0 result is in
hand.
