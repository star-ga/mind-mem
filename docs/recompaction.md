# Iterative Re-Compression Engine (Recompaction)

**Module:** `src/mind_mem/recompaction.py` (268 lines, 18 tests, 99% coverage)

**Status:** The re-compression engine is **fully implemented and tested**, and the injected compressor now has concrete implementations (`compressors.py`: `EchoCompressor` control + `OllamaCompressor`) plus a fact-retention benchmark (`bench/recompaction_bench.py`) driven by an autoresearch harness against the real corpus. **What is NOT yet wired: a `mm recompact` CLI command that clusters and routes results through `propose_update`, and a before/after recall benchmark on the governed corpus.**

---

## What It Is (One-Sentence Scope)

A deterministic engine that takes a cluster of semantically-related memory blocks, re-reads them together, re-compresses them to a tighter summary, and repeats until the rewrite stops changing (reaches a fixed point) — then emits the settled text as a human-reviewable proposal.

The compressor implementation is injected by the caller, so the loop, convergence detection, ordering-independence, and retention checks are testable with zero model calls in CI.

---

## Why Fixed Point, Not Fixed Count

Hyperparameter-based iteration caps hide failure to converge. Example: a compressor that misunderstands or paraphrases the input blocks might never settle — on the 4th iteration it might still be producing new text. Stopping at iteration 4 and emitting that text as "converged" is silently returning a non-fixed-point rewrite that looks indistinguishable from a real one.

MIND-Mem follows the same discipline as the mic@1 self-host fixed-point gate: the loop stops when the rewrite is byte-identical to its input (`rewritten == current`), or raises `NonConvergenceError` at a hard bound (default 6 iterations). The caller can choose to retry with a different compressor or config, but can never accidentally emit a partial rewrite as settled.

---

## Why the Compressor Is Injected

The scoring loop, convergence detection, order-independent cluster digest, and retention floor are side-effect-free functions of the input blocks and the compressor's output. By making the compressor a `Callable[[str, list[dict]] -> str]`, the entire engine becomes **testable deterministically with zero model API calls**.

This is what `tests/test_recompaction.py` does: it injects stand-in compressors whose convergence behavior is fixed by their own code — one that returns its input unchanged (settles on the first pass), one that shrinks by a known step count, and one that oscillates and must therefore raise `NonConvergenceError`. Together they exercise the loop, convergence detection, cluster-digest stability, and the retention floor without ever calling a model.

Note what this does *not* establish: these fixtures converge because they were written to. Nothing here predicts whether a real compressor will.

---

## Load-Bearing Invariants

### 1. Fixed Point, Not Fixed Count
The loop runs until `rewritten == current` or raises `NonConvergenceError(max_iterations)` at the bound. Never silently truncates.

### 2. Order-Independent Cluster Digest
A cluster is a **set** of blocks; its identity must not depend on retrieval order. The `cluster_digest()` function sorts the per-block body digests before hashing, so permutation is invisible but content changes are always caught.

### 3. Retention Floor
The converged text must be at least 25% (configurable via `min_retention_ratio`) of the **largest single source block** body length. A rewrite that collapses below this floor is treated as data loss and rejected with a `ValueError`, not silently accepted.

### 4. Never Mutates Source of Truth
The input `blocks` list and all block dicts are never modified. The source of truth is untouched. The result is a **proposal** — applying it to the store is a separate, HITL-gated step through `propose_update`.

### 5. Full Provenance
Every result carries:
- `source_ids`: tuple of all source block IDs
- `input_digest`: the order-independent cluster digest at start
- `output_digest`: the converged text digest
- `trajectory`: tuple of all intermediate rewrites (for audit)

---

## API

```python
from mind_mem.recompaction import recompact_cluster, RecompactionConfig, NonConvergenceError

result = recompact_cluster(
    blocks=[...],  # list of block dicts from find_similar
    compressor=my_llm_compress,  # (current_text, blocks) -> new_text
    config=RecompactionConfig(max_iterations=6, min_retention_ratio=0.25),
    seed_text=None,  # optional starting summary; defaults to concatenated block bodies
)

if result.converged:
    if result.changed:
        # Emit proposal via propose_update
        propose_update(..., body=result.text, supersedes=result.source_ids)
    else:
        # Cluster already converged; no tightening found
        pass
else:
    # Should not happen (exception raised instead), but defensive:
    log.error(f"Unexpected non-converged result: {result}")
```

### Exception Handling

- **`NonConvergenceError(iterations: int)`**: Raised when no fixed point is reached within `config.max_iterations`. This is a **diagnostic**, not a bug — it means the compressor is unsuitable for this cluster (e.g., it paraphrases or grinds on the input).
- **`ValueError`** (retention floor): Raised when the converged text falls below the retention floor. This signals data loss, not a win.

---

## Failure Modes and Their Meanings

| Failure | Meaning | Action |
|---------|---------|--------|
| `NonConvergenceError` | Compressor doesn't settle (paraphrases, adds new info, or grinds). | Try a different compressor model (e.g., a more careful one, or a larger one with fewer paraphrases). This is a **model selection problem**, not an algorithm bug. |
| `ValueError` (retention floor) | Converged text deleted information. | Reject the rewrite. Increase `min_retention_ratio` if floor was too strict, or retry with a compressor that preserves fidelity. |
| `changed=False` | Cluster was already converged on input. | No action needed; this cluster is already in its stable form. |

---

## Open Work

### (a) Real LLM Behind the Compressor — DONE
`src/mind_mem/compressors.py` ships two concrete `Compressor` implementations:
`EchoCompressor` (returns its input verbatim — the trivially-converging control
any harness must score at perfect convergence and perfect retention) and
`OllamaCompressor` (calls a local ollama model over HTTP, pinning
`temperature=0` and a fixed integer `seed` so a byte-identical fixed point is
legitimately reachable). A fact-retention benchmark
(`src/mind_mem/bench/recompaction_bench.py`,
`--model <echo|ollama-tag> --clusters N --db <path>`) prints a
machine-greppable `recompaction_score: <float>` where score =
`fact_retention * convergence_rate`; an autoresearch harness
(`config.mind-recompaction.yaml` + `program.mind-recompaction.md` +
`run_mind_recompaction.sh`) drives it against the real ~1469-block corpus.

**Empirical finding (live bench, 2026-07-10):** `mind-mem:4b` **converges** —
reaches a fixed point in ~2 passes and changes text (not a no-op) — but is
**lossy**: it over-compresses and trips the retention floor on a meaningful
fraction of clusters, so the safety gate rejects those rewrites. This is the
"converges but degrades" outcome, i.e. the case for a **narrow retrain** (see
(d)).

### (b) `mm recompact` CLI Command + Dream-Cycle Pass
A CLI verb that:
1. Clusters memory blocks via `find_similar`
2. Runs recompaction on each cluster
3. Routes non-trivial results (where `changed=True`) through `propose_update`
4. Integrates with the existing dream-cycle idle-time enrichment pass

**Scope:** Likely a new `dream_cycle_pass_6_recompaction.py` function, wired into the scheduler.

### (c) Before/After Recall Benchmark (Real Corpus)
Run the LoCoMo benchmark on the governed corpus with and without recompacted clusters. Measure recall accuracy delta. **Do not take the prior-art numbers (reported 10-15pp gains on long-context math) on faith** — verify on our own corpus with our own model.

**Gate:** This result determines whether recompaction is worth the CPU cost at scale.

### (d) `mind-mem:4b` Retrain Decision
The 2026-07-10 bench (item (a)) already establishes that `mind-mem:4b` converges
but degrades. The open decision is **use-as-is vs a narrow retrain**, gated on
the recall benchmark in (c). Retrain target, if pursued: given N related blocks,
emit one block such that (1) re-feeding its own output reproduces it (fixed
point) **and** (2) recall over the merged block answers every question the source
blocks answered. Still worth measuring alongside: iterations-to-convergence
distribution, retention-floor rejection rate, and whether a larger or oracle
model (e.g. a frontier model) clears the floor without retraining.

---

## Naestro Warning (Load-Bearing)

**Do NOT port this to Naestro's memory vault as an in-place rewrite.**

Naestro's vault is **append-only with provenance chains**. Every decision (a memory write) is immutable once written and carries a chain of evidence linking it to prior decisions. Re-compressing a vault block **in place** would:

1. Break the append-only guarantee
2. Invalidate the evidence chain — a replayed decision would reference a different version of the block than the chain proves

The correct port to Naestro is to **append a superseding block** with a back-pointer to the original cluster. This is a **materialized view**, not "sleep":

```
[original block 1]      → [new superseding block with source_ids]
[original block 2]      → (points to originals, proves derivation)
[original block 3]      →
```

The vault remains append-only, the chain remains intact, and governance can replay the decision exactly. This is a separate feature from recompaction; don't conflate them.

---

## Performance and Metrics

**No recall-improvement number is claimed, because recall has not been measured.**

A real model (`mind-mem:4b` via `OllamaCompressor`) has now run through the loop
against the governed corpus (bench, 2026-07-10): it converges in ~2 passes but
trips the retention floor on a meaningful fraction of clusters (see (a)). What
this does *not* establish is an end-to-end **recall** delta — the retention
bench is a regex fact-preservation proxy, not a recall test. The CI test suite
still injects stand-in compressors chosen *because* they converge (or oscillate)
on demand, so a convergence rate computed over them is a tautology about the
fixtures, not a property of the engine under a real compressor.

Producing the recall number is open work items (c) and (d) above. Until the
recall benchmark runs against the governed corpus with a real compressor:

- Do not assume any local model reaches a fixed point. A compressor that
  paraphrases never will, and `NonConvergenceError` is the diagnostic that says so.
- Do not assume re-compression helps recall. Prior-art gains reported elsewhere
  were measured on other corpora with other models and do not transfer on faith.

The only bounded claim the code supports today: the loop is `O(cluster size)` in
memory with no unbounded buffers, and it either reaches a byte-identical fixed
point or raises.

---

## Tests

- **`tests/test_recompaction.py`** — 18 tests covering:
  - Fixed-point convergence with mock compressors
  - Non-convergence detection (raises `NonConvergenceError`)
  - Retention floor validation
  - Order-independent cluster digest (permutation stability)
  - Source-of-truth immutability (blocks dict never modified)
  - Trajectory tracking (intermediate rewrites logged)
  - Single-block edge case (early exit, no model call)
  - Config validation (bounds checking)
- **`tests/test_compressors.py`** — 25 tests covering the `EchoCompressor`
  control (verbatim echo, instant fixed point) and `OllamaCompressor`
  (deterministic `temperature=0` + fixed seed request options, HTTP error paths
  raising `CompressorError` rather than faking convergence via an input echo).

All 43 tests pass; `recompaction.py` holds 99% code coverage.

---

## See Also

- **ROADMAP.md** §Group H, item "Iterative re-compression" — strategic context and open work
- **`src/mind_mem/recompaction.py`** — source code (268 lines, extensively documented)
- **`src/mind_mem/compressors.py`** — `EchoCompressor` control + `OllamaCompressor`
- **`src/mind_mem/bench/recompaction_bench.py`** — fact-retention benchmark (`recompaction_score:`)
- **Naestro warning above** — append-only semantics incompatibility
