# MIND configuration vs MIND language — clarifying the .mind extension

## TL;DR

The `.mind` files in this repository are NOT the MIND programming language.
They are declarative configuration files in an INI-style format
(`[section]` / `key = value`) used to tune mind-mem's scoring kernels at
runtime. The MIND programming language itself lives at
[github.com/star-ga/mind](https://github.com/star-ga/mind).

---

## What's in `mind/*.mind` in this repo

mind-mem ships 26 configuration files under the `mind/` directory. Each file
tunes a specific pipeline stage and is parsed at runtime by
`load_kernel_config()` in `src/mind_mem/mind_ffi.py`. They contain no
compiled code — only declarative key/value tuning data.

| File | What it tunes |
|------|---------------|
| `hybrid.mind` | RRF fusion parameters — `rrf_k`, BM25/vector weight multipliers, vector similarity threshold |
| `ranking.mind` | Ranking-pipeline scoring weights and cutoff strategy |
| `governance.mind` | Rationale gate, audit-chain invariants, concurrency lock timeout |
| `recall.mind` | Recall-flow parameters — candidate limits, RM3 expansion, knee cutoff |
| `truth.mind` | Truth-evaluation thresholds and confidence bands |
| `cognitive.mind` | Cognitive-kernel parameters — KernelKind dispatch, surprise threshold |
| `cross_encoder.mind` | Cross-encoder pipeline — model selection, batch size, score floor |
| `graph.mind` | Knowledge-graph edge scoring, co-retrieval boost multiplier |
| `query_plan.mind` | Query-planning budget, intent-routing weights |
| `intent.mind` | Intent-classification confidence weights per intent type |
| `bm25.mind` | BM25F field weights, stemmer, IDF floor |
| `rm3.mind` | RM3 query-expansion — feedback term count, lambda |
| `reranker.mind` | Reranker stage parameters |
| `rerank.mind` | Rerank score combination weights |
| `rrf.mind` | RRF standalone parameters (used by the federation merge path) |
| `temporal.mind` | Temporal-decay curve and freshness weight |
| `trajectory.mind` | Trajectory-scoring parameters for session continuity |
| `prefetch.mind` | Prefetch-trigger thresholds and TTL |
| `adversarial.mind` | Adversarial-probe guard thresholds |
| `category.mind` | Category-summary scoring parameters |
| `importance.mind` | Importance scoring decay and floor |
| `ensemble.mind` | Ensemble combiner weights |
| `evidence.mind` | Evidence-tier routing parameters |
| `session.mind` | Session-context scoring and decay |
| `abstention.mind` | Abstention-threshold tuning |
| `answer.mind` | Answer-extraction scoring parameters |

Each file parses as `{ section: { key: value } }` via `load_kernel_config()`.
Values are typed automatically: integers (`60`), floats (`1.0`), booleans
(`true` / `false`), and comma-separated lists are all recognized.

Example — the first few lines of `mind/hybrid.mind`:

```ini
[fusion]
rrf_k = 60
bm25_weight = 1.0
vector_weight = 1.0

[vector]
model = all-MiniLM-L6-v2
enabled = false
min_similarity = 0.3
```

---

## What MIND-the-language actually is

MIND is a statically-typed, tensor-oriented systems language designed for
deterministic, auditable computation across heterogeneous substrates.

Key properties:

- **First-class fixed-point arithmetic** — `q16` (Q16.16) is a native dtype,
  not a library type.
- **Cross-substrate bit-identity contract** — for integer and Q16.16
  fixed-point computation, given the same input bytes, the CPU backends
  (x86 AVX2, ARM NEON) produce the exact same output bytes; scalar-float
  cross-ISA verification and GPU backends are on the roadmap.
- **Determinism as a type property** — the `#[deterministic]`, `#[target(...)]`,
  and `#[q16]` annotations are enforced by the type checker, not just by
  convention.
- **Compiler with a self-hosting bootstrap/front-end** — `mindc` at
  [github.com/star-ga/mind](https://github.com/star-ga/mind)
  compiles `.mind` files that contain MIND language source.
- **Normative specification** at [github.com/star-ga/mind-spec](https://github.com/star-ga/mind-spec).
- **Public landing** at [https://mindlang.dev](https://mindlang.dev).

Sample MIND language code — this is what `mindc` compiles, and it is
**not** what any file in this repository's `mind/` directory contains:

```mind
pub fn dot_q16(a: Tensor<q16, [N]>, b: Tensor<q16, [N]>) -> q16 {
    (a .* b).sum()
}
```

```mind
module scoring {
    #[deterministic]
    #[q16]
    pub fn rrf_score(rank: u32, k: q16) -> q16 {
        q16::ONE / (k + rank as q16)
    }
}
```

If you see `pub fn`, `module`, `Tensor<...>`, or `#[...]` attribute syntax,
you are reading MIND language. If you see `[section]` headers and `key = value`
lines, you are reading mind-mem configuration data.

---

## Why the same extension

The `mind/` configuration files predate the public `mindc` compiler shipping.
When mind-mem first introduced its Q16.16 scoring kernels, `.mind` was an
internal-only convention meaning "tuning data consumed by the MIND kernel
runtime." As mind-mem went public on PyPI and the MIND programming language
matured as a separate, publicly released project, the overloaded extension
created an ambiguity at the public-facing surface.

A user browsing [mind-mem on GitHub](https://github.com/star-ga/mind-mem) or
inspecting their `pip` environment may open a `.mind` file, see `[fusion]` and
`rrf_k = 60`, and conclude that "MIND is an INI-style config language." That
conclusion is incorrect and dilutes the language identity that the compiler and
specification define.

---

## How to tell which is which at a glance

| Signal | Meaning |
|--------|---------|
| `[section]` / `key = value` lines | mind-mem **configuration** (this repo) |
| `pub fn name(...) -> ...` | MIND **language** |
| `module name { ... }` | MIND **language** |
| `Tensor<dtype, [dims]>` | MIND **language** |
| `#[deterministic]` / `#[q16]` / `#[target(...)]` | MIND **language** attribute annotations |

When in doubt: if the file compiles with `mindc build`, it is MIND language.
If it is parsed by `load_kernel_config()`, it is configuration data.

---

## Resolution roadmap

Tasks **#297** and **#298** in the STARGA strategic queue cover renaming
mind-mem's configuration files from `.mind` to `.mindcfg` and shipping them
properly inside the PyPI wheel.

The rename is deferred from this point because it is a coordinated batch
change: every call site that constructs a kernel config path must be updated
in the same commit (`load_kernel_config`, `get_mind_kernel`, the MCP tool that
surfaces kernel names to callers, and the wheel's `package_data` manifest).
Doing the rename piecemeal would break the v4.0.x production surface mid-flight.

This document is the disambiguation users need **today**, while the rename is
in flight. Once tasks #297 + #298 land:

- All configuration files in this repo will carry the `.mindcfg` extension.
- The `load_kernel_config()` loader will be updated to the new paths.
- The wheel will include the renamed files under `package_data`.
- This document will be updated to reflect the resolved state.
- The glossary entry for "MIND kernel" will be tightened accordingly.

Task **#299** (README repositioning on byte-identical replay) is tracked
separately and handled by a parallel workstream.

---

## Related projects — the open MIND ecosystem

**mindc** ([github.com/star-ga/mind](https://github.com/star-ga/mind))
The compiler for the MIND programming language (its bootstrap/front-end
self-hosts). Accepts `.mind`
files containing MIND language source (`pub fn`, `module`, `Tensor`, etc.) and
produces compiled artifacts. This is the canonical tool that defines what "a
MIND file" means in the language sense. Does not parse INI syntax.

**mind-spec** ([github.com/star-ga/mind-spec](https://github.com/star-ga/mind-spec))
The normative specification for the MIND language, covering syntax, type
system, determinism contract, and the Q16.16 fixed-point semantics.

**mind-nerve** ([github.com/star-ga/mind-nerve](https://github.com/star-ga/mind-nerve))
Agent-routing preselector written in and compiled from MIND language source.
Its `.mind` files contain actual MIND language (compiled by `mindc`), not
configuration data.

**rfn-mind** ([github.com/star-ga/rfn-mind](https://github.com/star-ga/rfn-mind))
Training framework written in MIND language. Similarly compiled by `mindc` —
its `.mind` files are MIND language source.

**mind-inference** ([github.com/star-ga/mind-inference](https://github.com/star-ga/mind-inference))
MIND-native inference pipeline. Also compiled by `mindc` from MIND language
source.

None of these projects use the INI-style `[section]` / `key = value` format.
The INI convention is specific to mind-mem's scoring-kernel configuration layer.

---

## Cross-references

- `docs/glossary.md` — "MIND kernel" entry (refers to the configuration files
  described in this document; will be updated alongside tasks #297 + #298).
- `src/mind_mem/mind_ffi.py` — `load_kernel_config()` at line 579 is the
  runtime parser for all files in `mind/*.mind`.
- `mind/hybrid.mind` — canonical example of the INI-style configuration format.
- Tasks #297 (`.mindcfg` rename), #298 (wheel packaging fix), #299 (README
  repositioning) are tracked in STARGA's strategic queue.
