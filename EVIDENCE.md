# Evidence Matrix

> Every headline claim → the artifact that backs it → the exact command to reproduce it
> → when it was last verified → whether an **independent** (non-STARGA) party has reproduced
> it. We would rather you **rerun** this than trust it.
>
> The discipline: a claim that cannot be reproduced from a pinned command is a marketing
> claim, not evidence. Anything not yet independently reproduced is labelled as such — no
> claim is dressed up as more-verified than it is.

## How to reproduce (one command each)

```bash
pip install -e ".[benchmark]"      # or: pip install mind-mem
make repro-niah                    # NIAH 250/250 — local, no API key, ~15-19 min
#   -> writes benchmarks/repro/niah/{results.jsonl,dataset.json,aggregate.json,environment.json,manifest.json}
#   rerun and diff manifest.json: the deterministic artifacts hash identically.
pytest tests/test_mind_ffi.py -q   # MIND kernel <-> Python baseline equivalence
```

The NIAH repro harness (`benchmarks/repro_niah.py`) imports the **same** code the test
suite runs (`tests/test_niah.py`) — it does not reimplement the benchmark — and emits raw
per-case results plus a `manifest.json` of sha256 hashes so a third party can rerun and
diff byte-for-byte.

## Matrix

| # | Claim | Evidence artifact | Repro command | Last verified | Independent? |
|---|-------|-------------------|---------------|---------------|--------------|
| 1 | **NIAH 250/250** (100% top-5 retrieval, 5 sizes × 5 depths × 10 needles) | `benchmarks/repro/niah/manifest.json` + `results.jsonl` (raw per-case) | `make repro-niah` | 2026-06-03 (`benchmarks/repro/niah/`) | ❌ not yet — **rerunnable + byte-diffable**, third-party repro wanted |
| 2 | **Deterministic / byte-identical replay** of the retrieval result set | `manifest.json` sha256s match across independent runs | `make repro-niah` twice; `diff` the two `manifest.json` | 2026-06-03 (verified: 2 runs → identical `results.jsonl`/`dataset.json`/`aggregate.json` hashes) | ❌ not yet — deterministic by construction |
| 3 | **Pinned dataset + config** (no hidden inputs) | `benchmarks/repro/niah/dataset.json` with `dataset_sha256` (needles + matrix + recall config) | `make repro-niah` (writes `dataset.json`) | 2026-06-03 | ✅ self-verifying (hash in artifact) |
| 4 | **Governed write prevents silent mutation** (propose → review → apply, never direct) | `propose_update` writes to `SIGNALS.md`; never touches `DECISIONS.md`/`TASKS.md` until `approve_apply` | `pytest tests/ -k "governance or propose or apply"` | see `CHANGELOG.md` | ❌ not yet — covered by repo tests |
| 5 | **MIND kernels equivalent to the Python baseline** | `tests/test_mind_ffi.py` (Q16.16 FFI vs Python scoring) | `pytest tests/test_mind_ffi.py -q` | see `CHANGELOG.md` | ❌ not yet — covered by repo tests |
| 6 | **Zero-infra / SQLite core** (no external service for the default backend) | default `mind-mem.json` backend = SQLite (`src/mind_mem/core/`); Postgres/pgvector is opt-in | `mm init <ws> && mm recall "q" <ws>` with no services running | see `CHANGELOG.md` | ❌ not yet — covered by install-smoke CI |
| 7 | **LoCoMo** (mean 77.9 / adversarial 82.3 / temporal 88.5) | `benchmarks/locomo_harness.py` + `benchmarks/locomo_*results*.json` | `python3 benchmarks/locomo_harness.py` (needs a judge LLM) | `locomo_*_20260304.json` | ❌ **self-published** — repro harness exists; raw outputs checked in; independent rerun wanted |
| 8 | **LongMemEval** | — | — | — | ⏸️ **HELD** — provenance hold active (`benchmarks/STATUS.md`); intentionally excluded from positioning until reconciled |

## What "10/10" requires (and what we are NOT claiming yet)

Per an external rubric, the score is gated on **external** proof, not more code:

1. **One independent reproduction** of NIAH (row 1) — a third-party issue/PR/CI fork that
   runs `make repro-niah` and reports the same `manifest.json` hashes. This is the single
   biggest lever and the most welcome contribution. Open an issue with your `manifest.json`.
2. **Independent security review** + SLSA L3 / signed releases / SBOM (roadmap, not done).
3. **A named external user / integration** not controlled by STARGA (roadmap, not done).

We are **not** claiming independent reproduction, a third-party audit, or external
production users today. Those rows are marked ❌ on purpose. The honest current state:
strong, reproducible **first-party** evidence with a clear, one-command path for anyone
outside STARGA to verify it.

> Positioning note: the scoring core is **Python today**; the MIND-language port that
> compiles to a native `.so` is forward-looking (see README "MIND Kernels"). Row 5 proves
> the MIND kernels match the Python baseline where wired; it does not claim the core is
> already native MIND.
