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
make repro-verify                  # recompute EVERY committed number from its committed raw rows (seconds)
make repro-niah-smoke              # 7 cells spanning every size/depth — proves the harness, ~2 min
make repro-niah                    # FULL NIAH matrix, 250 cells — local, no API key, ~1 h
#   -> writes benchmarks/repro/<name>/{raw.ndjson,metrics.json,dataset.json,environment.json,manifest.json}
pytest tests/test_mind_ffi.py -q   # MIND kernel <-> Python baseline equivalence
```

The NIAH repro harness (`benchmarks/repro_niah.py`) imports the **same** code the test
suite runs (`tests/test_niah.py`) — it does not reimplement the benchmark — and writes a
repro package: raw per-case NDJSON, the metrics recomputed from those rows, and a
`manifest.json` pinning the commit, the effective config and its sha256, the seeds, the
adapter, the embedder (name / dims / device), k, the exclusion rule, the killed-or-crashed
count, the hardware and the wall clock, plus the sha256 of every file.

**The verifier is the point.** `make repro-verify` re-derives each unit's verdict from its
own retrieved results, recomputes the published metrics from `raw.ndjson`, and exits
non-zero if a committed number does not follow from its committed evidence. It is proven
to fail: `tests/test_repro_package.py` edits a raw row, a metric, a manifest counter and a
scorecard cell in turn and requires the verifier to reject each one. A number without a
package that passes this is not evidence, whoever published it.

## Matrix

| # | Claim | Evidence artifact | Repro command | Last verified | Independent? |
|---|-------|-------------------|---------------|---------------|--------------|
| 1 | **NIAH 250/250** (100% top-5 retrieval, 5 sizes × 5 depths × 10 needles) | **none committed.** The full-matrix package has never been in this repository; earlier revisions of this row cited `benchmarks/repro/niah/` artifacts that did not exist. The harness and the verifier are committed; the run is not. | `make repro-niah` (~1 h, writes the package) then `make repro-verify` | **never** — no committed artifact | ❌ not yet, and not first-party-verified either while no package is committed |
| 2 | **The harness produces a verifiable package** (raw rows → metrics → manifest, checkable end to end) | `benchmarks/repro/niah-smoke/` — 7 of 250 cells spanning every haystack size and depth. A machinery smoke fixture: `headline_claim: false`, and it is **not** the 250/250 figure | `make repro-verify` | 2026-09-05 (7/7 cells, 21 checks) | ❌ not yet — deterministic by construction; `metrics.determinism.decision_fingerprint` is the cross-box comparison |
| 3 | **Pinned dataset + config** (no hidden inputs) | every package's `manifest.json` carries the dataset `content_sha256` **and** the generator's own source hash, the effective config + its sha256, the seeds, k, and the exclusion rule with counts | `make repro-verify` | 2026-09-05 | ✅ self-verifying (hashes in artifact, re-checked by the verifier) |
| 4 | **Governed write prevents silent mutation** (propose → review → apply, never direct) | `propose_update` writes to `SIGNALS.md`; never touches `DECISIONS.md`/`TASKS.md` until `approve_apply` | `pytest tests/ -k "governance or propose or apply"` | see `CHANGELOG.md` | ❌ not yet — covered by repo tests |
| 5 | **MIND kernels equivalent to the Python baseline** | `tests/test_mind_ffi.py` (Q16.16 FFI vs Python scoring) | `pytest tests/test_mind_ffi.py -q` | see `CHANGELOG.md` | ❌ not yet — covered by repo tests |
| 6 | **Zero-infra / SQLite core** (no external service for the default backend) | default `mind-mem.json` backend = SQLite (`src/mind_mem/core/`); Postgres/pgvector is opt-in | `mm init <ws> && mm recall "q" <ws>` with no services running | see `CHANGELOG.md` | ❌ not yet — covered by install-smoke CI |
| 7 | **LoCoMo, full 10-conv 1986Q** (Acc>=50 73.8% / mean 70.5; canonical — see `docs/benchmarks.md`) | `benchmarks/locomo_judge.py` + `benchmarks/locomo_v1.1.0_mistral_large_full.json` (raw, 1986 rows) | `python benchmarks/locomo_judge.py --answerer-model <model> --judge-model <model> --top-k 18` (needs a judge LLM) | 2026-02-23 (`benchmarks/REPORT.md`) | ❌ **self-published** — repro harness exists; raw outputs checked in; independent rerun wanted |
| 8 | **LongMemEval-S** — measured runs, 2 reps per adapter | `docs/benchmarks/2026-09-03-longmemeval-s-full-*.{ndjson,md}` (raw per-question rows beside each scorecard) | `python3 benchmarks/repro_verify.py all` | 2026-09-05 — all 6 committed scorecards recompute exactly from their own rows (138 checks) | ❌ not yet — first-party. The prior `R@5 = 85.3` stays **RETRACTED** (`benchmarks/REPORT.md`, `benchmarks/STATUS.md`); nothing here restates it |
| 9 | **Published numbers are recomputable from committed raw rows** | `benchmarks/repro_verify.py` + `tests/test_repro_package.py` (which proves the verifier fails on a tampered row, metric, counter or scorecard cell) | `make repro-verify` | 2026-09-05 (7 targets, 159 checks) | ❌ not yet — the check anyone outside STARGA can run first |

## What "10/10" requires (and what we are NOT claiming yet)

Per an external rubric, the score is gated on **external** proof, not more code:

1. **One independent reproduction** of NIAH (row 1) — a third-party issue/PR/CI fork that
   runs `make repro-niah`, then `make repro-verify`, and reports the same
   `metrics.determinism.decision_fingerprint`. (Do not diff whole files: latency does not
   reproduce across boxes, which is why the fingerprint covers the retrieval decisions and
   leaves timing out.) This is the single biggest lever and the most welcome contribution.
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
