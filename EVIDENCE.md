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
| 1 | **NIAH 250/250** (100% top-5 retrieval, 5 sizes × 5 depths × 10 needles) | `benchmarks/repro/niah/{raw.ndjson,metrics.json,dataset.json,environment.json,manifest.json}` — committed in b36ff3a on a clean tree (`repo_tracked_files_dirty_at_run: false`), pinned to commit `acb5a8b`. Two earlier 250/250 runs were discarded because they ran against a dirty tree and were not reproducible by a checkout. | `make repro-verify` (recomputes the metrics from the committed raw rows) or `make repro-niah` to regenerate from scratch | 2026-09-05 (250/250 cells, 22 checks, `decision_fingerprint` `33d5f8e6282e9844455c4d20c8701a2fe96f29cf6d33bf47278f5ab1dcc9570b`) | ❌ not yet — first-party only. The package is now committed and checkable end to end; nobody outside STARGA has re-run it and reported the same fingerprint |
| 2 | **The harness produces a verifiable package** (raw rows → metrics → manifest, checkable end to end) | `benchmarks/repro/niah-smoke/` — 7 of 250 cells spanning every haystack size and depth. A machinery smoke fixture: `headline_claim: false`, and it is **not** the 250/250 figure | `make repro-verify` | 2026-09-05 (7/7 cells, 22 checks) | ❌ not yet — deterministic by construction; `metrics.determinism.decision_fingerprint` is the cross-box comparison |
| 3 | **Pinned dataset + config** (no hidden inputs) | every package's `manifest.json` carries the dataset `content_sha256` **and** the generator's own source hash, the effective config + its sha256, the seeds, k, and the exclusion rule with counts | `make repro-verify` | 2026-09-05 | ✅ self-verifying (hashes in artifact, re-checked by the verifier) |
| 4 | **Governed write prevents silent mutation** (propose → review → apply, never direct) | `propose_update` writes to `SIGNALS.md`; never touches `DECISIONS.md`/`TASKS.md` until `approve_apply` | `pytest tests/ -k "governance or propose or apply"` | see `CHANGELOG.md` | ❌ not yet — covered by repo tests |
| 5 | **MIND kernels equivalent to the Python baseline** | `tests/test_mind_ffi.py` (Q16.16 FFI vs Python scoring) | `pytest tests/test_mind_ffi.py -q` | see `CHANGELOG.md` | ❌ not yet — covered by repo tests |
| 6 | **Zero-infra / SQLite core** (no external service for the default backend) | default `mind-mem.json` backend = SQLite (`src/mind_mem/core/`); Postgres/pgvector is opt-in | `mm init <ws> && mm recall "q" <ws>` with no services running | see `CHANGELOG.md` | ❌ not yet — covered by install-smoke CI |
| 7 | **LoCoMo, full 10-conv 1986Q** (Acc>=50 73.8% / mean 70.5; canonical — see `docs/benchmarks.md`) | `benchmarks/locomo_judge.py` + `benchmarks/locomo_v1.1.0_mistral_large_full.json` (raw, 1986 rows) | `python benchmarks/locomo_judge.py --answerer-model <model> --judge-model <model> --top-k 18` (needs a judge LLM) | 2026-02-23 (`benchmarks/REPORT.md`) | ❌ **self-published** — repro harness exists; raw outputs checked in; independent rerun wanted |
| 8 | **LongMemEval-S** — measured runs, both arms same box same day | `docs/benchmarks/head-20260907/*.{ndjson,md}` (raw per-question rows beside each scorecard) with the write-up in `docs/benchmarks/2026-09-07-longmemeval-s-HEAD.md`; the 2026-09-03 artifacts remain committed and are superseded, not deleted | `python3 benchmarks/lme_ablation_report.py --dir docs/benchmarks/head-20260907 --floor docs/benchmarks/head-20260907/lme-floor-head.ndjson` | 2026-09-07 — HEAD 0.9745 / 0.8447 / 0.8954 against a floor of 0.9702 / 0.8298 / 0.9081; all three paired comparisons not significant (0.7905 / 0.3240 / 0.0929), where on 09-03 two of them were. The floor arm reproduces its 09-03 committed values to every digit, which is the determinism control | ❌ not yet — first-party. `chroma_baseline` (2026-09-07) adds an EXTERNAL SYSTEM to compare against, which is not the same thing as an external PARTY verifying us, and this row stays ❌ until someone outside STARGA reruns it. The prior `R@5 = 85.3` stays **RETRACTED** |
| 9 | **Published numbers are recomputable from committed raw rows** | `benchmarks/repro_verify.py` + `tests/test_repro_package.py` (which proves the verifier fails on a tampered row, metric, counter or scorecard cell) | `make repro-verify` | 2026-09-05 (7 targets, 160 checks) | ❌ not yet — the check anyone outside STARGA can run first |
| 10 | **`supersedes` edges demote their target at least as fast as `contradicts`** (bi-temporal supersession) | `tests/test_block_lineage.py::TestKindDecay`, `tests/test_typed_edges_group_h.py`, `tests/test_lineage_staleness.py` — new tests added this pass | `pytest tests/test_block_lineage.py tests/test_lineage_staleness.py tests/test_typed_edges_group_h.py -q` | see `CHANGELOG.md` | ❌ not yet — covered by repo tests, not independently reproduced |

**Deliberately not closed this pass:** a content-category decay policy (infra/status facts on a short fixed
TTL regardless of recall frequency, decision/architecture facts long-or-no TTL, credential blocks never
auto-decay) per recent research — see `ROADMAP.md`, "Governance
— content-category decay policy". `memory_tiers.py` already does recency-based TTL/LRU decay, which is a
different axis; the category taxonomy and its default TTL values are a product decision that risks silently
mis-flagging real users' data if guessed wrong, so it is left open for the maintainer rather than implemented
speculatively.

## What "10/10" requires (and what we are NOT claiming yet)

Per an external rubric, the score is gated on **external** proof, not more code:

1. **One independent reproduction** of NIAH (row 1) — the package is committed
   (`benchmarks/repro/niah/`); what's missing is a third-party issue/PR/CI fork that
   runs `make repro-verify` (or regenerates via `make repro-niah`) and reports the same
   `metrics.determinism.decision_fingerprint` (`33d5f8e6282e9844455c4d20c8701a2fe96f29cf6d33bf47278f5ab1dcc9570b`).
   (Do not diff whole files: latency does not reproduce across boxes, which is why the
   fingerprint covers the retrieval decisions and leaves timing out.) This is the single
   biggest lever and the most welcome contribution.
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
