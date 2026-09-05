# Repro packages

A repro package is everything a third party needs to re-run one benchmark and
check that our published number follows from it. One directory per package.

```
raw.ndjson        one JSON object per unit of work -- the evidence. Every
                  headline number for this benchmark is recomputable from this
                  file alone.
metrics.json      the numbers, produced by running the metric function over
                  raw.ndjson. Never authored by hand.
dataset.json      the exact pinned inputs, with a content hash (and, for a
                  generated dataset, the generator's own source hash).
environment.json  hardware, OS, python, dependency versions, capture time.
manifest.json     the pins -- commit, effective config + sha256, seeds, adapter,
                  embedder (name/dims/device), k, exclusion rule with counts,
                  wall clock, killed-or-crashed count -- plus the sha256 of every
                  file above and the two commands: one that regenerates the
                  package, one that verifies it.
```

`manifest.json` is the only file not covered by its own hash, because it carries
the hashes.

## Verify one, or all

```bash
python3 benchmarks/repro_verify.py package benchmarks/repro/<name>
make repro-verify        # every committed package AND every scorecard/NDJSON pair
```

The verifier re-hashes each artifact, re-derives each unit's verdict from its own
retrieved results (not from the row's `found` field), recomputes the metrics from
`raw.ndjson`, cross-checks the manifest's unit counters against the rows, and
confirms the manifest's headline is the recomputed one. Any mismatch prints both
values and exits non-zero. It repairs nothing.

A package in which *every* unit was killed or crashed is rejected as **VACUOUS**:
its metrics recompute perfectly from rows that record only failures, and would
otherwise read as a clean bill of health. An empty run is not a pass.

## Comparing two runs

Do **not** diff whole files. Latency does not reproduce across boxes, and
`environment.json` is not supposed to. Compare
`metrics.determinism.decision_fingerprint`: a sha256 over each unit id and the
ids it retrieved, in rank order, with timing left out. Two runs that agree there
made the same retrieval decisions.

## What a package does not prove

- `headline_claim: false` means the package is a machinery fixture or a subset.
  It is not a citable figure, whatever its numbers say.
- A package verifying means the published number follows from the committed
  rows. It does not mean the benchmark is a good measure of anything, and it is
  not an independent reproduction -- that requires someone outside STARGA
  running it and reporting the same fingerprint.
