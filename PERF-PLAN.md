# Performance plan of record — 2026-09-11

**Goal: beat every comparable system on measured performance.** Ordered, and each step
gated on a NUMBER, because "faster" without a measurement is the class of claim this repo
spends most of its gates refusing.

## The rule this plan follows

No optimisation lands without: a baseline number, a profile naming the bottleneck, the
delta after the change, and a one-sided decision gate. A speedup nobody measured is a
story; a speedup measured once on a warm cache is a coincidence.

## P0 — THE LATENCY CLAIM IS MEASURED ON A PATH A DEFAULT INSTALL CANNOT REACH

`docs/benchmarks.md` publishes:

| Operation | Latency |
|---|---|
| Vector search (warm) | 52-64ms |

and attributes it to "Ollama GPU embeddings", with the CPU alternative recorded in the
same file as **60-300s per query**.

Measured 2026-09-11: the ollama embed leg was asking ollama for `all-MiniLM-L6-v2` — a
fastembed model name — so it 404'd on **every call** and fell through to CPU. 76 failed
round-trips in 48 seconds. The GPU path is reachable only when an operator explicitly sets
`recall.ollama_embed_model`, and nothing in a default install does.

So the published headline latency may describe a configuration almost no install has. That
is the first thing to measure, because if true it is the largest performance fact in the
product and it is currently hidden.

**Step 1 (now): measure both paths on the same corpus and queries.** Default config vs
`ollama_embed_model=mxbai-embed-large`. Report cold index, warm search, p50/p95.

**Step 2: make the fast path reachable without a manual opt-in** — when ollama is serving
a suitable embedding model, use it. The capability probe landed today already answers
"does ollama serve this model"; the missing half is choosing a served embedding model
instead of demanding the operator name it. Fail closed: never silently pick a model whose
dimensions disagree with the index.

**Step 3: correct `docs/benchmarks.md` to state which configuration each number belongs
to.** A latency table that does not name its embedder is not reproducible.

## P0 RESULT — MY HYPOTHESIS WAS WRONG, AND THE MEASUREMENT FOUND SOMETHING BIGGER

Measured both configurations on the live 2,726-block corpus, 15 recalls each:

| config | cold | p50 | p95 |
|---|---|---|---|
| DEFAULT (no `ollama_embed_model`) | 613ms | **279ms** | 313ms |
| `ollama_embed_model=mxbai-embed-large` | 613ms | **295ms** | 338ms |

**They are the same, so the embedder was never the lever.** The attestation says why:
`legs_ran: "bm25"`. The VECTOR LEG DOES NOT RUN in either configuration, so which embedder
is configured cannot matter. My P0 hypothesis is refuted and recorded as such.

Two facts survive it, and both are larger:

1. **Real default recall latency is ~280ms p50 — the published table says 52-64ms.** That
   published figure is for a leg a default install does not run at all, so the honest
   number for a default install was never in the docs.
2. The 52-64ms line describes vector search specifically. It is not wrong about vector
   search; it is presented where a reader takes it for recall latency.

## P1 — PROFILED. THE BOTTLENECK IS RE-PARSING THE CORPUS ON EVERY RECALL

`cProfile`, 15 recalls, sorted by cumulative time:

| cost | cumulative | share |
|---|---|---|
| `block_parser.parse_file` -> `parse_blocks` | **8.53s of 10.48s** | **81%** |
| `re.match` (module-level form) | 2,696,520 calls / 3.65s | — |
| `_enrich_fact_keys` | 3.05s | — |
| `re._compile` (pattern cache lookups) | 2,715,446 calls / 1.60s | — |

285 `parse_file` calls for 15 recalls: **19 corpus files re-parsed per recall, every
time.** Nothing caches the parse, so every query re-reads and re-tokenises the entire
corpus from markdown.

**Fix 1 (highest value): cache the parsed corpus, invalidated on a cheap stat.** Keyed on
`(path, mtime_ns, size)` — a stat, not a read, following the rule this codebase already
learned about off-path probes. 81% of recall time is the ceiling this removes.

**Fix 2: precompile the hot patterns.** 2.7M module-level `re.match(pattern, s)` calls each
re-look-up the compiled pattern; `re._compile` alone is 1.6s cumulative.

Each lands with a before/after number and a regression gate, not a claim.

## P1 FIX 1 — LANDED. 279.4ms -> 34.9ms p50 (8.0x), measured both sides

| | baseline | stat-keyed (WRONG) | content-hash (shipped) |
|---|---|---|---|
| p50 | 279.4ms | 32.7ms | **34.9ms** |
| p95 | 312.5ms | 38.9ms | **43.8ms** |
| cold | 613ms | 382ms | **358ms** |

**My first key was `(mtime_ns, size)` and it was wrong on both counts.** The reasoning --
"a stat, not a read", citing the off-path config probe -- does not transfer: there the READ
was the whole cost, here the read is 7.9ms and the PARSE is 81%. And it was incorrect, not
merely suboptimal: `tests/test_recall_hot_path_5_0_2.py` already carried a fixture whose
docstring names the hole ("keeps byte size AND st_mtime_ns identical ... the one class of
change size+mtime cannot see") and it caught the stale serve.

Keying on the content hash cost 39ms at first, and ~23ms of that was `str.encode()`
re-encoding 2 MB that had just been decoded. Reading BYTES, hashing bytes with blake2b
(3.1ms vs sha256's 8.3ms on this corpus) and deferring the decode to a cache MISS got the
correct version back to the incorrect version's speed.

Measured on the live corpus: 19 files, 1.96 MB; read 7.9ms, blake2b 3.1ms, sha256 8.3ms.

**34.9ms p50 is now faster than the 52-64ms the docs advertise for vector search alone.**

## P1b — PROFILE AGAIN (the 81% is gone, so the ranking has changed; do not guess)

Not guesses. `cProfile` over a realistic recall on the 2,726-block live corpus, ranked by
cumulative time. Candidates already visible from today's reading, to be confirmed or
refuted by the profile rather than assumed:

- per-call config reads on hot paths (the "silent is not free" rule: an OFF feature that
  re-reads and re-parses config per event was already measured at 1000 reads per 1000
  flag-off publishes elsewhere in this codebase);
- `_load_corpus` parsing the whole markdown corpus per operation;
- the embedding cache's hit rate on a warm workspace.

Each confirmed cost gets a fix, a before/after number, and a regression gate.

## P2 — BEAT THE COMPARISON, ON THE RECORD

`docs/benchmarks.md` already compares LoCoMo accuracy against other systems. Do the same
for LATENCY and INGEST THROUGHPUT on an identical corpus and query set, with the
competitor's own default configuration — ours must win on its default, not on a tuned one,
or the comparison is rigged in our favour and worthless.

## P3 — THE PURE-MIND HOT PATH

Already the chosen direction (the Rust/PyO3 port is marked superseded). Gated on `mindc`
capability and tracked in the MIND repo. It is the ceiling-raiser, not the first move:
P0-P2 are cheap and unblocked, and it would be foolish to port a path before knowing which
part of it costs anything.

## What is NOT in this plan, and why

The 27 open roadmap boxes are classified in `ROADMAP.md`: 5 are status lines, 10 need money
or a credential or a publish decision, 12 are blocked on a prerequisite in another repo.
None of them is a performance item. Performance is a separate axis and this file is its
plan of record.
