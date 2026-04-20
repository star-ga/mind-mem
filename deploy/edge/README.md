# mind-mem-edge — single-binary distribution (v4.0 prep)

Self-contained `mind-mem-edge` binary built with PyOxidizer.
Runs on on-device agents (laptops, phones via Termux, embedded)
with no pip install step. Proxies recall / governance calls up to
a central mind-mem cluster when connected; falls back to local
Markdown+SQLite when offline.

## Build

```bash
pip install pyoxidizer
cd deploy/edge
pyoxidizer build --release
# Output: build/<target>/release/install/mind-mem-edge
```

Development (CPython, fast rebuild):

```bash
pyoxidizer build
build/x86_64-unknown-linux-gnu/debug/install/mind-mem-edge recall "hi"
```

## What's embedded

- mind-mem core: block_store, recall, governance, cache,
  retrieval_trace, feature_gate, evidence_bundle, session_boost,
  truth_score.
- Python 3.12 interpreter (single binary, zero external deps).
- Markdown backend (MarkdownBlockStore) — no Postgres needed.

## What's opt-in (extended profile)

Heavy features stay cluster-side by default to keep the edge binary
under 40 MB:

- Postgres backend
- Vector / ONNX / sentence-transformers
- Reranker ensemble (BGE)
- LLM-backed query decomposition

Operators that need them build with the `extended` profile:

```bash
pyoxidizer build --release --var profile extended
```

## Edge daemon mode

```bash
mind-mem-edge daemon --upstream https://mind-mem.internal:8080 \
                     --cache-dir ~/.mind-mem-edge
```

Runs as a local agent: recall queries hit the upstream cluster when
online, fall back to the local Markdown snapshot when not. Any local
writes are queued and flushed upstream on reconnect (see
``streaming.StreamingIngestQueue`` for the back-pressure primitives
this reuses).

## Target platforms

PyOxidizer supports:

- Linux x86_64 / aarch64
- macOS arm64 / x86_64
- Windows x86_64

Mobile is not a target — iOS / Android devices reach mind-mem via
the cluster REST API, not via the edge binary.

## Size budget

| Profile | Target binary size |
|---|---|
| default | < 40 MB |
| extended | < 180 MB (pulls in ONNX + sentence-transformers) |

Track actual size via CI: `scripts/measure_edge_binary.sh`. A
regression > 10% fails the release gate.
