# MIND kernels and pipeline configuration

This directory contains two kinds of `.mind` files: eight MIND compiler sources
and 18 INI-style pipeline configuration files. Configuration files are read by
Python; they are not compiler inputs.

mind-mem currently retains its Python implementations and an optional C scoring
library. The MIND sources are migration work: no complete compiler-to-shared-library
replacement of that scoring library has been verified. Installing the compiler
alone does not enable a native MIND backend.

## Compiler sources

| File | Purpose |
| --- | --- |
| `bm25.mind` | BM25F scoring and score boosts |
| `rrf.mind` | Reciprocal rank fusion |
| `reranker.mind` | Date, category and negation reranking |
| `abstention.mind` | Entity overlap and confidence |
| `ranking.mind` | Weighted ranking and threshold mask prototype |
| `importance.mind` | Importance score prototype |
| `category.mind` | Category affinity and assignment |
| `prefetch.mind` | Prefetch signal scoring |

The prototypes do not yet establish parity with the serving implementations.
For example, `ranking.mind` uses a sigmoid threshold approximation; the C
`top_k_mask` selects a fixed number of entries. Their names do not imply identical
behavior or interchangeable ABIs.

For compiler development, verify one source at a time:

```bash
mindc mind/rrf.mind --verify-only
```

Verification is separate from native emission and execution. A source may pass
verification while shared-library emission rejects its tensor parameter ABI.
The CLI's shared-library option is `--emit-shared OUTPUT`, with one source input;
`--emit=shared` and a `mind/*.mind` input glob are not supported build recipes.
Keep the existing implementation until the replacement exports the required
symbols and passes actual consumer, numerical parity and performance gates.

## Pipeline configuration

The configuration files are `adversarial`, `answer`, `cognitive`, `cross_encoder`,
`ensemble`, `evidence`, `governance`, `graph`, `hybrid`, `intent`, `query_plan`,
`recall`, `rerank`, `rm3`, `session`, `temporal`, `trajectory` and `truth`, each with
the `.mind` extension. See [configuration documentation](../docs/mind-kernels.md).

## Existing native bridge

[`src/mind_mem/mind_ffi.py`](../src/mind_mem/mind_ffi.py) loads the optional
`libmindmem` library with `ctypes`. Its current symbols and argument declarations
are the consumer ABI. [`lib/kernels.c`](../lib/kernels.c) provides C implementations;
a library built from that file is a C backend, not evidence of pure MIND migration.
Current wheels do not ship this shared library. The supported Python path remains
available when no compatible native library is present.
