# Pipeline configuration and MIND language sources

The `.mind` extension has two uses in this repository. Of the 26 files in
`mind/`, 18 are INI-style pipeline configuration and eight are MIND-language
compiler sources. The [inventory](../mind/README.md) lists both groups.

## Configuration

Configuration files contain sections and key/value assignments, for example
[`mind/hybrid.mind`](../mind/hybrid.mind):

```ini
[fusion]
rrf_k = 60
bm25_weight = 1.0
vector_weight = 1.0
```

[`load_kernel_config()`](../src/mind_mem/mind_ffi.py) reads this format. It
converts integers, floats, booleans and comma-separated lists into Python
values. The Python scoring and retrieval code consumes those values; configuration
files are not inputs to `mindc`. Workspace overrides use the same configuration
format. See [configuration guidance](mind-kernels.md).

## Compiler sources

The eight source files are `abstention`, `bm25`, `category`, `importance`,
`prefetch`, `ranking`, `reranker` and `rrf`, each ending in `.mind`. They contain
MIND language syntax such as `import std.tensor;` and `fn rrf_fuse(...)`.
`load_kernel_config()` returns an empty configuration for these sources; it does
not execute them.

These files are migration prototypes. They do not establish a working native
MIND scoring backend. Current compiler probes can fail during verification or
shared-library emission, and a successfully emitted artifact would still need
to pass the existing consumer ABI and numerical parity gates. For example, the
MIND ranking prototype's sigmoid threshold is not equivalent to the C backend's
fixed-count top-k selection.

The existing native bridge is implemented in
[`src/mind_mem/mind_ffi.py`](../src/mind_mem/mind_ffi.py), with C implementations in
[`lib/kernels.c`](../lib/kernels.c). The Python path remains supported while the
MIND replacement is incomplete. See [migration and build status](../mind/README.md).

## Language and ecosystem references

The language compiler lives in [star-ga/mind](https://github.com/star-ga/mind),
with its specification in [star-ga/mind-spec](https://github.com/star-ga/mind-spec).
Use those sources for language syntax and supported compiler features.
A `.mind` extension, successful parsing, or the presence of source in a wheel
is not proof of native execution, complete Rust independence, or performance.

Other consumers include [mind-inference](https://github.com/star-ga/mind-inference),
[mind-nerve](https://github.com/star-ga/mind-nerve) and
[rfn-mind](https://github.com/star-ga/rfn-mind). Their source, executable coverage
and migration progress must be checked per project.

## Configuration rename

A future rename of configuration files to `.mindcfg` would need coordinated
updates to configuration lookup, MCP tools, workspace compatibility and wheel
data paths. This rename has not shipped. Existing `.mind` configuration paths
remain the compatibility contract; do not rename individual files in isolation.
