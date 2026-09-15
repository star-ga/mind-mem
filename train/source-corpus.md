# Source MCP contract corpus (preparation only)

`build_source_corpus.py` creates a deterministic, source-only preparation
artifact for the current MCP contract surface. It parses the checked-in
registration wiring in `src/mind_mem/mcp/server.py`, the compatibility shim
`src/mind_mem/mcp_server.py`, and `src/mind_mem/mcp/tools/*.py`. It reads those
files as UTF-8 bytes and Python ASTs; it never imports `mind_mem`, a model, the
historical corpus builder, or an evaluation module.

Run it with an explicit absolute output directory:

```console
python train/build_source_corpus.py --repo-root "$PWD" \
  --output-dir /absolute/path/to/a/new/source-corpus
```

The output directory must be empty (the command refuses to clobber files) and
must not be a symlink or contain symlink components. Source files must be
regular files inside the checkout. Per-file, source-set, docstring, message,
and record limits are recorded in the manifest. A manifest and JSONL corpus
are written only after registration and source checks pass.

There is one record per actual registered MCP symbol in registration order.
Each record carries a family key of `module:function`, the function signature,
literal docstring, registration and definition locations, and the exact
source-file SHA-256. The manifest binds the generator bytes, every source
file, output bytes/hash, record count, and the 107-symbol measured surface.
Changing a registered source changes both its binding and the generated
artifact; duplicate, missing, ambiguous, undocumented, malformed, or
symlinked inputs refuse before an artifact is published.

This is `PREPARATION_ONLY` with
`independent_evaluation: NOT_ESTABLISHED`. It is not a clean holdout, a proof
of train/evaluation separation, a trained corpus, or a model-readiness gate.
The existing corpus builders, probes, manifests, and historical artifacts are
preserved and are not imported or copied by this preparation slice.
