# mind-mem 4B full fine-tune — Qwen3.8-4B preparation plan

**Status: preparation only. No model download, paid RunPod job, checkpoint, or
public upload has been performed.** This document replaces the old v2/v3.3
recipe, whose dispatcher-generator and validation script names no longer match
the repository.

The requested run is one new full fine-tune of an explicitly pinned Qwen3.8-4B
base over the post-Pure-MIND MIND-Mem surface. It is not a continuation of an
older `star-ga/mind-mem-4b` checkpoint. The repository does not currently
establish an official `Qwen/Qwen3.8-4B` checkpoint: the official Hugging Face
metadata check on 2026-09-14 returned HTTP 401 for that exact API path, while
official Qwen entries were available for Qwen3.8-27B and Qwen3.5-4B-Base. A
401 is recorded as **UNVERIFIED**, not as proof that the requested model is
absent. A future operator must supply and independently verify the exact model
revision before any run.

The recorded primary metadata endpoints are
[`Qwen/Qwen3.8-4B`](https://huggingface.co/api/models/Qwen/Qwen3.8-4B),
[`Qwen/Qwen3.8-27B`](https://huggingface.co/api/models/Qwen/Qwen3.8-27B), and
[`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/api/models/Qwen/Qwen3.5-4B-Base).
The first returned 401 in the local check; the latter two returned metadata.
These URLs are evidence locations for a future recheck, not download
instructions.

## Required gates

The sequence is deliberately ordered:

1. Finish and independently accept the relevant Pure-MIND port. The current
   roadmap keeps this gate open; no model run closes it.
2. Freeze the post-port source tree and generate a new corpus from independently
   reviewed source inputs. `train/build_source_corpus.py` provides a source-only
   MCP contract inventory for this preparation; it is not a complete training
   corpus or a launch gate. The existing `train/build_corpus.py` corpus is
   development data: its
   holdout-targeted harvest makes it contaminated even where exact prompt
   overlap is zero. It cannot be used as an independent evaluation set.
3. Create a locked, family-disjoint evaluation set from a separate source
   review. Record its digest and review receipt. Paraphrasing an inspected
   holdout does not restore independence.
4. Snapshot the selected base locally and write an operator-controlled base
   manifest containing the model ID, immutable 40-hex revision, license,
   tokenizer SHA-256, config SHA-256, snapshot path, source URL, and retrieval
   timestamp. The existing evaluation receipt records and verifies the bytes
   selected by an evaluation; this preparation manifest does not download or
   independently approve a model snapshot.
5. Obtain the normal spend approval marker for a fresh run tag. The existing
   `spend_guard preflight` is a required launch step. The deployer checks that
   approval matches the selected training configuration; changing that
   configuration invalidates the approval. This does not establish the remote
   source/base/path configuration or satisfy the other readiness gates.
6. Treat the existing training-readiness manifest command as the local
   preflight. It must exit successfully and report `training_eval_separation_status:
   READY`; otherwise no RunPod command is authorized. Pure-MIND acceptance,
   base availability, independent evaluation, and funding remain external
   gates until their existing authoritative checks produce receipts.

The manifest checks textual coverage of the current registered MCP surface.
That is a corpus inventory control, not model competence. It recomputes
holdout overlap and scans the current generator for holdout-targeted harvests;
caller-supplied `PASS`, `READY`, or `locked` fields are not accepted as a
replacement for those checks. It never treats historical trained-tool counts
or a previous checkpoint as proof for this new run.

## Local preparation commands

First, inventory the current MCP source contracts without importing the runtime,
historical corpus, evaluation modules or a model:

```bash
: "${MM_CONTRACT_ROOT:?export MM_CONTRACT_ROOT to a new absolute directory}"
python3 train/build_source_corpus.py --repo-root "$PWD" \
  --output-dir "$MM_CONTRACT_ROOT"
```

The command requires an empty output directory outside the checkout. Its
manifest binds source, generator and output hashes and reports
`PREPARATION_ONLY` / `independent_evaluation: NOT_ESTABLISHED`. The current
source resolves 107 registered tools into 107 contract families. Unsupported
registration forms refuse rather than silently omit or rename tools. See
[the source-corpus contract](../train/source-corpus.md).

The commands below reproduce the **historical development-corpus diagnostic**
from an exact frozen checkout. They do not produce an approved training input.
Use a fresh output directory:

```bash
set -eu
: "${MM_BASE_MODEL:?export MM_BASE_MODEL=Qwen/Qwen3.8-4B for this preparation}"
: "${MM_TRAIN_ROOT:?export MM_TRAIN_ROOT to a fresh run directory}"
test "$MM_BASE_MODEL" = "Qwen/Qwen3.8-4B" || {
  echo "refusing a non-Qwen3.8-4B base: $MM_BASE_MODEL" >&2
  exit 2
}
mkdir -p -- "$MM_TRAIN_ROOT"
export MM_BASE_MODEL MM_CORPUS="$MM_TRAIN_ROOT/corpus.jsonl"
test ! -e "$MM_CORPUS" || {
  echo "refusing to overwrite existing corpus: $MM_CORPUS" >&2
  exit 2
}
test ! -e "$MM_TRAIN_ROOT/training-readiness-manifest.json" || {
  echo "refusing to overwrite existing readiness manifest" >&2
  exit 2
}
python3 train/build_corpus.py --output "$MM_CORPUS"
python3 train/training_readiness_manifest.py \
  --corpus "$MM_CORPUS" \
  --output "$MM_TRAIN_ROOT/training-readiness-manifest.json"
```

The current development corpus is expected to make the readiness manifest exit
nonzero. That is a useful refusal. Do not change the holdout strings or delete
targeted examples to make it green; produce a fresh locked evaluation artifact
after the source freeze. The repository's existing `tests/test_training_readiness.py`
covers both the targeted-paraphrase and exact-overlap refusals.

## Post-training evaluation and release checks

After a successful full-FT run has produced the exact checkpoint directory,
run the existing evaluation and release checks. Do not run this block before
training: it requires the new full-FT output. There is no separate Qwen3.8 gate
command that accepts hand-authored readiness flags:

```bash
set -eu
: "${MM_BASE_MODEL:?export MM_BASE_MODEL=Qwen/Qwen3.8-4B for evaluation}"
: "${MM_TRAIN_ROOT:?export MM_TRAIN_ROOT to the trained run directory}"
: "${MM_CORPUS:?export MM_CORPUS to the exact evaluated corpus.jsonl}"
: "${MM_FULLFT_DIR:?export MM_FULLFT_DIR to the exact full-ft checkpoint directory}"
: "${MM_HOLDOUT_REPORT:?export MM_HOLDOUT_REPORT to a new holdout receipt path}"
test "$MM_BASE_MODEL" = "Qwen/Qwen3.8-4B" || {
  echo "refusing evaluation of a non-Qwen3.8-4B base: $MM_BASE_MODEL" >&2
  exit 2
}
test -f "$MM_CORPUS"
test -d "$MM_FULLFT_DIR"
test -f "$MM_FULLFT_DIR/model.safetensors" || test -f "$MM_FULLFT_DIR/model.safetensors.index.json" || {
  echo "refusing evaluation without a full-FT weight manifest: $MM_FULLFT_DIR" >&2
  exit 2
}
export MM_BASE_MODEL MM_TRAIN_ROOT MM_CORPUS MM_FULLFT_DIR MM_HOLDOUT_REPORT
# The card/uploader reads MM_WEIGHTS_DIR; it does not consume MM_FULLFT_DIR.
export MM_WEIGHTS_DIR="$MM_FULLFT_DIR"
test ! -e "$MM_TRAIN_ROOT/eval_report.json"
test ! -e "$MM_HOLDOUT_REPORT"
python3 train/eval_harness.py
python3 train/eval_holdout.py
python3 train/build_model_card.py
python3 train/upload_to_hf.py --help
```

`train/eval_receipt.py` is an import-only support module, not a command-line
preflight. The evaluation and upload entrypoints use its existing content
addressing, complete-weight closure, tokenizer/base binding, and receipt
verification functions. A future operator must supply an immutable base
manifest with config/tokenizer/weights/license/provenance, but no new wrapper
is allowed to turn an operator-written manifest into a training approval.

## RunPod execution after approval

The existing `train/runpod_full_ft.py` is the actual full-FT entrypoint. It
accepts `MM_BASE_MODEL`, `MM_TRAIN_ROOT`, and `MM_CORPUS`; the old Qwen3.5
fallback remains a compatibility default and must not be used for this run.
Copy the frozen checkout, the corpus, the verified base snapshot/manifest, and
all evaluation scripts as one source-bound bundle. The actual deployment
entrypoint is `train/runpod_deploy.py`; its approval enforcement and remote
source/base/path configuration must pass launch-readiness review before an
executable launch recipe can be supplied. Setting local training paths alone
does not configure the deployer's remote paths. No provisioning command is
part of these preparation steps.

The training mode is full fine-tuning: no LoRA or quantization. Exact batch,
sequence, optimizer, and GPU settings remain operator inputs after the selected
base's configuration and memory requirements are measured. This document does
not invent a current RunPod price or capacity claim.

After training, retain the source, base, tokenizer, config, corpus, and output
manifests together. The bounded environment block above binds every evaluation
and model-card command to the explicit corpus and full-FT checkpoint paths.
`MM_BASE_MODEL` prevents the legacy adapter path from silently selecting another
base; a full-FT evaluation loads the checkpoint directly. Its receipt does not
independently establish which base produced those weights. Retain the separate
training/base provenance artifacts for that claim.
A release requires both evaluations to bind the same source/base/checkpoint and
the locked holdout to remain untouched. A score, tool-call rate, latency number,
or model-card generation does not establish general competence or independent
observation. Upload remains a separate operator decision through
`train/upload_to_hf.py` after its existing gates pass.

## Explicit non-goals and open inputs

- No current official Qwen3.8-4B availability or revision is asserted.
- The historical 83-tool checkpoint and its 133/133 evaluation are preserved
  as historical facts; neither is evidence for this new model.
- The current live registry count is measured by the preflight at execution
  time; it is not hardcoded here.
- The current holdout-targeted corpus is development/contaminated data, not an
  independent eval split.
- Pure-MIND completion, the locked post-port corpus/eval, immutable base
  manifest, operator funding approval, GPU selection, and any training result
  remain open inputs.
