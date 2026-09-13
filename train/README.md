# mind-mem training pipeline

These scripts build and evaluate a governance-aware memory-assistant model
from a fresh checkout of the MIND-Mem repository. They do not carry model
weights. The current default base is `Qwen/Qwen3.5-4B`; set `MM_BASE_MODEL`
explicitly when using another base and record that choice with the resulting
receipts. Availability of any other model, including Qwen3.8-4B, is not
established by this repository.

Artifacts use `MM_TRAIN_ROOT` (default
`/data/checkpoints/mm-workspace/train-output`). The corpus output can be
overridden independently with `MM_CORPUS_OUT`.

## Pipeline

1. **Harvest a deterministic corpus.**

   ```bash
   export MM_TRAIN_ROOT=/data/checkpoints/mm-workspace/train-output
   export MM_CORPUS_OUT="$MM_TRAIN_ROOT/corpus.jsonl"
   python3 train/build_corpus.py
   ```

   This reads the checked-out tool documentation, schemas, changelog, docs,
   and governance workflows. It makes no model or network call.

2. **Train an adapter or full-FT checkpoint.**

   ```bash
   MM_BASE_MODEL=Qwen/Qwen3.5-4B python3 train/train_qlora.py
   ```

   QLoRA writes `$MM_TRAIN_ROOT/adapter`. Full fine-tuning is run by
   `train/runpod_full_ft.py` (normally through `train/runpod_deploy.py`) and
   writes `$MM_TRAIN_ROOT/full-ft`. RunPod receives the canonical causal
   loader beside the training script. Its publication message names a
   configured base generically, so an `MM_BASE_MODEL` override is not
   misrepresented as Qwen3.5.

3. **Run both evaluations.**

   ```bash
   python3 train/eval_harness.py
   python3 train/eval_holdout.py
   ```

   The main report is `$MM_TRAIN_ROOT/eval_report.json`; the holdout report
   defaults to `$MM_TRAIN_ROOT/../full-ft/eval_holdout_report.json` and can be
   set with `MM_HOLDOUT_REPORT`. Both reports carry model/base and corpus
   bindings and are required by the release gate. A report records the
   measured result and its inputs; it is not proof of model availability,
   execution provenance, or a passed benchmark when a required field is
   missing.

4. **Build the model card.**

   ```bash
   python3 train/build_model_card.py
   ```

   The card is generated from the current checkpoint and evaluation receipts.
   Keep the main and holdout receipt files with the release artifacts.

5. **Upload the release artifacts only after the gate passes.**

   ```bash
   HF_TOKEN=hf_... python3 train/upload_to_hf.py
   ```

   The uploader requires the evaluation pair, tokenizer, model metadata, and
   a complete safetensors file or index with all shard files. It publishes
   the adapter when only an adapter is present, or the full-FT safetensors
   checkpoint when that is present. A write-scope token is required; the
   default cached read-only token is rejected before any upload bytes are
   sent.

## Optional GGUF conversion

`train/export_gguf.py` is a local conversion step for llama.cpp or Ollama
users. It does not create a release receipt and the uploader deliberately
omits GGUF until derived-artifact parent binding and post-quantization
evaluation exist.

```bash
MM_GGUF_SOURCE=fullft MM_BASE_MODEL=Qwen/Qwen3.5-4B \
  python3 train/export_gguf.py
```

Set `MM_GGUF_SOURCE=fullft` or `adapter` explicitly. Unknown values are
rejected. The `fullft` source is converted directly. The `adapter` source is
merged into the selected base after `adapter/adapter_config.json` has
declared the same base; a mismatch or malformed configuration is rejected
before heavyweight imports, merged-directory deletion, or conversion. The
canonical causal loader is used for the adapter merge, so the selected base
and its verified configuration follow the same loading path as training.

The converter writes a temporary F16 file and then a Q4_K_M file beneath the
training root. Successful conversion alone does not establish quality,
compatibility, provenance, or upload eligibility. Keep the output separate
from the safetensors release until those controls are implemented.

## Runtime requirements

- Python 3.10+ with `transformers`, `peft`, `bitsandbytes`, `accelerate`,
  `trl`, `datasets`, and `torch` for training.
- A suitable GPU and disk budget for the selected base and checkpoint.
- A local `llama.cpp` checkout with `convert_hf_to_gguf.py` and its quantizer
  for GGUF conversion.
- A Hugging Face token with write scope for upload. Training still requires
  access to the selected base model; this README does not assert that a model
  identifier is available.
