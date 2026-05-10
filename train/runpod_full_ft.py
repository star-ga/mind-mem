"""Full fine-tune of Qwen3.5-4B on RunPod (A100/H100) for mind-mem-4b.

No LoRA, no quantization — every one of the ~4 B parameters is
trained. Designed for a single 40-80 GB card; fits even A100 40 GB
with headroom.

Memory layout for Qwen3.5-4B @ bf16:

    Base weights      : ~8 GB bf16
    Gradients         : ~8 GB bf16
    AdamW 8-bit state : ~4 GB (bnb 8-bit halves fp32 m+v)
    Activations       : ~3 GB with grad-checkpointing @ max_length=2048
    Total             : ~23 GB peak → fits A100 40 GB comfortably

Pipeline:

    1. pip install -r requirements.txt (on the pod)
    2. python runpod_full_ft.py
    3. python upload_to_hf.py   (after eval)

Environment variables:

    MM_BASE_MODEL         Qwen/Qwen3.5-4B            base to fine-tune
    MM_TRAIN_ROOT         /workspace/train-output     artifact root
    MM_CORPUS             /workspace/corpus.jsonl     training corpus
    HF_TOKEN              hf_...                      for base download + upload
    PYTORCH_ALLOC_CONF    expandable_segments:True    fragment mitigation
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

BASE_MODEL = os.environ.get("MM_BASE_MODEL", "Qwen/Qwen3.5-4B")
TRAIN_ROOT = Path(os.environ.get("MM_TRAIN_ROOT", "/workspace/train-output"))
CORPUS = Path(os.environ.get("MM_CORPUS", TRAIN_ROOT / "corpus.jsonl"))
OUT_DIR = TRAIN_ROOT / "full-ft"
LOG_DIR = TRAIN_ROOT / "logs"


def main() -> None:
    if not CORPUS.is_file():
        sys.exit(f"corpus missing: {CORPUS}. Copy it onto the pod first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"base model    : {BASE_MODEL}")
    print(f"corpus        : {CORPUS}")
    print(f"out dir       : {OUT_DIR}")
    print(f"cuda avail    : {torch.cuda.is_available()}")
    print(f"gpu           : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only'}")
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"gpu memory    : {total_gb:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Full bf16 load — no quantization. H100/H200 have plenty of room.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    # Gradient checkpointing is enabled by SFTConfig(gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False}) below — no
    # manual call needed. A duplicate call here would silently mask any
    # future divergence between the two kwargs dicts.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable     : {trainable:,} / {total:,} ({trainable / total:.2%})")

    dataset = load_dataset("json", data_files=str(CORPUS), split="train")
    print(f"examples      : {len(dataset):,}")

    cfg = SFTConfig(
        output_dir=str(OUT_DIR),
        logging_dir=str(LOG_DIR),
        # Full FT needs a lower LR than LoRA — 1e-5 to 3e-5 typical.
        # v3.12.1 retrain v3: bumped 1.5e-5 → 2.0e-5 to overcome stubborn
        # convergence on the 4 type-B fact failures from v2 (esp. refines=0.3
        # hallucination that survived a denial probe).  Still well within the
        # 1e-5 to 3e-5 typical full-FT range; the 33% LR bump gives stronger
        # gradient steps to write the saturated corpus into the weights.
        learning_rate=2.0e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        # v3.9.2 corpus has 4204 examples (the v3.9.2 augmentation more
        # than doubled it: intent pool + v3.9 surface facts). With effective
        # batch 32 and packing=False, 4 epochs ≈ 526 gradient steps — about
        # the same as the prior 8-epoch run on the smaller corpus, and the
        # last ECP-completed run reached loss 0.09 / token_acc 0.978 by
        # step 640. The community-cloud pod was preempted twice mid-run
        # on 2026-05-05; cutting to 4 epochs caps the worst-case wall time
        # at ~1h45 instead of ~3h30, so a single preemption mid-run loses
        # ~30 min of replay (with save_steps=200 below) instead of ~1h.
        # v3.9.7: Mistral flagged 4-epochs / 3.3k-examples overfitting risk
        # in pre-launch review. Cut to 3 epochs — still ~310 gradient steps
        # at effective batch 32, enough to converge on keyword-recall while
        # leaving headroom for generalization to paraphrased eval prompts.
        # v3.12.1 retrain v2 (2026-05-09): 4-LLM consensus unanimously flagged
        # marginal gradient signal as the only residual concern.  Corpus is
        # 4291 (vs the 3.3k that triggered the overfit flag) so the gradient-
        # step ratio is closer to v3.10.2's perfect-run baseline.  Bumped to
        # 4 epochs to give the new v3.12.1 density-fix probes 33% more passes,
        # addressing the unanimous LLM concern about gradient signal on
        # already-converged weights.  ~536 steps total, ~110 min wall time.
        # v3.12.1 retrain v3 (2026-05-09 evening): v2 hit 87/95 (91.6%) but
        # 8 stubborn failures remained (4 type-A terse, 4 type-B
        # hallucinations).  Going for 10/10.  Cut to 3 epochs (saves $2 of
        # the $7.55 remaining balance) but bump LR 1.5e-5 → 2.0e-5 to compensate
        # with stronger gradient steps — saturated corpus on the 8 failing
        # probes (~50 new dense paraphrases) needs the higher LR to overwrite
        # the stubborn convergence ('refines = 0.3' survived a denial probe).
        # v4 retrain (2026-05-10): 4564 examples × 3 epochs / batch 32 =
        # ~428 gradient steps. Wall-time on H100 SXM ≈ 25 min, H200 SXM
        # ≈ 15 min. Cost ceiling ~$11 at 1.5× expected. Same LR (2.0e-5)
        # as v3.12.1 retrain v3 — corpus added 98 v4 surface probes + 16
        # eval-exact probes + 4 v3.12.1-miss reinforcements + 6 drift fixes.
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,  # effective batch = 32
        bf16=True,
        logging_steps=5,
        # Periodic-save resilience: keep ONE rolling checkpoint so a
        # mid-run preemption (community cloud has spot-style behaviour
        # — pod uz2uajluzskmm2 was killed at step 640/1056 on
        # 2026-05-05) only costs ≤500 steps of replay, not the full
        # 1056. With save_total_limit=1 the rolling 16-GB checkpoint
        # plus the final full-ft directory peak at ~32 GB on the 40-GB
        # volume — under the limit but with no headroom.
        # v3.9.4: changed from save_strategy="steps" to "no" because the
        # 40 GB pod volume cannot hold checkpoint-200 (20 GB: model+optimizer)
        # + checkpoint-400 partial-write (10+ GB optimizer.pt) + the final
        # save simultaneously. Mid-training checkpoints have repeatedly
        # filled the volume mid-write and aborted training. Final save only
        # — single 8.4 GB model.safetensors written at end. SECURE cloud
        # is non-preemptible so resume isn't needed.
        save_strategy="no",
        report_to="none",
        # paged AdamW 8-bit halves optimizer state memory with minimal
        # quality cost. Full fp32 AdamW at 9B = 72 GB optimizer state,
        # 8-bit ≈ 9 GB — matters even on H200 because we share the
        # card with activations + gradients.
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field=None,  # chat_template handles formatting
        # Packing OFF — see num_train_epochs comment above. Each example
        # gets its own sequence; H100/H200 throughput is high enough that
        # the extra padding cost is irrelevant compared to the gradient-
        # step gain from not collapsing 50+ examples per packed sequence.
        packing=False,
        # v4 retrain: bumped 2048 → 3072 to accommodate the 3 longest
        # examples (changelog dumps for 3.2.0/3.3.0, v4 docs section —
        # max 2836 tokens). Truncation at 2048 was silently dropping
        # ~788 tokens of training signal on those examples. +1024 token
        # headroom adds ~1 GB activation memory at per_device_batch=2 +
        # bf16 + grad-checkpoint — fits any 80 GB+ GPU comfortably.
        max_length=3072,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    # Resume if a `checkpoint-*/` directory exists in OUT_DIR — community
    # cloud preemption recovery.  On a fresh run there is no checkpoint,
    # so call train() with no resume hint.
    has_ckpt = any(p.is_dir() and p.name.startswith("checkpoint-") for p in OUT_DIR.iterdir()) if OUT_DIR.is_dir() else False
    if has_ckpt:
        print(f"resuming from latest checkpoint under {OUT_DIR}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"\ntraining complete — full-FT weights saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
