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

    # Full bf16 load — no quantization. H200 has plenty of room.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable     : {trainable:,} / {total:,} ({trainable / total:.2%})")

    dataset = load_dataset("json", data_files=str(CORPUS), split="train")
    print(f"examples      : {len(dataset):,}")

    cfg = SFTConfig(
        output_dir=str(OUT_DIR),
        logging_dir=str(LOG_DIR),
        # Full FT needs a lower LR than LoRA — 1e-5 to 3e-5 typical.
        learning_rate=1.5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=5,  # fewer epochs than LoRA — full FT fits faster
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,  # effective batch = 32
        bf16=True,
        logging_steps=5,
        # Disk on the pod volume (40 GB) can't hold concurrent 16 GB
        # checkpoints. Skip intermediate saves — `trainer.save_model`
        # at the end writes the single final model. Resumability is
        # sacrificed but a fresh 45-min train is cheaper than a failed
        # 2-day checkpoint strategy.
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
        packing=True,  # pack multiple examples per sequence for efficiency
        max_length=2048,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"\ntraining complete — full-FT weights saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
