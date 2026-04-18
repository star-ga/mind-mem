"""QLoRA fine-tune for mind-mem-4b on the harvested corpus.

Base model:  Qwen/Qwen3.5-4B  (fits in 10 GB VRAM with INT4 + gradient
             checkpointing + max_length=768; tighter than the old 7B
             config but matches the HF repo's qwen3.5 lineage).

LoRA config: r=16, alpha=32, dropout=0.05, target=all linear.
Training:    3 epochs, per_device_batch_size=1, grad_accum_steps=16,
             lr=2e-4 with cosine decay + 3% warmup, bf16.

Output:      /home/n/mm-train-output/adapter/   (PEFT adapter files)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

BASE_MODEL = os.environ.get("MM_BASE_MODEL", "Qwen/Qwen3.5-4B")
# All training artifacts live on /data (916 GB, 303 GB free) —
# / is a 468 GB partition that hits 100% fast when HF caches land
# on it (4B base + F16 GGUF + Q4 GGUF + merged = ~35 GB).
_BASE = Path(os.environ.get("MM_TRAIN_ROOT", "/data/checkpoints/mm-workspace/train-output"))
CORPUS = _BASE / "corpus.jsonl"
OUT_DIR = _BASE / "adapter"
LOG_DIR = _BASE / "logs"


def main() -> None:
    if not CORPUS.is_file():
        sys.exit(f"corpus missing: {CORPUS}. run build_corpus.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"base model : {BASE_MODEL}")
    print(f"corpus     : {CORPUS}")
    print(f"out dir    : {OUT_DIR}")
    print(f"cuda avail : {torch.cuda.is_available()}")
    print(f"gpu name   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    # QLoRA 4-bit config keeps the 7B base at ~4.5 GB VRAM, leaving
    # headroom for optimizer state + activations on a 10 GB card.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Skip peft.prepare_model_for_kbit_training — the default path casts
    # every frozen parameter to fp32 (~28 GB for a 7B model), which
    # blows the 10 GB 3080 budget. Instead we keep the base in its
    # native 4-bit state, manually enable gradient checkpointing, and
    # disable grad on all frozen params. LoRA adapter weights (the only
    # trainable tensors) live in bf16 regardless.
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(CORPUS), split="train")

    sft_config = SFTConfig(
        output_dir=str(OUT_DIR),
        logging_dir=str(LOG_DIR),
        num_train_epochs=6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field=None,  # chat_template handles formatting
        packing=False,
        max_length=768,  # Qwen3.5-4B is tighter on 10 GB VRAM than 7B
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"\ntraining complete — adapter saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
