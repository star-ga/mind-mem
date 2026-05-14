"""Local QLoRA fallback on RTX 3080 (10GB VRAM).

Identical recipe to the Kaggle script, but loads from the local retry-2e
weights directly (no HF round-trip) and runs entirely off-cloud.

Memory budget on 10GB 3080:
  4-bit base:           ~2.5 GB
  LoRA rank 32:         ~50  MB
  paged_adamw_8bit:     ~100 MB
  KV cache + activations: ~3-5 GB
  Total estimated:      ~6-8 GB  (fits with grad checkpointing)

Output: LoRA adapter at /data/checkpoints/mm-workspace/lora-v4.1.0/
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE = Path("/data/checkpoints/mm-workspace/full-ft.retry2e-109of109+18of22")
CORPUS = Path("/data/checkpoints/mm-workspace/train-output/corpus.jsonl")
ADDENDUM = Path("/data/checkpoints/mm-workspace/train-output/corpus-addendum-v4.1.1.jsonl")
OUTPUT_DIR = Path("/data/checkpoints/mm-workspace/lora-v4.1.3")
ADDENDUM_UPWEIGHT = 32  # r4 (v4.1.1): 26 addendum (18 r3 + 8 r4 KernelKind) * 32 = 832 weighted samples (~13% mass)

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

LR = 1.5e-4  # lowered from 2e-4 to reduce catastrophic forgetting on existing facts
EPOCHS = 3
BATCH = 1
GRAD_ACCUM = 16
MAX_SEQ_LEN = 1024
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01

def main():
    print(f"CUDA: {torch.version.cuda}  devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  "
              f"({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)")

    print(f"\nLoading tokenizer from {BASE}")
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading 4-bit base from {BASE}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    def _load(p: Path) -> list[dict]:
        rows = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    corpus = _load(CORPUS)
    addendum = _load(ADDENDUM)
    combined = corpus + (addendum * ADDENDUM_UPWEIGHT)
    print(f"corpus={len(corpus)}  addendum={len(addendum)}  upweight={ADDENDUM_UPWEIGHT}  combined={len(combined)}")

    def _format(row):
        text = tok.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    ds = Dataset.from_list(combined).map(_format, remove_columns=["messages"])

    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        max_length=MAX_SEQ_LEN,
        packing=False,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg, processing_class=tok)
    print("starting QLoRA training on RTX 3080...")
    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"\n✓ Adapter saved to {OUTPUT_DIR}")
    print("Next: python3 train/merge_and_eval_v4.1.0.py "
          f"--adapter {OUTPUT_DIR} --ship-dir /data/checkpoints/mm-workspace/full-ft.v4.1.0-candidate")


if __name__ == "__main__":
    main()
