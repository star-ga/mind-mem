#!/usr/bin/env python3
"""
Mind7B: Purpose-trained 7B model for mind-mem memory operations.
UNSLOTH QLoRA fine-tuning on RunPod A100.

Tasks trained:
  - Entity extraction (text → JSON entities)
  - Fact extraction (text → JSON facts)
  - Observation compression (blocks → focused observations)
  - LLM reranking (query + candidates → relevance scores)
  - Governance analysis (evidence assessment)
  - Contradiction detection (block pairs → conflict analysis)
  - Axis-aware retrieval classification
  - Intent classification (9-type router)

Base: Qwen/Qwen3.5-7B
Method: UNSLOTH FastLanguageModel + QLoRA + SFTTrainer
Output: star-ga/mind7b on HuggingFace (PUBLIC)

Run: python3 train_mind7b_runpod.py
"""
import os
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "unsloth", "trl", "datasets", "huggingface_hub"], check=True)

import json, time, torch
from huggingface_hub import login, hf_hub_download

# === CONFIG ===
MODEL_NAME = "Qwen/Qwen3.5-4B"
DATASET_REPO = "star-ga/mind7b-training"
DATASET_FILE = "mind7b_train.jsonl"
OUTPUT_DIR = "/workspace/mind7b"
HF_WRITE_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "star-ga/mind7b"

MAX_SEQ_LENGTH = 1024
EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_RANK = 16
BATCH_SIZE = 4  # 7B fits larger batches on A100
GRAD_ACCUM = 4

login(token=HF_WRITE_TOKEN, add_to_git_credential=False)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === GPU INFO ===
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name}, {mem:.1f}GB")

# === DOWNLOAD DATASET ===
print(f"\nDownloading dataset from {DATASET_REPO}...")
dataset_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename=DATASET_FILE,
    repo_type="dataset",
    cache_dir="/workspace/hf_cache",
)
print(f"Dataset downloaded: {dataset_path}")

# === LOAD MODEL (UNSLOTH) ===
from unsloth import FastLanguageModel

print(f"\nLoading {MODEL_NAME} in 4-bit via UNSLOTH...")
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)
print(f"Loaded in {time.time()-t0:.0f}s")

# === LORA ===
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"LoRA: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")

# === DATASET ===
from datasets import Dataset

examples = []
with open(dataset_path) as f:
    for line in f:
        d = json.loads(line)
        if "messages" in d:
            parts = []
            for msg in d["messages"]:
                role = msg["role"]
                content = msg["content"]
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            text = "\n".join(parts)
            examples.append({"text": text})

dataset = Dataset.from_list(examples)
print(f"Dataset: {len(examples)} examples")

# === TRAIN ===
from trl import SFTTrainer
from transformers import TrainingArguments

tokenizer.model_max_length = MAX_SEQ_LENGTH
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

total_steps = len(dataset) * EPOCHS // (BATCH_SIZE * GRAD_ACCUM)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=5,
    optim="adamw_8bit",
    save_strategy="epoch",
    save_total_limit=2,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
    max_seq_length=MAX_SEQ_LENGTH,
)

print(f"\nTraining Mind7B: {EPOCHS} epochs, {total_steps} steps")
print(f"  rank={LORA_RANK}, lr={LEARNING_RATE}, seq_len={MAX_SEQ_LENGTH}")
print(f"  batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"\nTraining complete! {elapsed/60:.1f} min")

# === SAVE LORA ===
lora_dir = f"{OUTPUT_DIR}/lora"
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
print(f"LoRA saved to: {lora_dir}")

# === MERGE + SAVE FULL MODEL ===
print("\nMerging LoRA into base model...")
merged_dir = f"{OUTPUT_DIR}/merged"
model.save_pretrained_merged(merged_dir, tokenizer)
print(f"Merged model saved to: {merged_dir}")

# === QUANTIZE TO GGUF (Q4_K_M for RTX 3080) ===
print("\nQuantizing to GGUF Q4_K_M (for local deployment)...")
gguf_dir = f"{OUTPUT_DIR}/gguf"
model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
print(f"GGUF saved to: {gguf_dir}")

# === PUSH TO HUGGINGFACE ===
print(f"\nPushing to HuggingFace: {HF_REPO}...")
try:
    model.push_to_hub(HF_REPO)
    tokenizer.push_to_hub(HF_REPO)
    print(f"LoRA pushed to {HF_REPO}")
except Exception as e:
    print(f"Push failed: {e}")

# Push merged model
try:
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path=merged_dir,
        repo_id=HF_REPO,
        path_in_repo="merged",
    )
    print(f"Merged model pushed to {HF_REPO}/merged")
except Exception as e:
    print(f"Merged push failed: {e}")

# Push GGUF
try:
    import glob
    gguf_files = glob.glob(f"{gguf_dir}/*.gguf")
    for gf in gguf_files:
        api.upload_file(
            path_or_fileobj=gf,
            path_in_repo=os.path.basename(gf),
            repo_id=HF_REPO,
        )
    print(f"GGUF pushed to {HF_REPO}")
except Exception as e:
    print(f"GGUF push failed: {e}")

print(f"""
+==========================================+
|  Mind7B Training Complete!               |
|  Time: {elapsed/60:.0f} min                          |
|  Examples: {len(examples):<30d}|
|  LoRA: {lora_dir:<34s}|
|  Merged: {merged_dir:<32s}|
|  GGUF: {gguf_dir:<34s}|
|  HF: {HF_REPO:<36s}|
+==========================================+
""")
